"""CPU reference for 4-tile 2-projection Q4NX GEMV (M=32/tile, K=1024)."""

import numpy as np
from ml_dtypes import bfloat16


M_PER_TILE = 32
NUM_TILES = 4
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
NUM_PROJECTIONS = 2
TOTAL_OUTPUT = NUM_TILES * M_PER_TILE * NUM_PROJECTIONS


def pack_q4nx_chunk(scales, zeros, int4_data, M, K_chunk, group_size=32):
    """Pack one Q4NX chunk: [scales][zeros][int4_payload]."""
    groups_per_row = K_chunk // group_size
    assert scales.shape == (M, groups_per_row)
    assert zeros.shape == (M, groups_per_row)
    assert int4_data.shape == (M, K_chunk)
    assert int4_data.max() <= 15

    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(M):
        for col in range(0, K_chunk, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))

    return np.frombuffer(bytes(packed), dtype=np.uint8)


def pack_tile_projection_weights(scales_full, zeros_full, int4_full):
    """Pack weights for one tile, one projection: K=1024 → 4 sequential chunks."""
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunks = []
    for c in range(NUM_CHUNKS):
        col_start = c * K_CHUNK
        col_end = col_start + K_CHUNK
        g_start = c * groups_per_row
        g_end = g_start + groups_per_row
        chunk = pack_q4nx_chunk(
            scales_full[:, g_start:g_end],
            zeros_full[:, g_start:g_end],
            int4_full[:, col_start:col_end],
            M_PER_TILE, K_CHUNK, GROUP_SIZE,
        )
        chunks.append(chunk)
    return np.concatenate(chunks)


def pack_all_weights(all_scales, all_zeros, all_int4):
    """Pack weight BO: [tile0_proj0 | tile0_proj1 | tile1_proj0 | ... | tile3_proj1].

    all_scales[tile][proj] shape: (M_PER_TILE, K//GROUP_SIZE)
    all_zeros[tile][proj] shape: (M_PER_TILE, K//GROUP_SIZE)
    all_int4[tile][proj] shape: (M_PER_TILE, K)
    """
    parts = []
    for tile in range(NUM_TILES):
        for proj in range(NUM_PROJECTIONS):
            packed = pack_tile_projection_weights(
                all_scales[tile][proj],
                all_zeros[tile][proj],
                all_int4[tile][proj],
            )
            parts.append(packed)
    return np.concatenate(parts)


def q4nx_matvec_single(packed_bytes, activation):
    """Compute M_PER_TILE x K Q4NX matvec for one tile, one projection."""
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    accum = np.zeros(M_PER_TILE, dtype=np.float32)

    for c in range(NUM_CHUNKS):
        chunk_data = packed_bytes[c * chunk_bytes:(c + 1) * chunk_bytes]
        scale_bytes = M_PER_TILE * groups_per_row * 2
        zero_bytes = M_PER_TILE * groups_per_row * 2
        data_offset = scale_bytes + zero_bytes

        scales = np.frombuffer(
            chunk_data[:scale_bytes], dtype=bfloat16
        ).reshape(M_PER_TILE, groups_per_row)
        zeros = np.frombuffer(
            chunk_data[scale_bytes:data_offset], dtype=bfloat16
        ).reshape(M_PER_TILE, groups_per_row)
        int4_raw = chunk_data[data_offset:]

        weights_u4 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
        for row in range(M_PER_TILE):
            for col in range(0, K_CHUNK, 2):
                b = int(int4_raw[row * (K_CHUNK // 2) + col // 2])
                weights_u4[row, col] = b & 0x0F
                weights_u4[row, col + 1] = (b >> 4) & 0x0F

        weights_f32 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
        for row in range(M_PER_TILE):
            for g in range(groups_per_row):
                s = float(scales[row, g])
                z = float(zeros[row, g])
                c0 = g * GROUP_SIZE
                c1 = c0 + GROUP_SIZE
                weights_f32[row, c0:c1] = (weights_u4[row, c0:c1] - z) * s

        act_chunk = activation[c * K_CHUNK:(c + 1) * K_CHUNK].astype(np.float32)
        accum += weights_f32 @ act_chunk

    return accum.astype(bfloat16)


def projection_fabric_reference(packed_all, activation):
    """Full 4-tile 2-projection reference. Returns (TOTAL_OUTPUT,) bf16 array.

    Output layout: [tile0_proj0(32) | tile0_proj1(32) | tile1_proj0(32) | ... | tile3_proj1(32)]
    """
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    per_proj_bytes = NUM_CHUNKS * chunk_bytes
    per_tile_bytes = NUM_PROJECTIONS * per_proj_bytes

    output = np.zeros(TOTAL_OUTPUT, dtype=bfloat16)

    for tile in range(NUM_TILES):
        for proj in range(NUM_PROJECTIONS):
            offset = tile * per_tile_bytes + proj * per_proj_bytes
            tile_proj_packed = packed_all[offset:offset + per_proj_bytes]
            result = q4nx_matvec_single(tile_proj_packed, activation)
            out_idx = tile * M_PER_TILE * NUM_PROJECTIONS + proj * M_PER_TILE
            output[out_idx:out_idx + M_PER_TILE] = result

    return output


if __name__ == "__main__":
    np.random.seed(42)
    activation = np.random.randn(K).astype(bfloat16)

    all_scales = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]
    all_zeros = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]
    all_int4 = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]

    for tile in range(NUM_TILES):
        for proj in range(NUM_PROJECTIONS):
            all_scales[tile][proj] = np.random.uniform(
                0.01, 0.5, (M_PER_TILE, K // GROUP_SIZE)
            ).astype(bfloat16)
            all_zeros[tile][proj] = np.random.uniform(
                4, 12, (M_PER_TILE, K // GROUP_SIZE)
            ).astype(bfloat16)
            all_int4[tile][proj] = np.random.randint(
                0, 16, (M_PER_TILE, K), dtype=np.uint8
            )

    packed = pack_all_weights(all_scales, all_zeros, all_int4)
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    per_tile_bytes = NUM_PROJECTIONS * NUM_CHUNKS * chunk_bytes

    print(f"Config: {NUM_TILES} tiles x {NUM_PROJECTIONS} projections")
    print(f"  M_PER_TILE={M_PER_TILE}, K={K}, chunks={NUM_CHUNKS}, chunk_bytes={chunk_bytes}")
    print(f"  Per-tile weight: {per_tile_bytes} bytes")
    print(f"  Total weight BO: {len(packed)} bytes")
    print(f"  Activation: {K * 2} bytes")
    print(f"  Output: {TOTAL_OUTPUT * 2} bytes ({TOTAL_OUTPUT} bf16)")

    output = projection_fabric_reference(packed, activation)
    print(f"\nOutput[0:8] (tile0_proj0): {output[:8]}")
    print(f"Output[32:40] (tile0_proj1): {output[32:40]}")
    print(f"Output[64:72] (tile1_proj0): {output[64:72]}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print("\nPASS: Reference computed successfully")
