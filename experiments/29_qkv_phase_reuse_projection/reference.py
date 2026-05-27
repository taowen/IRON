"""CPU reference for exp29: same physical tiles compute Q, then K, then V."""

import numpy as np
from ml_dtypes import bfloat16


M_PER_TILE = 32
NUM_COLS = 2
ROWS_PER_COL = 2
NUM_TILES = NUM_COLS * ROWS_PER_COL
K = 4096
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
NUM_PROJECTIONS = 3
PROJECTION_NAMES = ("Q", "K", "V")
TOTAL_OUTPUT = NUM_TILES * M_PER_TILE * NUM_PROJECTIONS
CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COL


def pack_q4nx_chunk(scales, zeros, int4_data, M, K_chunk, group_size=32):
    groups_per_row = K_chunk // group_size
    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(M):
        for col in range(0, K_chunk, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def pack_all_weights_fat(all_scales, all_zeros, all_int4):
    """Pack weight BO with fat-chunk layout grouped by column.

    Layout: [col0_fat_chunks | col1_fat_chunks]
    Per column: Q chunks, then K chunks, then V chunks
    Each fat chunk: [row0_chunk(5120 bytes) | row1_chunk(5120 bytes)]

    all_scales[col][row][proj] shape: (M_PER_TILE, K//GROUP_SIZE)
    """
    groups_per_row = K_CHUNK // GROUP_SIZE
    parts = []
    for col in range(NUM_COLS):
        for proj in range(NUM_PROJECTIONS):
            for chunk_idx in range(NUM_CHUNKS):
                for row in range(ROWS_PER_COL):
                    s = all_scales[col][row][proj]
                    z = all_zeros[col][row][proj]
                    d = all_int4[col][row][proj]
                    g_start = chunk_idx * groups_per_row
                    g_end = g_start + groups_per_row
                    col_start = chunk_idx * K_CHUNK
                    col_end = col_start + K_CHUNK
                    chunk = pack_q4nx_chunk(
                        s[:, g_start:g_end],
                        z[:, g_start:g_end],
                        d[:, col_start:col_end],
                        M_PER_TILE, K_CHUNK, GROUP_SIZE,
                    )
                    parts.append(chunk)
    return np.concatenate(parts)


def q4nx_matvec_single(packed_bytes, activation):
    """Compute M_PER_TILE x K Q4NX matvec for one tile, one projection (4 chunks)."""
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    accum = np.zeros(M_PER_TILE, dtype=np.float32)

    for c in range(NUM_CHUNKS):
        chunk_data = packed_bytes[c * chunk_bytes:(c + 1) * chunk_bytes]
        scale_bytes = M_PER_TILE * groups_per_row * 2
        zero_bytes = M_PER_TILE * groups_per_row * 2
        data_offset = scale_bytes + zero_bytes

        scales = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(M_PER_TILE, groups_per_row)
        zeros = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(M_PER_TILE, groups_per_row)
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


def projection_column_reference(packed_all, activation):
    """Full reference.

    Output layout:
    [col0_r0_Q | col0_r0_K | col0_r0_V | col0_r1_Q | ... | col1_...]
    """
    chunk_bytes = CHUNK_BF16 * 2  # 5120 bytes per Q4NX chunk
    fat_chunk_bytes = chunk_bytes * ROWS_PER_COL  # 10240 bytes
    per_col_bytes = NUM_PROJECTIONS * NUM_CHUNKS * fat_chunk_bytes

    output = np.zeros(TOTAL_OUTPUT, dtype=bfloat16)

    for col in range(NUM_COLS):
        for proj in range(NUM_PROJECTIONS):
            for row in range(ROWS_PER_COL):
                # Extract this tile's chunks from fat layout
                tile_chunks = bytearray()
                for chunk_idx in range(NUM_CHUNKS):
                    fat_offset = col * per_col_bytes + (proj * NUM_CHUNKS + chunk_idx) * fat_chunk_bytes
                    row_offset = fat_offset + row * chunk_bytes
                    tile_chunks.extend(packed_all[row_offset:row_offset + chunk_bytes])

                tile_packed = np.frombuffer(bytes(tile_chunks), dtype=np.uint8)
                result = q4nx_matvec_single(tile_packed, activation)

                out_idx = (col * ROWS_PER_COL * NUM_PROJECTIONS + row * NUM_PROJECTIONS + proj) * M_PER_TILE
                output[out_idx:out_idx + M_PER_TILE] = result

    return output


if __name__ == "__main__":
    np.random.seed(42)
    activation = np.random.randn(K).astype(bfloat16)

    all_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            all_scales[col][row] = [
                np.random.uniform(0.01, 0.5, (M_PER_TILE, K // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PROJECTIONS)
            ]
            all_zeros[col][row] = [
                np.random.uniform(4, 12, (M_PER_TILE, K // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PROJECTIONS)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K), dtype=np.uint8)
                for _ in range(NUM_PROJECTIONS)
            ]

    packed = pack_all_weights_fat(all_scales, all_zeros, all_int4)
    print(f"Config: {NUM_COLS} cols x {ROWS_PER_COL} rows x Q/K/V phases")
    print(f"  Weight BO: {len(packed)} bytes ({len(packed)//2} bf16)")
    print(f"  Activation: {K*2} bytes")
    print(f"  Output: {TOTAL_OUTPUT*2} bytes ({TOTAL_OUTPUT} bf16)")

    output = projection_column_reference(packed, activation)
    print(f"\nOutput[0:4] (col0_r0_Q): {output[:4]}")
    print(f"Output[32:36] (col0_r0_K): {output[32:36]}")
    print(f"Output[64:68] (col0_r0_V): {output[64:68]}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print("\nPASS: Reference computed successfully")
