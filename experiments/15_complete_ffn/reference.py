"""CPU reference for exp 15: complete FFN (gate + up + SwiGLU + down)."""

import numpy as np
from ml_dtypes import bfloat16


M_PER_TILE = 32
NUM_COLS = 2
ROWS_PER_COL = 2
K_HIDDEN = 512
K_CHUNK = 256
NUM_CHUNKS = K_HIDDEN // K_CHUNK
GROUP_SIZE = 32
NUM_PHASES_PROJ = 2  # gate + up

INTERMEDIATE = ROWS_PER_COL * M_PER_TILE  # 64
K_DOWN = INTERMEDIATE
K_CHUNK_DOWN = 64
NUM_CHUNKS_DOWN = 1
GROUPS_PER_ROW_DOWN = K_CHUNK_DOWN // GROUP_SIZE  # 2

CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COL
DOWN_CHUNK_BF16 = 640
FAT_DOWN_CHUNK_BF16 = DOWN_CHUNK_BF16 * ROWS_PER_COL

TOTAL_OUTPUT = NUM_COLS * ROWS_PER_COL * M_PER_TILE  # 128

PER_COL_WT_BYTES = (NUM_PHASES_PROJ * NUM_CHUNKS * FAT_CHUNK_BF16 + FAT_DOWN_CHUNK_BF16) * 2


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


def pack_all_weights(all_scales, all_zeros, all_int4, down_scales, down_zeros, down_int4):
    """Pack gate/up weights (fat-chunk layout) + down weights per column.

    Layout per column: [gate_fat_chunks | up_fat_chunks | down_fat_chunk]
    """
    groups_per_row = K_CHUNK // GROUP_SIZE
    parts = []
    for col in range(NUM_COLS):
        # Gate + Up fat chunks (same layout as exp 14)
        for phase in range(NUM_PHASES_PROJ):
            for chunk_idx in range(NUM_CHUNKS):
                for row in range(ROWS_PER_COL):
                    s = all_scales[col][row][phase]
                    z = all_zeros[col][row][phase]
                    d = all_int4[col][row][phase]
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

        # Down fat chunk (1 chunk, different K)
        for row in range(ROWS_PER_COL):
            s = down_scales[col][row]
            z = down_zeros[col][row]
            d = down_int4[col][row]
            chunk = pack_q4nx_chunk(s, z, d, M_PER_TILE, K_CHUNK_DOWN, GROUP_SIZE)
            parts.append(chunk)

    return np.concatenate(parts)


def q4nx_matvec_single(packed_bytes, activation, K, K_chunk, group_size=32):
    """Compute M_PER_TILE x K Q4NX matvec."""
    groups_per_row = K_chunk // group_size
    num_chunks = K // K_chunk
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_chunk // 2
    accum = np.zeros(M_PER_TILE, dtype=np.float32)

    for c in range(num_chunks):
        chunk_data = packed_bytes[c * chunk_bytes:(c + 1) * chunk_bytes]
        scale_bytes = M_PER_TILE * groups_per_row * 2
        zero_bytes = M_PER_TILE * groups_per_row * 2
        data_offset = scale_bytes + zero_bytes

        scales = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(M_PER_TILE, groups_per_row)
        zeros = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(M_PER_TILE, groups_per_row)
        int4_raw = chunk_data[data_offset:]

        weights_u4 = np.zeros((M_PER_TILE, K_chunk), dtype=np.float32)
        for row in range(M_PER_TILE):
            for col in range(0, K_chunk, 2):
                b = int(int4_raw[row * (K_chunk // 2) + col // 2])
                weights_u4[row, col] = b & 0x0F
                weights_u4[row, col + 1] = (b >> 4) & 0x0F

        weights_f32 = np.zeros((M_PER_TILE, K_chunk), dtype=np.float32)
        for row in range(M_PER_TILE):
            for g in range(groups_per_row):
                s = float(scales[row, g])
                z = float(zeros[row, g])
                c0 = g * group_size
                c1 = c0 + group_size
                weights_f32[row, c0:c1] = (weights_u4[row, c0:c1] - z) * s

        act_chunk = activation[c * K_chunk:(c + 1) * K_chunk].astype(np.float32)
        accum += weights_f32 @ act_chunk

    return accum.astype(bfloat16)


def complete_ffn_reference(packed_all, activation):
    """Full reference: gate + up + SwiGLU + down for each column."""
    gate_up_chunk_bytes = CHUNK_BF16 * 2
    fat_chunk_bytes = gate_up_chunk_bytes * ROWS_PER_COL
    down_chunk_bytes = DOWN_CHUNK_BF16 * 2
    fat_down_bytes = down_chunk_bytes * ROWS_PER_COL
    per_col_bytes = NUM_PHASES_PROJ * NUM_CHUNKS * fat_chunk_bytes + fat_down_bytes

    output = np.zeros(TOTAL_OUTPUT, dtype=bfloat16)

    for col in range(NUM_COLS):
        col_base = col * per_col_bytes
        intermediates = []

        for row in range(ROWS_PER_COL):
            # Gate projection (phase 0)
            gate_chunks = bytearray()
            for chunk_idx in range(NUM_CHUNKS):
                fat_offset = col_base + (0 * NUM_CHUNKS + chunk_idx) * fat_chunk_bytes
                row_offset = fat_offset + row * gate_up_chunk_bytes
                gate_chunks.extend(packed_all[row_offset:row_offset + gate_up_chunk_bytes])
            gate_packed = np.frombuffer(bytes(gate_chunks), dtype=np.uint8)
            gate = q4nx_matvec_single(gate_packed, activation, K_HIDDEN, K_CHUNK)

            # Up projection (phase 1)
            up_chunks = bytearray()
            for chunk_idx in range(NUM_CHUNKS):
                fat_offset = col_base + (1 * NUM_CHUNKS + chunk_idx) * fat_chunk_bytes
                row_offset = fat_offset + row * gate_up_chunk_bytes
                up_chunks.extend(packed_all[row_offset:row_offset + gate_up_chunk_bytes])
            up_packed = np.frombuffer(bytes(up_chunks), dtype=np.uint8)
            up = q4nx_matvec_single(up_packed, activation, K_HIDDEN, K_CHUNK)

            # SwiGLU
            gate_f32 = gate.astype(np.float32)
            up_f32 = up.astype(np.float32)
            silu_gate = gate_f32 / (1.0 + np.exp(-gate_f32))
            inter = (silu_gate * up_f32).astype(bfloat16)
            intermediates.append(inter)

        # Gather: concatenate intermediates from all rows in column
        gathered = np.concatenate(intermediates)  # 64 bf16

        # Down projection for each row
        for row in range(ROWS_PER_COL):
            down_fat_offset = col_base + NUM_PHASES_PROJ * NUM_CHUNKS * fat_chunk_bytes
            down_row_offset = down_fat_offset + row * down_chunk_bytes
            down_packed = np.frombuffer(
                bytes(packed_all[down_row_offset:down_row_offset + down_chunk_bytes]),
                dtype=np.uint8
            )
            down_out = q4nx_matvec_single(down_packed, gathered, K_DOWN, K_CHUNK_DOWN)

            out_idx = (col * ROWS_PER_COL + row) * M_PER_TILE
            output[out_idx:out_idx + M_PER_TILE] = down_out

    return output


if __name__ == "__main__":
    np.random.seed(42)
    activation = np.random.randn(K_HIDDEN).astype(bfloat16)

    # Gate/Up weights
    all_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            all_scales[col][row] = [
                np.random.uniform(0.01, 0.5, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_zeros[col][row] = [
                np.random.uniform(4, 12, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K_HIDDEN), dtype=np.uint8)
                for _ in range(NUM_PHASES_PROJ)
            ]

    # Down weights (K=64)
    down_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            down_scales[col][row] = np.random.uniform(0.01, 0.5, (M_PER_TILE, GROUPS_PER_ROW_DOWN)).astype(bfloat16)
            down_zeros[col][row] = np.random.uniform(4, 12, (M_PER_TILE, GROUPS_PER_ROW_DOWN)).astype(bfloat16)
            down_int4[col][row] = np.random.randint(0, 16, (M_PER_TILE, K_DOWN), dtype=np.uint8)

    packed = pack_all_weights(all_scales, all_zeros, all_int4, down_scales, down_zeros, down_int4)
    print(f"Config: {NUM_COLS} cols x {ROWS_PER_COL} rows, complete FFN")
    print(f"  K_hidden={K_HIDDEN}, intermediate={INTERMEDIATE}, K_down={K_DOWN}")
    print(f"  Weight BO: {len(packed)} bytes ({len(packed)//2} bf16)")
    print(f"  Activation: {K_HIDDEN*2} bytes")
    print(f"  Output: {TOTAL_OUTPUT*2} bytes ({TOTAL_OUTPUT} bf16)")

    output = complete_ffn_reference(packed, activation)
    print(f"\nOutput[0:4] (col0_r0): {output[:4]}")
    print(f"Output[32:36] (col0_r1): {output[32:36]}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print("\nPASS: Reference computed successfully")
