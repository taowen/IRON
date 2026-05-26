"""CPU reference for exp 14: FFN SwiGLU fusion with cross-phase residency."""

import numpy as np
from ml_dtypes import bfloat16


M_PER_TILE = 32
NUM_COLS = 2
ROWS_PER_COL = 2
NUM_TILES = NUM_COLS * ROWS_PER_COL
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COL
TOTAL_OUTPUT = NUM_TILES * M_PER_TILE  # 128

NUM_PHASES = 2  # gate + up (weight phases)


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
    """Pack weights with fat-chunk layout grouped by column.

    Layout: [col0_fat_chunks | col1_fat_chunks]
    Per column: NUM_PHASES * NUM_CHUNKS fat chunks
    Each fat chunk: [row0_chunk(5120 bytes) | row1_chunk(5120 bytes)]

    all_scales[col][row][phase] shape: (M_PER_TILE, K//GROUP_SIZE)
    """
    groups_per_row = K_CHUNK // GROUP_SIZE
    parts = []
    for col in range(NUM_COLS):
        for phase in range(NUM_PHASES):
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
    return np.concatenate(parts)


def q4nx_matvec_single(packed_bytes, activation):
    """Compute M_PER_TILE x K Q4NX matvec for one tile (4 chunks)."""
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


def swiglu_reference(packed_all, activation):
    """Full reference: gate + up projections, then SwiGLU.

    Output layout: [col0_r0 | col0_r1 | col1_r0 | col1_r1] (32 bf16 each)
    """
    chunk_bytes = CHUNK_BF16 * 2  # 5120 bytes per Q4NX chunk
    fat_chunk_bytes = chunk_bytes * ROWS_PER_COL  # 10240 bytes
    per_col_bytes = NUM_PHASES * NUM_CHUNKS * fat_chunk_bytes

    output = np.zeros(TOTAL_OUTPUT, dtype=bfloat16)

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            # Gate projection (phase 0)
            gate_chunks = bytearray()
            for chunk_idx in range(NUM_CHUNKS):
                fat_offset = col * per_col_bytes + (0 * NUM_CHUNKS + chunk_idx) * fat_chunk_bytes
                row_offset = fat_offset + row * chunk_bytes
                gate_chunks.extend(packed_all[row_offset:row_offset + chunk_bytes])
            gate_packed = np.frombuffer(bytes(gate_chunks), dtype=np.uint8)
            gate = q4nx_matvec_single(gate_packed, activation)

            # Up projection (phase 1)
            up_chunks = bytearray()
            for chunk_idx in range(NUM_CHUNKS):
                fat_offset = col * per_col_bytes + (1 * NUM_CHUNKS + chunk_idx) * fat_chunk_bytes
                row_offset = fat_offset + row * chunk_bytes
                up_chunks.extend(packed_all[row_offset:row_offset + chunk_bytes])
            up_packed = np.frombuffer(bytes(up_chunks), dtype=np.uint8)
            up = q4nx_matvec_single(up_packed, activation)

            # SwiGLU: silu(gate) * up
            gate_f32 = gate.astype(np.float32)
            up_f32 = up.astype(np.float32)
            silu_gate = gate_f32 / (1.0 + np.exp(-gate_f32))
            swiglu = (silu_gate * up_f32).astype(bfloat16)

            out_idx = (col * ROWS_PER_COL + row) * M_PER_TILE
            output[out_idx:out_idx + M_PER_TILE] = swiglu

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
                for _ in range(NUM_PHASES)
            ]
            all_zeros[col][row] = [
                np.random.uniform(4, 12, (M_PER_TILE, K // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K), dtype=np.uint8)
                for _ in range(NUM_PHASES)
            ]

    packed = pack_all_weights_fat(all_scales, all_zeros, all_int4)
    print(f"Config: {NUM_COLS} cols x {ROWS_PER_COL} rows, gate+up+swiglu")
    print(f"  Weight BO: {len(packed)} bytes ({len(packed)//2} bf16)")
    print(f"  Activation: {K*2} bytes")
    print(f"  Output: {TOTAL_OUTPUT*2} bytes ({TOTAL_OUTPUT} bf16)")

    output = swiglu_reference(packed, activation)
    print(f"\nOutput[0:4] (col0_r0): {output[:4]}")
    print(f"Output[32:36] (col0_r1): {output[32:36]}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print("\nPASS: Reference computed successfully")
