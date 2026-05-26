"""CPU reference for multi-chunk Q4NX dequant + GEMV (M=32, K=1024)."""

import numpy as np
from ml_dtypes import bfloat16


M = 32
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32


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


def pack_all_chunks(scales_full, zeros_full, int4_full):
    """Pack K=1024 weight matrix into 4 sequential Q4NX chunks."""
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
            M, K_CHUNK, GROUP_SIZE,
        )
        chunks.append(chunk)
    return np.concatenate(chunks)


def q4nx_matvec_reference(packed_all, activation):
    """Full M=32 K=1024 Q4NX matmul via chunk-wise unpack + FP32 accumulate."""
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M * groups_per_row * 2 + M * groups_per_row * 2 + M * K_CHUNK // 2
    accum = np.zeros(M, dtype=np.float32)

    for c in range(NUM_CHUNKS):
        chunk_data = packed_all[c * chunk_bytes:(c + 1) * chunk_bytes]
        scale_bytes = M * groups_per_row * 2
        zero_bytes = M * groups_per_row * 2
        data_offset = scale_bytes + zero_bytes

        scales = np.frombuffer(
            chunk_data[:scale_bytes], dtype=bfloat16
        ).reshape(M, groups_per_row)
        zeros = np.frombuffer(
            chunk_data[scale_bytes:data_offset], dtype=bfloat16
        ).reshape(M, groups_per_row)
        int4_raw = chunk_data[data_offset:]

        weights_u4 = np.zeros((M, K_CHUNK), dtype=np.float32)
        for row in range(M):
            for col in range(0, K_CHUNK, 2):
                b = int(int4_raw[row * (K_CHUNK // 2) + col // 2])
                weights_u4[row, col] = b & 0x0F
                weights_u4[row, col + 1] = (b >> 4) & 0x0F

        weights_f32 = np.zeros((M, K_CHUNK), dtype=np.float32)
        for row in range(M):
            for g in range(groups_per_row):
                s = float(scales[row, g])
                z = float(zeros[row, g])
                c0 = g * GROUP_SIZE
                c1 = c0 + GROUP_SIZE
                weights_f32[row, c0:c1] = (weights_u4[row, c0:c1] - z) * s

        act_chunk = activation[c * K_CHUNK:(c + 1) * K_CHUNK].astype(np.float32)
        accum += weights_f32 @ act_chunk

    return accum.astype(bfloat16)


if __name__ == "__main__":
    np.random.seed(42)
    activation = np.random.randn(K).astype(bfloat16)
    scales = np.random.uniform(0.01, 0.5, (M, K // GROUP_SIZE)).astype(bfloat16)
    zeros = np.random.uniform(4, 12, (M, K // GROUP_SIZE)).astype(bfloat16)
    int4_data = np.random.randint(0, 16, (M, K), dtype=np.uint8)

    packed = pack_all_chunks(scales, zeros, int4_data)
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M * groups_per_row * 2 + M * groups_per_row * 2 + M * K_CHUNK // 2
    print(f"Q4NX config: M={M}, K={K}, chunks={NUM_CHUNKS}, chunk_bytes={chunk_bytes}")
    print(f"Total packed weight: {len(packed)} bytes ({NUM_CHUNKS} x {chunk_bytes})")

    output = q4nx_matvec_reference(packed, activation)
    print(f"Output[0:8]: {output[:8]}")
    print(f"Output dtype: {output.dtype}, shape: {output.shape}")

    # Cross-check: full matrix multiply
    full_weights_f32 = np.zeros((M, K), dtype=np.float32)
    for c in range(NUM_CHUNKS):
        chunk_data = packed[c * chunk_bytes:(c + 1) * chunk_bytes]
        scale_bytes = M * groups_per_row * 2
        zero_bytes = M * groups_per_row * 2
        data_offset = scale_bytes + zero_bytes
        s = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(M, groups_per_row)
        z = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(M, groups_per_row)
        int4_raw = chunk_data[data_offset:]
        for row in range(M):
            for col in range(0, K_CHUNK, 2):
                b = int(int4_raw[row * (K_CHUNK // 2) + col // 2])
                full_weights_f32[row, c * K_CHUNK + col] = b & 0x0F
                full_weights_f32[row, c * K_CHUNK + col + 1] = (b >> 4) & 0x0F
            for g in range(groups_per_row):
                sc = float(s[row, g])
                zz = float(z[row, g])
                c0 = c * K_CHUNK + g * GROUP_SIZE
                c1 = c0 + GROUP_SIZE
                full_weights_f32[row, c0:c1] = (full_weights_f32[row, c0:c1] - zz) * sc

    ref_full = (full_weights_f32 @ activation.astype(np.float32)).astype(bfloat16)
    max_diff = np.max(np.abs(output.astype(np.float32) - ref_full.astype(np.float32)))
    print(f"\nCross-check vs full matmul: max_diff = {max_diff}")
    if max_diff < 0.01:
        print("PASS: Chunked reference matches full matmul")
    else:
        print("FAIL: Chunked reference differs from full matmul")
