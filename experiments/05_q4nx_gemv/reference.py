"""CPU reference for Q4NX dequant + GEMV."""

import numpy as np
from ml_dtypes import bfloat16


def pack_q4nx_chunk(scales, zeros, int4_data, M, K, group_size=32):
    """Pack Q4NX data into the chunk format: [scales][zeros][int4_payload]."""
    groups_per_row = K // group_size
    assert scales.shape == (M, groups_per_row)
    assert zeros.shape == (M, groups_per_row)
    assert int4_data.shape == (M, K)
    assert int4_data.max() <= 15

    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(M):
        for col in range(0, K, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))

    return np.frombuffer(bytes(packed), dtype=np.uint8)


def q4nx_matvec_reference(packed_bytes, activation, M=32, K=256, group_size=32):
    """Unpack Q4NX chunk and compute matmul with activation vector."""
    groups_per_row = K // group_size
    scale_bytes = M * groups_per_row * 2
    zero_bytes = M * groups_per_row * 2
    data_offset = scale_bytes + zero_bytes

    scales = np.frombuffer(
        packed_bytes[:scale_bytes], dtype=bfloat16
    ).reshape(M, groups_per_row)
    zeros = np.frombuffer(
        packed_bytes[scale_bytes:data_offset], dtype=bfloat16
    ).reshape(M, groups_per_row)
    int4_raw = packed_bytes[data_offset:]

    weights_u4 = np.zeros((M, K), dtype=np.float32)
    for row in range(M):
        for col in range(0, K, 2):
            b = int(int4_raw[row * (K // 2) + col // 2])
            weights_u4[row, col] = b & 0x0F
            weights_u4[row, col + 1] = (b >> 4) & 0x0F

    weights_f32 = np.zeros((M, K), dtype=np.float32)
    for row in range(M):
        for g in range(groups_per_row):
            s = float(scales[row, g])
            z = float(zeros[row, g])
            c0 = g * group_size
            c1 = c0 + group_size
            weights_f32[row, c0:c1] = (weights_u4[row, c0:c1] - z) * s

    output = (weights_f32 @ activation.astype(np.float32)).astype(bfloat16)
    return output
