"""Shared Q4NX reference math for qwen3-layer integration checks."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    C1R2_PACKET_DWORDS,
    CHUNK_BF16,
    GROUP_SIZE,
    MAIN_COLUMNS,
    M_PER_TILE,
    ROWS_PER_COLUMN,
)

HIDDEN_DWORDS = C1R2_PACKET_DWORDS - 1
GROUPS_PER_CHUNK = ACT_SLICE_BF16 // GROUP_SIZE
CHUNK_BYTES = CHUNK_BF16 * 2
OUT_RECORD_BF16 = M_PER_TILE + 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_RECORD_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


def make_hidden_bf16() -> np.ndarray:
    values = np.empty(HIDDEN_DWORDS * 2, dtype=bfloat16)
    for lane in range(values.shape[0]):
        raw = ((lane * 7 + (lane >> 5) * 13) % 127) - 63
        values[lane] = bfloat16(raw / 64.0)
    return values


def hidden_as_i32() -> np.ndarray:
    return np.frombuffer(make_hidden_bf16().tobytes(), dtype=np.int32).copy()


def make_q4nx_chunk(rng: np.random.Generator) -> np.ndarray:
    scales = rng.uniform(0.001, 0.012, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
    zeros = rng.uniform(6.0, 9.0, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
    packed_data = rng.integers(0, 256, M_PER_TILE * (ACT_SLICE_BF16 // 2), dtype=np.uint8)
    packed = bytearray()
    packed += scales.view(np.uint8).tobytes()
    packed += zeros.view(np.uint8).tobytes()
    packed += packed_data.tobytes()
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def packed_as_i32(packed: np.ndarray) -> np.ndarray:
    return np.frombuffer(packed.tobytes(), dtype=np.int32).copy()


def q4nx_matvec_from_chunk(packed_chunk: np.ndarray, activation_slice: np.ndarray) -> np.ndarray:
    scale_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    zero_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    data_offset = scale_bytes + zero_bytes
    scales = np.frombuffer(packed_chunk[:scale_bytes], dtype=bfloat16).reshape(
        M_PER_TILE,
        GROUPS_PER_CHUNK,
    )
    zeros = np.frombuffer(packed_chunk[scale_bytes:data_offset], dtype=bfloat16).reshape(
        M_PER_TILE,
        GROUPS_PER_CHUNK,
    )
    packed_u8 = packed_chunk[data_offset:].reshape(M_PER_TILE, ACT_SLICE_BF16 // 2)

    weights_u4 = np.empty((M_PER_TILE, ACT_SLICE_BF16), dtype=np.float32)
    weights_u4[:, 0::2] = (packed_u8 & 0x0F).astype(np.float32)
    weights_u4[:, 1::2] = (packed_u8 >> 4).astype(np.float32)

    grouped_weights = weights_u4.reshape(M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
    grouped_act = activation_slice.astype(np.float32).reshape(GROUPS_PER_CHUNK, GROUP_SIZE)
    dequant = (
        (grouped_weights - zeros.astype(np.float32)[:, :, None])
        * scales.astype(np.float32)[:, :, None]
    ).astype(bfloat16).astype(np.float32)
    return np.sum(
        dequant * grouped_act[None, :, :],
        axis=(1, 2),
    )
