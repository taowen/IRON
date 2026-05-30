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
Q4NX_SCALE_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2
Q4NX_ZERO_BYTES = Q4NX_SCALE_BYTES
Q4NX_DATA_OFFSET = Q4NX_SCALE_BYTES + Q4NX_ZERO_BYTES
Q4NX_LANES = 2
Q4NX_ROWS_PER_LANE = M_PER_TILE // Q4NX_LANES
Q4NX_DATA_BYTES_PER_LANE = ACT_SLICE_BF16 * (Q4NX_ROWS_PER_LANE // 2)
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
    zeros = rng.uniform(-0.095, -0.015, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
    weights_u4 = rng.integers(0, 16, (M_PER_TILE, ACT_SLICE_BF16), dtype=np.uint8)

    scale_words = np.empty(M_PER_TILE * GROUPS_PER_CHUNK, dtype=bfloat16)
    zero_words = np.empty(M_PER_TILE * GROUPS_PER_CHUNK, dtype=bfloat16)
    for group in range(GROUPS_PER_CHUNK):
        for row in range(M_PER_TILE):
            scale_words[group * M_PER_TILE + row] = scales[row, group]
            zero_words[group * M_PER_TILE + row] = zeros[row, group]

    packed_data = np.empty(M_PER_TILE * (ACT_SLICE_BF16 // 2), dtype=np.uint8)
    for lane in range(Q4NX_LANES):
        lane_base = lane * Q4NX_DATA_BYTES_PER_LANE
        row_base = lane * Q4NX_ROWS_PER_LANE
        for col in range(ACT_SLICE_BF16):
            for byte_idx in range(Q4NX_ROWS_PER_LANE // 2):
                row0 = row_base + byte_idx * 2
                row1 = row0 + 1
                packed_data[lane_base + col * (Q4NX_ROWS_PER_LANE // 2) + byte_idx] = (
                    int(weights_u4[row0, col]) | (int(weights_u4[row1, col]) << 4)
                )

    packed = bytearray()
    packed += scale_words.view(np.uint8).tobytes()
    packed += zero_words.view(np.uint8).tobytes()
    packed += packed_data.tobytes()
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def packed_as_i32(packed: np.ndarray) -> np.ndarray:
    return np.frombuffer(packed.tobytes(), dtype=np.int32).copy()


def _q4nx_dequant_bf16(
    grouped_weights: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
) -> np.ndarray:
    scaled = (grouped_weights * scales[..., None]).astype(bfloat16).astype(np.float32)
    return (scaled + zeros[..., None]).astype(bfloat16).astype(np.float32)


def q4nx_matvec_from_chunk(packed_chunk: np.ndarray, activation_slice: np.ndarray) -> np.ndarray:
    if packed_chunk.shape[0] != CHUNK_BYTES:
        raise ValueError(f"bad Q4NX chunk bytes: {packed_chunk.shape[0]} != {CHUNK_BYTES}")

    scales_raw = np.frombuffer(packed_chunk[:Q4NX_SCALE_BYTES], dtype=bfloat16).astype(np.float32)
    zeros_raw = np.frombuffer(
        packed_chunk[Q4NX_SCALE_BYTES:Q4NX_DATA_OFFSET],
        dtype=bfloat16,
    ).astype(np.float32)
    scales = scales_raw.reshape(GROUPS_PER_CHUNK, M_PER_TILE).T
    zeros = zeros_raw.reshape(GROUPS_PER_CHUNK, M_PER_TILE).T

    packed_u8 = packed_chunk[Q4NX_DATA_OFFSET:]
    weights_u4 = np.empty((M_PER_TILE, ACT_SLICE_BF16), dtype=np.float32)
    for lane in range(Q4NX_LANES):
        lane_base = lane * Q4NX_DATA_BYTES_PER_LANE
        row_base = lane * Q4NX_ROWS_PER_LANE
        lane_bytes = packed_u8[
            lane_base : lane_base + Q4NX_DATA_BYTES_PER_LANE
        ].reshape(ACT_SLICE_BF16, Q4NX_ROWS_PER_LANE // 2)
        for byte_idx in range(Q4NX_ROWS_PER_LANE // 2):
            row0 = row_base + byte_idx * 2
            row1 = row0 + 1
            byte_values = lane_bytes[:, byte_idx]
            weights_u4[row0, :] = (byte_values & 0x0F).astype(np.float32)
            weights_u4[row1, :] = (byte_values >> 4).astype(np.float32)

    grouped_weights = weights_u4.reshape(M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
    grouped_act = activation_slice.astype(np.float32).reshape(GROUPS_PER_CHUNK, GROUP_SIZE)
    dequant = _q4nx_dequant_bf16(grouped_weights, scales, zeros)
    output = np.zeros((M_PER_TILE,), dtype=np.float32)
    for group in range(GROUPS_PER_CHUNK):
        for dim in range(GROUP_SIZE):
            np.add(output, dequant[:, group, dim] * grouped_act[group, dim], out=output)
    return output


def q4nx_matvec_from_chunks(
    packed_chunks: np.ndarray,
    activation_slice: np.ndarray,
    batch_chunks: int = 512,
) -> np.ndarray:
    if packed_chunks.ndim != 2 or packed_chunks.shape[1] != CHUNK_BYTES:
        raise ValueError(f"bad Q4NX chunk array shape: {packed_chunks.shape}")
    if activation_slice.shape != (ACT_SLICE_BF16,):
        raise ValueError(f"activation slice shape mismatch: {activation_slice.shape} != {(ACT_SLICE_BF16,)}")
    grouped_act = activation_slice.astype(np.float32).reshape(GROUPS_PER_CHUNK, GROUP_SIZE)
    output = np.empty((packed_chunks.shape[0], M_PER_TILE), dtype=np.float32)
    for start in range(0, packed_chunks.shape[0], batch_chunks):
        end = min(start + batch_chunks, packed_chunks.shape[0])
        chunks = np.ascontiguousarray(packed_chunks[start:end])
        count = chunks.shape[0]
        scales_raw = chunks[:, :Q4NX_SCALE_BYTES].view(bfloat16).astype(np.float32)
        zeros_raw = chunks[:, Q4NX_SCALE_BYTES:Q4NX_DATA_OFFSET].view(bfloat16).astype(np.float32)
        scales = scales_raw.reshape(count, GROUPS_PER_CHUNK, M_PER_TILE).transpose(0, 2, 1)
        zeros = zeros_raw.reshape(count, GROUPS_PER_CHUNK, M_PER_TILE).transpose(0, 2, 1)

        packed_u8 = chunks[:, Q4NX_DATA_OFFSET:]
        weights_u4 = np.empty((count, M_PER_TILE, ACT_SLICE_BF16), dtype=np.float32)
        for lane in range(Q4NX_LANES):
            lane_base = lane * Q4NX_DATA_BYTES_PER_LANE
            row_base = lane * Q4NX_ROWS_PER_LANE
            lane_bytes = packed_u8[
                :, lane_base : lane_base + Q4NX_DATA_BYTES_PER_LANE
            ].reshape(count, ACT_SLICE_BF16, Q4NX_ROWS_PER_LANE // 2)
            for byte_idx in range(Q4NX_ROWS_PER_LANE // 2):
                row0 = row_base + byte_idx * 2
                row1 = row0 + 1
                byte_values = lane_bytes[:, :, byte_idx]
                weights_u4[:, row0, :] = (byte_values & 0x0F).astype(np.float32)
                weights_u4[:, row1, :] = (byte_values >> 4).astype(np.float32)

        grouped_weights = weights_u4.reshape(count, M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
        dequant = _q4nx_dequant_bf16(grouped_weights, scales, zeros)
        chunk_output = np.zeros((count, M_PER_TILE), dtype=np.float32)
        for group in range(GROUPS_PER_CHUNK):
            for dim in range(GROUP_SIZE):
                np.add(
                    chunk_output,
                    dequant[:, :, group, dim] * grouped_act[group, dim],
                    out=chunk_output,
                )
        output[start:end] = chunk_output
    return output
