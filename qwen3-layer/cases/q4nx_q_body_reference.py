"""CPU reference for host hidden -> c1r2 replay -> main16 Q4NX Q body."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    C1R2_PACKET_DWORDS,
    CHUNK_BF16,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    M_PER_TILE,
    RECORD_DWORDS,
    ROWS_PER_COLUMN,
    ROWS_PER_PATCH,
)
from projection_schedule import (
    PATCHES_PER_COLUMN,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    Q_WEIGHT_CHUNKS,
)
from q4nx_reference import CHUNK_BYTES, make_q4nx_chunk, packed_as_i32, q4nx_matvec_from_chunk
from qkv_compact_reference import column_compact_from_records, global_compact_from_columns

CASE_NAME = "q4nx-q-body-bridge"
HIDDEN_DWORDS = C1R2_PACKET_DWORDS - 1
OUTPUT_DWORDS = Q_BODY_RECORDS * (COMPACT_PACKET_DWORDS - 1)
PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * Q_WEIGHT_CHUNKS * CHUNK_BF16
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
Q_PHASE = 0


def make_hidden_bf16() -> np.ndarray:
    values = np.empty(HIDDEN_DWORDS * 2, dtype=bfloat16)
    for lane in range(values.shape[0]):
        raw = ((lane * 7 + (lane >> 5) * 13) % 127) - 63
        values[lane] = bfloat16(raw / 64.0)
    return values


def hidden_as_i32() -> np.ndarray:
    return np.frombuffer(make_hidden_bf16().tobytes(), dtype=np.int32).copy()


def make_packed_weights(seed: int = 313) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _group in range(len(MAIN_COLUMNS)):
        for _patch in range(PATCHES_PER_COLUMN):
            for _chunk in range(Q_WEIGHT_CHUNKS):
                for _row_in_patch in range(ROWS_PER_PATCH):
                    parts.append(make_q4nx_chunk(rng))
    packed = np.concatenate(parts)
    if packed.shape[0] != TOTAL_WEIGHT_BF16 * 2:
        raise RuntimeError(f"bad packed weight bytes: {packed.shape[0]}")
    return packed


def _activation_slice(values: np.ndarray, chunk: int) -> np.ndarray:
    start = chunk * ACT_SLICE_BF16
    return values[start : start + ACT_SLICE_BF16]


def _chunk_for_tile(packed: np.ndarray, group: int, row: int, chunk: int) -> np.ndarray:
    patch = row // ROWS_PER_PATCH
    row_in_patch = row % ROWS_PER_PATCH
    offset_bf16 = (
        group * COLUMN_WEIGHT_BF16
        + patch * PATCH_WEIGHT_BF16
        + chunk * ROWS_PER_PATCH * CHUNK_BF16
        + row_in_patch * CHUNK_BF16
    )
    offset_bytes = offset_bf16 * 2
    return packed[offset_bytes : offset_bytes + CHUNK_BYTES]


def _q_record(packed: np.ndarray, hidden: np.ndarray, group: int, row: int, block: int) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    weight_base = Q_WEIGHT_CHUNK_BASE + block * Q_CHUNKS_PER_RECORD
    for chunk in range(Q_CHUNKS_PER_RECORD):
        accum += q4nx_matvec_from_chunk(
            _chunk_for_tile(packed, group, row, weight_base + chunk),
            _activation_slice(hidden, chunk),
        )
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = (Q_PHASE << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0
    record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
    return record


def q_body_payload(packed: np.ndarray | None = None, hidden: np.ndarray | None = None) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    hidden_values = make_hidden_bf16() if hidden is None else hidden
    packets: list[np.ndarray] = []
    for block in range(Q_BODY_RECORDS):
        columns = []
        for group in range(len(MAIN_COLUMNS)):
            records = [
                _q_record(weights, hidden_values, group, row, block)
                for row in range(ROWS_PER_COLUMN)
            ]
            columns.append(column_compact_from_records(records))
        packets.append(global_compact_from_columns(columns)[1:])
    return np.concatenate(packets).astype(np.int32)


def validate_output(expected: np.ndarray, got: np.ndarray) -> list[str]:
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    errors: list[str] = []
    expected_values = np.frombuffer(expected.tobytes(), dtype=bfloat16).astype(np.float32)
    got_values = np.frombuffer(got.tobytes(), dtype=bfloat16).astype(np.float32)
    abs_err = np.abs(expected_values - got_values)
    mismatch = np.where(abs_err > 1.5)[0]
    for idx in mismatch[:32]:
        errors.append(
            f"value[{int(idx)}]: expected={float(expected_values[idx]):.6f} "
            f"got={float(got_values[idx]):.6f} abs={float(abs_err[idx]):.6f}"
        )
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional value mismatches")
    return errors


def route_summary() -> list[str]:
    return [
        f"case={CASE_NAME}",
        "hidden=host 4096-bf16 -> c1r2-position full-vector station -> packet0 replay",
        "activation=c1r1 DMA4/DMA1 multicast -> all main16 DMA0 chunks",
        "weights=row1 S2MM4/5 Q4NX ingress -> row1 MM2S0..3 -> main16 DMA1",
        f"q_body=main16 Q4NX Q phase emits {Q_BODY_RECORDS} payload blocks",
        f"output={OUTPUT_DWORDS} dwords stripped Q body payload for c1r3",
    ]
