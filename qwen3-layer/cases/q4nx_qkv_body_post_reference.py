"""CPU reference for Q4NX Q/K/V body handoff into c1r3 postprocess."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    CHUNK_BF16,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    M_PER_TILE,
    RECORD_DWORDS,
    ROWS_PER_COLUMN,
    ROWS_PER_PATCH,
)
from projection_schedule import (
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    PATCHES_PER_COLUMN,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    V_CHUNKS_PER_RECORD,
    V_WEIGHT_CHUNK_BASE,
)
from q4nx_reference import CHUNK_BYTES, make_q4nx_chunk, packed_as_i32, q4nx_matvec_from_chunk
from q4nx_reference import HIDDEN_DWORDS, hidden_as_i32, make_hidden_bf16
from qkv_compact_reference import column_compact_from_records, global_compact_from_columns

CASE_NAME = "q4nx-qkv-body-post-bridge"
Q_DWORDS = Q_BODY_RECORDS * (COMPACT_PACKET_DWORDS - 1)
CURRENT_DWORDS = KV_BODY_RECORDS * (COMPACT_PACKET_DWORDS - 1)
OUTPUT_DWORDS = Q_DWORDS + CURRENT_DWORDS + CURRENT_DWORDS
PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * QKV_BODY_WEIGHT_CHUNKS * CHUNK_BF16
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
Q_PHASE = 0
K_PHASE = 1
V_PHASE = 2


def make_packed_weights(seed: int = 419) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _group in range(len(MAIN_COLUMNS)):
        for _patch in range(PATCHES_PER_COLUMN):
            for _chunk in range(QKV_BODY_WEIGHT_CHUNKS):
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


def _body_record(
    packed: np.ndarray,
    hidden: np.ndarray,
    group: int,
    row: int,
    phase: int,
    block: int,
) -> np.ndarray:
    if phase == Q_PHASE:
        weight_base = Q_WEIGHT_CHUNK_BASE + block * Q_CHUNKS_PER_RECORD
        chunks_per_record = Q_CHUNKS_PER_RECORD
    elif phase == K_PHASE:
        weight_base = K_WEIGHT_CHUNK_BASE + block * K_CHUNKS_PER_RECORD
        chunks_per_record = K_CHUNKS_PER_RECORD
    elif phase == V_PHASE:
        weight_base = V_WEIGHT_CHUNK_BASE + block * V_CHUNKS_PER_RECORD
        chunks_per_record = V_CHUNKS_PER_RECORD
    else:
        raise ValueError(f"bad body phase: {phase}")

    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for chunk in range(chunks_per_record):
        accum += q4nx_matvec_from_chunk(
            _chunk_for_tile(packed, group, row, weight_base + chunk),
            _activation_slice(hidden, chunk),
        )
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = (phase << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0
    record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
    return record


def _body_payload(packed: np.ndarray, hidden: np.ndarray, phase: int, records: int) -> np.ndarray:
    packets: list[np.ndarray] = []
    for block in range(records):
        columns = []
        for group in range(len(MAIN_COLUMNS)):
            tile_records = [
                _body_record(packed, hidden, group, row, phase, block)
                for row in range(ROWS_PER_COLUMN)
            ]
            columns.append(column_compact_from_records(tile_records))
        packets.append(global_compact_from_columns(columns)[1:])
    return np.concatenate(packets).astype(np.int32)


def q_body_payload(packed: np.ndarray | None = None, hidden: np.ndarray | None = None) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _body_payload(weights, values, Q_PHASE, Q_BODY_RECORDS)


def k_body_payload(packed: np.ndarray | None = None, hidden: np.ndarray | None = None) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _body_payload(weights, values, K_PHASE, KV_BODY_RECORDS)


def v_body_payload(packed: np.ndarray | None = None, hidden: np.ndarray | None = None) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _body_payload(weights, values, V_PHASE, KV_BODY_RECORDS)


def _current_from_body(body: np.ndarray) -> np.ndarray:
    kv_heads = 8
    head_dwords = CURRENT_DWORDS // kv_heads
    half_current_dwords = CURRENT_DWORDS // 2
    half_head_dwords = head_dwords // 2
    current = np.empty(CURRENT_DWORDS, dtype=np.int32)
    for head in range(kv_heads):
        for pair in range(half_head_dwords):
            even_idx = head * head_dwords + pair * 2
            odd_idx = even_idx + 1
            even_stream_idx = head * half_head_dwords + pair
            odd_stream_idx = half_current_dwords + even_stream_idx
            current[even_stream_idx] = body[even_idx]
            current[odd_stream_idx] = body[odd_idx]
    return current


def expected_output(packed: np.ndarray | None = None) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    hidden = make_hidden_bf16()
    q_body = q_body_payload(weights, hidden)
    k_body = k_body_payload(weights, hidden)
    v_body = v_body_payload(weights, hidden)
    return np.concatenate((q_body, _current_from_body(k_body), _current_from_body(v_body))).astype(np.int32)


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
        "hidden=host 4096-bf16 -> c1r2-position full-vector station -> 12 packet0 replays",
        "weights=row1 S2MM4/5 Q/K/V Q4NX ingress -> row1 MM2S0..3 -> main16 DMA1",
        "main16 emits Q/K/V body payloads with Q4NX kernels instead of qkv_emit_qkv_body_records",
        "c1r3 postprocess drains Q payload plus current K/V layout to host",
        f"output={OUTPUT_DWORDS} dwords = Q {Q_DWORDS} + K {CURRENT_DWORDS} + V {CURRENT_DWORDS}",
    ]
