"""CPU reference for current K/V full-layer tail with Q4NX up/gate and down."""

from __future__ import annotations

import math

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    C1R2_PACKET_DWORDS,
    C6R2_HALF_DWORDS,
    CHUNK_BF16,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    M_PER_TILE,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
    ROWS_PER_PATCH,
    SWIGLU_SLICES,
)
from c1r2_reference import _trunc_div
from compact_dataflow import down_record_header
from projection_schedule import (
    DOWN_CHUNKS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_TOTAL_WEIGHT_CHUNKS,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    O_WEIGHT_CHUNKS,
    PATCHES_PER_COLUMN,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    UPGATE_CHUNKS_PER_REPLAY,
    UPGATE_WEIGHT_CHUNKS,
    V_CHUNKS_PER_RECORD,
    V_WEIGHT_CHUNK_BASE,
)
from q4nx_reference import (
    CHUNK_BYTES,
    HIDDEN_DWORDS,
    OUT_RECORD_BF16,
    OUT_TOTAL_I32,
    hidden_as_i32,
    make_q4nx_chunk,
    make_hidden_bf16,
    packed_as_i32 as q4nx_packed_as_i32,
    q4nx_matvec_from_chunk,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    CURRENT_DWORDS,
    DecodeSchedule,
    attention_payload_from_qkv,
    make_decode_schedule,
    make_history_k_cache_payload,
    make_history_v_cache_payload,
    validate_cache_layout_contract,
    _write_current,
)
from qkv_compact_reference import (
    column_compact_from_records,
    global_compact_from_columns,
)

CASE_NAME = "currentkv-full-layer-q4nx-down-bridge"
DEFAULT_SCHEDULE = make_decode_schedule(None)
OUTPUT_DWORDS = COMPACT_PACKET_DWORDS
Q_BODY_COMPACT_DWORDS = Q_BODY_RECORDS * COMPACT_PACKET_DWORDS
KV_BODY_COMPACT_DWORDS = KV_BODY_RECORDS * COMPACT_PACKET_DWORDS
PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * FULL_LAYER_TOTAL_WEIGHT_CHUNKS * CHUNK_BF16
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
COMPACT_NUMERIC_LANES = RECORD_PAYLOAD_DWORDS * len(MAIN_COLUMNS) * ROWS_PER_COLUMN * 2
REPLAY_FIXED_SCALE = 256.0
FULL_PIPELINE_ABS_TOL = 16.0
FULL_PIPELINE_REL_TOL = 1.00
ATTENTION_QKV_SCALE = 16.0
CURRENT_CACHE_S16_TOL = 1


def make_packed_weights(seed: int = 197) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _group in range(len(MAIN_COLUMNS)):
        for _patch in range(PATCHES_PER_COLUMN):
            for _chunk in range(FULL_LAYER_TOTAL_WEIGHT_CHUNKS):
                for _row_in_patch in range(ROWS_PER_PATCH):
                    parts.append(make_q4nx_chunk(rng))
    packed = np.concatenate(parts)
    if packed.shape[0] != TOTAL_WEIGHT_BF16 * 2:
        raise RuntimeError(f"bad packed weight bytes: {packed.shape[0]}")
    return packed


def packed_as_i32(packed: np.ndarray) -> np.ndarray:
    return q4nx_packed_as_i32(packed)


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


Q_PHASE = 0
K_PHASE = 1
V_PHASE = 2


def _qkv_phase_config(phase: int, block: int) -> tuple[int, int]:
    if phase == Q_PHASE:
        return Q_WEIGHT_CHUNK_BASE + block * Q_CHUNKS_PER_RECORD, Q_CHUNKS_PER_RECORD
    if phase == K_PHASE:
        return K_WEIGHT_CHUNK_BASE + block * K_CHUNKS_PER_RECORD, K_CHUNKS_PER_RECORD
    if phase == V_PHASE:
        return V_WEIGHT_CHUNK_BASE + block * V_CHUNKS_PER_RECORD, V_CHUNKS_PER_RECORD
    raise ValueError(f"bad Q/K/V phase: {phase}")


def _qkv_body_record(
    packed: np.ndarray,
    hidden: np.ndarray,
    phase: int,
    block: int,
    group: int,
    row: int,
) -> np.ndarray:
    weight_base, chunks_per_record = _qkv_phase_config(phase, block)
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


def _qkv_body_compact(
    packed: np.ndarray,
    hidden: np.ndarray,
    phase: int,
    records: int,
) -> np.ndarray:
    packets = []
    for block in range(records):
        columns = []
        for group in range(len(MAIN_COLUMNS)):
            tile_records = [
                _qkv_body_record(packed, hidden, phase, block, group, row)
                for row in range(ROWS_PER_COLUMN)
            ]
            columns.append(column_compact_from_records(tile_records))
        packets.append(global_compact_from_columns(columns))
    return np.concatenate(packets).astype(np.int32)


def q_body_compact(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _qkv_body_compact(weights, values, Q_PHASE, Q_BODY_RECORDS)


def k_body_compact(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _qkv_body_compact(weights, values, K_PHASE, KV_BODY_RECORDS)


def v_body_compact(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return _qkv_body_compact(weights, values, V_PHASE, KV_BODY_RECORDS)


def _body_payload_word(body: np.ndarray, word: int) -> np.int32:
    block = word // (COMPACT_PACKET_DWORDS - 1)
    payload_word = word - block * (COMPACT_PACKET_DWORDS - 1)
    return body[block * COMPACT_PACKET_DWORDS + 1 + payload_word]


def _bf16_word_to_attention_s16(word: np.int32) -> np.int32:
    values = np.frombuffer(np.array([word], dtype=np.int32).tobytes(), dtype=bfloat16).astype(np.float32)
    scaled = values * ATTENTION_QKV_SCALE
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)).astype(np.int32)
    clipped = np.clip(rounded, -32768, 32767)
    low = int(clipped[0]) & 0xFFFF
    high = int(clipped[1]) & 0xFFFF
    return np.array(low | (high << 16), dtype=np.uint32).view(np.int32)


def _body_attention_word(body: np.ndarray, word: int) -> np.int32:
    return _bf16_word_to_attention_s16(_body_payload_word(body, word))


def q_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    compact = q_body_compact(packed, hidden)
    return np.array(
        [_body_attention_word(compact, word) for word in range(Q_BODY_RECORDS * (COMPACT_PACKET_DWORDS - 1))],
        dtype=np.int32,
    )


def _current_cache_payload_from_body(body: np.ndarray) -> np.ndarray:
    return np.array([_body_attention_word(body, word) for word in range(CURRENT_DWORDS)], dtype=np.int32)


def current_k_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    return _current_cache_payload_from_body(k_body_compact(packed, hidden))


def current_v_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    return _current_cache_payload_from_body(v_body_compact(packed, hidden))


def merged_k_cache_payload_body(
    schedule: DecodeSchedule,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    return _write_current(schedule, make_history_k_cache_payload(schedule), current_k_payload_body(packed, hidden))


def merged_v_cache_payload_body(
    schedule: DecodeSchedule,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    return _write_current(schedule, make_history_v_cache_payload(schedule), current_v_payload_body(packed, hidden))


def _normalized_replay_bf16(compact: np.ndarray) -> np.ndarray:
    payload_dwords = C1R2_PACKET_DWORDS - 1
    lanes = payload_dwords * 2
    compact_values = np.frombuffer(compact[1:].tobytes(), dtype=bfloat16).astype(np.float32)
    compact_fixed = np.trunc(compact_values * REPLAY_FIXED_SCALE).astype(np.int32)
    sum_sq = int(np.sum(compact_fixed.astype(np.int64) * compact_fixed.astype(np.int64)))
    denominator = math.isqrt(sum_sq // COMPACT_NUMERIC_LANES + 1)
    replay = np.empty(lanes, dtype=bfloat16)
    for lane in range(lanes):
        scaled = _trunc_div(int(compact_fixed[lane & (COMPACT_NUMERIC_LANES - 1)]) * 1024, denominator)
        replay[lane] = bfloat16(scaled / 1024.0)
    return replay


def _activation_slice(values: np.ndarray, chunk: int) -> np.ndarray:
    start = chunk * ACT_SLICE_BF16
    return values[start : start + ACT_SLICE_BF16]


def _unpack_s16(payload: np.ndarray, lane: int) -> int:
    word = int(payload[lane >> 1]) & 0xFFFF_FFFF
    raw = (word >> 16) & 0xFFFF if lane & 1 else word & 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def attention_payload_bf16(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    hidden: np.ndarray,
) -> np.ndarray:
    attention_words = attention_payload_from_qkv(
        schedule,
        q_payload_body(packed, hidden),
        merged_k_cache_payload_body(schedule, packed, hidden),
        merged_v_cache_payload_body(schedule, packed, hidden),
    )
    values = np.empty(attention_words.shape[0] * 2, dtype=bfloat16)
    for lane in range(values.shape[0]):
        values[lane] = bfloat16(_unpack_s16(attention_words, lane) / 1024.0)
    return np.frombuffer(values.tobytes(), dtype=np.int32).copy()


def q4nx_o_global_compact(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    activation_values = np.frombuffer(attention_payload_bf16(schedule, packed, hidden).tobytes(), dtype=bfloat16)
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        compact_records = []
        for row in range(ROWS_PER_COLUMN):
            accum = np.zeros(M_PER_TILE, dtype=np.float32)
            for chunk in range(O_WEIGHT_CHUNKS):
                accum += q4nx_matvec_from_chunk(
                    _chunk_for_tile(packed, group, row, FULL_LAYER_O_WEIGHT_CHUNK_BASE + chunk),
                    _activation_slice(activation_values, chunk),
                )
            record = np.empty(RECORD_DWORDS, dtype=np.int32)
            record[0] = (3 << 24) | (group << 16) | (row << 8) | 0xD0
            record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
            compact_records.append(record)
        columns.append(column_compact_from_records(compact_records))
    return global_compact_from_columns(columns)


def _upgate_tile_output(
    replay_values: np.ndarray,
    packed: np.ndarray,
    group: int,
    row: int,
    replay: int,
) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    weight_base = FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE + replay * UPGATE_CHUNKS_PER_REPLAY
    for chunk in range(UPGATE_CHUNKS_PER_REPLAY):
        accum += q4nx_matvec_from_chunk(
            _chunk_for_tile(packed, group, row, weight_base + chunk),
            _activation_slice(replay_values, chunk),
        )
    return accum.astype(bfloat16)


def _upgate_record(
    replay_values: np.ndarray,
    packed: np.ndarray,
    group: int,
    row: int,
    replay: int,
) -> np.ndarray:
    phase = 4 if (replay & 1) == 0 else 5
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = (phase << 24) | (group << 16) | (row << 8) | 0xD0
    output = _upgate_tile_output(replay_values, packed, group, row, replay)
    record[1:] = np.frombuffer(output.tobytes(), dtype=np.int32)
    return record


def _q4nx_upgate_global_compact_from_replay(
    replay_values: np.ndarray,
    packed: np.ndarray,
    replay: int,
) -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [
            _upgate_record(replay_values, packed, group, row, replay)
            for row in range(ROWS_PER_COLUMN)
        ]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def q4nx_upgate_global_compact(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    hidden: np.ndarray,
    replay: int,
) -> np.ndarray:
    replay_values = _normalized_replay_bf16(q4nx_o_global_compact(schedule, packed, hidden))
    return _q4nx_upgate_global_compact_from_replay(replay_values, packed, replay)


def _swiglu_bf16_inputs(input_slice: np.ndarray, slice_index: int) -> np.ndarray:
    values = np.frombuffer(input_slice.tobytes(), dtype=bfloat16).astype(np.float32)
    up = values[: C6R2_HALF_DWORDS * 2]
    gate = values[C6R2_HALF_DWORDS * 2 :]
    sigmoid = np.clip(0.5 + gate * 0.125, 0.0, 1.0)
    scale = 1.0 + slice_index / 256.0
    return (up * gate * sigmoid * scale).astype(bfloat16)


def swiglu_activation_payload(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    replay_values = _normalized_replay_bf16(q4nx_o_global_compact(schedule, packed, hidden))
    slices = []
    for slice_index in range(SWIGLU_SLICES):
        up = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2)[1:]
        gate = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2 + 1)[1:]
        if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
            raise RuntimeError(f"bad q4nx up/gate halves: {up.shape[0]}/{gate.shape[0]}")
        slices.append(_swiglu_bf16_inputs(np.concatenate((up, gate)).astype(np.int32), slice_index))
    activation = np.concatenate(slices).astype(bfloat16)
    return np.frombuffer(activation.tobytes(), dtype=np.int32).copy()


def q4nx_down_global_compact(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    activation = swiglu_activation_payload(schedule, packed, hidden)
    activation_values = np.frombuffer(activation.tobytes(), dtype=bfloat16)
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        compact_records = []
        for row in range(ROWS_PER_COLUMN):
            accum = np.zeros(M_PER_TILE, dtype=np.float32)
            for chunk in range(DOWN_CHUNKS):
                accum += q4nx_matvec_from_chunk(
                    _chunk_for_tile(packed, group, row, FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE + chunk),
                    _activation_slice(activation_values, chunk),
                )
            record = np.empty(RECORD_DWORDS, dtype=np.int32)
            record[0] = down_record_header(group, row)
            record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
            compact_records.append(record)
        columns.append(column_compact_from_records(compact_records))
    return global_compact_from_columns(columns)


def expected_output(
    schedule: DecodeSchedule = DEFAULT_SCHEDULE,
    packed: np.ndarray | None = None,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    hidden = make_hidden_bf16()
    return q4nx_down_global_compact(schedule, weights, hidden)


def validate_cache_writeback(
    schedule: DecodeSchedule,
    got_k: np.ndarray,
    got_v: np.ndarray,
    packed: np.ndarray,
    hidden: np.ndarray,
) -> list[str]:
    errors = validate_cache_layout_contract(schedule)
    expected_k = merged_k_cache_payload_body(schedule, packed, hidden)
    expected_v = merged_v_cache_payload_body(schedule, packed, hidden)
    if got_k.shape != expected_k.shape:
        errors.append(f"K cache shape mismatch: {got_k.shape} != {expected_k.shape}")
    if got_v.shape != expected_v.shape:
        errors.append(f"V cache shape mismatch: {got_v.shape} != {expected_v.shape}")
    if errors:
        return errors
    _validate_s16_cache("K", expected_k, got_k, errors)
    _validate_s16_cache("V", expected_v, got_v, errors)
    return errors


def _validate_s16_cache(label: str, expected: np.ndarray, got: np.ndarray, errors: list[str]) -> None:
    mismatch_count = 0
    for word_idx in range(expected.shape[0]):
        for lane in range(2):
            expected_lane = _unpack_s16(expected, word_idx * 2 + lane)
            got_lane = _unpack_s16(got, word_idx * 2 + lane)
            diff = abs(expected_lane - got_lane)
            if diff > CURRENT_CACHE_S16_TOL:
                if mismatch_count < 16:
                    errors.append(
                        f"{label} cache word {word_idx} lane {lane}: "
                        f"expected={expected_lane} got={got_lane} diff={diff}"
                    )
                mismatch_count += 1
    if mismatch_count > 16:
        errors.append(f"{mismatch_count - 16} additional {label} cache lane mismatches")


def validate_expected_output(expected: np.ndarray, got: np.ndarray) -> list[str]:
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    errors: list[str] = []
    if int(got[0]) != int(expected[0]):
        errors.append(f"compact header mismatch: expected={int(expected[0])} got={int(got[0])}")
    expected_values = np.frombuffer(expected[1:].tobytes(), dtype=bfloat16).astype(np.float32)
    got_values = np.frombuffer(got[1:].tobytes(), dtype=bfloat16).astype(np.float32)
    abs_err = np.abs(expected_values - got_values)
    rel_err = abs_err / np.maximum(np.abs(expected_values), 1e-6)
    mask = abs_err > np.maximum(FULL_PIPELINE_ABS_TOL, FULL_PIPELINE_REL_TOL * np.abs(expected_values))
    mismatch = np.where(mask)[0]
    for idx in mismatch[:32]:
        errors.append(
            f"value[{idx}]: expected={float(expected_values[idx]):.6f} "
            f"got={float(got_values[idx]):.6f} abs={float(abs_err[idx]):.6f} "
            f"rel={float(rel_err[idx]):.6f}"
        )
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional value mismatches")
    return errors


def route_summary(schedule: DecodeSchedule) -> list[str]:
    return [
        f"case={CASE_NAME}",
        "closed_loop_1=host hidden -> c1r2 packet0 replay -> main16 Q4NX Q/K/V -> c1r3 bf16-to-s16 attention ABI -> current K/V writeback -> KV scan -> kv16 attention -> packet2 -> main16 O",
        "closed_loop_2=bf16 attention packet2 plus Q4NX O -> c1r2 bf16 replay -> main16 Q4NX up/gate -> c6r2 bf16-input SwiGLU",
        "closed_loop_3=packet1 down activation plus row1 S2MM4/5 Q4NX weights -> main16 DMA0/DMA1",
        "output=main16 Q4NX down compact records -> row1/c1r1 compact -> c1r2 compact drain",
        f"decode_token={schedule.current_token}, blocks={schedule.kv_blocks}, tail={schedule.tail_tokens}",
        f"hidden={HIDDEN_DWORDS} dwords, qkv_weight_chunks={QKV_BODY_WEIGHT_CHUNKS}, tail_weight_chunks={FULL_LAYER_TOTAL_WEIGHT_CHUNKS - QKV_BODY_WEIGHT_CHUNKS}, host_output={OUTPUT_DWORDS} dwords",
    ]
