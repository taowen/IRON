"""CPU reference for current K/V full-layer tail with Q4NX up/gate and down."""

from __future__ import annotations

import math

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    C6R2_HALF_DWORDS,
    CHUNK_BF16,
    COMPACT_PACKET_DWORDS,
    HIDDEN_DIM,
    MAIN_COLUMNS,
    M_PER_TILE,
    RECORD_DWORDS,
    ROWS_PER_COLUMN,
    ROWS_PER_PATCH,
    SWIGLU_SLICES,
)
from projection_schedule import (
    DOWN_CHUNKS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_TOTAL_WEIGHT_CHUNKS,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    DOWN_BODY_RECORDS,
    O_BODY_RECORDS,
    O_CHUNKS_PER_RECORD,
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
OUTPUT_DWORDS = HIDDEN_DIM // 2
Q_BODY_COMPACT_DWORDS = Q_BODY_RECORDS * COMPACT_PACKET_DWORDS
KV_BODY_COMPACT_DWORDS = KV_BODY_RECORDS * COMPACT_PACKET_DWORDS
PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * FULL_LAYER_TOTAL_WEIGHT_CHUNKS * CHUNK_BF16
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
REPLAY_FIXED_SCALE = 256.0
FULL_PIPELINE_ABS_TOL = 16.0
FULL_PIPELINE_REL_TOL = 1.00
ATTENTION_QKV_SCALE = 16.0
CURRENT_CACHE_S16_TOL = 1
SIGMOID_TABLE_SCALE = 8.0
SIGMOID_TABLE = np.array(
    [
        0.5000000000,
        0.5312093734,
        0.5621765009,
        0.5926666000,
        0.6224593312,
        0.6513548647,
        0.6791786992,
        0.7057850278,
        0.7310585786,
        0.7549149869,
        0.7772998612,
        0.7981867777,
        0.8175744762,
        0.8354835371,
        0.8519528020,
        0.8670357598,
        0.8807970780,
        0.8933094061,
        0.9046505351,
        0.9149009550,
        0.9241418200,
        0.9324533089,
        0.9399133498,
        0.9465966702,
        0.9525741268,
        0.9579122721,
        0.9626731127,
        0.9669140216,
        0.9706877692,
        0.9740426428,
        0.9770226301,
        0.9796676467,
        0.9820137900,
        0.9840936083,
        0.9859363730,
        0.9875683491,
        0.9890130574,
        0.9902915235,
        0.9914225146,
        0.9924227587,
        0.9933071491,
        0.9940889311,
        0.9947798743,
        0.9953904278,
        0.9959298623,
        0.9964063974,
        0.9968273172,
        0.9971990730,
        0.9975273768,
        0.9978172836,
        0.9980732653,
        0.9982992776,
        0.9984988177,
        0.9986749776,
        0.9988304897,
        0.9989677690,
        0.9990889488,
        0.9991959141,
        0.9992903296,
        0.9993736658,
        0.9994472214,
        0.9995121429,
        0.9995694429,
        0.9996200155,
        0.9996646499,
    ],
    dtype=np.float32,
)


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


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


def _normalized_full_vector_bf16(values: np.ndarray) -> np.ndarray:
    if values.shape != (HIDDEN_DIM,):
        raise ValueError(f"full vector shape mismatch: {values.shape} != {(HIDDEN_DIM,)}")
    vector = values.astype(np.float32)
    fixed = np.trunc(vector * REPLAY_FIXED_SCALE).astype(np.int32)
    sum_sq = int(np.sum(fixed.astype(np.int64) * fixed.astype(np.int64)))
    denominator = math.isqrt(sum_sq // HIDDEN_DIM + 1)
    replay = np.empty(HIDDEN_DIM, dtype=bfloat16)
    for lane in range(HIDDEN_DIM):
        scaled = _trunc_div(int(fixed[lane]) * 1024, denominator)
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


def _compact_payload_bf16(compact: np.ndarray) -> np.ndarray:
    return np.frombuffer(compact[1:].tobytes(), dtype=bfloat16).copy()


def _full_vector_from_compacts(compacts: np.ndarray, records: int) -> np.ndarray:
    if compacts.shape != (records * COMPACT_PACKET_DWORDS,):
        raise ValueError(f"compact shape mismatch: {compacts.shape} != {(records * COMPACT_PACKET_DWORDS,)}")
    parts = [
        _compact_payload_bf16(compacts[record * COMPACT_PACKET_DWORDS : (record + 1) * COMPACT_PACKET_DWORDS])
        for record in range(records)
    ]
    return np.concatenate(parts).astype(bfloat16)


def q4nx_o_global_compacts(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    activation_values = np.frombuffer(attention_payload_bf16(schedule, packed, hidden).tobytes(), dtype=bfloat16)
    compacts = []
    for block in range(O_BODY_RECORDS):
        columns = []
        for group in range(len(MAIN_COLUMNS)):
            compact_records = []
            for row in range(ROWS_PER_COLUMN):
                accum = np.zeros(M_PER_TILE, dtype=np.float32)
                for chunk in range(O_CHUNKS_PER_RECORD):
                    weight_chunk = FULL_LAYER_O_WEIGHT_CHUNK_BASE + chunk * O_BODY_RECORDS + block
                    accum += q4nx_matvec_from_chunk(
                        _chunk_for_tile(packed, group, row, weight_chunk),
                        _activation_slice(activation_values, chunk),
                    )
                record = np.empty(RECORD_DWORDS, dtype=np.int32)
                record[0] = (3 << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0
                record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
                compact_records.append(record)
            columns.append(column_compact_from_records(compact_records))
        compacts.append(global_compact_from_columns(columns))
    return np.concatenate(compacts).astype(np.int32)


def q4nx_o_global_compact(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    return q4nx_o_global_compacts(schedule, packed, hidden)[:COMPACT_PACKET_DWORDS]


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
    o_values = _full_vector_from_compacts(q4nx_o_global_compacts(schedule, packed, hidden), O_BODY_RECORDS)
    replay_values = _normalized_full_vector_bf16(o_values)
    return _q4nx_upgate_global_compact_from_replay(replay_values, packed, replay)


def _sigmoid_approx(values: np.ndarray) -> np.ndarray:
    scaled = np.minimum(np.abs(values), 8.0) * SIGMOID_TABLE_SCALE
    index = np.trunc(scaled).astype(np.int32)
    base_index = np.minimum(index, SIGMOID_TABLE.shape[0] - 2)
    fraction = scaled - base_index.astype(np.float32)
    low = SIGMOID_TABLE[base_index]
    high = SIGMOID_TABLE[base_index + 1]
    positive = np.where(
        index >= SIGMOID_TABLE.shape[0] - 1,
        SIGMOID_TABLE[-1],
        low + (high - low) * fraction,
    )
    sigmoid = np.where(values >= 0.0, positive, 1.0 - positive)
    return np.where(values > 8.0, 1.0, np.where(values < -8.0, 0.0, sigmoid))


def _swiglu_bf16_inputs(input_slice: np.ndarray) -> np.ndarray:
    values = np.frombuffer(input_slice.tobytes(), dtype=bfloat16).astype(np.float32)
    up = values[: C6R2_HALF_DWORDS * 2]
    gate = values[C6R2_HALF_DWORDS * 2 :]
    return (up * gate * _sigmoid_approx(gate)).astype(bfloat16)


def swiglu_activation_payload(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    o_values = _full_vector_from_compacts(q4nx_o_global_compacts(schedule, packed, hidden), O_BODY_RECORDS)
    replay_values = _normalized_full_vector_bf16(o_values)
    slices = []
    for slice_index in range(SWIGLU_SLICES):
        up = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2)[1:]
        gate = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2 + 1)[1:]
        if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
            raise RuntimeError(f"bad q4nx up/gate halves: {up.shape[0]}/{gate.shape[0]}")
        slices.append(_swiglu_bf16_inputs(np.concatenate((up, gate)).astype(np.int32)))
    activation = np.concatenate(slices).astype(bfloat16)
    return np.frombuffer(activation.tobytes(), dtype=np.int32).copy()


def q4nx_down_global_compacts(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    activation = swiglu_activation_payload(schedule, packed, hidden)
    activation_values = np.frombuffer(activation.tobytes(), dtype=bfloat16)
    compacts = []
    for block in range(DOWN_BODY_RECORDS):
        columns = []
        for group in range(len(MAIN_COLUMNS)):
            compact_records = []
            for row in range(ROWS_PER_COLUMN):
                accum = np.zeros(M_PER_TILE, dtype=np.float32)
                for chunk in range(DOWN_CHUNKS):
                    weight_chunk = FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE + chunk * DOWN_BODY_RECORDS + block
                    accum += q4nx_matvec_from_chunk(
                        _chunk_for_tile(packed, group, row, weight_chunk),
                        _activation_slice(activation_values, chunk),
                    )
                record = np.empty(RECORD_DWORDS, dtype=np.int32)
                record[0] = (6 << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0
                record[1:] = np.frombuffer(accum.astype(bfloat16).tobytes(), dtype=np.int32)
                compact_records.append(record)
            columns.append(column_compact_from_records(compact_records))
        compacts.append(global_compact_from_columns(columns))
    return np.concatenate(compacts).astype(np.int32)


def q4nx_down_global_compact(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    return q4nx_down_global_compacts(schedule, packed, hidden)[:COMPACT_PACKET_DWORDS]


def q4nx_down_hidden_output(schedule: DecodeSchedule, packed: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    values = _full_vector_from_compacts(q4nx_down_global_compacts(schedule, packed, hidden), DOWN_BODY_RECORDS)
    return np.frombuffer(values.tobytes(), dtype=np.int32).copy()


def expected_output(
    schedule: DecodeSchedule = DEFAULT_SCHEDULE,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    values = make_hidden_bf16() if hidden is None else hidden
    return q4nx_down_hidden_output(schedule, weights, values)


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
    expected_values = np.frombuffer(expected.tobytes(), dtype=bfloat16).astype(np.float32)
    got_values = np.frombuffer(got.tobytes(), dtype=bfloat16).astype(np.float32)
    if not bool(np.all(np.isfinite(expected_values))):
        bad = int(np.flatnonzero(~np.isfinite(expected_values))[0])
        errors.append(f"expected output is not finite at bf16 lane {bad}: {float(expected_values[bad])}")
    if not bool(np.all(np.isfinite(got_values))):
        bad = int(np.flatnonzero(~np.isfinite(got_values))[0])
        errors.append(f"NPU output is not finite at bf16 lane {bad}: {float(got_values[bad])}")
    if errors:
        return errors
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
