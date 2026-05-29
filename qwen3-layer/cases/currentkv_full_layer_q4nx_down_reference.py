"""CPU reference for current K/V full-layer tail with Q4NX up/gate and down."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    ACT_SLICE_BF16,
    C6R2_HALF_DWORDS,
    CHUNK_BF16,
    COMPACT_PACKET_DWORDS,
    HEAD_DIM,
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
    make_q4nx_chunk,
    make_hidden_bf16,
    packed_as_i32 as q4nx_packed_as_i32,
    q4nx_matvec_from_chunk,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    CURRENT_DWORDS,
    DecodeSchedule,
    HEAD_DWORDS,
    KV_HEADS,
    logical_cache_index,
    make_decode_schedule,
    pack_logical_cache_to_npu,
    unpack_npu_cache_to_logical,
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
RMS_NORM_DWORDS = HIDDEN_DWORDS * 2
QK_ROPE_BF16 = HEAD_DIM + HEAD_DIM + HEAD_DIM
QK_ROPE_DWORDS = QK_ROPE_BF16 // 2
AUX_DWORDS = RMS_NORM_DWORDS + QK_ROPE_DWORDS
Q_BODY_COMPACT_DWORDS = Q_BODY_RECORDS * COMPACT_PACKET_DWORDS
KV_BODY_COMPACT_DWORDS = KV_BODY_RECORDS * COMPACT_PACKET_DWORDS
PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * FULL_LAYER_TOTAL_WEIGHT_CHUNKS * CHUNK_BF16
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
TOTAL_WEIGHT_AND_AUX_I32 = TOTAL_WEIGHT_I32 + AUX_DWORDS
FULL_PIPELINE_ABS_TOL = 16.0
FULL_PIPELINE_REL_TOL = 1.00
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


def _default_norm_weight() -> np.ndarray:
    return np.ones(HIDDEN_DIM, dtype=bfloat16)


def _default_head_norm_weight() -> np.ndarray:
    return np.ones(HEAD_DIM, dtype=bfloat16)


def hidden_input_as_i32(hidden: np.ndarray | None = None) -> np.ndarray:
    raw_hidden = make_hidden_bf16() if hidden is None else hidden
    if raw_hidden.shape != (HIDDEN_DIM,):
        raise ValueError(f"hidden shape mismatch: {raw_hidden.shape} != {(HIDDEN_DIM,)}")
    packed = np.frombuffer(raw_hidden.astype(bfloat16).tobytes(), dtype=np.int32).copy()
    if packed.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"hidden input shape mismatch: {packed.shape} != {(HIDDEN_DWORDS,)}")
    return packed


def qk_rope_side_bf16(
    current_token: int,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    q_weight = _default_head_norm_weight() if q_norm_weight is None else q_norm_weight
    k_weight = _default_head_norm_weight() if k_norm_weight is None else k_norm_weight
    for label, values in (
        ("q_norm_weight", q_weight),
        ("k_norm_weight", k_weight),
    ):
        if values.shape != (HEAD_DIM,):
            raise ValueError(f"{label} shape mismatch: {values.shape} != {(HEAD_DIM,)}")
    dims = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
    inv_freq = np.power(np.float32(rope_theta), -dims / np.float32(HEAD_DIM))
    angles = np.float32(current_token) * inv_freq
    side = np.concatenate(
        (
            q_weight.astype(bfloat16),
            k_weight.astype(bfloat16),
            np.cos(angles).astype(bfloat16),
            np.sin(angles).astype(bfloat16),
        )
    )
    if side.shape != (QK_ROPE_BF16,):
        raise RuntimeError(f"q/k norm+RoPE side shape mismatch: {side.shape} != {(QK_ROPE_BF16,)}")
    return side


def aux_as_i32(
    current_token: int = DEFAULT_SCHEDULE.current_token,
    input_norm_weight: np.ndarray | None = None,
    post_norm_weight: np.ndarray | None = None,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    input_weight = _default_norm_weight() if input_norm_weight is None else input_norm_weight
    post_weight = _default_norm_weight() if post_norm_weight is None else post_norm_weight
    for label, values in (
        ("input_norm_weight", input_weight),
        ("post_norm_weight", post_weight),
    ):
        if values.shape != (HIDDEN_DIM,):
            raise ValueError(f"{label} shape mismatch: {values.shape} != {(HIDDEN_DIM,)}")
    qk_side = qk_rope_side_bf16(current_token, q_norm_weight, k_norm_weight, rope_theta)
    payload = np.concatenate(
        (
            input_weight.astype(bfloat16),
            post_weight.astype(bfloat16),
            qk_side,
        )
    )
    packed = np.frombuffer(payload.tobytes(), dtype=np.int32).copy()
    if packed.shape != (AUX_DWORDS,):
        raise RuntimeError(f"aux shape mismatch: {packed.shape} != {(AUX_DWORDS,)}")
    return packed


def weights_with_aux_i32(
    packed: np.ndarray,
    current_token: int = DEFAULT_SCHEDULE.current_token,
    input_norm_weight: np.ndarray | None = None,
    post_norm_weight: np.ndarray | None = None,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    weight_words = packed_as_i32(packed)
    aux_words = aux_as_i32(
        current_token,
        input_norm_weight,
        post_norm_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )
    combined = np.concatenate((weight_words, aux_words)).astype(np.int32)
    if combined.shape != (TOTAL_WEIGHT_AND_AUX_I32,):
        raise RuntimeError(
            f"weight+aux shape mismatch: {combined.shape} != {(TOTAL_WEIGHT_AND_AUX_I32,)}"
        )
    return combined


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


def _weighted_normalized_full_vector_bf16(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if values.shape != (HIDDEN_DIM,):
        raise ValueError(f"full vector shape mismatch: {values.shape} != {(HIDDEN_DIM,)}")
    if weight.shape != (HIDDEN_DIM,):
        raise ValueError(f"norm weight shape mismatch: {weight.shape} != {(HIDDEN_DIM,)}")
    vector = values.astype(np.float32)
    weights = weight.astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(float(np.mean(vector * vector)) + 0.000001))
    return (vector * scale * weights).astype(bfloat16)


def input_norm_activation(hidden: np.ndarray, input_norm_weight: np.ndarray) -> np.ndarray:
    return _weighted_normalized_full_vector_bf16(hidden, input_norm_weight)


def _post_attention_residual_and_replay(
    raw_hidden: np.ndarray,
    o_values: np.ndarray,
    post_norm_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_hidden.shape != (HIDDEN_DIM,) or o_values.shape != (HIDDEN_DIM,):
        raise ValueError(f"post-attention vector shape mismatch: {raw_hidden.shape}/{o_values.shape}")
    residual = (raw_hidden.astype(np.float32) + o_values.astype(np.float32)).astype(bfloat16)
    return residual, _weighted_normalized_full_vector_bf16(residual, post_norm_weight)


def _activation_slice(values: np.ndarray, chunk: int) -> np.ndarray:
    start = chunk * ACT_SLICE_BF16
    return values[start : start + ACT_SLICE_BF16]


def _body_payload_bf16(body: np.ndarray, records: int) -> np.ndarray:
    words = np.array(
        [_body_payload_word(body, word) for word in range(records * (COMPACT_PACKET_DWORDS - 1))],
        dtype=np.int32,
    )
    return np.frombuffer(words.tobytes(), dtype=bfloat16).copy()


def _head_rms_norm(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if values.shape[0] % HEAD_DIM != 0:
        raise ValueError(f"head vector shape mismatch: {values.shape}")
    if weight.shape != (HEAD_DIM,):
        raise ValueError(f"head norm shape mismatch: {weight.shape} != {(HEAD_DIM,)}")
    heads = values.astype(np.float32).reshape(-1, HEAD_DIM)
    weights = weight.astype(np.float32)
    output = np.empty_like(heads, dtype=np.float32)
    for head in range(heads.shape[0]):
        scale = np.float32(1.0 / np.sqrt(float(np.mean(heads[head] * heads[head])) + 0.000001))
        output[head] = heads[head] * scale * weights
    return output.reshape(values.shape).astype(bfloat16)


def _apply_rope(values: np.ndarray, current_token: int, rope_theta: float) -> np.ndarray:
    heads = values.astype(np.float32).reshape(-1, HEAD_DIM)
    dims = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
    inv_freq = np.power(np.float32(rope_theta), -dims / np.float32(HEAD_DIM))
    angles = np.float32(current_token) * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    output = np.empty_like(heads)
    even = heads[:, 0::2]
    odd = heads[:, 1::2]
    output[:, 0::2] = even * cos - odd * sin
    output[:, 1::2] = even * sin + odd * cos
    return output.reshape(values.shape).astype(bfloat16)


def q_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    q_norm_weight: np.ndarray | None = None,
    current_token: int = DEFAULT_SCHEDULE.current_token,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    compact = q_body_compact(packed, hidden)
    q_values = _body_payload_bf16(compact, Q_BODY_RECORDS)
    q_norm = _default_head_norm_weight() if q_norm_weight is None else q_norm_weight
    q_payload = _apply_rope(_head_rms_norm(q_values, q_norm), current_token, rope_theta)
    return np.frombuffer(q_payload.tobytes(), dtype=np.int32).copy()


def _current_cache_payload(values: np.ndarray) -> np.ndarray:
    words = np.frombuffer(values.astype(bfloat16).tobytes(), dtype=np.int32).copy()
    if words.shape != (CURRENT_DWORDS,):
        raise RuntimeError(f"current cache payload shape mismatch: {words.shape} != {(CURRENT_DWORDS,)}")
    return words


def current_k_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    current_token: int = DEFAULT_SCHEDULE.current_token,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    compact = k_body_compact(packed, hidden)
    k_values = _body_payload_bf16(compact, KV_BODY_RECORDS)
    k_norm = _default_head_norm_weight() if k_norm_weight is None else k_norm_weight
    current_k = _apply_rope(_head_rms_norm(k_values, k_norm), current_token, rope_theta)
    return _current_cache_payload(current_k)


def current_v_payload_body(
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
) -> np.ndarray:
    compact = v_body_compact(packed, hidden)
    return _current_cache_payload(_body_payload_bf16(compact, KV_BODY_RECORDS))


def _history_value(token: int, head: int, dim: int, is_v: bool) -> float:
    if is_v:
        raw = ((head + 5) * 7 + token * 11 + dim * 2) % 127 - 63
    else:
        raw = ((head + 3) * 9 + token * 5 + dim * 3) % 127 - 63
    return raw / 256.0


def bf16_cache_payload(schedule: DecodeSchedule, cache_values: np.ndarray) -> np.ndarray:
    expected = (schedule.total_context, KV_HEADS, HEAD_DIM)
    if cache_values.shape != expected:
        raise ValueError(f"bf16 cache shape mismatch: {cache_values.shape} != {expected}")
    logical = np.empty(schedule.logical_cache_dwords, dtype=np.int32)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim_pair in range(HEAD_DWORDS):
                low_dim = dim_pair * 2
                pair = cache_values[token, head, low_dim : low_dim + 2].astype(bfloat16)
                logical[logical_cache_index(token, head, dim_pair)] = np.frombuffer(pair.tobytes(), dtype=np.int32)[0]
    return pack_logical_cache_to_npu(schedule, logical)


def make_history_k_cache_payload_bf16(schedule: DecodeSchedule) -> np.ndarray:
    values = np.empty((schedule.total_context, KV_HEADS, HEAD_DIM), dtype=bfloat16)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim in range(HEAD_DIM):
                values[token, head, dim] = bfloat16(_history_value(token, head, dim, False))
    return bf16_cache_payload(schedule, values)


def make_history_v_cache_payload_bf16(schedule: DecodeSchedule) -> np.ndarray:
    values = np.empty((schedule.total_context, KV_HEADS, HEAD_DIM), dtype=bfloat16)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim in range(HEAD_DIM):
                values[token, head, dim] = bfloat16(_history_value(token, head, dim, True))
    return bf16_cache_payload(schedule, values)


def merged_k_cache_payload_body(
    schedule: DecodeSchedule,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
    history_cache: np.ndarray | None = None,
) -> np.ndarray:
    base = make_history_k_cache_payload_bf16(schedule) if history_cache is None else history_cache
    return _write_current(
        schedule,
        base,
        current_k_payload_body(packed, hidden, k_norm_weight, schedule.current_token, rope_theta),
    )


def merged_v_cache_payload_body(
    schedule: DecodeSchedule,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    history_cache: np.ndarray | None = None,
) -> np.ndarray:
    base = make_history_v_cache_payload_bf16(schedule) if history_cache is None else history_cache
    return _write_current(schedule, base, current_v_payload_body(packed, hidden))


def attention_payload_bf16(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    hidden: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
    history_k_cache: np.ndarray | None = None,
    history_v_cache: np.ndarray | None = None,
) -> np.ndarray:
    q_words = q_payload_body(packed, hidden, q_norm_weight, schedule.current_token, rope_theta)
    k_cache = merged_k_cache_payload_body(schedule, packed, hidden, k_norm_weight, rope_theta, history_k_cache)
    v_cache = merged_v_cache_payload_body(schedule, packed, hidden, history_v_cache)
    q = np.frombuffer(q_words.tobytes(), dtype=bfloat16).astype(np.float32).reshape(-1, HEAD_DIM)
    k_values = np.frombuffer(unpack_npu_cache_to_logical(schedule, k_cache).tobytes(), dtype=bfloat16).astype(np.float32)
    v_values = np.frombuffer(unpack_npu_cache_to_logical(schedule, v_cache).tobytes(), dtype=bfloat16).astype(np.float32)
    k_values = k_values.reshape(schedule.total_context, KV_HEADS, HEAD_DIM)[: schedule.current_token + 1]
    v_values = v_values.reshape(schedule.total_context, KV_HEADS, HEAD_DIM)[: schedule.current_token + 1]
    output = np.empty((q.shape[0], HEAD_DIM), dtype=bfloat16)
    score_scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    for q_head in range(q.shape[0]):
        kv_head = q_head // (q.shape[0] // KV_HEADS)
        scores = np.einsum("d,td->t", q[q_head], k_values[:, kv_head, :]) * score_scale
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights)
        output[q_head] = np.einsum("t,td->d", weights, v_values[:, kv_head, :]).astype(bfloat16)
    return np.frombuffer(output.reshape(-1).tobytes(), dtype=np.int32).copy()

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


def q4nx_o_global_compacts(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    hidden: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
    history_k_cache: np.ndarray | None = None,
    history_v_cache: np.ndarray | None = None,
) -> np.ndarray:
    activation_values = np.frombuffer(
        attention_payload_bf16(
            schedule,
            packed,
            hidden,
            q_norm_weight,
            k_norm_weight,
            rope_theta,
            history_k_cache,
            history_v_cache,
        ).tobytes(),
        dtype=bfloat16,
    )
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


def q4nx_o_global_compact(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    hidden: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    return q4nx_o_global_compacts(schedule, packed, hidden, q_norm_weight, k_norm_weight, rope_theta)[
        :COMPACT_PACKET_DWORDS
    ]


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
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    replay: int,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    o_values = _full_vector_from_compacts(
        q4nx_o_global_compacts(schedule, packed, qkv_activation, q_norm_weight, k_norm_weight, rope_theta),
        O_BODY_RECORDS,
    )
    _post_residual, replay_values = _post_attention_residual_and_replay(
        raw_hidden,
        o_values,
        post_norm_weight,
    )
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


def _ffn_replay_values(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    o_values = _full_vector_from_compacts(
        q4nx_o_global_compacts(schedule, packed, qkv_activation, q_norm_weight, k_norm_weight, rope_theta),
        O_BODY_RECORDS,
    )
    return _post_attention_residual_and_replay(raw_hidden, o_values, post_norm_weight)


def swiglu_activation_payload(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    _post_residual, replay_values = _ffn_replay_values(
        schedule,
        packed,
        qkv_activation,
        raw_hidden,
        post_norm_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )
    slices = []
    for slice_index in range(SWIGLU_SLICES):
        up = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2)[1:]
        gate = _q4nx_upgate_global_compact_from_replay(replay_values, packed, slice_index * 2 + 1)[1:]
        if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
            raise RuntimeError(f"bad q4nx up/gate halves: {up.shape[0]}/{gate.shape[0]}")
        slices.append(_swiglu_bf16_inputs(np.concatenate((up, gate)).astype(np.int32)))
    activation = np.concatenate(slices).astype(bfloat16)
    return np.frombuffer(activation.tobytes(), dtype=np.int32).copy()


def q4nx_down_global_compacts(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    activation = swiglu_activation_payload(
        schedule,
        packed,
        qkv_activation,
        raw_hidden,
        post_norm_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )
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


def q4nx_down_global_compact(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    return q4nx_down_global_compacts(
        schedule,
        packed,
        qkv_activation,
        raw_hidden,
        post_norm_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )[:COMPACT_PACKET_DWORDS]


def q4nx_down_hidden_output(
    schedule: DecodeSchedule,
    packed: np.ndarray,
    qkv_activation: np.ndarray,
    raw_hidden: np.ndarray,
    post_norm_weight: np.ndarray,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    residual, _ffn_replay = _ffn_replay_values(
        schedule,
        packed,
        qkv_activation,
        raw_hidden,
        post_norm_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )
    down_values = _full_vector_from_compacts(
        q4nx_down_global_compacts(
            schedule,
            packed,
            qkv_activation,
            raw_hidden,
            post_norm_weight,
            q_norm_weight,
            k_norm_weight,
            rope_theta,
        ),
        DOWN_BODY_RECORDS,
    )
    output = (residual.astype(np.float32) + down_values.astype(np.float32)).astype(bfloat16)
    return np.frombuffer(output.tobytes(), dtype=np.int32).copy()


def expected_output(
    schedule: DecodeSchedule = DEFAULT_SCHEDULE,
    packed: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    input_norm_weight: np.ndarray | None = None,
    post_norm_weight: np.ndarray | None = None,
    q_norm_weight: np.ndarray | None = None,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
) -> np.ndarray:
    weights = make_packed_weights() if packed is None else packed
    raw_hidden = make_hidden_bf16() if hidden is None else hidden
    input_weight = _default_norm_weight() if input_norm_weight is None else input_norm_weight
    post_weight = _default_norm_weight() if post_norm_weight is None else post_norm_weight
    qkv_activation = input_norm_activation(raw_hidden, input_weight)
    return q4nx_down_hidden_output(
        schedule,
        weights,
        qkv_activation,
        raw_hidden,
        post_weight,
        q_norm_weight,
        k_norm_weight,
        rope_theta,
    )


def validate_cache_writeback(
    schedule: DecodeSchedule,
    got_k: np.ndarray,
    got_v: np.ndarray,
    packed: np.ndarray,
    hidden: np.ndarray,
    k_norm_weight: np.ndarray | None = None,
    rope_theta: float = 1_000_000.0,
    history_k_cache: np.ndarray | None = None,
    history_v_cache: np.ndarray | None = None,
) -> list[str]:
    errors = validate_cache_layout_contract(schedule)
    expected_k = merged_k_cache_payload_body(schedule, packed, hidden, k_norm_weight, rope_theta, history_k_cache)
    expected_v = merged_v_cache_payload_body(schedule, packed, hidden, history_v_cache)
    if got_k.shape != expected_k.shape:
        errors.append(f"K cache shape mismatch: {got_k.shape} != {expected_k.shape}")
    if got_v.shape != expected_v.shape:
        errors.append(f"V cache shape mismatch: {got_v.shape} != {expected_v.shape}")
    if errors:
        return errors
    _validate_bf16_cache("K", expected_k, got_k, errors, 0.035)
    _validate_bf16_cache("V", expected_v, got_v, errors, 0.05)
    return errors


def _validate_bf16_cache(
    label: str,
    expected: np.ndarray,
    got: np.ndarray,
    errors: list[str],
    abs_tol: float,
) -> None:
    expected_values = np.frombuffer(expected.tobytes(), dtype=bfloat16).astype(np.float32)
    got_values = np.frombuffer(got.tobytes(), dtype=bfloat16).astype(np.float32)
    expected_bad = np.flatnonzero(~np.isfinite(expected_values))
    got_bad = np.flatnonzero(~np.isfinite(got_values))
    for lane in expected_bad[:16]:
        errors.append(f"{label} cache expected lane {int(lane)} is not finite: {float(expected_values[lane])}")
    for lane in got_bad[:16]:
        errors.append(f"{label} cache got lane {int(lane)} is not finite: {float(got_values[lane])}")
    if expected_bad.size > 16:
        errors.append(f"{expected_bad.size - 16} additional non-finite expected {label} cache lanes")
    if got_bad.size > 16:
        errors.append(f"{got_bad.size - 16} additional non-finite got {label} cache lanes")
    diff = np.abs(expected_values - got_values)
    finite = np.isfinite(expected_values) & np.isfinite(got_values)
    mismatch = np.flatnonzero((diff > abs_tol) & finite)
    for lane in mismatch[:16]:
        errors.append(
            f"{label} cache lane {int(lane)}: expected={float(expected_values[lane]):.6f} "
            f"got={float(got_values[lane]):.6f} abs={float(diff[lane]):.6f}"
        )
    if mismatch.size > 16:
        errors.append(f"{mismatch.size - 16} additional {label} cache lane mismatches")


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
        "closed_loop_1=host raw hidden+RMSNorm weights -> c1r2 input RMSNorm replay -> main16 Q4NX Q/K/V -> c1r3 Q/K norm+RoPE bf16 ABI -> current K/V writeback -> KV scan -> bf16 attention -> packet2 -> main16 O",
        "closed_loop_2=bf16 attention packet2 plus Q4NX O -> c1r2 residual+post RMSNorm replay -> main16 Q4NX up/gate -> c6r2 bf16-input SwiGLU",
        "closed_loop_3=packet1 down activation plus row1 S2MM4/5 Q4NX weights -> main16 DMA0/DMA1",
        "output=main16 Q4NX down compact records -> row1/c1r1 compact -> c1r2 compact drain",
        f"decode_token={schedule.current_token}, blocks={schedule.kv_blocks}, tail={schedule.tail_tokens}",
        f"hidden={HIDDEN_DWORDS} dwords, aux={AUX_DWORDS} dwords, qkv_weight_chunks={QKV_BODY_WEIGHT_CHUNKS}, tail_weight_chunks={FULL_LAYER_TOTAL_WEIGHT_CHUNKS - QKV_BODY_WEIGHT_CHUNKS}, host_output={OUTPUT_DWORDS} dwords",
    ]
