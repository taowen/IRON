"""CPU reference for current K/V cache writeback and rounded KV scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cases import attention_block_reference as attention
from cases import kv_scan_reference as kvscan
from cases import q_projection_attention_reference as mainq
from qkv_compact_reference import K_PHASE, V_PHASE, qkv_global_compact

CASE_NAME = "decode-cache-reference"
CURRENT_DWORDS = 512
CURRENT_PACKET_K = 8
CURRENT_PACKET_V = 9
MAX_ITERATED_SCAN_BLOCKS = 63

CONTEXT = attention.CONTEXT
BLOCK_TOKENS = attention.CONTEXT
HEAD_DIM = attention.HEAD_DIM
HEAD_DWORDS = attention.HEAD_DIM // 2
HEADS_PER_WINDOW = attention.HEADS_PER_WINDOW
HOST_OUTPUT_DWORDS = mainq.HOST_OUTPUT_DWORDS
K_CACHE_SIDE_DWORDS = kvscan.K_CACHE_SIDE_DWORDS
K_WINDOW_DWORDS = kvscan.K_WINDOW_DWORDS
KV_HEADS = 8
KV_HEADS_PER_WINDOW = attention.KV_HEADS_PER_WINDOW
KV_SIDE_DWORDS = kvscan.KV_SIDE_DWORDS
OUTPUT_DWORDS = kvscan.OUTPUT_DWORDS
Q_DWORDS = kvscan.Q_DWORDS
SCALAR_DWORDS = kvscan.SCALAR_DWORDS
V_CACHE_SIDE_DWORDS = kvscan.V_CACHE_SIDE_DWORDS
V_WINDOW_DWORDS = kvscan.V_WINDOW_DWORDS
WEIGHT_DWORDS = kvscan.WEIGHT_DWORDS
WINDOW_DWORDS = kvscan.WINDOW_DWORDS
WINDOW_HEAD_DWORDS = BLOCK_TOKENS * HEAD_DWORDS
CACHE_BLOCK_DWORDS = KV_HEADS * WINDOW_HEAD_DWORDS
ACCUM_LANES = HEADS_PER_WINDOW * HEAD_DIM
SOFTMAX_SCALE = attention.SOFTMAX_SCALE
DEFAULT_CURRENT_TOKEN = 127


@dataclass(frozen=True)
class DecodeSchedule:
    current_token: int

    def __post_init__(self) -> None:
        if self.current_token < 0:
            raise ValueError("current_token must be non-negative")
        if self.kv_blocks > MAX_ITERATED_SCAN_BLOCKS:
            raise ValueError(
                f"current_token {self.current_token} needs {self.kv_blocks} scan blocks; "
                f"iterated shim BD supports at most {MAX_ITERATED_SCAN_BLOCKS}"
            )

    @property
    def kv_blocks(self) -> int:
        return self.current_token // BLOCK_TOKENS + 1

    @property
    def total_context(self) -> int:
        return self.kv_blocks * BLOCK_TOKENS

    @property
    def tail_tokens(self) -> int:
        return self.current_token % BLOCK_TOKENS + 1

    @property
    def current_block(self) -> int:
        return self.current_token // BLOCK_TOKENS

    @property
    def current_token_offset(self) -> int:
        return (self.current_token % BLOCK_TOKENS) * HEAD_DWORDS

    @property
    def kv_cache_dwords(self) -> int:
        return self.kv_blocks * CACHE_BLOCK_DWORDS

    @property
    def logical_cache_dwords(self) -> int:
        return self.total_context * KV_HEADS * HEAD_DWORDS

    @property
    def current_write_byte_offset(self) -> int:
        return (self.current_block * CACHE_BLOCK_DWORDS + self.current_token_offset) * 4


DEFAULT_SCHEDULE = DecodeSchedule(DEFAULT_CURRENT_TOKEN)
KV_BLOCKS = DEFAULT_SCHEDULE.kv_blocks
TOTAL_CONTEXT = DEFAULT_SCHEDULE.total_context
CURRENT_TOKEN = DEFAULT_SCHEDULE.current_token
CURRENT_TOKEN_OFFSET = DEFAULT_SCHEDULE.current_token_offset
KV_CACHE_DWORDS = DEFAULT_SCHEDULE.kv_cache_dwords


def make_decode_schedule(current_token: int | None) -> DecodeSchedule:
    if current_token is None:
        return DEFAULT_SCHEDULE
    return DecodeSchedule(current_token)


def make_q_payload() -> np.ndarray:
    return mainq.make_q_payload()


def _pack_s16_pair(low: int, high: int) -> np.int32:
    low_u16 = low & 0xFFFF
    high_u16 = high & 0xFFFF
    return np.array(low_u16 | (high_u16 << 16), dtype=np.uint32).view(np.int32)


def _current_lane(schedule: DecodeSchedule, compact: np.ndarray, lane: int, is_v: bool) -> int:
    head = lane // HEAD_DIM
    dim = lane % HEAD_DIM
    if is_v:
        seed = int(compact[1 + ((lane * 7) & 255)]) & 15
        return ((head + 5) * 7 + schedule.current_token * 11 + dim * 2 + seed) % 63 - 31
    seed = int(compact[1 + ((lane * 5) & 255)]) & 7
    return ((head + 3) * 9 + schedule.current_token * 5 + dim * 3 + seed) % 31 - 15


def _current_payload(schedule: DecodeSchedule, phase: int, is_v: bool) -> np.ndarray:
    compact = qkv_global_compact(phase)
    payload = np.empty(CURRENT_DWORDS, dtype=np.int32)
    for idx in range(CURRENT_DWORDS):
        low_lane = idx * 2
        high_lane = low_lane + 1
        payload[idx] = _pack_s16_pair(
            _current_lane(schedule, compact, low_lane, is_v),
            _current_lane(schedule, compact, high_lane, is_v),
        )
    return payload


def current_k_payload(schedule: DecodeSchedule) -> np.ndarray:
    return _current_payload(schedule, K_PHASE, False)


def current_v_payload(schedule: DecodeSchedule) -> np.ndarray:
    return _current_payload(schedule, V_PHASE, True)


def _history_lane(global_kv_head: int, token: int, dim: int, is_v: bool) -> int:
    if is_v:
        return ((global_kv_head + 5) * 7 + token * 11 + dim * 2) % 63 - 31
    return ((global_kv_head + 3) * 9 + token * 5 + dim * 3) % 31 - 15


def logical_cache_index(token: int, head: int, dim_pair: int) -> int:
    return (token * KV_HEADS + head) * HEAD_DWORDS + dim_pair


def npu_cache_index(token: int, head: int, dim_pair: int) -> int:
    block = token // BLOCK_TOKENS
    token_in_block = token % BLOCK_TOKENS
    return block * CACHE_BLOCK_DWORDS + head * WINDOW_HEAD_DWORDS + token_in_block * HEAD_DWORDS + dim_pair


def pack_logical_cache_to_npu(schedule: DecodeSchedule, logical: np.ndarray) -> np.ndarray:
    if logical.shape != (schedule.logical_cache_dwords,):
        raise ValueError(f"logical cache shape mismatch: {logical.shape} != {(schedule.logical_cache_dwords,)}")
    cache = np.empty(schedule.kv_cache_dwords, dtype=np.int32)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim_pair in range(HEAD_DWORDS):
                cache[npu_cache_index(token, head, dim_pair)] = logical[logical_cache_index(token, head, dim_pair)]
    return cache


def unpack_npu_cache_to_logical(schedule: DecodeSchedule, cache: np.ndarray) -> np.ndarray:
    if cache.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"npu cache shape mismatch: {cache.shape} != {(schedule.kv_cache_dwords,)}")
    logical = np.empty(schedule.logical_cache_dwords, dtype=np.int32)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim_pair in range(HEAD_DWORDS):
                logical[logical_cache_index(token, head, dim_pair)] = cache[npu_cache_index(token, head, dim_pair)]
    return logical


def _history_logical_cache(schedule: DecodeSchedule, is_v: bool) -> np.ndarray:
    cache = np.empty(schedule.logical_cache_dwords, dtype=np.int32)
    for token in range(schedule.total_context):
        for head in range(KV_HEADS):
            for dim_pair in range(HEAD_DWORDS):
                low_dim = dim_pair * 2
                high_dim = low_dim + 1
                cache[logical_cache_index(token, head, dim_pair)] = _pack_s16_pair(
                    _history_lane(head, token, low_dim, is_v),
                    _history_lane(head, token, high_dim, is_v),
                )
    return cache


def _poison_current_slots(schedule: DecodeSchedule, cache: np.ndarray, poison_value: int) -> np.ndarray:
    poisoned = cache.copy()
    poison = np.array(poison_value, dtype=np.uint32).view(np.int32)
    for head in range(KV_HEADS):
        start = logical_cache_index(schedule.current_token, head, 0)
        poisoned[start : start + HEAD_DWORDS] = poison
    return poisoned


def make_history_k_cache_payload(schedule: DecodeSchedule) -> np.ndarray:
    return pack_logical_cache_to_npu(schedule, _poison_current_slots(schedule, _history_logical_cache(schedule, False), 0x13572468))


def make_history_v_cache_payload(schedule: DecodeSchedule) -> np.ndarray:
    return pack_logical_cache_to_npu(schedule, _poison_current_slots(schedule, _history_logical_cache(schedule, True), 0x13572469))


def _write_current(schedule: DecodeSchedule, cache: np.ndarray, current: np.ndarray) -> np.ndarray:
    merged = cache.copy()
    block_offset = schedule.current_block * CACHE_BLOCK_DWORDS
    for head in range(KV_HEADS):
        dst = block_offset + head * WINDOW_HEAD_DWORDS + schedule.current_token_offset
        src = head * HEAD_DWORDS
        merged[dst : dst + HEAD_DWORDS] = current[src : src + HEAD_DWORDS]
    return merged


def merged_k_cache_payload(schedule: DecodeSchedule) -> np.ndarray:
    return _write_current(schedule, make_history_k_cache_payload(schedule), current_k_payload(schedule))


def merged_v_cache_payload(schedule: DecodeSchedule) -> np.ndarray:
    return _write_current(schedule, make_history_v_cache_payload(schedule), current_v_payload(schedule))


def q_window(window: int) -> np.ndarray:
    start = window * WINDOW_DWORDS
    return make_q_payload()[start : start + WINDOW_DWORDS]


def k_window(schedule: DecodeSchedule, window: int, block: int) -> np.ndarray:
    return _k_window_from_cache(merged_k_cache_payload(schedule), window, block)


def v_window(schedule: DecodeSchedule, window: int, block: int) -> np.ndarray:
    return _v_window_from_cache(merged_v_cache_payload(schedule), window, block)


def _k_window_from_cache(cache: np.ndarray, window: int, block: int) -> np.ndarray:
    start = block * CACHE_BLOCK_DWORDS + window * K_WINDOW_DWORDS
    return cache[start : start + K_WINDOW_DWORDS]


def _v_window_from_cache(cache: np.ndarray, window: int, block: int) -> np.ndarray:
    start = block * CACHE_BLOCK_DWORDS + window * V_WINDOW_DWORDS
    return cache[start : start + V_WINDOW_DWORDS]


def _score(q: np.ndarray, k: np.ndarray, q_head: int, token: int) -> int:
    kv_head = q_head // attention.GQA_RATIO
    total = 0
    q_base = q_head * HEAD_DIM
    k_base = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM
    for dim in range(HEAD_DIM):
        total += attention._unpack_s16(q, q_base + dim) * attention._unpack_s16(k, k_base + dim)
    return attention._trunc_div(total, HEAD_DIM)


def _score_from_q(q: np.ndarray, k: np.ndarray, q_head: int, token: int) -> int:
    return _score(q, k, q_head, token)


def make_carrier(schedule: DecodeSchedule, window: int, block: int) -> np.ndarray:
    return make_carrier_from_k(schedule, window, block, merged_k_cache_payload(schedule))


def make_carrier_from_k(schedule: DecodeSchedule, window: int, block: int, k_cache: np.ndarray) -> np.ndarray:
    q = q_window(window)
    return make_carrier_from_qk(schedule, window, block, q, k_cache)


def make_carrier_from_qk(
    schedule: DecodeSchedule,
    window: int,
    block: int,
    q: np.ndarray,
    k_cache: np.ndarray,
) -> np.ndarray:
    k = _k_window_from_cache(k_cache, window, block)
    carrier = np.zeros(attention.WEIGHT_DWORDS + attention.SCALAR_DWORDS, dtype=np.int32)
    valid_tokens = schedule.tail_tokens if block + 1 == schedule.kv_blocks else CONTEXT
    for q_head in range(HEADS_PER_WINDOW):
        scores = [_score_from_q(q, k, q_head, token) for token in range(valid_tokens)]
        running_max = max(scores)
        weights = [attention._softmax_weight(running_max - score) for score in scores]
        weights.extend([0] * (CONTEXT - valid_tokens))
        for token in range(0, CONTEXT, 2):
            word = (q_head * CONTEXT + token) // 2
            carrier[word] = attention._pack_weight_pair(weights[token], weights[token + 1])
        scalar = WEIGHT_DWORDS + q_head * 2
        carrier[scalar] = running_max
        carrier[scalar + 1] = sum(weights)
    return carrier


def return_window(schedule: DecodeSchedule, window: int) -> np.ndarray:
    return return_window_from_cache(schedule, window, merged_k_cache_payload(schedule), merged_v_cache_payload(schedule))


def return_window_from_cache(
    schedule: DecodeSchedule,
    window: int,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
) -> np.ndarray:
    return return_window_from_qkv(schedule, window, q_window(window), k_cache, v_cache)


def return_window_from_qkv(
    schedule: DecodeSchedule,
    window: int,
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
) -> np.ndarray:
    accum = np.zeros(ACCUM_LANES, dtype=np.int64)
    running_max = np.zeros(HEADS_PER_WINDOW, dtype=np.int64)
    running_sum = np.zeros(HEADS_PER_WINDOW, dtype=np.int64)
    for block in range(schedule.kv_blocks):
        carrier = make_carrier_from_qk(schedule, window, block, q, k_cache)
        v = _v_window_from_cache(v_cache, window, block)
        for q_head in range(HEADS_PER_WINDOW):
            kv_head = q_head // attention.GQA_RATIO
            block_max = int(carrier[WEIGHT_DWORDS + q_head * 2])
            block_sum = int(carrier[WEIGHT_DWORDS + q_head * 2 + 1])
            if running_sum[q_head] == 0:
                new_max = block_max
                old_scale = 0
                new_scale = SOFTMAX_SCALE
            else:
                new_max = max(int(running_max[q_head]), block_max)
                old_scale = attention._softmax_weight(new_max - int(running_max[q_head]))
                new_scale = attention._softmax_weight(new_max - block_max)

            for dim in range(HEAD_DIM):
                block_total = 0
                for token in range(CONTEXT):
                    weight = attention._unpack_weight(carrier, q_head, token)
                    v_lane = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM + dim
                    block_total += weight * attention._unpack_s16(v, v_lane)
                lane = q_head * HEAD_DIM + dim
                accum[lane] = attention._trunc_div(
                    int(accum[lane]) * old_scale + block_total * new_scale,
                    SOFTMAX_SCALE,
                )

            running_sum[q_head] = attention._trunc_div(
                int(running_sum[q_head]) * old_scale + block_sum * new_scale,
                SOFTMAX_SCALE,
            )
            running_max[q_head] = new_max

    lanes = np.zeros(OUTPUT_DWORDS * 2, dtype=np.int32)
    for q_head in range(HEADS_PER_WINDOW):
        weight_sum = int(running_sum[q_head])
        for dim in range(HEAD_DIM):
            lane = q_head * HEAD_DIM + dim
            lanes[lane] = attention._trunc_div(int(accum[lane]), weight_sum)
    return attention._pack_lanes(lanes)


def attention_payload(schedule: DecodeSchedule) -> np.ndarray:
    k_cache = merged_k_cache_payload(schedule)
    v_cache = merged_v_cache_payload(schedule)
    return attention_payload_from_qkv(schedule, make_q_payload(), k_cache, v_cache)


def attention_payload_from_qkv(
    schedule: DecodeSchedule,
    q_payload: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            return_window_from_qkv(
                schedule,
                window,
                q_payload[window * WINDOW_DWORDS : (window + 1) * WINDOW_DWORDS],
                k_cache,
                v_cache,
            )
            for window in range(4)
        ]
    ).astype(np.int32)


def expected_output(schedule: DecodeSchedule) -> np.ndarray:
    return mainq.expected_output_from_attention(attention_payload(schedule))


def validate_output(schedule: DecodeSchedule, got: np.ndarray) -> list[str]:
    return validate_expected_output(expected_output(schedule), got)


def validate_expected_output(expected: np.ndarray, got: np.ndarray) -> list[str]:
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    mismatch = np.where(got != expected)[0]
    errors = [
        f"out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}"
        for idx in mismatch[:32]
    ]
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional mismatches")
    return errors


def validate_cache_writeback(schedule: DecodeSchedule, got_k: np.ndarray, got_v: np.ndarray) -> list[str]:
    expected_k = merged_k_cache_payload(schedule)
    expected_v = merged_v_cache_payload(schedule)
    errors: list[str] = []
    for name, got, expected in (("k_cache", got_k, expected_k), ("v_cache", got_v, expected_v)):
        if got.shape != expected.shape:
            errors.append(f"{name} shape mismatch: {got.shape} != {expected.shape}")
            continue
        mismatches: list[int] = []
        block_offset = schedule.current_block * CACHE_BLOCK_DWORDS
        for head in range(KV_HEADS):
            start = block_offset + head * WINDOW_HEAD_DWORDS + schedule.current_token_offset
            local = np.where(got[start : start + HEAD_DWORDS] != expected[start : start + HEAD_DWORDS])[0]
            mismatches.extend((start + int(idx)) for idx in local[: 32 - len(mismatches)])
            if len(mismatches) >= 32:
                break
        for idx in mismatches:
            errors.append(f"{name}[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
        if len(mismatches) == 32:
            errors.append(f"{name}: additional current-token mismatches omitted")
    return errors


def validate_cache_layout_contract(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    logical = np.arange(schedule.logical_cache_dwords, dtype=np.int32)
    packed = pack_logical_cache_to_npu(schedule, logical)
    unpacked = unpack_npu_cache_to_logical(schedule, packed)
    errors: list[str] = []
    mismatch = np.where(unpacked != logical)[0]
    for idx in mismatch[:8]:
        errors.append(f"logical cache roundtrip[{idx}]: expected={int(logical[idx])} got={int(unpacked[idx])}")
    if mismatch.size > 8:
        errors.append(f"{mismatch.size - 8} additional logical cache roundtrip mismatches")

    for window in range(4):
        head_base = window * KV_HEADS_PER_WINDOW
        for block in range(schedule.kv_blocks):
            token_base = block * BLOCK_TOKENS
            window_start = block * CACHE_BLOCK_DWORDS + window * K_WINDOW_DWORDS
            for local_head in range(KV_HEADS_PER_WINDOW):
                head = head_base + local_head
                for token_offset in range(BLOCK_TOKENS):
                    token = token_base + token_offset
                    for dim_pair in range(HEAD_DWORDS):
                        npu_idx = window_start + local_head * WINDOW_HEAD_DWORDS + token_offset * HEAD_DWORDS + dim_pair
                        logical_idx = logical_cache_index(token, head, dim_pair)
                        if packed[npu_idx] != logical[logical_idx]:
                            errors.append(
                                f"window{window} block{block} scan mapping mismatch: "
                                f"packed[{npu_idx}]={int(packed[npu_idx])} logical[{logical_idx}]={int(logical[logical_idx])}"
                            )
                            return errors
    return errors


def route_summary(schedule: DecodeSchedule) -> list[str]:
    return [
        f"case={CASE_NAME}",
        f"main16 emits Q/K/V compacts; c1r3 emits Q plus packet{CURRENT_PACKET_K} current K and packet{CURRENT_PACKET_V} current V",
        f"packet{CURRENT_PACKET_K}/{CURRENT_PACKET_V} write token{schedule.current_token} into block-major K/V cache BOs before shim KV scan starts",
        f"shim scans {schedule.kv_blocks} rounded {BLOCK_TOKENS}-token blocks; tail block has {schedule.tail_tokens} valid token(s)",
        "logical Qwen3 KV cache is token-major [token][kv_head][dim_pair], packed to NPU block-major scan BOs",
        "host history cache intentionally poisons current-token slots so write-then-scan is validated",
        f"current_k={CURRENT_DWORDS} dwords, current_v={CURRENT_DWORDS} dwords",
        f"k_cache={schedule.kv_cache_dwords} dwords, v_cache={schedule.kv_cache_dwords} dwords",
        f"host_output={HOST_OUTPUT_DWORDS} dwords",
    ]
