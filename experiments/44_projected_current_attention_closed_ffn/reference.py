"""CPU reference for exp44 projected current-write attention closed FFN."""

from __future__ import annotations

import numpy as np

CONTEXT_LEN = 31
TOKENS_PER_TILE = 16
ROUNDED_TOKENS = 32
HEAD_DIM = 128
Q_HEADS = 4
HIDDEN_DWORDS = 512
QUERY_DWORDS = Q_HEADS * HEAD_DIM
HALF_DWORDS = 256
PLANE_DWORDS = ROUNDED_TOKENS * HEAD_DIM
KV_CACHE_DWORDS = 2 * PLANE_DWORDS
O_WEIGHT_DWORDS = 2048
FFN_WEIGHT_DWORDS = 4096
OUTPUT_DWORDS = 512
CURRENT_TOKEN = CONTEXT_LEN - 1


def trunc_div(value: int, divisor: int) -> int:
    sign = -1 if value < 0 else 1
    return sign * (abs(value) // divisor)


def make_hidden_payload() -> np.ndarray:
    hidden = np.empty(HIDDEN_DWORDS, dtype=np.int32)
    for i in range(HIDDEN_DWORDS):
        hidden[i] = np.int32(((i * 5 + 3) % 23) - 11)
    return hidden


def make_kv_cache_without_current() -> np.ndarray:
    cache = np.empty(KV_CACHE_DWORDS, dtype=np.int32)
    for token in range(ROUNDED_TOKENS):
        for dim in range(HEAD_DIM):
            idx = token * HEAD_DIM + dim
            if token < CONTEXT_LEN and token != CURRENT_TOKEN:
                cache[idx] = np.int32(((token * 7 + dim * 3) % 19) - 9)
                cache[PLANE_DWORDS + idx] = np.int32(((token * 5 + dim * 11) % 23) - 11)
            else:
                cache[idx] = np.int32(91 + token + (dim % 5))
                cache[PLANE_DWORDS + idx] = np.int32(-93 - token - (dim % 7))
    return cache


def make_o_weight_payload() -> np.ndarray:
    weight = np.empty(O_WEIGHT_DWORDS, dtype=np.int32)
    for i in range(O_WEIGHT_DWORDS):
        value = ((i * 13 + 7) % 7) - 3
        weight[i] = np.int32(value if value != 0 else 2)
    return weight


def make_ffn_weight_payload() -> np.ndarray:
    weight = np.empty(FFN_WEIGHT_DWORDS, dtype=np.int32)
    for i in range(FFN_WEIGHT_DWORDS):
        value = ((i * 17 + 11) % 9) - 4
        weight[i] = np.int32(value if value != 0 else -2)
    return weight


def project_query_current(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.empty(QUERY_DWORDS, dtype=np.int32)
    current_k = np.empty(HEAD_DIM, dtype=np.int32)
    current_v = np.empty(HEAD_DIM, dtype=np.int32)

    for i in range(QUERY_DWORDS):
        head = i // HEAD_DIM
        dim = i & (HEAD_DIM - 1)
        query[i] = np.int32(
            2 * hidden[(dim * 3 + head * 17) & (HIDDEN_DWORDS - 1)]
            - hidden[(dim * 5 + head * 29 + 7) & (HIDDEN_DWORDS - 1)]
            + ((dim + head) & 7)
        )

    for dim in range(HEAD_DIM):
        current_k[dim] = np.int32(
            hidden[(dim * 7 + 11) & (HIDDEN_DWORDS - 1)]
            - 2 * hidden[(dim * 13 + 3) & (HIDDEN_DWORDS - 1)]
            + (dim & 3)
        )
        current_v[dim] = np.int32(
            3 * hidden[(dim * 5 + 19) & (HIDDEN_DWORDS - 1)]
            + hidden[(dim * 9 + 23) & (HIDDEN_DWORDS - 1)]
            - (dim & 5)
        )
    return query, current_k, current_v


def apply_current_to_cache(cache: np.ndarray, current_k: np.ndarray, current_v: np.ndarray) -> np.ndarray:
    out = cache.copy()
    offset = CURRENT_TOKEN * HEAD_DIM
    out[offset:offset + HEAD_DIM] = current_k
    out[PLANE_DWORDS + offset:PLANE_DWORDS + offset + HEAD_DIM] = current_v
    return out


def attention_from_cache(query: np.ndarray, cache: np.ndarray) -> np.ndarray:
    output = np.empty(QUERY_DWORDS, dtype=np.int32)
    k_plane = cache[:PLANE_DWORDS]
    v_plane = cache[PLANE_DWORDS:]
    for i in range(QUERY_DWORDS):
        head = i // HEAD_DIM
        dim = i & (HEAD_DIM - 1)
        q = int(query[i])
        acc = 0
        for token in range(CONTEXT_LEN):
            idx = token * HEAD_DIM + dim
            acc += (q + head + 1) * int(k_plane[idx])
            acc += (head + 2) * int(v_plane[idx])
            acc += (token + dim) & 7
        output[i] = np.int32(trunc_div(acc, 32))
    return output


def o_project(attention: np.ndarray, weight: np.ndarray) -> np.ndarray:
    output = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    for i in range(OUTPUT_DWORDS):
        base = i * 4
        output[i] = np.int32(
            attention[i] * weight[base]
            + attention[(i + 17) & (OUTPUT_DWORDS - 1)] * weight[base + 1]
            - attention[(i + 29) & (OUTPUT_DWORDS - 1)] * weight[base + 2]
            + attention[(i + 43) & (OUTPUT_DWORDS - 1)] * weight[base + 3]
            + (i & 31)
        )
    return output


def ffn_tail(o_output: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gate = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    up = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    swiglu = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    output = np.empty(OUTPUT_DWORDS, dtype=np.int32)

    for i in range(OUTPUT_DWORDS):
        norm = int(o_output[i]) - ((i % 17) - 8)
        gate[i] = np.int32(norm * int(weight[i]) + int(weight[512 + i]) + (i & 7))
        up[i] = np.int32(norm * int(weight[1024 + i]) - int(weight[1536 + i]) + (i & 15))
        swiglu[i] = np.int32(trunc_div(int(gate[i]) * int(up[i]), 64))

    for i in range(OUTPUT_DWORDS):
        output[i] = np.int32(
            int(o_output[i])
            + int(swiglu[i]) * int(weight[2048 + i])
            + int(swiglu[(i + 19) & (OUTPUT_DWORDS - 1)]) * int(weight[2560 + i])
            - int(swiglu[(i + 37) & (OUTPUT_DWORDS - 1)]) * int(weight[3072 + i])
            + int(weight[3584 + i])
            + (i & 31)
        )
    return output


def expected_output() -> np.ndarray:
    hidden = make_hidden_payload()
    cache_before = make_kv_cache_without_current()
    o_weight = make_o_weight_payload()
    ffn_weight = make_ffn_weight_payload()
    query, current_k, current_v = project_query_current(hidden)
    cache_after = apply_current_to_cache(cache_before, current_k, current_v)
    attention = attention_from_cache(query, cache_after)
    o_output = o_project(attention, o_weight)
    return ffn_tail(o_output, ffn_weight)


if __name__ == "__main__":
    hidden_ref = make_hidden_payload()
    cache_ref = make_kv_cache_without_current()
    query_ref, k_ref, v_ref = project_query_current(hidden_ref)
    cache_after_ref = apply_current_to_cache(cache_ref, k_ref, v_ref)
    attn_ref = attention_from_cache(query_ref, cache_after_ref)
    o_ref = o_project(attn_ref, make_o_weight_payload())
    out_ref = ffn_tail(o_ref, make_ffn_weight_payload())
    print(f"query[0:8]={query_ref[:8].tolist()}")
    print(f"current_k[0:8]={k_ref[:8].tolist()}")
    print(f"current_v[0:8]={v_ref[:8].tolist()}")
    print(f"attention[0:8]={attn_ref[:8].tolist()}")
    print(f"output[0:8]={out_ref[:8].tolist()}")
