"""CPU reference for exp43 attention + O projection + FFN closed tail."""

from __future__ import annotations

import numpy as np

CURRENT_DWORDS = 512
HISTORY_DWORDS = 2048
ATTENTION_DWORDS = 512
ATTENTION_HALF_DWORDS = 256
O_WEIGHT_DWORDS = 2048
FFN_WEIGHT_DWORDS = 4096
OUTPUT_DWORDS = 512


def make_current_payload() -> np.ndarray:
    current = np.empty(CURRENT_DWORDS, dtype=np.int32)
    for i in range(CURRENT_DWORDS):
        current[i] = np.int32(((i * 5 + 3) % 47) - 23)
    return current


def make_history_payload() -> np.ndarray:
    history = np.empty(HISTORY_DWORDS, dtype=np.int32)
    for i in range(HISTORY_DWORDS):
        history[i] = np.int32(((i * 7 + 19) % 61) - 30)
    return history


def make_o_weight_payload() -> np.ndarray:
    weight = np.empty(O_WEIGHT_DWORDS, dtype=np.int32)
    for i in range(O_WEIGHT_DWORDS):
        value = ((i * 13 + 7) % 9) - 4
        weight[i] = np.int32(value if value != 0 else 2)
    return weight


def make_ffn_weight_payload() -> np.ndarray:
    weight = np.empty(FFN_WEIGHT_DWORDS, dtype=np.int32)
    for i in range(FFN_WEIGHT_DWORDS):
        value = ((i * 17 + 11) % 11) - 5
        weight[i] = np.int32(value if value != 0 else -3)
    return weight


def make_attention_output(current: np.ndarray, history: np.ndarray) -> np.ndarray:
    attention = np.empty(ATTENTION_DWORDS, dtype=np.int32)
    for i in range(ATTENTION_DWORDS):
        attention[i] = np.int32(
            3 * current[i]
            - 2 * current[(i + 13) & (CURRENT_DWORDS - 1)]
            + 5 * history[(i * 7) & (HISTORY_DWORDS - 1)]
            + history[(i * 11 + 3) & (HISTORY_DWORDS - 1)]
            + (i % 17)
        )
    return attention


def o_project(attention: np.ndarray, weight: np.ndarray) -> np.ndarray:
    output = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    for i in range(OUTPUT_DWORDS):
        base = i * 4
        output[i] = np.int32(
            attention[i] * weight[base]
            + attention[(i + 17) & (ATTENTION_DWORDS - 1)] * weight[base + 1]
            - attention[(i + 29) & (ATTENTION_DWORDS - 1)] * weight[base + 2]
            + attention[(i + 43) & (ATTENTION_DWORDS - 1)] * weight[base + 3]
            + (i & 31)
        )
    return output


def ffn_tail(o_output: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gate = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    up = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    swiglu = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    output = np.empty(OUTPUT_DWORDS, dtype=np.int32)

    for i in range(OUTPUT_DWORDS):
        norm = np.int32(o_output[i] - ((i % 17) - 8))
        gate[i] = np.int32(norm * weight[i] + weight[512 + i] + (i & 7))
        up[i] = np.int32(norm * weight[1024 + i] - weight[1536 + i] + (i & 15))
        product = int(gate[i]) * int(up[i])
        sign = -1 if product < 0 else 1
        swiglu[i] = np.int32(sign * (abs(product) // 64))

    for i in range(OUTPUT_DWORDS):
        output[i] = np.int32(
            o_output[i]
            + swiglu[i] * weight[2048 + i]
            + swiglu[(i + 19) & (OUTPUT_DWORDS - 1)] * weight[2560 + i]
            - swiglu[(i + 37) & (OUTPUT_DWORDS - 1)] * weight[3072 + i]
            + weight[3584 + i]
            + (i & 31)
        )
    return output


def expected_output() -> np.ndarray:
    current = make_current_payload()
    history = make_history_payload()
    o_weight = make_o_weight_payload()
    ffn_weight = make_ffn_weight_payload()
    attention = make_attention_output(current, history)
    o_output = o_project(attention, o_weight)
    return ffn_tail(o_output, ffn_weight)


if __name__ == "__main__":
    current_ref = make_current_payload()
    history_ref = make_history_payload()
    attention_ref = make_attention_output(current_ref, history_ref)
    o_ref = o_project(attention_ref, make_o_weight_payload())
    out_ref = ffn_tail(o_ref, make_ffn_weight_payload())
    print(f"attention[0:8]={attention_ref[:8].tolist()}")
    print(f"o_output[0:8]={o_ref[:8].tolist()}")
    print(f"output[0:8]={out_ref[:8].tolist()}")
