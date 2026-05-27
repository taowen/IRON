"""CPU reference for exp42 attention-output -> O-projection handoff."""

from __future__ import annotations

import numpy as np

CURRENT_DWORDS = 512
HISTORY_DWORDS = 2048
ATTENTION_DWORDS = 512
O_WEIGHT_DWORDS = 2048
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


def expected_output() -> np.ndarray:
    current = make_current_payload()
    history = make_history_payload()
    weight = make_o_weight_payload()
    attention = make_attention_output(current, history)
    return o_project(attention, weight)


if __name__ == "__main__":
    attention_ref = make_attention_output(make_current_payload(), make_history_payload())
    out_ref = expected_output()
    print(f"attention[0:8]={attention_ref[:8].tolist()}")
    print(f"output[0:8]={out_ref[:8].tolist()}")
