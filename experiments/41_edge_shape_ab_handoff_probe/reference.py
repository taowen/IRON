"""CPU reference for exp41 edge shape-A -> shape-B handoff probe."""

from __future__ import annotations

import numpy as np

CURRENT_DWORDS = 512
HISTORY_DWORDS = 2048
STATE_DWORDS = 17
OUTPUT_DWORDS = 512


def make_current_payload() -> np.ndarray:
    current = np.empty(CURRENT_DWORDS, dtype=np.int32)
    for i in range(CURRENT_DWORDS):
        current[i] = np.int32(1000 + i * 3 + (i % 7))
    return current


def make_history_a_payload() -> np.ndarray:
    history = np.empty(HISTORY_DWORDS, dtype=np.int32)
    for i in range(HISTORY_DWORDS):
        history[i] = np.int32(200000 + i * 5 - (i % 11))
    return history


def make_history_b_payload() -> np.ndarray:
    history = np.empty(HISTORY_DWORDS, dtype=np.int32)
    for i in range(HISTORY_DWORDS):
        history[i] = np.int32(-50000 + i * 7 + (i % 13))
    return history


def make_state(current: np.ndarray, history_a: np.ndarray) -> np.ndarray:
    cur_sum = np.int64(0)
    hist_sum = np.int64(0)
    for i in range(0, CURRENT_DWORDS, 32):
        cur_sum += np.int64(current[i])
    for i in range(0, HISTORY_DWORDS, 128):
        hist_sum += np.int64(history_a[i])

    state = np.empty(STATE_DWORDS, dtype=np.int32)
    state[0] = np.int32(0x41000000 | ((int(cur_sum) & 0xFF) << 8) | (int(hist_sum) & 0xFF))
    cur_bias = np.int32(int(cur_sum) & 0xFFFF)
    hist_bias = np.int32(int(hist_sum) & 0x7FFF)
    for lane in range(16):
        state[1 + lane] = np.int32(
            current[(lane * 31) % CURRENT_DWORDS]
            + 3 * history_a[(lane * 97) % HISTORY_DWORDS]
            + lane * 17
            + cur_bias
            - hist_bias
        )
    return state


def consume_state(state: np.ndarray, history_b: np.ndarray) -> np.ndarray:
    output = np.empty(OUTPUT_DWORDS, dtype=np.int32)
    header_low = np.int32(state[0] & 0xFFFF)
    for i in range(OUTPUT_DWORDS):
        lane = i & 15
        block = i >> 4
        output[i] = np.int32(
            state[1 + lane]
            + history_b[(block * 37 + lane * 13) & (HISTORY_DWORDS - 1)]
            + header_low
            + block
        )
    return output


def expected_output() -> np.ndarray:
    current = make_current_payload()
    history_a = make_history_a_payload()
    history_b = make_history_b_payload()
    state = make_state(current, history_a)
    return consume_state(state, history_b)


if __name__ == "__main__":
    state_ref = make_state(make_current_payload(), make_history_a_payload())
    out_ref = expected_output()
    print(f"state={state_ref.tolist()}")
    print(f"out[0:16]={out_ref[:16].tolist()}")
