"""CPU reference for exp47 main16 multi-phase O/gate/up/down replay."""

from __future__ import annotations

import numpy as np

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
SHARD_DWORDS = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = 16
O_WEIGHT_DWORDS = 32
GATE_WEIGHT_DWORDS = 32
UP_WEIGHT_DWORDS = 32
DOWN_WEIGHT_DWORDS = 32
TILE_WEIGHT_DWORDS = O_WEIGHT_DWORDS + GATE_WEIGHT_DWORDS + UP_WEIGHT_DWORDS + DOWN_WEIGHT_DWORDS
COLUMN_OUTPUT_DWORDS = ROWS_PER_COLUMN * SHARD_DWORDS
COLUMN_WEIGHT_DWORDS = ROWS_PER_COLUMN * TILE_WEIGHT_DWORDS
TOTAL_OUTPUT_DWORDS = len(MAIN_COLUMNS) * COLUMN_OUTPUT_DWORDS
TOTAL_WEIGHT_DWORDS = len(MAIN_COLUMNS) * COLUMN_WEIGHT_DWORDS


def record_header(group: int, row: int) -> np.int32:
    return np.int32(0x45000000 | (group << 12) | (row << 4) | 0xA)


def record_payload(group: int, row: int, lane: int) -> np.int32:
    return np.int32(10000 * group + 100 * row + lane)


def make_record(group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = record_payload(group, row, lane)
    return record


def trunc_div(num: int, den: int) -> int:
    abs_q = abs(num) // abs(den)
    return -abs_q if (num < 0) != (den < 0) else abs_q


def edge_attention_from_record(group: int, row: int, record: np.ndarray) -> np.ndarray:
    attention = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        a = int(record[1 + lane % RECORD_PAYLOAD_DWORDS])
        b = int(record[1 + (lane + 5) % RECORD_PAYLOAD_DWORDS])
        attention[lane] = np.int32(a + 2 * b + group * 1000 + row * 31 + lane)
    return attention


def make_weights() -> np.ndarray:
    weights = np.empty(TOTAL_WEIGHT_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            base = group * COLUMN_WEIGHT_DWORDS + row * TILE_WEIGHT_DWORDS
            for lane in range(TILE_WEIGHT_DWORDS):
                raw = ((group * 7 + row * 5 + lane * 3) % 13) - 6
                weights[base + lane] = np.int32(raw if raw != 0 else 4)
    return weights


def main_o_phase(group: int, row: int, attention: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty(SHARD_DWORDS, dtype=np.int32)
    group_base = group * COLUMN_WEIGHT_DWORDS
    shard_base = group_base + row * TILE_WEIGHT_DWORDS
    weight = weights[shard_base:shard_base + O_WEIGHT_DWORDS]
    for lane in range(SHARD_DWORDS):
        output[lane] = np.int32(
            int(attention[lane]) * int(weight[lane])
            + int(attention[(lane + 7) & (SHARD_DWORDS - 1)])
            - int(weight[(lane + 11) & (SHARD_DWORDS - 1)])
            + group * 13
            + row * 5
            + lane
        )
    return output


def _phase_weight(group: int, row: int, weights: np.ndarray, phase_offset: int) -> np.ndarray:
    group_base = group * COLUMN_WEIGHT_DWORDS
    shard_base = group_base + row * TILE_WEIGHT_DWORDS
    return weights[shard_base + phase_offset:shard_base + phase_offset + SHARD_DWORDS]


def _norm(group: int, row: int, lane: int, o_output: np.ndarray) -> int:
    return trunc_div(int(o_output[lane]), 128) - ((group * 5 + row * 3 + lane) & 15) + 7


def main_gate_phase(group: int, row: int, o_output: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight = _phase_weight(group, row, weights, O_WEIGHT_DWORDS)
    gate = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        norm = _norm(group, row, lane, o_output)
        gate[lane] = np.int32(norm * int(weight[lane]) + int(weight[(lane + 11) & (SHARD_DWORDS - 1)]) + group * 11 + row)
    return gate


def main_up_phase(group: int, row: int, o_output: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight = _phase_weight(group, row, weights, O_WEIGHT_DWORDS + GATE_WEIGHT_DWORDS)
    up = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        norm = _norm(group, row, lane, o_output)
        up[lane] = np.int32(norm * int(weight[lane]) - int(weight[(lane + 7) & (SHARD_DWORDS - 1)]) + row * 17 + lane)
    return up


def main_swiglu_phase(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    swiglu = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        swiglu[lane] = np.int32(trunc_div(int(gate[lane]) * int(up[lane]), 64))
    return swiglu


def main_down_phase(group: int, row: int, o_output: np.ndarray, swiglu: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight = _phase_weight(group, row, weights, O_WEIGHT_DWORDS + GATE_WEIGHT_DWORDS + UP_WEIGHT_DWORDS)
    output = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        output[lane] = np.int32(
            int(o_output[lane])
            + int(swiglu[lane])
            - trunc_div(int(swiglu[(lane + 9) & (SHARD_DWORDS - 1)]) * int(weight[lane]), 32)
            + int(weight[(lane + 13) & (SHARD_DWORDS - 1)])
            + group * 19
            + row * 23
            + lane
        )
    return output


def expected_output(weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = make_weights()

    output = np.empty(TOTAL_OUTPUT_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            attention = edge_attention_from_record(group, row, make_record(group, row))
            o_output = main_o_phase(group, row, attention, weights)
            gate = main_gate_phase(group, row, o_output, weights)
            up = main_up_phase(group, row, o_output, weights)
            swiglu = main_swiglu_phase(gate, up)
            start = group * COLUMN_OUTPUT_DWORDS + row * SHARD_DWORDS
            output[start:start + SHARD_DWORDS] = main_down_phase(group, row, o_output, swiglu, weights)
    return output


if __name__ == "__main__":
    weights_data = make_weights()
    expected = expected_output(weights_data)
    print(f"weights[0:8]={weights_data[:8].tolist()}")
    print(f"output[0:8]={expected[:8].tolist()}")
