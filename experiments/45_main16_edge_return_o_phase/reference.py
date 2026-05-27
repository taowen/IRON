"""CPU reference for exp45 main16 -> edge -> main16 O-phase handoff."""

from __future__ import annotations

import numpy as np

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
SHARD_DWORDS = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = 16
COLUMN_DWORDS = ROWS_PER_COLUMN * SHARD_DWORDS
TOTAL_DWORDS = len(MAIN_COLUMNS) * COLUMN_DWORDS


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


def edge_attention_from_record(group: int, row: int, record: np.ndarray) -> np.ndarray:
    attention = np.empty(SHARD_DWORDS, dtype=np.int32)
    for lane in range(SHARD_DWORDS):
        a = int(record[1 + lane % RECORD_PAYLOAD_DWORDS])
        b = int(record[1 + (lane + 5) % RECORD_PAYLOAD_DWORDS])
        attention[lane] = np.int32(a + 2 * b + group * 1000 + row * 31 + lane)
    return attention


def make_weights() -> np.ndarray:
    weights = np.empty(TOTAL_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            base = group * COLUMN_DWORDS + row * SHARD_DWORDS
            for lane in range(SHARD_DWORDS):
                raw = ((group * 7 + row * 5 + lane * 3) % 13) - 6
                weights[base + lane] = np.int32(raw if raw != 0 else 4)
    return weights


def main_o_phase(group: int, row: int, attention: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty(SHARD_DWORDS, dtype=np.int32)
    group_base = group * COLUMN_DWORDS
    shard_base = group_base + row * SHARD_DWORDS
    weight = weights[shard_base:shard_base + SHARD_DWORDS]
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


def expected_output(weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = make_weights()

    output = np.empty(TOTAL_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            attention = edge_attention_from_record(group, row, make_record(group, row))
            start = group * COLUMN_DWORDS + row * SHARD_DWORDS
            output[start:start + SHARD_DWORDS] = main_o_phase(group, row, attention, weights)
    return output


if __name__ == "__main__":
    weights_data = make_weights()
    expected = expected_output(weights_data)
    print(f"weights[0:8]={weights_data[:8].tolist()}")
    print(f"output[0:8]={expected[:8].tolist()}")
