"""CPU reference for the c1r2 full-vector replay integration case."""

from __future__ import annotations

import numpy as np

from contract import (
    C1R2_PACKET_DWORDS,
    C1R2_UPGATE_REPLAYS,
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)

CASE_NAME = "c1r2-o-upgate-bridge"
O_PHASE = 3
UP_PHASE = 4
GATE_PHASE = 5
MAIN_CHUNK_DWORDS = 128
MAIN_CHUNKS_PER_REPLAY = C1R2_PACKET_DWORDS - 1
MAIN_CHUNKS_PER_REPLAY //= MAIN_CHUNK_DWORDS
TOTAL_MAIN_CHUNKS = C1R2_UPGATE_REPLAYS * MAIN_CHUNKS_PER_REPLAY
MAIN_RECORD_DWORDS = RECORD_DWORDS * 3
MAIN_ACCUM_DWORDS = RECORD_PAYLOAD_DWORDS * 2
SWIGLU_OUTPUT_DWORDS = C6R2_HALF_DWORDS
MAIN_PACKET_BASE = 16
COLUMN_PACKET_BASE = 4
O_GLOBAL_PACKET_ID = 8
FFN_GLOBAL_PACKET_ID = 9


def main_packet(group: int, row: int) -> int:
    return MAIN_PACKET_BASE + group * ROWS_PER_COLUMN + row


def column_packet(group: int) -> int:
    return COLUMN_PACKET_BASE + group


def record_header(phase: int, group: int, row: int) -> int:
    return (phase << 24) | (group << 16) | (row << 8) | 0xA5


def o_payload_value(group: int, row: int, lane: int) -> int:
    return O_PHASE * 4096 + group * 1024 + row * 256 + lane


def make_o_record(group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(O_PHASE, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = o_payload_value(group, row, lane)
    return record


def column_compact_from_records(records: list[np.ndarray]) -> np.ndarray:
    compact = np.empty(
        RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS,
        dtype=np.int32,
    )
    offset = 0
    for row, record in enumerate(records):
        segment = record if row == 0 else record[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def global_compact_from_columns(columns: list[np.ndarray]) -> np.ndarray:
    compact = np.empty(COMPACT_PACKET_DWORDS, dtype=np.int32)
    offset = 0
    for group, column in enumerate(columns):
        segment = column if group == 0 else column[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def o_global_compact() -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_o_record(group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def replay_payload(replay: int, payload_idx: int, compact: np.ndarray) -> int:
    seed = int(compact[1 + payload_idx % (COMPACT_PACKET_DWORDS - 1)])
    value = replay * 1024 + (payload_idx & 1023) + (seed & 127)
    return value & 0xFFFF


def replay_chunk(replay: int, chunk: int, compact: np.ndarray) -> np.ndarray:
    start = chunk * MAIN_CHUNK_DWORDS
    values = [
        replay_payload(replay, start + lane, compact)
        for lane in range(MAIN_CHUNK_DWORDS)
    ]
    return np.array(values, dtype=np.int32)


def main_accum(group: int, row: int, compact: np.ndarray) -> np.ndarray:
    accum = np.zeros(MAIN_ACCUM_DWORDS, dtype=np.int32)
    accum[0] = group
    accum[1] = row
    for chunk_idx in range(TOTAL_MAIN_CHUNKS):
        replay = chunk_idx // MAIN_CHUNKS_PER_REPLAY
        chunk = chunk_idx - replay * MAIN_CHUNKS_PER_REPLAY
        parity = replay & 1
        payload = replay_chunk(replay, chunk, compact)
        for lane in range(RECORD_PAYLOAD_DWORDS):
            source = (lane * 7 + group * 13 + row * 5 + chunk_idx) & (MAIN_CHUNK_DWORDS - 1)
            base = parity * RECORD_PAYLOAD_DWORDS + lane
            accum[base] = (int(accum[base]) + int(payload[source]) + chunk_idx + lane) & 0xFFFF
    return accum


def make_upgate_records(group: int, row: int, compact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    accum = main_accum(group, row, compact)
    up = np.empty(RECORD_DWORDS, dtype=np.int32)
    gate = np.empty(RECORD_DWORDS, dtype=np.int32)
    up[0] = record_header(UP_PHASE, group, row)
    gate[0] = record_header(GATE_PHASE, group, row)
    up[1:] = accum[:RECORD_PAYLOAD_DWORDS]
    gate[1:] = accum[RECORD_PAYLOAD_DWORDS:]
    return up, gate


def ffn_global_compact(phase: int) -> np.ndarray:
    compact = o_global_compact()
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = []
        for row in range(ROWS_PER_COLUMN):
            up, gate = make_upgate_records(group, row, compact)
            records.append(up if phase == UP_PHASE else gate)
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def expected_c6r2_input() -> np.ndarray:
    up = ffn_global_compact(UP_PHASE)[1:]
    gate = ffn_global_compact(GATE_PHASE)[1:]
    if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
        raise RuntimeError(f"bad c6r2 halves: {up.shape[0]}/{gate.shape[0]}")
    return np.concatenate((up, gate)).astype(np.int32)


def expected_output() -> np.ndarray:
    c6r2_input = expected_c6r2_input()
    if c6r2_input.shape[0] != C6R2_INPUT_DWORDS:
        raise RuntimeError(f"bad c6r2 input: {c6r2_input.shape[0]}")
    output = np.empty(SWIGLU_OUTPUT_DWORDS, dtype=np.int32)
    for idx in range(SWIGLU_OUTPUT_DWORDS):
        output[idx] = (int(c6r2_input[idx]) << 16) | int(c6r2_input[C6R2_HALF_DWORDS + idx])
    return output


def validate_output(got: np.ndarray) -> list[str]:
    expected = expected_output()
    errors: list[str] = []
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    mismatch = np.where(got != expected)[0]
    for idx in mismatch[:32]:
        errors.append(f"out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional mismatches")
    return errors


def route_summary() -> list[str]:
    return [
        f"case={CASE_NAME}",
        "o_compact=main16 O records -> row1 column compact -> c1r1 -> c1r2",
        f"c1r2_replays={C1R2_UPGATE_REPLAYS}x{C1R2_PACKET_DWORDS} dwords",
        f"main_chunks={TOTAL_MAIN_CHUNKS} per main tile",
        "ffn_compact=main16 summary up/gate records -> row1/c1r1 -> c6r2",
        f"c6r2_output={SWIGLU_OUTPUT_DWORDS} dwords",
    ]
