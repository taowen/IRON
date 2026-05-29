"""CPU reference for the c1r2 full-vector replay integration case."""

from __future__ import annotations

import math

import numpy as np

from swiglu_reference import fixed_swiglu_slice_output
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
    SWIGLU_SLICES,
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
MAIN_ACCUM_DWORDS = RECORD_PAYLOAD_DWORDS * 4
SWIGLU_OUTPUT_DWORDS = C6R2_HALF_DWORDS
OUTPUT_DWORDS = SWIGLU_OUTPUT_DWORDS * SWIGLU_SLICES
MAIN_PACKET_BASE = 16
COLUMN_PACKET_BASE = 4
O_GLOBAL_PACKET_ID = 8
FFN_GLOBAL_PACKET_ID = 9
RMSNORM_SCALE = 1024
PROJECTION_TAPS = 4
PROJECTION_SHIFT = 8


def main_packet(group: int, row: int) -> int:
    return MAIN_PACKET_BASE + group * ROWS_PER_COLUMN + row


def column_packet(group: int) -> int:
    return COLUMN_PACKET_BASE + group


def record_header(phase: int, group: int, row: int, slice_index: int = 0xA5) -> int:
    return (phase << 24) | (group << 16) | (row << 8) | (slice_index & 0xFF)


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


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def _clamp_s16(value: int) -> int:
    if value < -32768:
        return -32768
    if value > 32767:
        return 32767
    return value


def _pack_s16_pair(low: int, high: int) -> np.int32:
    low_u16 = _clamp_s16(low) & 0xFFFF
    high_u16 = _clamp_s16(high) & 0xFFFF
    return np.array(low_u16 | (high_u16 << 16), dtype=np.uint32).view(np.int32)


def _unpack_s16_word(word: int, lane: int) -> int:
    raw = ((int(word) & 0xFFFF_FFFF) >> 16) & 0xFFFF if lane else int(word) & 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def _compact_raw_lane(compact: np.ndarray, lane: int) -> int:
    seed = int(compact[1 + ((lane * 13 + (lane >> 5)) & 255)])
    mixed = seed + lane * 17 + (lane >> 3) * 31 + (lane >> 7) * 127
    return (mixed & 1023) - 512


def _rms_denominator(compact: np.ndarray, lanes: int) -> int:
    sum_sq = 0
    for lane in range(lanes):
        raw = _compact_raw_lane(compact, lane)
        sum_sq += raw * raw
    return math.isqrt(sum_sq // lanes + 1)


def _normalized_lane(compact: np.ndarray, lane: int, denominator: int) -> int:
    raw = _compact_raw_lane(compact, lane)
    return _trunc_div(raw * RMSNORM_SCALE, denominator)


def normalized_replay(compact: np.ndarray) -> np.ndarray:
    payload_dwords = C1R2_PACKET_DWORDS - 1
    lanes = payload_dwords * 2
    denominator = _rms_denominator(compact, lanes)
    replay = np.empty(payload_dwords, dtype=np.int32)
    for idx in range(payload_dwords):
        replay[idx] = _pack_s16_pair(
            _normalized_lane(compact, idx * 2, denominator),
            _normalized_lane(compact, idx * 2 + 1, denominator),
        )
    return replay


def o_global_compact() -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_o_record(group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def replay_payload(replay: int, payload_idx: int, compact: np.ndarray) -> int:
    del replay
    return int(normalized_replay(compact)[payload_idx])


def replay_chunk(replay: int, chunk: int, compact: np.ndarray) -> np.ndarray:
    start = chunk * MAIN_CHUNK_DWORDS
    del replay
    replay = normalized_replay(compact)
    return replay[start : start + MAIN_CHUNK_DWORDS].copy()


def _projection_weight(replay: int, group: int, row: int, out_lane: int, source_lane: int, tap: int) -> int:
    return ((replay * 5 + group * 3 + row * 7 + out_lane * 11 + source_lane * 13 + tap) & 15) - 8


def _scaled_projection_lane(value: int) -> int:
    return _trunc_div(value, 1 << PROJECTION_SHIFT)


def main_replay_accum(group: int, row: int, compact: np.ndarray, replay_idx: int) -> np.ndarray:
    accum = np.zeros(MAIN_ACCUM_DWORDS, dtype=np.int32)
    replay = normalized_replay(compact)
    for chunk in range(MAIN_CHUNKS_PER_REPLAY):
        payload = replay[chunk * MAIN_CHUNK_DWORDS : (chunk + 1) * MAIN_CHUNK_DWORDS]
        for out_lane in range(MAIN_ACCUM_DWORDS):
            total = 0
            for tap in range(PROJECTION_TAPS):
                source_lane = (out_lane * 17 + tap * 37 + group * 11 + row * 7 + chunk * 13) & 255
                activation = _unpack_s16_word(int(payload[source_lane >> 1]), source_lane & 1)
                weight = _projection_weight(replay_idx, group, row, out_lane, source_lane, tap)
                total += activation * weight
            accum[out_lane] = int(accum[out_lane]) + total
    return accum


def make_projection_record(
    phase: int,
    group: int,
    row: int,
    compact: np.ndarray,
    replay_idx: int,
    slice_index: int,
) -> np.ndarray:
    accum = main_replay_accum(group, row, compact, replay_idx)
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase, group, row, slice_index)
    for word in range(RECORD_PAYLOAD_DWORDS):
        record[1 + word] = _pack_s16_pair(
            _scaled_projection_lane(int(accum[word * 2])),
            _scaled_projection_lane(int(accum[word * 2 + 1])),
        )
    return record


def make_upgate_slice_records(
    group: int,
    row: int,
    compact: np.ndarray,
    slice_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    up = make_projection_record(UP_PHASE, group, row, compact, slice_index * 2, slice_index)
    gate = make_projection_record(GATE_PHASE, group, row, compact, slice_index * 2 + 1, slice_index)
    return up, gate


def ffn_global_compact(phase: int, slice_index: int = 0) -> np.ndarray:
    compact = o_global_compact()
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = []
        for row in range(ROWS_PER_COLUMN):
            up, gate = make_upgate_slice_records(group, row, compact, slice_index)
            records.append(up if phase == UP_PHASE else gate)
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def expected_c6r2_input(slice_index: int = 0) -> np.ndarray:
    up = ffn_global_compact(UP_PHASE, slice_index)[1:]
    gate = ffn_global_compact(GATE_PHASE, slice_index)[1:]
    if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
        raise RuntimeError(f"bad c6r2 halves: {up.shape[0]}/{gate.shape[0]}")
    return np.concatenate((up, gate)).astype(np.int32)


def expected_output() -> np.ndarray:
    slices = []
    for slice_index in range(SWIGLU_SLICES):
        c6r2_input = expected_c6r2_input(slice_index)
        if c6r2_input.shape[0] != C6R2_INPUT_DWORDS:
            raise RuntimeError(f"bad c6r2 input: {c6r2_input.shape[0]}")
        slices.append(fixed_swiglu_slice_output(c6r2_input, slice_index))
    return np.concatenate(slices).astype(np.int32)


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
        f"main_chunks={TOTAL_MAIN_CHUNKS} per main tile, per-replay tapped int4 projection",
        f"ffn_compact={SWIGLU_SLICES} adjacent up/gate pairs -> row1/c1r1 -> c6r2",
        f"c6r2_output={SWIGLU_SLICES}x{SWIGLU_OUTPUT_DWORDS} dwords",
    ]
