"""CPU reference for exp40 projection-record handoff ABI."""

import numpy as np

RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = 16
ROWS_PER_COLUMN = 4
COLUMNS_PER_BLOCK = 4
RECORDS_PER_BLOCK = ROWS_PER_COLUMN * COLUMNS_PER_BLOCK
HIDDEN_BLOCKS = 8
FFN_BLOCKS = 24
ALL_RECORDS = FFN_BLOCKS * RECORDS_PER_BLOCK
ALL_RECORD_DWORDS = ALL_RECORDS * RECORD_DWORDS

RAW_COLUMN_DWORDS = ROWS_PER_COLUMN * RECORD_DWORDS
COLUMN65_DWORDS = 1 + ROWS_PER_COLUMN * RECORD_PAYLOAD_DWORDS
REPLAY64_DWORDS = ROWS_PER_COLUMN * RECORD_PAYLOAD_DWORDS
BLOCK257_DWORDS = 1 + RECORDS_PER_BLOCK * RECORD_PAYLOAD_DWORDS
HIDDEN2049_DWORDS = 1 + HIDDEN_BLOCKS * RECORDS_PER_BLOCK * RECORD_PAYLOAD_DWORDS
FFN6144_DWORDS = FFN_BLOCKS * RECORDS_PER_BLOCK * RECORD_PAYLOAD_DWORDS

RAW_COLUMN_OFFSET = 0
COLUMN65_OFFSET = RAW_COLUMN_OFFSET + RAW_COLUMN_DWORDS
REPLAY64_OFFSET = COLUMN65_OFFSET + COLUMN65_DWORDS
BLOCK257_OFFSET = REPLAY64_OFFSET + REPLAY64_DWORDS
HIDDEN2049_OFFSET = BLOCK257_OFFSET + BLOCK257_DWORDS
FFN6144_OFFSET = HIDDEN2049_OFFSET + HIDDEN2049_DWORDS
TOTAL_OUTPUT_DWORDS = FFN6144_OFFSET + FFN6144_DWORDS


def record_header(phase: int, block: int, col: int, row: int) -> np.int32:
    value = 0x40000000 | (phase << 20) | (block << 12) | (col << 8) | (row << 4) | 0xA
    return np.int32(value)


def record_payload(phase: int, block: int, col: int, row: int, lane: int) -> np.int32:
    return np.int32(phase * 1_000_000 + block * 10_000 + col * 1_000 + row * 100 + lane)


def column_header(phase: int, block: int, col: int) -> np.int32:
    return np.int32(0x51000000 | (phase << 20) | (block << 12) | (col << 8) | 0x5)


def block_header(phase: int, block: int) -> np.int32:
    return np.int32(0x52000000 | (phase << 20) | (block << 12) | 0x7)


def hidden_header(phase: int) -> np.int32:
    return np.int32(0x53000000 | (phase << 20) | 0x9)


def make_record(phase: int, block: int, col: int, row: int) -> np.ndarray:
    record = np.zeros(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase, block, col, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = record_payload(phase, block, col, row, lane)
    return record


def make_raw_column() -> np.ndarray:
    return np.concatenate([make_record(1, 0, 0, row) for row in range(ROWS_PER_COLUMN)])


def make_all_records() -> np.ndarray:
    records = []
    for block in range(FFN_BLOCKS):
        for col in range(COLUMNS_PER_BLOCK):
            for row in range(ROWS_PER_COLUMN):
                records.append(make_record(2, block, col, row))
    return np.concatenate(records).astype(np.int32)


def _record_at(records: np.ndarray, block: int, col: int, row: int) -> np.ndarray:
    index = (block * RECORDS_PER_BLOCK + col * ROWS_PER_COLUMN + row) * RECORD_DWORDS
    return records[index:index + RECORD_DWORDS]


def column65_from_raw(raw_column: np.ndarray) -> np.ndarray:
    out = np.zeros(COLUMN65_DWORDS, dtype=np.int32)
    out[0] = column_header(1, 0, 0)
    for row in range(ROWS_PER_COLUMN):
        src = raw_column[row * RECORD_DWORDS:(row + 1) * RECORD_DWORDS]
        dst = 1 + row * RECORD_PAYLOAD_DWORDS
        out[dst:dst + RECORD_PAYLOAD_DWORDS] = src[1:]
    return out


def replay64_from_raw(raw_column: np.ndarray) -> np.ndarray:
    return column65_from_raw(raw_column)[1:].copy()


def block257_from_records(records: np.ndarray, block: int = 0) -> np.ndarray:
    out = np.zeros(BLOCK257_DWORDS, dtype=np.int32)
    out[0] = block_header(2, block)
    cursor = 1
    for col in range(COLUMNS_PER_BLOCK):
        for row in range(ROWS_PER_COLUMN):
            out[cursor:cursor + RECORD_PAYLOAD_DWORDS] = _record_at(records, block, col, row)[1:]
            cursor += RECORD_PAYLOAD_DWORDS
    return out


def hidden2049_from_records(records: np.ndarray) -> np.ndarray:
    out = np.zeros(HIDDEN2049_DWORDS, dtype=np.int32)
    out[0] = hidden_header(2)
    cursor = 1
    for block in range(HIDDEN_BLOCKS):
        for col in range(COLUMNS_PER_BLOCK):
            for row in range(ROWS_PER_COLUMN):
                out[cursor:cursor + RECORD_PAYLOAD_DWORDS] = _record_at(records, block, col, row)[1:]
                cursor += RECORD_PAYLOAD_DWORDS
    return out


def ffn6144_from_records(records: np.ndarray) -> np.ndarray:
    out = np.zeros(FFN6144_DWORDS, dtype=np.int32)
    cursor = 0
    for block in range(FFN_BLOCKS):
        for col in range(COLUMNS_PER_BLOCK):
            for row in range(ROWS_PER_COLUMN):
                out[cursor:cursor + RECORD_PAYLOAD_DWORDS] = _record_at(records, block, col, row)[1:]
                cursor += RECORD_PAYLOAD_DWORDS
    return out


def expected_output() -> np.ndarray:
    raw = make_raw_column()
    records = make_all_records()
    out = np.zeros(TOTAL_OUTPUT_DWORDS, dtype=np.int32)
    out[RAW_COLUMN_OFFSET:COLUMN65_OFFSET] = raw
    out[COLUMN65_OFFSET:REPLAY64_OFFSET] = column65_from_raw(raw)
    out[REPLAY64_OFFSET:BLOCK257_OFFSET] = replay64_from_raw(raw)
    out[BLOCK257_OFFSET:HIDDEN2049_OFFSET] = block257_from_records(records)
    out[HIDDEN2049_OFFSET:FFN6144_OFFSET] = hidden2049_from_records(records)
    out[FFN6144_OFFSET:] = ffn6144_from_records(records)
    return out


SEGMENTS = (
    ("raw68", RAW_COLUMN_OFFSET, RAW_COLUMN_DWORDS),
    ("column65", COLUMN65_OFFSET, COLUMN65_DWORDS),
    ("replay64", REPLAY64_OFFSET, REPLAY64_DWORDS),
    ("block257", BLOCK257_OFFSET, BLOCK257_DWORDS),
    ("hidden2049", HIDDEN2049_OFFSET, HIDDEN2049_DWORDS),
    ("ffn6144", FFN6144_OFFSET, FFN6144_DWORDS),
)
