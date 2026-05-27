#include <stdint.h>

namespace {

constexpr int RECORD_DWORDS = 17;
constexpr int RECORD_PAYLOAD_DWORDS = 16;
constexpr int ROWS_PER_COLUMN = 4;
constexpr int COLUMNS_PER_BLOCK = 4;
constexpr int RECORDS_PER_BLOCK = ROWS_PER_COLUMN * COLUMNS_PER_BLOCK;
constexpr int HIDDEN_BLOCKS = 8;
constexpr int FFN_BLOCKS = 24;

constexpr int RAW_COLUMN_DWORDS = ROWS_PER_COLUMN * RECORD_DWORDS;
constexpr int COLUMN65_DWORDS = 1 + ROWS_PER_COLUMN * RECORD_PAYLOAD_DWORDS;
constexpr int REPLAY64_DWORDS = ROWS_PER_COLUMN * RECORD_PAYLOAD_DWORDS;
constexpr int BLOCK257_DWORDS = 1 + RECORDS_PER_BLOCK * RECORD_PAYLOAD_DWORDS;
constexpr int HIDDEN2049_DWORDS = 1 + HIDDEN_BLOCKS * RECORDS_PER_BLOCK * RECORD_PAYLOAD_DWORDS;

constexpr int RAW_COLUMN_OFFSET = 0;
constexpr int COLUMN65_OFFSET = RAW_COLUMN_OFFSET + RAW_COLUMN_DWORDS;
constexpr int REPLAY64_OFFSET = COLUMN65_OFFSET + COLUMN65_DWORDS;
constexpr int BLOCK257_OFFSET = REPLAY64_OFFSET + REPLAY64_DWORDS;
constexpr int HIDDEN2049_OFFSET = BLOCK257_OFFSET + BLOCK257_DWORDS;
constexpr int FFN6144_OFFSET = HIDDEN2049_OFFSET + HIDDEN2049_DWORDS;
constexpr int OUTPUT_SPLIT_DWORDS = 4096;

static inline int32_t record_header(int32_t phase, int32_t block, int32_t col, int32_t row) {
    return 0x40000000 | (phase << 20) | (block << 12) | (col << 8) | (row << 4) | 0xA;
}

static inline int32_t record_payload(int32_t phase, int32_t block, int32_t col, int32_t row, int32_t lane) {
    return phase * 1000000 + block * 10000 + col * 1000 + row * 100 + lane;
}

static inline int32_t column_header(int32_t phase, int32_t block, int32_t col) {
    return 0x51000000 | (phase << 20) | (block << 12) | (col << 8) | 0x5;
}

static inline int32_t block_header(int32_t phase, int32_t block) {
    return 0x52000000 | (phase << 20) | (block << 12) | 0x7;
}

static inline int32_t hidden_header(int32_t phase) {
    return 0x53000000 | (phase << 20) | 0x9;
}

static inline int32_t source_row_from_header(int32_t header) {
    return (header >> 4) & 0xF;
}

static inline int32_t *record_at(int32_t *records, int32_t block, int32_t col, int32_t row) {
    int32_t record_index = block * RECORDS_PER_BLOCK + col * ROWS_PER_COLUMN + row;
    return records + record_index * RECORD_DWORDS;
}

static inline void store_output(int32_t *out0, int32_t *out1, int index, int32_t value) {
    if (index < OUTPUT_SPLIT_DWORDS) {
        out0[index] = value;
    } else {
        out1[index - OUTPUT_SPLIT_DWORDS] = value;
    }
}

} // namespace

extern "C" {

void emit_projection_record(int32_t *record, int32_t phase, int32_t block, int32_t col, int32_t row) {
    record[0] = record_header(phase, block, col, row);
    for (int32_t lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
        record[1 + lane] = record_payload(phase, block, col, row, lane);
    }
}

void aggregate_record_ladder(int32_t *raw_arrival, int32_t *all_records, int32_t *out0, int32_t *out1) {
    int32_t raw_sorted[RAW_COLUMN_DWORDS];
    for (int row = 0; row < ROWS_PER_COLUMN; row++) {
        for (int lane = 0; lane < RECORD_DWORDS; lane++) {
            raw_sorted[row * RECORD_DWORDS + lane] = -1;
        }
    }

    for (int arrival = 0; arrival < ROWS_PER_COLUMN; arrival++) {
        int32_t *record = raw_arrival + arrival * RECORD_DWORDS;
        int row = source_row_from_header(record[0]);
        if (row >= 0 && row < ROWS_PER_COLUMN) {
            for (int lane = 0; lane < RECORD_DWORDS; lane++) {
                raw_sorted[row * RECORD_DWORDS + lane] = record[lane];
            }
        }
    }

    for (int idx = 0; idx < RAW_COLUMN_DWORDS; idx++) {
        store_output(out0, out1, RAW_COLUMN_OFFSET + idx, raw_sorted[idx]);
    }

    store_output(out0, out1, COLUMN65_OFFSET, column_header(1, 0, 0));
    for (int row = 0; row < ROWS_PER_COLUMN; row++) {
        int32_t *record = raw_sorted + row * RECORD_DWORDS;
        for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
            int idx = COLUMN65_OFFSET + 1 + row * RECORD_PAYLOAD_DWORDS + lane;
            store_output(out0, out1, idx, record[1 + lane]);
        }
    }

    for (int lane = 0; lane < REPLAY64_DWORDS; lane++) {
        int row = lane / RECORD_PAYLOAD_DWORDS;
        int payload_lane = lane - row * RECORD_PAYLOAD_DWORDS;
        store_output(out0, out1, REPLAY64_OFFSET + lane, raw_sorted[row * RECORD_DWORDS + 1 + payload_lane]);
    }

    store_output(out0, out1, BLOCK257_OFFSET, block_header(2, 0));
    int cursor = 1;
    for (int col = 0; col < COLUMNS_PER_BLOCK; col++) {
        for (int row = 0; row < ROWS_PER_COLUMN; row++) {
            int32_t *record = record_at(all_records, 0, col, row);
            for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
                store_output(out0, out1, BLOCK257_OFFSET + cursor, record[1 + lane]);
                cursor++;
            }
        }
    }

    store_output(out0, out1, HIDDEN2049_OFFSET, hidden_header(2));
    cursor = 1;
    for (int block = 0; block < HIDDEN_BLOCKS; block++) {
        for (int col = 0; col < COLUMNS_PER_BLOCK; col++) {
            for (int row = 0; row < ROWS_PER_COLUMN; row++) {
                int32_t *record = record_at(all_records, block, col, row);
                for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
                    store_output(out0, out1, HIDDEN2049_OFFSET + cursor, record[1 + lane]);
                    cursor++;
                }
            }
        }
    }

    cursor = 0;
    for (int block = 0; block < FFN_BLOCKS; block++) {
        for (int col = 0; col < COLUMNS_PER_BLOCK; col++) {
            for (int row = 0; row < ROWS_PER_COLUMN; row++) {
                int32_t *record = record_at(all_records, block, col, row);
                for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
                    store_output(out0, out1, FFN6144_OFFSET + cursor, record[1 + lane]);
                    cursor++;
                }
            }
        }
    }
}

} // extern "C"
