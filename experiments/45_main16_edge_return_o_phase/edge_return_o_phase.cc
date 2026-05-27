#include <stdint.h>

namespace {

constexpr int ROWS_PER_COLUMN = 4;
constexpr int SHARD_DWORDS = 32;
constexpr int RECORD_DWORDS = 17;
constexpr int RECORD_PAYLOAD_DWORDS = 16;
constexpr int COLUMN_DWORDS = ROWS_PER_COLUMN * SHARD_DWORDS;

static inline int32_t record_header(int32_t group, int32_t row) {
    return 0x45000000 | (group << 12) | (row << 4) | 0xA;
}

static inline int32_t record_payload(int32_t group, int32_t row, int32_t lane) {
    return 10000 * group + 100 * row + lane;
}

} // namespace

extern "C" {

void emit_main_record(int32_t *record, int32_t group, int32_t row) {
    record[0] = record_header(group, row);
    for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
        record[1 + lane] = record_payload(group, row, lane);
    }
}

void edge_make_attention_row(int32_t *record, int32_t *attention, int32_t group, int32_t row) {
    for (int lane = 0; lane < SHARD_DWORDS; lane++) {
        int32_t a = record[1 + (lane & (RECORD_PAYLOAD_DWORDS - 1))];
        int32_t b = record[1 + ((lane + 5) & (RECORD_PAYLOAD_DWORDS - 1))];
        attention[lane] = a + 2 * b + group * 1000 + row * 31 + lane;
    }
}

void main_o_phase(int32_t *attention, int32_t *weight, int32_t *output, int32_t group, int32_t row) {
    for (int lane = 0; lane < SHARD_DWORDS; lane++) {
        output[lane] = attention[lane] * weight[lane] + attention[(lane + 7) & (SHARD_DWORDS - 1)] -
                       weight[(lane + 11) & (SHARD_DWORDS - 1)] + group * 13 + row * 5 + lane;
    }
}

} // extern "C"
