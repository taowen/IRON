#include <stdint.h>

namespace {

constexpr int ROWS_PER_COLUMN = 4;
constexpr int SHARD_DWORDS = 32;
constexpr int RECORD_DWORDS = 17;
constexpr int RECORD_PAYLOAD_DWORDS = 16;
constexpr int O_WEIGHT_DWORDS = 32;
constexpr int FFN_WEIGHT_OFFSET = O_WEIGHT_DWORDS;

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

void main_ffn_tail(int32_t *o_output, int32_t *weight, int32_t *gate, int32_t *up,
                   int32_t *swiglu, int32_t *output, int32_t group, int32_t row) {
    int32_t *ffn_weight = weight + FFN_WEIGHT_OFFSET;
    for (int lane = 0; lane < SHARD_DWORDS; lane++) {
        int32_t norm = o_output[lane] / 128 - ((group * 5 + row * 3 + lane) & 15) + 7;
        gate[lane] = norm * ffn_weight[lane] + ffn_weight[32 + lane] + group * 11 + row;
        up[lane] = norm * ffn_weight[64 + lane] - ffn_weight[96 + lane] + row * 17 + lane;
        swiglu[lane] = (gate[lane] * up[lane]) / 64;
    }

    for (int lane = 0; lane < SHARD_DWORDS; lane++) {
        output[lane] = o_output[lane] + swiglu[lane] -
                       ((swiglu[(lane + 9) & (SHARD_DWORDS - 1)] * ffn_weight[96 + lane]) / 32) +
                       ffn_weight[32 + lane] - ffn_weight[(lane + 13) & (SHARD_DWORDS - 1)] +
                       group * 19 + row * 23 + lane;
    }
}

} // extern "C"
