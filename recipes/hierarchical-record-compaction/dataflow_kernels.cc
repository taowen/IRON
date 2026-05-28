#include <stdint.h>

extern "C" {

static int32_t record_header_value(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0x5A;
}

static int32_t record_payload_value(int32_t phase, int32_t group, int32_t row, int32_t lane) {
    return phase * 4096 + group * 1024 + row * 256 + lane;
}

void emit_paired_records(int32_t *records, int32_t group, int32_t row) {
    constexpr int32_t low_phase = 4;
    constexpr int32_t high_phase = 5;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t payload_dwords = 16;

    records[0] = record_header_value(low_phase, group, row);
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[1 + lane] = record_payload_value(low_phase, group, row, lane);
    }

    records[record_dwords] = record_header_value(high_phase, group, row);
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[record_dwords + 1 + lane] =
            record_payload_value(high_phase, group, row, lane);
    }
}

void merge_paired_halves(int32_t *input, int32_t *output, int32_t dwords) {
    const int32_t half = dwords / 2;
    for (int32_t idx = 0; idx < half; idx++) {
        const uint32_t low = static_cast<uint32_t>(input[idx]);
        const uint32_t high = static_cast<uint32_t>(input[half + idx]);
        output[idx] = static_cast<int32_t>((low << 16) | (high & 0xffff));
    }
}

}
