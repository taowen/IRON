#include <stdint.h>

namespace {

static int32_t unpack_s16(int32_t word, int32_t lane) {
    const uint32_t raw_word = static_cast<uint32_t>(word);
    const uint32_t raw = lane != 0 ? (raw_word >> 16) & 0xffff : raw_word & 0xffff;
    return (raw & 0x8000) != 0 ? static_cast<int32_t>(raw) - 0x10000 : static_cast<int32_t>(raw);
}

static int32_t clamp_s16(int32_t value) {
    if (value < -32768) {
        return -32768;
    }
    if (value > 32767) {
        return 32767;
    }
    return value;
}

static int32_t pack_s16_pair(int32_t low, int32_t high) {
    const uint32_t low_u16 = static_cast<uint32_t>(clamp_s16(low)) & 0xffff;
    const uint32_t high_u16 = static_cast<uint32_t>(clamp_s16(high)) & 0xffff;
    return static_cast<int32_t>(low_u16 | (high_u16 << 16));
}

static int32_t div_i32(int32_t numerator, int32_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -((-numerator) / denominator);
}

static int32_t ffn_record_header(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0x5A;
}

static int32_t ffn_payload_value(int32_t phase, int32_t group, int32_t row, int32_t lane) {
    return phase * 4096 + group * 1024 + row * 256 + lane;
}

static int32_t qkv_record_header(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0xC3;
}

static int32_t qkv_payload_value(int32_t phase, int32_t group, int32_t row, int32_t lane) {
    return phase * 4096 + group * 1024 + row * 256 + lane * 3 + 7;
}

static void emit_projection_record(
    int32_t *records,
    int32_t *accum,
    int32_t phase,
    int32_t record_offset,
    int32_t group,
    int32_t row,
    int32_t slice
) {
    constexpr int32_t payload_dwords = 16;
    constexpr int32_t projection_shift = 8;
    records[record_offset] = (phase << 24) | (group << 16) | (row << 8) | (slice & 0xff);
    for (int32_t word = 0; word < payload_dwords; word++) {
        const int32_t low = div_i32(accum[word * 2], 1 << projection_shift);
        const int32_t high = div_i32(accum[word * 2 + 1], 1 << projection_shift);
        records[record_offset + 1 + word] = pack_s16_pair(low, high);
    }
}

static void init_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
}

} // namespace

extern "C" {

void bridge_init_summary(int32_t *summary, int32_t group, int32_t row) {
    init_summary(summary, group, row);
}

void bridge_accum_chunk(int32_t *chunk, int32_t *summary, int32_t dwords) {
    if (summary[2] == 0) {
        summary[3] = chunk[0];
    }
    summary[4] = chunk[dwords - 1];
    for (int idx = 0; idx < dwords; idx++) {
        summary[5] += chunk[idx];
        summary[6] ^= chunk[idx];
    }
    summary[2] += 1;
    summary[7] += dwords;
}

void ffn_emit_records(int32_t *records, int32_t group, int32_t row) {
    constexpr int32_t up_phase = 4;
    constexpr int32_t gate_phase = 5;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t payload_dwords = 16;
    records[0] = ffn_record_header(up_phase, group, row);
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[1 + lane] = ffn_payload_value(up_phase, group, row, lane);
    }
    records[record_dwords] = ffn_record_header(gate_phase, group, row);
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[record_dwords + 1 + lane] = ffn_payload_value(gate_phase, group, row, lane);
    }
}

void c1r2_emit_o_record(int32_t *records, int32_t group, int32_t row) {
    constexpr int32_t o_phase = 3;
    constexpr int32_t payload_dwords = 16;
    records[0] = (o_phase << 24) | (group << 16) | (row << 8) | 0xA5;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[1 + lane] = o_phase * 4096 + group * 1024 + row * 256 + lane;
    }
}

void c1r2_main_init_accum(int32_t *accum, int32_t group, int32_t row) {
    constexpr int32_t accum_dwords = 64;
    for (int32_t idx = 0; idx < accum_dwords; idx++) {
        accum[idx] = 0;
    }
    (void)group;
    (void)row;
}

void c1r2_main_accum_chunk(
    int32_t *chunk,
    int32_t *accum,
    int32_t chunk_idx,
    int32_t group,
    int32_t row,
    int32_t dwords
) {
    constexpr int32_t replay_chunks = 16;
    constexpr int32_t accum_lanes = 64;
    constexpr int32_t projection_taps = 4;
    const int32_t replay = chunk_idx / replay_chunks;
    const int32_t chunk_in_replay = chunk_idx - replay * replay_chunks;
    const int32_t lanes = dwords * 2;
    for (int32_t out_lane = 0; out_lane < accum_lanes; out_lane++) {
        int32_t total = 0;
        for (int32_t tap = 0; tap < projection_taps; tap++) {
            const int32_t source_lane =
                (out_lane * 17 + tap * 37 + group * 11 + row * 7 + chunk_in_replay * 13) &
                (lanes - 1);
            const int32_t activation = unpack_s16(chunk[source_lane >> 1], source_lane & 1);
            const int32_t weight =
                ((replay * 5 + group * 3 + row * 7 + out_lane * 11 + source_lane * 13 + tap) &
                 15) -
                8;
            total += activation * weight;
        }
        accum[out_lane] += total;
    }
}

void c1r2_main_emit_up_record(
    int32_t *records,
    int32_t *accum,
    int32_t group,
    int32_t row,
    int32_t slice
) {
    constexpr int32_t up_phase = 4;
    constexpr int32_t record_dwords = 17;
    emit_projection_record(records, accum, up_phase, record_dwords, group, row, slice);
}

void c1r2_main_emit_gate_record(
    int32_t *records,
    int32_t *accum,
    int32_t group,
    int32_t row,
    int32_t slice
) {
    constexpr int32_t gate_phase = 5;
    constexpr int32_t record_dwords = 17;
    emit_projection_record(records, accum, gate_phase, record_dwords * 2, group, row, slice);
}

void shape_make_carrier(
    int32_t *q_window,
    int32_t *k_window,
    int32_t *carrier,
    int32_t window,
    int32_t window_dwords,
    int32_t carrier_dwords
) {
    for (int32_t idx = 0; idx < carrier_dwords; idx++) {
        const int32_t q_idx = (idx * 7 + window) & (window_dwords - 1);
        const int32_t k_idx = (idx * 13 + window * 3) & (window_dwords - 1);
        carrier[idx] =
            (window + 1) * 100000 + (q_window[q_idx] & 0xffff) +
            ((k_window[k_idx] & 0xffff) << 1) + idx;
    }
}

void shape_make_return(
    int32_t *v_window,
    int32_t *carrier,
    int32_t *output,
    int32_t window,
    int32_t window_dwords,
    int32_t carrier_dwords
) {
    for (int32_t idx = 0; idx < window_dwords; idx++) {
        const int32_t c_idx = (idx * 3 + window) % carrier_dwords;
        const int32_t v_idx = (idx * 5 + window * 11) & (window_dwords - 1);
        output[idx] = (window + 1) * 1000000 + carrier[c_idx] + (v_window[v_idx] & 0xffff) +
            idx * 17;
    }
}

void shape_init_summary(int32_t *summary, int32_t group, int32_t row) {
    init_summary(summary, group, row);
}

void shape_accum_chunk(int32_t *chunk, int32_t *summary, int32_t chunk_idx, int32_t dwords) {
    if (summary[2] == 0) {
        summary[3] = chunk[0];
    }
    summary[4] = chunk[dwords - 1];
    uint32_t sum = static_cast<uint32_t>(summary[5]);
    uint32_t hash = static_cast<uint32_t>(summary[6]);
    const uint32_t base = static_cast<uint32_t>(chunk_idx * dwords);
    for (int32_t idx = 0; idx < dwords; idx++) {
        const uint32_t value = static_cast<uint32_t>(chunk[idx]);
        sum += value;
        hash = (hash * 16777619u) ^ (value + base + static_cast<uint32_t>(idx));
    }
    summary[2] += 1;
    summary[5] = static_cast<int32_t>(sum);
    summary[6] = static_cast<int32_t>(hash);
    summary[7] += dwords;
}

void qkv_emit_qkv_records(int32_t *records, int32_t group, int32_t row) {
    constexpr int32_t record_dwords = 17;
    constexpr int32_t payload_dwords = 16;
    for (int32_t phase = 0; phase < 3; phase++) {
        const int32_t offset = phase * record_dwords;
        records[offset] = qkv_record_header(phase, group, row);
        for (int32_t lane = 0; lane < payload_dwords; lane++) {
            records[offset + 1 + lane] = qkv_payload_value(phase, group, row, lane);
        }
    }
}

void mainq_emit_q_record(int32_t *records, int32_t group, int32_t row) {
    constexpr int32_t q_phase = 0;
    constexpr int32_t payload_dwords = 16;
    records[0] = qkv_record_header(q_phase, group, row);
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[1 + lane] = qkv_payload_value(q_phase, group, row, lane);
    }
}

void mainq_emit_o_record(int32_t *records, int32_t *summary, int32_t group, int32_t row) {
    constexpr int32_t o_phase = 3;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t offset = record_dwords;
    records[offset] = qkv_record_header(o_phase, group, row);
    records[offset + 1] = summary[2];
    records[offset + 2] = summary[3];
    records[offset + 3] = summary[4];
    records[offset + 4] = summary[5];
    records[offset + 5] = summary[6];
    records[offset + 6] = summary[7];
    records[offset + 7] = group;
    records[offset + 8] = row;
    records[offset + 9] = summary[2] ^ group;
    records[offset + 10] = summary[3] ^ row;
    records[offset + 11] = summary[4] ^ group;
    records[offset + 12] = summary[5] ^ row;
    records[offset + 13] = summary[6] ^ group;
    records[offset + 14] = summary[7] ^ row;
    records[offset + 15] = 0x51564F;
    records[offset + 16] = 0x4F434D50;
}

void qkv_main_init_summary(int32_t *summary, int32_t group, int32_t row) {
    init_summary(summary, group, row);
}

void qkv_main_accum_chunk(int32_t *chunk, int32_t *summary, int32_t chunk_idx, int32_t dwords) {
    shape_accum_chunk(chunk, summary, chunk_idx, dwords);
}

void qkv_main_emit_o_record(int32_t *records, int32_t *summary, int32_t group, int32_t row) {
    constexpr int32_t o_phase = 3;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t offset = o_phase * record_dwords;
    records[offset] = qkv_record_header(o_phase, group, row);
    records[offset + 1] = summary[2];
    records[offset + 2] = summary[3];
    records[offset + 3] = summary[4];
    records[offset + 4] = summary[5];
    records[offset + 5] = summary[6];
    records[offset + 6] = summary[7];
    records[offset + 7] = group;
    records[offset + 8] = row;
    records[offset + 9] = summary[2] ^ group;
    records[offset + 10] = summary[3] ^ row;
    records[offset + 11] = summary[4] ^ group;
    records[offset + 12] = summary[5] ^ row;
    records[offset + 13] = summary[6] ^ group;
    records[offset + 14] = summary[7] ^ row;
    records[offset + 15] = 0x51564F;
    records[offset + 16] = 0x4F434D50;
}

} // extern "C"
