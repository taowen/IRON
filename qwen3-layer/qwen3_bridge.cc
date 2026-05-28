#include <stdint.h>

extern "C" {

void bridge_init_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
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

static int32_t ffn_record_header(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0x5A;
}

static int32_t ffn_payload_value(int32_t phase, int32_t group, int32_t row, int32_t lane) {
    return phase * 4096 + group * 1024 + row * 256 + lane;
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

void ffn_swiglu_contract(int32_t *input, int32_t *output, int32_t dwords) {
    const int32_t half = dwords / 2;
    for (int32_t idx = 0; idx < half; idx++) {
        const uint32_t low = static_cast<uint32_t>(input[idx]);
        const uint32_t high = static_cast<uint32_t>(input[half + idx]);
        output[idx] = static_cast<int32_t>((low << 16) | (high & 0xffff));
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

void c1r2_make_replay(
    int32_t *compact,
    int32_t *replay,
    int32_t replay_idx,
    int32_t payload_dwords
) {
    replay[0] = 0xC1000000 | replay_idx;
    for (int32_t idx = 0; idx < payload_dwords; idx++) {
        const int32_t seed = compact[1 + (idx & 255)];
        replay[1 + idx] = (replay_idx * 1024 + (idx & 1023) + (seed & 127)) & 0xffff;
    }
}

void c1r2_main_init_accum(int32_t *accum, int32_t group, int32_t row) {
    constexpr int32_t accum_dwords = 32;
    for (int32_t idx = 0; idx < accum_dwords; idx++) {
        accum[idx] = 0;
    }
    accum[0] = group;
    accum[1] = row;
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
    constexpr int32_t payload_dwords = 16;
    const int32_t replay = chunk_idx / replay_chunks;
    const int32_t parity = replay & 1;
    const int32_t base = parity * payload_dwords;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        const int32_t source = (lane * 7 + group * 13 + row * 5 + chunk_idx) & (dwords - 1);
        accum[base + lane] = (accum[base + lane] + chunk[source] + chunk_idx + lane) & 0xffff;
    }
}

void c1r2_main_emit_upgate_records(
    int32_t *records,
    int32_t *accum,
    int32_t group,
    int32_t row
) {
    constexpr int32_t up_phase = 4;
    constexpr int32_t gate_phase = 5;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t payload_dwords = 16;
    constexpr int32_t up_offset = record_dwords;
    constexpr int32_t gate_offset = record_dwords * 2;

    records[up_offset] = (up_phase << 24) | (group << 16) | (row << 8) | 0xA5;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[up_offset + 1 + lane] = accum[lane];
    }

    records[gate_offset] = (gate_phase << 24) | (group << 16) | (row << 8) | 0xA5;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[gate_offset + 1 + lane] = accum[payload_dwords + lane];
    }
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
        carrier[idx] = (
            (window + 1) * 100000 +
            (q_window[q_idx] & 0xffff) +
            ((k_window[k_idx] & 0xffff) << 1) +
            idx
        );
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
        output[idx] = (
            (window + 1) * 1000000 +
            carrier[c_idx] +
            (v_window[v_idx] & 0xffff) +
            idx * 17
        );
    }
}

void shape_init_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
}

void shape_accum_chunk(
    int32_t *chunk,
    int32_t *summary,
    int32_t chunk_idx,
    int32_t dwords
) {
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

}
