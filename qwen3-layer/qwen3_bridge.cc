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

static int32_t attention_kv16_unpack_s16(int32_t *payload, int32_t lane) {
    const uint32_t word = static_cast<uint32_t>(payload[lane >> 1]);
    const uint32_t raw = (lane & 1) != 0 ? (word >> 16) & 0xffff : word & 0xffff;
    return (raw & 0x8000) != 0 ? static_cast<int32_t>(raw) - 0x10000 : static_cast<int32_t>(raw);
}

static int32_t attention_kv16_clamp_s16(int32_t value) {
    if (value < -32768) {
        return -32768;
    }
    if (value > 32767) {
        return 32767;
    }
    return value;
}

static int32_t attention_kv16_pack_s16_pair(int32_t low, int32_t high) {
    const uint32_t low_u16 = static_cast<uint32_t>(attention_kv16_clamp_s16(low)) & 0xffff;
    const uint32_t high_u16 = static_cast<uint32_t>(attention_kv16_clamp_s16(high)) & 0xffff;
    return static_cast<int32_t>(low_u16 | (high_u16 << 16));
}

static int32_t attention_kv16_div(int32_t numerator, int32_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -((-numerator) / denominator);
}

static int32_t attention_kv16_weight(int32_t delta) {
    if (delta <= 0) {
        return 4096;
    }
    if (delta <= 1) {
        return 3072;
    }
    if (delta <= 2) {
        return 2048;
    }
    if (delta <= 4) {
        return 1024;
    }
    if (delta <= 8) {
        return 512;
    }
    if (delta <= 16) {
        return 256;
    }
    if (delta <= 32) {
        return 128;
    }
    return 64;
}

static int32_t attention_kv16_score(
    int32_t *q_window,
    int32_t *k_window,
    int32_t q_head,
    int32_t token
) {
    constexpr int32_t head_dim = 128;
    constexpr int32_t context = 16;
    constexpr int32_t gqa_ratio = 4;
    const int32_t kv_head = q_head / gqa_ratio;
    const int32_t q_base = q_head * head_dim;
    const int32_t k_base = kv_head * context * head_dim + token * head_dim;
    int32_t dot = 0;
    for (int32_t dim = 0; dim < head_dim; dim++) {
        dot += attention_kv16_unpack_s16(q_window, q_base + dim) *
            attention_kv16_unpack_s16(k_window, k_base + dim);
    }
    return attention_kv16_div(dot, head_dim);
}

void attention_kv16_make_carrier(
    int32_t *q_window,
    int32_t *k_window,
    int32_t *carrier,
    int32_t window,
    int32_t q_dwords,
    int32_t k_dwords,
    int32_t carrier_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t context = 16;
    constexpr int32_t weight_dwords = 64;
    for (int32_t idx = 0; idx < carrier_dwords; idx++) {
        carrier[idx] = 0;
    }

    for (int32_t q_head = 0; q_head < heads; q_head++) {
        int32_t scores[context];
        int32_t running_max = attention_kv16_score(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < context; token++) {
            scores[token] = attention_kv16_score(q_window, k_window, q_head, token);
            if (scores[token] > running_max) {
                running_max = scores[token];
            }
        }

        int32_t weight_sum = 0;
        for (int32_t token = 0; token < context; token += 2) {
            const int32_t low = attention_kv16_weight(running_max - scores[token]);
            const int32_t high = attention_kv16_weight(running_max - scores[token + 1]);
            carrier[(q_head * context + token) >> 1] =
                static_cast<int32_t>((low & 0xffff) | ((high & 0xffff) << 16));
            weight_sum += low + high;
        }

        const int32_t scalar = weight_dwords + q_head * 2;
        carrier[scalar] = running_max;
        carrier[scalar + 1] = weight_sum;
    }

    (void)window;
    (void)q_dwords;
    (void)k_dwords;
}

static int32_t attention_kv16_unpack_weight(int32_t *carrier, int32_t q_head, int32_t token) {
    constexpr int32_t context = 16;
    const int32_t lane = q_head * context + token;
    const uint32_t word = static_cast<uint32_t>(carrier[lane >> 1]);
    return (lane & 1) != 0 ? static_cast<int32_t>((word >> 16) & 0xffff) : static_cast<int32_t>(word & 0xffff);
}

void attention_kv16_make_return(
    int32_t *v_window,
    int32_t *carrier,
    int32_t *output,
    int32_t window,
    int32_t v_dwords,
    int32_t output_dwords,
    int32_t carrier_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t context = 16;
    constexpr int32_t head_dim = 128;
    constexpr int32_t gqa_ratio = 4;
    constexpr int32_t weight_dwords = 64;
    for (int32_t idx = 0; idx < output_dwords; idx++) {
        output[idx] = 0;
    }

    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const int32_t kv_head = q_head / gqa_ratio;
        const int32_t weight_sum = carrier[weight_dwords + q_head * 2 + 1];
        for (int32_t dim = 0; dim < head_dim; dim += 2) {
            int32_t low_total = 0;
            int32_t high_total = 0;
            for (int32_t token = 0; token < context; token++) {
                const int32_t weight = attention_kv16_unpack_weight(carrier, q_head, token);
                const int32_t v_base = kv_head * context * head_dim + token * head_dim + dim;
                low_total += weight * attention_kv16_unpack_s16(v_window, v_base);
                high_total += weight * attention_kv16_unpack_s16(v_window, v_base + 1);
            }
            const int32_t low = attention_kv16_div(low_total, weight_sum);
            const int32_t high = attention_kv16_div(high_total, weight_sum);
            output[(q_head * head_dim + dim) >> 1] = attention_kv16_pack_s16_pair(low, high);
        }
    }

    (void)window;
    (void)v_dwords;
    (void)carrier_dwords;
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

static int32_t qkv_record_header(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0xC3;
}

static int32_t qkv_payload_value(int32_t phase, int32_t group, int32_t row, int32_t lane) {
    return phase * 4096 + group * 1024 + row * 256 + lane * 3 + 7;
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

void mainq_postprocess_payload(
    int32_t *q_compact,
    int32_t *q_payload,
    int32_t q_dwords
) {
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        const int32_t low_lane = idx * 2;
        const int32_t high_lane = low_lane + 1;
        const int32_t low_seed = q_compact[1 + (low_lane & 255)] & 31;
        const int32_t high_seed = q_compact[1 + (high_lane & 255)] & 31;
        const int32_t low = ((low_lane * 5 + low_seed) % 31) - 15;
        const int32_t high = ((high_lane * 5 + high_seed) % 31) - 15;
        q_payload[idx] = attention_kv16_pack_s16_pair(low, high);
    }
}

void mainq_emit_o_record(
    int32_t *records,
    int32_t *summary,
    int32_t group,
    int32_t row
) {
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

void qkv_postprocess_payload(
    int32_t *q_compact,
    int32_t *k_compact,
    int32_t *v_compact,
    int32_t *q_payload,
    int32_t *kv_payload,
    int32_t q_dwords,
    int32_t kv_side_dwords,
    int32_t window_dwords
) {
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        const int32_t q_seed = q_compact[1 + (idx & 255)] & 31;
        const int32_t k_seed = k_compact[1 + ((idx * 3) & 255)] & 7;
        q_payload[idx] = 10000 + idx * 3 + q_seed + k_seed;
    }

    for (int32_t side = 0; side < 2; side++) {
        int32_t *kv = kv_payload + side * kv_side_dwords;
        const int32_t base = 20000 + side * 10000;
        for (int32_t idx = 0; idx < kv_side_dwords; idx++) {
            const int32_t is_v = (idx / window_dwords) & 1;
            int32_t seed = 0;
            if (is_v) {
                seed = v_compact[1 + ((idx * 7 + side * 13) & 255)] & 31;
            } else {
                seed = k_compact[1 + ((idx * 5 + side * 11) & 255)] & 31;
            }
            kv[idx] = base + idx * 5 + seed;
        }
    }
}

void qkv_split_kv_payload(
    int32_t *kv_payload,
    int32_t *kv_left,
    int32_t *kv_right,
    int32_t kv_side_dwords
) {
    for (int32_t idx = 0; idx < kv_side_dwords; idx++) {
        kv_left[idx] = kv_payload[idx];
        kv_right[idx] = kv_payload[kv_side_dwords + idx];
    }
}

void qkv_main_init_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
}

void qkv_main_accum_chunk(
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

void qkv_main_emit_o_record(
    int32_t *records,
    int32_t *summary,
    int32_t group,
    int32_t row
) {
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

void qkv_c1r2_summarize_compact(int32_t *compact, int32_t *summary, int32_t dwords) {
    uint32_t sum = 0;
    uint32_t hash = 0;
    for (int32_t idx = 0; idx < dwords; idx++) {
        const uint32_t value = static_cast<uint32_t>(compact[idx]);
        sum += value;
        hash = (hash * 16777619u) ^ (value + static_cast<uint32_t>(idx));
    }
    summary[0] = 0x51564F43;
    summary[1] = dwords;
    summary[2] = compact[0];
    summary[3] = compact[dwords - 1];
    summary[4] = static_cast<int32_t>(sum);
    summary[5] = static_cast<int32_t>(hash);
    summary[6] = compact[1];
    summary[7] = compact[dwords - 2];
}

void full_main_emit_upgate_records(
    int32_t *records,
    int32_t *accum,
    int32_t group,
    int32_t row
) {
    constexpr int32_t up_phase = 4;
    constexpr int32_t gate_phase = 5;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t payload_dwords = 16;
    constexpr int32_t up_offset = record_dwords * up_phase;
    constexpr int32_t gate_offset = record_dwords * gate_phase;

    records[up_offset] = (up_phase << 24) | (group << 16) | (row << 8) | 0xA5;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[up_offset + 1 + lane] = accum[lane];
    }

    records[gate_offset] = (gate_phase << 24) | (group << 16) | (row << 8) | 0xA5;
    for (int32_t lane = 0; lane < payload_dwords; lane++) {
        records[gate_offset + 1 + lane] = accum[payload_dwords + lane];
    }
}

void full_main_init_down_summary(int32_t *summary, int32_t group, int32_t row) {
    summary[0] = group;
    summary[1] = row;
    summary[2] = 0;
    summary[3] = 0;
    summary[4] = 0;
    summary[5] = 0;
    summary[6] = 0;
    summary[7] = 0;
}

void full_main_accum_down_chunk(
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

void full_main_emit_down_record(
    int32_t *records,
    int32_t *summary,
    int32_t group,
    int32_t row
) {
    constexpr int32_t down_phase = 6;
    constexpr int32_t record_dwords = 17;
    constexpr int32_t offset = record_dwords * down_phase;
    records[offset] = (down_phase << 24) | (group << 16) | (row << 8) | 0xD0;
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
    records[offset + 15] = 0x444F574E;
    records[offset + 16] = 0x46494E;
}

void full_c1r2_summarize_down(int32_t *compact, int32_t *summary, int32_t dwords) {
    uint32_t sum = 0;
    uint32_t hash = 0;
    for (int32_t idx = 0; idx < dwords; idx++) {
        const uint32_t value = static_cast<uint32_t>(compact[idx]);
        sum += value;
        hash = (hash * 16777619u) ^ (value + static_cast<uint32_t>(idx));
    }
    summary[0] = 0x464C4C43;
    summary[1] = dwords;
    summary[2] = compact[0];
    summary[3] = compact[dwords - 1];
    summary[4] = static_cast<int32_t>(sum);
    summary[5] = static_cast<int32_t>(hash);
    summary[6] = compact[1];
    summary[7] = compact[dwords - 2];
}

}
