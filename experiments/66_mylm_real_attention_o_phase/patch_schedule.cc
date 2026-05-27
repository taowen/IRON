#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int M_PER_TILE = 32;
constexpr int ACT_SLICE_BF16 = 256;
constexpr int Q4_K_CHUNK = 256;
constexpr int GROUP_SIZE = 32;
constexpr int RECORD_DWORDS = 17;
constexpr int RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1;
constexpr int RECORD_PAYLOAD_BF16 = RECORD_PAYLOAD_DWORDS * 2;
constexpr int OUT_RECORD_BF16 = M_PER_TILE + 2;
constexpr int Q_PHASE = 0;
constexpr int K_PHASE = 1;
constexpr int V_PHASE = 2;
constexpr int O_PHASE = 3;
constexpr int Q_BLOCKS = 8;
constexpr int KV_BLOCKS = 2;
constexpr int LOCAL_Q_VALUES = Q_BLOCKS * M_PER_TILE;
constexpr int LOCAL_KV_VALUES = KV_BLOCKS * M_PER_TILE;
constexpr int CONTEXT_LEN = 31;
constexpr int TOKENS_PER_TILE = 16;

static float accum[M_PER_TILE];
static float local_q[LOCAL_Q_VALUES];
static float local_k[LOCAL_KV_VALUES];
static float local_v[LOCAL_KV_VALUES];
static bfloat16 attention_cache[ACT_SLICE_BF16 * 16];
static uint8_t attention_cache_valid[ACT_SLICE_BF16 * 16];

static inline int32_t record_header(int32_t tag, int32_t group, int32_t row) {
    return 0x54000000 | (tag << 16) | (group << 12) | (row << 4) | 0xA;
}

static inline float summary_scale(int32_t phase, int32_t block) {
    return 0.0078125f * static_cast<float>((phase + 1) * ((block & 3) + 1));
}

static inline int32_t record_tag(int32_t header) {
    return (header >> 16) & 0xFF;
}

static inline bfloat16 *record_payload(int32_t *record) {
    return reinterpret_cast<bfloat16 *>(record + 1);
}

static inline void clear_attention_cache() {
    for (int idx = 0; idx < ACT_SLICE_BF16 * 16; idx++) {
        attention_cache_valid[idx] = 0;
    }
}

static inline bfloat16 payload_value(bfloat16 value, int32_t phase, int32_t block) {
    float scaled = static_cast<float>(value) * 0.00390625f;
    scaled += static_cast<float>((phase + 1) * ((block & 3) + 1)) * 0.0009765625f;
    if (scaled > 2.0f) {
        scaled = 2.0f;
    }
    if (scaled < -2.0f) {
        scaled = -2.0f;
    }
    return static_cast<bfloat16>(scaled);
}

static inline void write_payload_from_phase(
    bfloat16 *phase_out,
    int32_t *record,
    int32_t phase,
    int32_t block
) {
    bfloat16 *payload = record_payload(record);
    for (int idx = 0; idx < RECORD_PAYLOAD_BF16; idx++) {
        payload[idx] = payload_value(phase_out[idx], phase, block);
    }
}

static inline void add_to_summary(
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t num_rows
) {
    float scale = summary_scale(phase, block);
    for (int idx = 0; idx < num_rows; idx++) {
        float next = static_cast<float>(summary[idx]) + static_cast<float>(phase_out[idx]) * scale;
        summary[idx] = static_cast<bfloat16>(next);
    }
}

static inline bfloat16 record_value(
    int32_t *record,
    int32_t phase,
    int32_t block,
    int32_t chunk_idx,
    int32_t group,
    int32_t row,
    int32_t global_idx
) {
    bfloat16 *payload = record_payload(record);
    int32_t lane = global_idx & (RECORD_PAYLOAD_BF16 - 1);
    int32_t other = (lane + phase + block + 5) & (RECORD_PAYLOAD_BF16 - 1);
    float a = static_cast<float>(payload[lane]);
    float b = static_cast<float>(payload[other]);
    int32_t raw = (
        global_idx * 7 + chunk_idx * 11 + block * 13 +
        group * 17 + row * 19 + phase * 23
    ) % 127;
    float mix = a * (0.5f + 0.03125f * static_cast<float>(phase + 1));
    mix += b * (0.25f + 0.015625f * static_cast<float>((block & 7) + 1));
    mix += static_cast<float>(raw - 63) * 0.00390625f;
    return static_cast<bfloat16>(mix);
}

static inline float max2(float a, float b) {
    return a > b ? a : b;
}

static inline float approx_exp(float x) {
    if (x <= -16.0f) {
        return 0.0f;
    }
    if (x > 0.0f) {
        x = 0.0f;
    }
    float y = 1.0f + x * 0.015625f;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    return y;
}

static inline float history_k_value(int32_t group, int32_t row, int32_t token, int32_t dim) {
    int32_t raw = (group * 13 + row * 5 + token * 7 + dim * 3) % 37;
    return static_cast<float>(raw - 18) * 0.00875f;
}

static inline float history_v_value(int32_t group, int32_t row, int32_t token, int32_t dim) {
    int32_t raw = (group * 11 + row * 3 + token * 5 + dim * 2) % 31;
    return static_cast<float>(raw - 15) * 0.0125f;
}

static inline void remember_projection_payload(int32_t *record) {
    int32_t tag = record_tag(record[0]);
    if (tag <= 0) {
        clear_attention_cache();
        return;
    }

    int32_t produced_phase = (tag - 1) / 32;
    int32_t produced_block = (tag - 1) - produced_phase * 32;
    bfloat16 *payload = record_payload(record);

    if (produced_phase == Q_PHASE && produced_block < Q_BLOCKS) {
        int base = produced_block * M_PER_TILE;
        for (int idx = 0; idx < M_PER_TILE; idx++) {
            local_q[base + idx] = static_cast<float>(payload[idx]);
        }
    } else if (produced_phase == K_PHASE && produced_block < KV_BLOCKS) {
        int base = produced_block * M_PER_TILE;
        for (int idx = 0; idx < M_PER_TILE; idx++) {
            local_k[base + idx] = static_cast<float>(payload[idx]);
        }
    } else if (produced_phase == V_PHASE && produced_block < KV_BLOCKS) {
        int base = produced_block * M_PER_TILE;
        for (int idx = 0; idx < M_PER_TILE; idx++) {
            local_v[base + idx] = static_cast<float>(payload[idx]);
        }
    }
}

static inline bfloat16 attention_output_value(
    int32_t global_idx,
    int32_t group,
    int32_t row
) {
    if (attention_cache_valid[global_idx] != 0) {
        return attention_cache[global_idx];
    }

    int32_t q_idx = global_idx & (LOCAL_Q_VALUES - 1);
    int32_t kv_idx = (global_idx + row * 7 + group * 11) & (LOCAL_KV_VALUES - 1);
    float query = local_q[q_idx];

    float running_max = -3.4028234663852886e38f;
    float running_sum = 0.0f;
    float out = 0.0f;
    for (int tile = 0; tile < 2; tile++) {
        int valid = tile == 0 ? TOKENS_PER_TILE : CONTEXT_LEN - TOKENS_PER_TILE;
        float local_max = -3.4028234663852886e38f;
        for (int token = 0; token < valid; token++) {
            int global_token = tile * TOKENS_PER_TILE + token;
            float key = global_token == CONTEXT_LEN - 1
                ? local_k[kv_idx]
                : history_k_value(group, row, global_token, kv_idx);
            float score = query * key * 0.08838834764831845f;
            score += static_cast<float>((global_idx + global_token * 3) & 7) * 0.0009765625f;
            local_max = max2(local_max, score);
        }

        float new_max = max2(running_max, local_max);
        float old_scale = running_sum == 0.0f ? 0.0f : approx_exp(running_max - new_max);
        out *= old_scale;

        float local_sum = 0.0f;
        for (int token = 0; token < valid; token++) {
            int global_token = tile * TOKENS_PER_TILE + token;
            float key = global_token == CONTEXT_LEN - 1
                ? local_k[kv_idx]
                : history_k_value(group, row, global_token, kv_idx);
            float value = global_token == CONTEXT_LEN - 1
                ? local_v[kv_idx]
                : history_v_value(group, row, global_token, kv_idx);
            float score = query * key * 0.08838834764831845f;
            score += static_cast<float>((global_idx + global_token * 3) & 7) * 0.0009765625f;
            float weight = approx_exp(score - new_max);
            local_sum += weight;
            out += weight * value;
        }

        running_max = new_max;
        running_sum = running_sum * old_scale + local_sum;
    }

    float result = running_sum == 0.0f ? 0.0f : out / running_sum;
    attention_cache[global_idx] = static_cast<bfloat16>(result);
    attention_cache_valid[global_idx] = 1;
    return attention_cache[global_idx];
}

} // namespace

extern "C" {

void clear_summary(bfloat16 *summary, int32_t num_rows) {
    for (int idx = 0; idx < num_rows; idx++) {
        summary[idx] = static_cast<bfloat16>(0.0f);
        accum[idx] = 0.0f;
    }
}

void emit_seed_sideband(int32_t *record, int32_t group, int32_t row) {
    record[0] = record_header(0, group, row);
    bfloat16 *payload = record_payload(record);
    for (int lane = 0; lane < RECORD_PAYLOAD_BF16; lane++) {
        int raw = (group * 37 + row * 11 + lane * 3 + 1) % 127;
        payload[lane] = static_cast<bfloat16>(static_cast<float>(raw - 63) / 64.0f);
    }
}

void emit_next_block_sideband(
    int32_t *record,
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    add_to_summary(phase_out, summary, phase, block, num_rows);

    int32_t tag = phase * 32 + block + 1;
    record[0] = record_header(tag, group, row);
    write_payload_from_phase(phase_out, record, phase, block);
}

void accumulate_block_summary(
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t num_rows
) {
    add_to_summary(phase_out, summary, phase, block, num_rows);
}

void edge_make_block_slice(
    int32_t *record,
    bfloat16 *activation_slice,
    int32_t phase,
    int32_t block,
    int32_t chunk_idx,
    int32_t group,
    int32_t row
) {
    remember_projection_payload(record);
    int32_t base = chunk_idx * ACT_SLICE_BF16;
    for (int idx = 0; idx < ACT_SLICE_BF16; idx++) {
        if (phase == O_PHASE) {
            activation_slice[idx] = attention_output_value(base + idx, group, row);
        } else {
            activation_slice[idx] = record_value(record, phase, block, chunk_idx, group, row, base + idx);
        }
    }
}

void q4nx_chunk_accum_slice(
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice,
    int32_t num_rows
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = M_PER_TILE * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);

    for (int row = 0; row < num_rows; row++) {
        float row_acc = 0.0f;

        for (int group = 0; group < groups_per_row; group++) {
            bfloat16 scale = scales[row * groups_per_row + group];
            bfloat16 zero = zeros[row * groups_per_row + group];
            uint4 *group_ptr = data + row * (Q4_K_CHUNK / 2) + group * (GROUP_SIZE / 2);

            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);
            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> zero_vec = aie::broadcast<bfloat16, GROUP_SIZE>(zero);
            aie::vector<bfloat16, GROUP_SIZE> scale_vec = aie::broadcast<bfloat16, GROUP_SIZE>(scale);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, zero_vec);
            aie::accum<accfloat, GROUP_SIZE> dequant_acc = aie::mul(shifted, scale_vec);
            aie::vector<bfloat16, GROUP_SIZE> dequant = dequant_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(activation_slice + group * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dequant, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();

            row_acc += aie::reduce_add(mac_f32);
        }

        accum[row] += row_acc;
    }
}

void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

void flush_schedule_output_with_header(
    bfloat16 *output,
    bfloat16 *summary,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    output[0] = static_cast<bfloat16>(static_cast<float>(group));
    output[1] = static_cast<bfloat16>(static_cast<float>(row));
    for (int idx = 0; idx < num_rows; idx++) {
        output[2 + idx] = summary[idx];
    }
    for (int idx = 2 + num_rows; idx < OUT_RECORD_BF16; idx++) {
        output[idx] = static_cast<bfloat16>(0.0f);
    }
}

} // extern "C"
