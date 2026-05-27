#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int NUM_HEADS = 1;
constexpr int HEAD_DIM = 32;
constexpr int TOKEN_DWORDS = NUM_HEADS * HEAD_DIM;
constexpr int TOKENS_PER_TILE = 16;
constexpr int PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE;
constexpr int Q4_ROWS = 32;
constexpr int HIDDEN_DIM = 4096;
constexpr int Q4_K_CHUNK = 256;
constexpr int Q4_CHUNKS = HIDDEN_DIM / Q4_K_CHUNK;
constexpr int GROUP_SIZE = 32;
constexpr int CHUNK_BF16 = 2560;

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

} // namespace

extern "C" {

void zero_token(float *dst) {
    for (int i = 0; i < TOKEN_DWORDS; i++) {
        dst[i] = 0.0f;
    }
}

void q4nx_project_chunk_accum(
    bfloat16 *packed_chunk,
    bfloat16 *activation,
    float *output,
    int32_t act_offset
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = Q4_ROWS * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);
    bfloat16 *act_slice = activation + act_offset;

    for (int row = 0; row < Q4_ROWS; row++) {
        float row_acc = 0.0f;
        for (int g = 0; g < groups_per_row; g++) {
            bfloat16 s = scales[row * groups_per_row + g];
            bfloat16 z = zeros[row * groups_per_row + g];

            uint4 *group_ptr = data + row * (Q4_K_CHUNK / 2) + g * (GROUP_SIZE / 2);
            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);
            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> z_vec = aie::broadcast<bfloat16, GROUP_SIZE>(z);
            aie::vector<bfloat16, GROUP_SIZE> s_vec = aie::broadcast<bfloat16, GROUP_SIZE>(s);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, z_vec);
            aie::accum<accfloat, GROUP_SIZE> dq_acc = aie::mul(shifted, s_vec);
            aie::vector<bfloat16, GROUP_SIZE> dq = dq_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(act_slice + g * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dq, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();
            row_acc += aie::reduce_add(mac_f32);
        }

        output[row] += row_acc;
    }
}

void copy_token(float *src, float *dst) {
    for (int i = 0; i < TOKEN_DWORDS; i++) {
        dst[i] = src[i];
    }
}

void init_attention_state(float *out_buf, float *running_max, float *running_sum) {
    for (int i = 0; i < TOKEN_DWORDS; i++) {
        out_buf[i] = 0.0f;
    }
    for (int h = 0; h < NUM_HEADS; h++) {
        running_max[h] = -3.4028234663852886e38f;
        running_sum[h] = 0.0f;
    }
}

void online_softmax_attention_pair(
    float *query,
    float *kv_buf,
    float *out_buf,
    float *running_max,
    float *running_sum,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid
) {
    const float scale = 0.1767766952966369f;
    int valid = (tile_idx < num_tiles - 1) ? TOKENS_PER_TILE : last_valid;
    float scores[TOKENS_PER_TILE];
    float weights[TOKENS_PER_TILE];

    for (int h = 0; h < NUM_HEADS; h++) {
        float local_max = -3.4028234663852886e38f;
        for (int t = 0; t < valid; t++) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += query[h * HEAD_DIM + d] * kv_buf[t * TOKEN_DWORDS + h * HEAD_DIM + d];
            }
            score *= scale;
            scores[t] = score;
            local_max = max2(local_max, score);
        }

        float old_max = running_max[h];
        float new_max = max2(old_max, local_max);
        float old_scale = running_sum[h] == 0.0f ? 0.0f : approx_exp(old_max - new_max);

        float local_sum = 0.0f;
        for (int t = 0; t < valid; t++) {
            float weight = approx_exp(scores[t] - new_max);
            weights[t] = weight;
            local_sum += weight;
        }

        float new_sum = running_sum[h] * old_scale + local_sum;
        int out_base = h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = out_buf[out_base + d] * old_scale;
            for (int t = 0; t < valid; t++) {
                acc += weights[t] * kv_buf[PLANE_TILE_DWORDS + t * TOKEN_DWORDS + h * HEAD_DIM + d];
            }
            out_buf[out_base + d] = acc;
        }

        running_max[h] = new_max;
        running_sum[h] = new_sum;
    }
}

void finalize_attention(float *out_buf, float *running_sum) {
    for (int h = 0; h < NUM_HEADS; h++) {
        float denom = running_sum[h];
        int out_base = h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            out_buf[out_base + d] = denom == 0.0f ? 0.0f : out_buf[out_base + d] / denom;
        }
    }
}

} // extern "C"
