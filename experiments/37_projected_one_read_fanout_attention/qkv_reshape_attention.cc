#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int TOKENS_PER_TILE = 16;
constexpr int DIM_GROUPS = 32;
constexpr int GROUP_DWORDS = 4;
constexpr int Q4_ROWS = 32;
constexpr int HIDDEN_DIM = 4096;
constexpr int Q4_K_CHUNK = 256;
constexpr int GROUP_SIZE = 32;

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

static inline int reshaped_index(int dim, int token) {
    int dim_group = dim / GROUP_DWORDS;
    int lane = dim - dim_group * GROUP_DWORDS;
    return (dim_group * TOKENS_PER_TILE + token) * GROUP_DWORDS + lane;
}

} // namespace

extern "C" {

void zero_head(float *dst) {
    for (int i = 0; i < HEAD_DIM; i++) {
        dst[i] = 0.0f;
    }
}

void q4nx_project_head_chunk(
    bfloat16 *packed_chunk,
    bfloat16 *activation,
    float *output,
    int32_t output_offset,
    int32_t act_offset
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = Q4_ROWS * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);
    bfloat16 *act_slice = activation + act_offset;

    for (int row = 0; row < Q4_ROWS; row++) {
        int out_row = output_offset + row;
        if (out_row >= HEAD_DIM) {
            continue;
        }

        float row_acc = 0.0f;
        for (int group = 0; group < groups_per_row; group++) {
            bfloat16 s = scales[row * groups_per_row + group];
            bfloat16 z = zeros[row * groups_per_row + group];

            uint4 *group_ptr = data + row * (Q4_K_CHUNK / 2) + group * (GROUP_SIZE / 2);
            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);
            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> z_vec = aie::broadcast<bfloat16, GROUP_SIZE>(z);
            aie::vector<bfloat16, GROUP_SIZE> s_vec = aie::broadcast<bfloat16, GROUP_SIZE>(s);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, z_vec);
            aie::accum<accfloat, GROUP_SIZE> dq_acc = aie::mul(shifted, s_vec);
            aie::vector<bfloat16, GROUP_SIZE> dq = dq_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(act_slice + group * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dq, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();
            row_acc += aie::reduce_add(mac_f32);
        }

        output[out_row] += row_acc;
    }
}

void copy_head(float *src, float *dst) {
    for (int i = 0; i < HEAD_DIM; i++) {
        dst[i] = src[i];
    }
}

void init_attention_state_head(float *out_buf, float *running_max, float *running_sum) {
    for (int i = 0; i < HEAD_DIM; i++) {
        out_buf[i] = 0.0f;
    }
    running_max[0] = -3.4028234663852886e38f;
    running_sum[0] = 0.0f;
}

void online_attention_reshaped_head(
    float *query,
    float *k_tile,
    float *v_tile,
    float *out_buf,
    float *running_max,
    float *running_sum,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid
) {
    const float scale = 0.08838834764831845f;
    int valid = (tile_idx < num_tiles - 1) ? TOKENS_PER_TILE : last_valid;

    float local_max = -3.4028234663852886e38f;
    for (int t = 0; t < valid; t++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += query[d] * k_tile[reshaped_index(d, t)];
        }
        score *= scale;
        local_max = max2(local_max, score);
    }

    float old_max = running_max[0];
    float new_max = max2(old_max, local_max);
    float old_scale = running_sum[0] == 0.0f ? 0.0f : approx_exp(old_max - new_max);

    for (int d = 0; d < HEAD_DIM; d++) {
        out_buf[d] *= old_scale;
    }

    float local_sum = 0.0f;
    for (int t = 0; t < valid; t++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += query[d] * k_tile[reshaped_index(d, t)];
        }
        score *= scale;

        float weight = approx_exp(score - new_max);
        local_sum += weight;
        for (int d = 0; d < HEAD_DIM; d++) {
            out_buf[d] += weight * v_tile[reshaped_index(d, t)];
        }
    }

    running_max[0] = new_max;
    running_sum[0] = running_sum[0] * old_scale + local_sum;
}

void finalize_attention_head(float *out_buf, float *running_sum) {
    float denom = running_sum[0];
    for (int d = 0; d < HEAD_DIM; d++) {
        out_buf[d] = denom == 0.0f ? 0.0f : out_buf[d] / denom;
    }
}

} // extern "C"
