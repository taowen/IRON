#include <stdint.h>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int TOKENS_PER_TILE = 16;
constexpr int DIM_GROUPS = 32;
constexpr int GROUP_DWORDS = 4;

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

void copy_head_from_current(float *current_kv, float *dst, int32_t offset) {
    for (int i = 0; i < HEAD_DIM; i++) {
        dst[i] = current_kv[offset + i];
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
