#include <stdint.h>

extern "C" {

void copy_token(float *src, float *dst) {
    for (int i = 0; i < 128; i++) {
        dst[i] = src[i];
    }
}

static inline float query_value(int head_offset, int local_head, int dim) {
    int lane = (dim % 7) - 3;
    return 0.02f * static_cast<float>(head_offset + local_head + 1) * static_cast<float>(lane);
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
    // exp(x) ~= (1 + x / 64) ^ 64.  Softmax only calls this with x <= 0.
    float y = 1.0f + x * 0.015625f;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    return y;
}

void init_attention_state(float *out_buf, float *running_max, float *running_sum) {
    for (int i = 0; i < 128; i++) {
        out_buf[i] = 0.0f;
    }
    for (int h = 0; h < 4; h++) {
        running_max[h] = -3.4028234663852886e38f;
        running_sum[h] = 0.0f;
    }
}

void online_softmax_attention(
    float *k_buf,
    float *v_buf,
    float *out_buf,
    float *running_max,
    float *running_sum,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid,
    int32_t head_offset
) {
    const int NUM_HEADS = 4;
    const int HEAD_DIM = 32;
    const int TOKENS_PER_TILE = 16;
    const int STRIDE = NUM_HEADS * HEAD_DIM;
    const float SCALE = 0.1767766952966369f; // 1 / sqrt(32)

    int valid = (tile_idx < num_tiles - 1) ? TOKENS_PER_TILE : last_valid;
    float scores[TOKENS_PER_TILE];
    float weights[TOKENS_PER_TILE];

    for (int h = 0; h < NUM_HEADS; h++) {
        float local_max = -3.4028234663852886e38f;
        for (int t = 0; t < valid; t++) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += query_value(head_offset, h, d) * k_buf[t * STRIDE + h * HEAD_DIM + d];
            }
            score *= SCALE;
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
                acc += weights[t] * v_buf[t * STRIDE + h * HEAD_DIM + d];
            }
            out_buf[out_base + d] = acc;
        }

        running_max[h] = new_max;
        running_sum[h] = new_sum;
    }
}

void finalize_attention(float *out_buf, float *running_sum) {
    const int NUM_HEADS = 4;
    const int HEAD_DIM = 32;
    for (int h = 0; h < NUM_HEADS; h++) {
        float denom = running_sum[h];
        int out_base = h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            out_buf[out_base + d] = denom == 0.0f ? 0.0f : out_buf[out_base + d] / denom;
        }
    }
}

}
