#include <stdint.h>

extern "C" {

void copy_token(float *src, float *dst) {
    for (int i = 0; i < 128; i++) {
        dst[i] = src[i];
    }
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
    // exp(x) ~= (1 + x / 64) ^ 64. Softmax only calls this with x <= 0.
    float y = 1.0f + x * 0.015625f;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    return y;
}

static inline float approx_sigmoid(float x) {
    if (x >= 0.0f) {
        float e = approx_exp(-x);
        return 1.0f / (1.0f + e);
    }
    float e = approx_exp(x);
    return e / (1.0f + e);
}

void derive_query(float *current_k, float *current_v, float *query) {
    const int NUM_HEADS = 4;
    const int HEAD_DIM = 32;
    for (int h = 0; h < NUM_HEADS; h++) {
        int base = h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            float k = current_k[base + d];
            float v = current_v[base + ((d + 3) & 31)];
            float neighbor = current_k[base + ((d + 1) & 31)];
            float lane = static_cast<float>((d % 7) - 3);
            query[base + d] = k * 0.625f + v * 0.1875f - neighbor * 0.0625f
                + lane * 0.00390625f * static_cast<float>(h + 1);
        }
    }
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
    float *query,
    float *k_buf,
    float *v_buf,
    float *out_buf,
    float *running_max,
    float *running_sum,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid
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
                score += query[h * HEAD_DIM + d] * k_buf[t * STRIDE + h * HEAD_DIM + d];
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

void layer_epilogue(float *query, float *out_buf, float *tmp_buf) {
    const int NUM_HEADS = 4;
    const int HEAD_DIM = 32;
    for (int h = 0; h < NUM_HEADS; h++) {
        int base = h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            float attn = out_buf[base + d];
            float next = out_buf[base + ((d + 1) & 31)];
            float far = out_buf[base + ((d + 7) & 31)];
            float residual = query[base + d] * 0.25f;
            float mixed = attn * 0.75f + next * 0.125f - far * 0.0625f + residual;
            float gate = mixed * approx_sigmoid(mixed * 0.5f);
            tmp_buf[base + d] = mixed + gate * 0.1f;
        }
    }
    for (int i = 0; i < 128; i++) {
        out_buf[i] = tmp_buf[i];
    }
}

}
