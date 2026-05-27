#include <stdint.h>

extern "C" {

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

void copy_current_512(float *src, float *dst) {
    for (int i = 0; i < 512; i++) {
        dst[i] = src[i];
    }
}

void shape_a_softmax_sideband(float *current, float *k_history, float *sideband) {
    const int HEAD_DIM = 128;
    const int TOKENS = 16;
    const float SCALE = 0.08838834764831845f; // 1 / sqrt(128)

    int valid = static_cast<int>(current[511]);
    if (valid < 1) {
        valid = 1;
    }
    if (valid > TOKENS) {
        valid = TOKENS;
    }

    float scores[TOKENS];
    float max_score = -3.4028234663852886e38f;
    for (int t = 0; t < valid; t++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += current[d] * k_history[t * HEAD_DIM + d];
        }
        score *= SCALE;
        scores[t] = score;
        max_score = max2(max_score, score);
    }

    float denom = 0.0f;
    for (int t = 0; t < TOKENS; t++) {
        float weight = 0.0f;
        if (t < valid) {
            weight = approx_exp(scores[t] - max_score);
            denom += weight;
        }
        sideband[t] = weight;
    }
    sideband[16] = denom;
}

void sideband_debug_and_forward(float *sideband, float *debug, float *forward) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        float value = sideband[i];
        sum += value;
        forward[i] = value;
    }
    forward[16] = sideband[16];
    debug[0] = sum;
    debug[1] = sideband[0];
    debug[2] = sideband[15];
    debug[3] = sideband[16];
}

void shape_b_weighted_value(float *sideband, float *v_history, float *output) {
    const int HEAD_DIM = 128;
    const int TOKENS = 16;
    float denom = sideband[16];

    for (int i = 0; i < 512; i++) {
        output[i] = 0.0f;
    }

    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0f;
        for (int t = 0; t < TOKENS; t++) {
            acc += sideband[t] * v_history[t * HEAD_DIM + d];
        }
        output[d] = denom == 0.0f ? 0.0f : acc / denom;
    }
    output[128] = denom;
    output[129] = sideband[0];
    output[130] = sideband[15];
    output[131] = v_history[0];
}

}
