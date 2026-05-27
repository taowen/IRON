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

void shape_a_tile_sideband(
    float *current,
    float *k_history,
    float *sideband,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_valid
) {
    const int HEAD_DIM = 128;
    const int TOKENS = 16;
    const float SCALE = 0.08838834764831845f; // 1 / sqrt(128)

    int valid = tile_idx < num_tiles - 1 ? TOKENS : last_valid;
    if (valid < 1) {
        valid = 1;
    }
    if (valid > TOKENS) {
        valid = TOKENS;
    }

    float scores[TOKENS];
    float local_max = -3.4028234663852886e38f;
    for (int t = 0; t < valid; t++) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += current[d] * k_history[t * HEAD_DIM + d];
        }
        score *= SCALE;
        scores[t] = score;
        local_max = max2(local_max, score);
    }

    for (int t = 0; t < TOKENS; t++) {
        sideband[t] = t < valid ? approx_exp(scores[t] - local_max) : 0.0f;
    }
    sideband[16] = local_max;
}

void sideband_debug_init(float *debug) {
    for (int i = 0; i < 4; i++) {
        debug[i] = 0.0f;
    }
}

void sideband_debug_and_forward_tile(
    float *sideband,
    float *debug,
    float *forward,
    int32_t tile_idx
) {
    float local_sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        float value = sideband[i];
        local_sum += value;
        forward[i] = value;
    }
    forward[16] = sideband[16];
    debug[0] += local_sum;
    debug[1] = sideband[0];
    debug[2] = sideband[15];
    debug[3] = static_cast<float>(tile_idx);
}

void shape_b_init(float *output) {
    for (int i = 0; i < 512; i++) {
        output[i] = 0.0f;
    }
    output[128] = 0.0f;                       // running_sum
    output[129] = -3.4028234663852886e38f;    // running_max
    output[130] = 0.0f;                       // last local_max
    output[131] = 0.0f;                       // last local_sum
}

void shape_b_accumulate_tile(float *sideband, float *v_history, float *output) {
    const int HEAD_DIM = 128;
    const int TOKENS = 16;

    float local_sum = 0.0f;
    for (int t = 0; t < TOKENS; t++) {
        local_sum += sideband[t];
    }

    float old_sum = output[128];
    float old_max = output[129];
    float local_max = sideband[16];
    float new_max = max2(old_max, local_max);
    float old_scale = old_sum == 0.0f ? 0.0f : approx_exp(old_max - new_max);
    float tile_scale = local_sum == 0.0f ? 0.0f : approx_exp(local_max - new_max);

    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = output[d] * old_scale;
        float tile_acc = 0.0f;
        for (int t = 0; t < TOKENS; t++) {
            tile_acc += sideband[t] * v_history[t * HEAD_DIM + d];
        }
        output[d] = acc + tile_scale * tile_acc;
    }

    output[128] = old_sum * old_scale + tile_scale * local_sum;
    output[129] = new_max;
    output[130] = local_max;
    output[131] = local_sum;
}

void shape_b_finalize(float *output) {
    float denom = output[128];
    for (int d = 0; d < 128; d++) {
        output[d] = denom == 0.0f ? 0.0f : output[d] / denom;
    }
}

}
