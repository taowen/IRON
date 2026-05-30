#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static bfloat16 bf16_lane(int32_t *payload, int32_t lane) {
    bfloat16 *values = reinterpret_cast<bfloat16 *>(payload);
    return values[lane];
}

static float *as_f32(int32_t *payload) {
    return reinterpret_cast<float *>(payload);
}

static int32_t floor_i32(float value) {
    const int32_t truncated = static_cast<int32_t>(value);
    return static_cast<float>(truncated) > value ? truncated - 1 : truncated;
}

static float pow2_i32(int32_t exponent) {
    if (exponent < -126) {
        return 0.0f;
    }
    if (exponent > 127) {
        exponent = 127;
    }
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = static_cast<uint32_t>(exponent + 127) << 23;
    return bits.f;
}

static float fast_exp(float value) {
    if (value <= -20.0f) {
        return 0.0f;
    }
    if (value >= 0.0f) {
        return 1.0f;
    }
    constexpr float inv_ln2 = 1.4426950409f;
    constexpr float ln2 = 0.6931471806f;
    const int32_t exponent = floor_i32(value * inv_ln2);
    const float reduced = value - static_cast<float>(exponent) * ln2;
    const float r2 = reduced * reduced;
    const float r3 = r2 * reduced;
    const float r4 = r3 * reduced;
    const float polynomial = 1.0f + reduced + 0.5f * r2 + 0.16666667f * r3 + 0.04166667f * r4;
    return pow2_i32(exponent) * polynomial;
}

static float attention_score_bf16(
    int32_t *q_window,
    int32_t *k_window,
    int32_t q_head,
    int32_t token
) {
    constexpr int32_t head_dim = 128;
    constexpr int32_t context = 16;
    constexpr int32_t gqa_ratio = 4;
    constexpr float rsqrt_head_dim = 0.0883883476f;
    const int32_t kv_head = q_head / gqa_ratio;
    const int32_t q_base = q_head * head_dim;
    const int32_t k_base = kv_head * context * head_dim + token * head_dim;
    float dot = 0.0f;
    for (int32_t dim = 0; dim < head_dim; dim++) {
        dot += static_cast<float>(bf16_lane(q_window, q_base + dim))
            * static_cast<float>(bf16_lane(k_window, k_base + dim));
    }
    return dot * rsqrt_head_dim;
}

} // namespace

extern "C" {

void qwen3_attention_bf16_make_carrier_masked(
    int32_t *q_window,
    int32_t *k_window,
    int32_t *carrier,
    int32_t window,
    int32_t block,
    int32_t blocks,
    int32_t tail_tokens,
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

    const int32_t valid_tokens = block + 1 == blocks ? tail_tokens : context;
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + weight_dwords);
    for (int32_t q_head = 0; q_head < heads; q_head++) {
        float scores[context];
        float running_max = attention_score_bf16(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < valid_tokens; token++) {
            scores[token] = attention_score_bf16(q_window, k_window, q_head, token);
            if (scores[token] > running_max) {
                running_max = scores[token];
            }
        }

        float weight_sum = 0.0f;
        for (int32_t token = 0; token < context; token++) {
            float weight = 0.0f;
            if (token < valid_tokens) {
                weight = fast_exp(scores[token] - running_max);
                weight_sum += weight;
            }
            weights[q_head * context + token] = static_cast<bfloat16>(weight);
        }

        scalars[q_head * 2] = running_max;
        scalars[q_head * 2 + 1] = weight_sum;
    }

    (void)window;
    (void)q_dwords;
    (void)k_dwords;
}

void qwen3_attention_bf16_init_accum(
    int32_t *accum,
    int32_t *state,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    for (int32_t idx = 0; idx < accum_lanes; idx++) {
        accum_values[idx] = 0.0f;
    }
    for (int32_t idx = 0; idx < state_dwords; idx++) {
        state_values[idx] = 0.0f;
    }
}

void qwen3_attention_bf16_accum_block(
    int32_t *v_window,
    int32_t *carrier,
    int32_t *accum,
    int32_t *state,
    int32_t block,
    int32_t v_dwords,
    int32_t carrier_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t context = 16;
    constexpr int32_t head_dim = 128;
    constexpr int32_t gqa_ratio = 4;
    constexpr int32_t weight_dwords = 64;
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + weight_dwords);
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);

    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const float block_max = scalars[q_head * 2];
        const float block_sum = scalars[q_head * 2 + 1];
        if (block_sum == 0.0f) {
            continue;
        }
        const float old_max = state_values[q_head * 2];
        const float old_sum = state_values[q_head * 2 + 1];
        float new_max = block_max;
        float old_scale = 0.0f;
        float block_scale = 1.0f;
        if (old_sum != 0.0f) {
            new_max = old_max > block_max ? old_max : block_max;
            old_scale = fast_exp(old_max - new_max);
            block_scale = fast_exp(block_max - new_max);
        }

        const int32_t kv_head = q_head / gqa_ratio;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            float block_total = 0.0f;
            for (int32_t token = 0; token < context; token++) {
                const float weight = static_cast<float>(weights[q_head * context + token]);
                const int32_t v_lane = kv_head * context * head_dim + token * head_dim + dim;
                block_total += weight * static_cast<float>(bf16_lane(v_window, v_lane));
            }
            const int32_t lane = q_head * head_dim + dim;
            accum_values[lane] = accum_values[lane] * old_scale + block_total * block_scale;
        }

        state_values[q_head * 2] = new_max;
        state_values[q_head * 2 + 1] = old_sum * old_scale + block_sum * block_scale;
    }

    (void)block;
    (void)v_dwords;
    (void)carrier_dwords;
    (void)accum_lanes;
    (void)state_dwords;
}

void qwen3_attention_bf16_finish_accum(
    int32_t *accum,
    int32_t *state,
    int32_t *output,
    int32_t output_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t head_dim = 128;
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(output);
    for (int32_t idx = 0; idx < output_dwords * 2; idx++) {
        payload[idx] = static_cast<bfloat16>(0.0f);
    }
    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const float weight_sum = state_values[q_head * 2 + 1];
        for (int32_t dim = 0; dim < head_dim; dim++) {
            const int32_t lane = q_head * head_dim + dim;
            payload[lane] = static_cast<bfloat16>(accum_values[lane] / weight_sum);
        }
    }

    (void)accum_lanes;
    (void)state_dwords;
}

} // extern "C"
