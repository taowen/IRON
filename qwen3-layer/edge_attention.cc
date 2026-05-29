#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static int32_t unpack_s16(int32_t *payload, int32_t lane) {
    const uint32_t word = static_cast<uint32_t>(payload[lane >> 1]);
    const uint32_t raw = (lane & 1) != 0 ? (word >> 16) & 0xffff : word & 0xffff;
    return (raw & 0x8000) != 0 ? static_cast<int32_t>(raw) - 0x10000 : static_cast<int32_t>(raw);
}

static int32_t clamp_s16(int32_t value) {
    if (value < -32768) {
        return -32768;
    }
    if (value > 32767) {
        return 32767;
    }
    return value;
}

static int32_t pack_s16_pair(int32_t low, int32_t high) {
    const uint32_t low_u16 = static_cast<uint32_t>(clamp_s16(low)) & 0xffff;
    const uint32_t high_u16 = static_cast<uint32_t>(clamp_s16(high)) & 0xffff;
    return static_cast<int32_t>(low_u16 | (high_u16 << 16));
}

static int32_t div_i32(int32_t numerator, int32_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -((-numerator) / denominator);
}

static constexpr int32_t softmax_scale = 4096;
static constexpr int32_t exp_delta_q12[] = {
    4096, 3615, 3190, 2815, 2484, 2192, 1935, 1707,
    1507, 1330, 1174, 1036, 914, 807, 712, 628,
    554, 489, 432, 381, 336, 297, 262, 231,
    204, 180, 159, 140, 124, 109, 96, 85,
    75, 66, 58, 52, 46, 40, 35, 31,
    28, 24, 21, 19, 17, 15, 13, 12,
    10, 9, 8, 7, 6, 5, 5, 4,
    4, 3, 3, 3, 2, 2, 2, 2,
    1,
};
static constexpr int32_t exp_delta_q12_len = sizeof(exp_delta_q12) / sizeof(exp_delta_q12[0]);

static int32_t attention_weight(int32_t delta) {
    if (delta <= 0) {
        return softmax_scale;
    }
    if (delta < exp_delta_q12_len) {
        return exp_delta_q12[delta];
    }
    return 1;
}

static int32_t attention_score(
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
        dot += unpack_s16(q_window, q_base + dim) * unpack_s16(k_window, k_base + dim);
    }
    return div_i32(dot, head_dim);
}

static int32_t unpack_weight(int32_t *carrier, int32_t q_head, int32_t token) {
    constexpr int32_t context = 16;
    const int32_t lane = q_head * context + token;
    const uint32_t word = static_cast<uint32_t>(carrier[lane >> 1]);
    return (lane & 1) != 0
        ? static_cast<int32_t>((word >> 16) & 0xffff)
        : static_cast<int32_t>(word & 0xffff);
}

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
        int32_t running_max = attention_score(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < context; token++) {
            scores[token] = attention_score(q_window, k_window, q_head, token);
            if (scores[token] > running_max) {
                running_max = scores[token];
            }
        }

        int32_t weight_sum = 0;
        for (int32_t token = 0; token < context; token += 2) {
            const int32_t low = attention_weight(running_max - scores[token]);
            const int32_t high = attention_weight(running_max - scores[token + 1]);
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

void attention_kv16_make_carrier_masked(
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
    for (int32_t q_head = 0; q_head < heads; q_head++) {
        int32_t scores[context];
        int32_t running_max = attention_score(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < valid_tokens; token++) {
            scores[token] = attention_score(q_window, k_window, q_head, token);
            if (scores[token] > running_max) {
                running_max = scores[token];
            }
        }

        int32_t weight_sum = 0;
        for (int32_t token = 0; token < context; token += 2) {
            int32_t low = 0;
            int32_t high = 0;
            if (token < valid_tokens) {
                low = attention_weight(running_max - scores[token]);
                weight_sum += low;
            }
            if (token + 1 < valid_tokens) {
                high = attention_weight(running_max - scores[token + 1]);
                weight_sum += high;
            }
            carrier[(q_head * context + token) >> 1] =
                static_cast<int32_t>((low & 0xffff) | ((high & 0xffff) << 16));
        }

        const int32_t scalar = weight_dwords + q_head * 2;
        carrier[scalar] = running_max;
        carrier[scalar + 1] = weight_sum;
    }

    (void)window;
    (void)q_dwords;
    (void)k_dwords;
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
                const int32_t weight = unpack_weight(carrier, q_head, token);
                const int32_t v_base = kv_head * context * head_dim + token * head_dim + dim;
                low_total += weight * unpack_s16(v_window, v_base);
                high_total += weight * unpack_s16(v_window, v_base + 1);
            }
            const int32_t low = div_i32(low_total, weight_sum);
            const int32_t high = div_i32(high_total, weight_sum);
            output[(q_head * head_dim + dim) >> 1] = pack_s16_pair(low, high);
        }
    }

    (void)window;
    (void)v_dwords;
    (void)carrier_dwords;
}

void attention_kv16_init_accum(
    int32_t *accum,
    int32_t *state,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    for (int32_t idx = 0; idx < accum_lanes; idx++) {
        accum[idx] = 0;
    }
    for (int32_t idx = 0; idx < state_dwords; idx++) {
        state[idx] = 0;
    }
}

void attention_kv16_accum_block(
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

    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const int32_t scalar = weight_dwords + q_head * 2;
        const int32_t state_offset = q_head * 2;
        const int32_t block_max = carrier[scalar];
        const int32_t block_sum = carrier[scalar + 1];
        const int32_t old_max = state[state_offset];
        const int32_t old_sum = state[state_offset + 1];
        int32_t new_max = block_max;
        int32_t old_scale = 0;
        int32_t new_scale = softmax_scale;

        if (old_sum != 0) {
            new_max = old_max > block_max ? old_max : block_max;
            old_scale = attention_weight(new_max - old_max);
            new_scale = attention_weight(new_max - block_max);
        }

        const int32_t kv_head = q_head / gqa_ratio;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            int32_t block_total = 0;
            for (int32_t token = 0; token < context; token++) {
                const int32_t weight = unpack_weight(carrier, q_head, token);
                const int32_t v_lane = kv_head * context * head_dim + token * head_dim + dim;
                block_total += weight * unpack_s16(v_window, v_lane);
            }
            const int32_t lane = q_head * head_dim + dim;
            const int32_t merged = accum[lane] * old_scale + block_total * new_scale;
            accum[lane] = div_i32(merged, softmax_scale);
        }

        const int32_t sum_merged = old_sum * old_scale + block_sum * new_scale;
        state[state_offset] = new_max;
        state[state_offset + 1] = div_i32(sum_merged, softmax_scale);
    }

    (void)block;
    (void)v_dwords;
    (void)carrier_dwords;
    (void)accum_lanes;
    (void)state_dwords;
}

void attention_kv16_finish_accum(
    int32_t *accum,
    int32_t *state,
    int32_t *output,
    int32_t output_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t head_dim = 128;
    for (int32_t idx = 0; idx < output_dwords; idx++) {
        output[idx] = 0;
    }
    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const int32_t weight_sum = state[q_head * 2 + 1];
        for (int32_t dim = 0; dim < head_dim; dim += 2) {
            const int32_t lane = q_head * head_dim + dim;
            const int32_t low = div_i32(accum[lane], weight_sum);
            const int32_t high = div_i32(accum[lane + 1], weight_sum);
            output[lane >> 1] = pack_s16_pair(low, high);
        }
    }

    (void)accum_lanes;
    (void)state_dwords;
}

void attention_kv16_finish_accum_bf16(
    int32_t *accum,
    int32_t *state,
    int32_t *output,
    int32_t output_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
    constexpr int32_t heads = 8;
    constexpr int32_t head_dim = 128;
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(output);
    for (int32_t idx = 0; idx < output_dwords * 2; idx++) {
        payload[idx] = static_cast<bfloat16>(0.0f);
    }
    for (int32_t q_head = 0; q_head < heads; q_head++) {
        const int32_t weight_sum = state[q_head * 2 + 1];
        for (int32_t dim = 0; dim < head_dim; dim++) {
            const int32_t lane = q_head * head_dim + dim;
            const int32_t value = div_i32(accum[lane], weight_sum);
            payload[lane] = static_cast<bfloat16>(static_cast<float>(value) / 1024.0f);
        }
    }

    (void)accum_lanes;
    (void)state_dwords;
}

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
