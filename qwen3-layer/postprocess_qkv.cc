#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

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

static int32_t current_word_from_compact(
    int32_t *compact,
    int32_t idx,
    int32_t current_token,
    bool is_v
) {
    const int32_t low_lane = idx * 2;
    const int32_t high_lane = low_lane + 1;
    const int32_t low_head = low_lane / 128;
    const int32_t high_head = high_lane / 128;
    const int32_t low_dim = low_lane % 128;
    const int32_t high_dim = high_lane % 128;
    const int32_t seed_mul = is_v ? 7 : 5;
    const int32_t seed_mask = is_v ? 15 : 7;
    const int32_t low_seed = compact[1 + ((low_lane * seed_mul) & 255)] & seed_mask;
    const int32_t high_seed = compact[1 + ((high_lane * seed_mul) & 255)] & seed_mask;
    int32_t low = 0;
    int32_t high = 0;
    if (is_v) {
        low = ((low_head + 5) * 7 + current_token * 11 + low_dim * 2 + low_seed) % 63 - 31;
        high = ((high_head + 5) * 7 + current_token * 11 + high_dim * 2 + high_seed) % 63 - 31;
    } else {
        low = ((low_head + 3) * 9 + current_token * 5 + low_dim * 3 + low_seed) % 31 - 15;
        high = ((high_head + 3) * 9 + current_token * 5 + high_dim * 3 + high_seed) % 31 - 15;
    }
    return pack_s16_pair(low, high);
}

static int32_t q4nx_body_attention_lane(bfloat16 value) {
    constexpr float scale = 16.0f;
    const float scaled = static_cast<float>(value) * scale;
    const int32_t rounded = static_cast<int32_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
    return clamp_s16(rounded);
}

static int32_t q4nx_body_attention_word_at(int32_t *body, int32_t logical_word) {
    bfloat16 *values = reinterpret_cast<bfloat16 *>(body);
    const int32_t low = q4nx_body_attention_lane(values[logical_word * 2]);
    const int32_t high = q4nx_body_attention_lane(values[logical_word * 2 + 1]);
    return pack_s16_pair(low, high);
}

static float fast_rsqrt(float value) {
    if (value <= 0.0f) {
        return 1.0f;
    }
    float y = 1.0f;
    if (value > 1.0f) {
        y = 0.5f;
    }
    if (value > 4.0f) {
        y = 0.25f;
    }
    if (value > 16.0f) {
        y = 0.125f;
    }
    const float half = value * 0.5f;
    for (int32_t iter = 0; iter < 6; iter++) {
        y = y * (1.5f - half * y * y);
    }
    return y;
}

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static float head_rms_scale(bfloat16 *body, int32_t head, int32_t head_dim) {
    const int32_t base = head * head_dim;
    float max_abs = 0.0f;
    for (int32_t dim = 0; dim < head_dim; dim++) {
        const float value = static_cast<float>(body[base + dim]);
        const float magnitude = abs_f32(value);
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }
    if (max_abs == 0.0f) {
        return 1.0f;
    }
    float sum_sq = 0.0f;
    for (int32_t dim = 0; dim < head_dim; dim++) {
        const float normalized = static_cast<float>(body[base + dim]) / max_abs;
        sum_sq += normalized * normalized;
    }
    constexpr float eps = 0.000001f;
    const float scaled_eps = eps / (max_abs * max_abs);
    return fast_rsqrt(sum_sq / static_cast<float>(head_dim) + scaled_eps) / max_abs;
}

static void write_rope_pair(
    bfloat16 *body,
    bfloat16 *norm_weight,
    bfloat16 *rope_cos,
    bfloat16 *rope_sin,
    bfloat16 *output,
    int32_t head,
    int32_t dim,
    float scale
) {
    constexpr int32_t head_dim = 128;
    const int32_t lane = head * head_dim + dim;
    const float even = static_cast<float>(body[lane]) * scale * static_cast<float>(norm_weight[dim]);
    const float odd = static_cast<float>(body[lane + 1]) * scale * static_cast<float>(norm_weight[dim + 1]);
    const int32_t pair = dim >> 1;
    const float c = static_cast<float>(rope_cos[pair]);
    const float s = static_cast<float>(rope_sin[pair]);
    output[lane] = static_cast<bfloat16>(even * c - odd * s);
    output[lane + 1] = static_cast<bfloat16>(even * s + odd * c);
}

static int32_t packed_rope_word(
    bfloat16 *body,
    bfloat16 *norm_weight,
    bfloat16 *rope_cos,
    bfloat16 *rope_sin,
    int32_t logical_word,
    float scale
) {
    constexpr int32_t head_dim = 128;
    const int32_t lane = logical_word * 2;
    const int32_t dim = lane % head_dim;
    const float even = static_cast<float>(body[lane]) * scale * static_cast<float>(norm_weight[dim]);
    const float odd = static_cast<float>(body[lane + 1]) * scale * static_cast<float>(norm_weight[dim + 1]);
    const int32_t pair = dim >> 1;
    const float c = static_cast<float>(rope_cos[pair]);
    const float s = static_cast<float>(rope_sin[pair]);
    bfloat16 packed[2];
    packed[0] = static_cast<bfloat16>(even * c - odd * s);
    packed[1] = static_cast<bfloat16>(even * s + odd * c);
    return reinterpret_cast<int32_t *>(packed)[0];
}

static void write_current_even_odd_from_compact(
    int32_t *k_compact,
    int32_t *v_compact,
    int32_t *current_k,
    int32_t *current_v,
    int32_t current_dwords,
    int32_t current_token
) {
    constexpr int32_t kv_heads = 8;
    const int32_t head_dwords = current_dwords / kv_heads;
    const int32_t half_current_dwords = current_dwords / 2;
    const int32_t half_head_dwords = head_dwords / 2;
    for (int32_t head = 0; head < kv_heads; head++) {
        for (int32_t pair = 0; pair < half_head_dwords; pair++) {
            const int32_t even_idx = head * head_dwords + pair * 2;
            const int32_t odd_idx = even_idx + 1;
            const int32_t even_stream_idx = head * half_head_dwords + pair;
            const int32_t odd_stream_idx = half_current_dwords + even_stream_idx;
            current_k[even_stream_idx] =
                current_word_from_compact(k_compact, even_idx, current_token, false);
            current_v[even_stream_idx] =
                current_word_from_compact(v_compact, even_idx, current_token, true);
            current_k[odd_stream_idx] =
                current_word_from_compact(k_compact, odd_idx, current_token, false);
            current_v[odd_stream_idx] =
                current_word_from_compact(v_compact, odd_idx, current_token, true);
        }
    }
}

static void write_current_even_odd_from_body(
    int32_t *k_body,
    int32_t *v_body,
    int32_t *current_k,
    int32_t *current_v,
    int32_t current_dwords,
    bool q4nx_body
) {
    constexpr int32_t kv_heads = 8;
    const int32_t head_dwords = current_dwords / kv_heads;
    const int32_t half_current_dwords = current_dwords / 2;
    const int32_t half_head_dwords = head_dwords / 2;
    for (int32_t head = 0; head < kv_heads; head++) {
        for (int32_t pair = 0; pair < half_head_dwords; pair++) {
            const int32_t even_idx = head * head_dwords + pair * 2;
            const int32_t odd_idx = even_idx + 1;
            const int32_t even_stream_idx = head * half_head_dwords + pair;
            const int32_t odd_stream_idx = half_current_dwords + even_stream_idx;
            if (q4nx_body) {
                current_k[even_stream_idx] = q4nx_body_attention_word_at(k_body, even_idx);
                current_v[even_stream_idx] = q4nx_body_attention_word_at(v_body, even_idx);
                current_k[odd_stream_idx] = q4nx_body_attention_word_at(k_body, odd_idx);
                current_v[odd_stream_idx] = q4nx_body_attention_word_at(v_body, odd_idx);
            } else {
                current_k[even_stream_idx] = k_body[even_idx];
                current_v[even_stream_idx] = v_body[even_idx];
                current_k[odd_stream_idx] = k_body[odd_idx];
                current_v[odd_stream_idx] = v_body[odd_idx];
            }
        }
    }
}

static void write_qwen3_current_even_odd(
    int32_t *k_body,
    int32_t *v_body,
    int32_t *qk_rope_side,
    int32_t *current_k,
    int32_t *current_v,
    int32_t current_dwords
) {
    constexpr int32_t kv_heads = 8;
    constexpr int32_t head_dim = 128;
    bfloat16 *k_values = reinterpret_cast<bfloat16 *>(k_body);
    bfloat16 *side = reinterpret_cast<bfloat16 *>(qk_rope_side);
    bfloat16 *k_norm = side + head_dim;
    bfloat16 *rope_cos = side + head_dim * 2;
    bfloat16 *rope_sin = rope_cos + head_dim / 2;
    const int32_t head_dwords = current_dwords / kv_heads;
    const int32_t half_current_dwords = current_dwords / 2;
    const int32_t half_head_dwords = head_dwords / 2;

    float scales[kv_heads];
    for (int32_t head = 0; head < kv_heads; head++) {
        scales[head] = head_rms_scale(k_values, head, head_dim);
    }

    for (int32_t head = 0; head < kv_heads; head++) {
        for (int32_t pair = 0; pair < half_head_dwords; pair++) {
            const int32_t even_idx = head * head_dwords + pair * 2;
            const int32_t odd_idx = even_idx + 1;
            const int32_t even_stream_idx = head * half_head_dwords + pair;
            const int32_t odd_stream_idx = half_current_dwords + even_stream_idx;
            current_k[even_stream_idx] = packed_rope_word(
                k_values,
                k_norm,
                rope_cos,
                rope_sin,
                even_idx,
                scales[head]
            );
            current_k[odd_stream_idx] = packed_rope_word(
                k_values,
                k_norm,
                rope_cos,
                rope_sin,
                odd_idx,
                scales[head]
            );
            current_v[even_stream_idx] = v_body[even_idx];
            current_v[odd_stream_idx] = v_body[odd_idx];
        }
    }
}

} // namespace

extern "C" {

void mainq_postprocess_payload(int32_t *q_compact, int32_t *q_payload, int32_t q_dwords) {
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        const int32_t low_lane = idx * 2;
        const int32_t high_lane = low_lane + 1;
        const int32_t low_seed = q_compact[1 + (low_lane & 255)] & 31;
        const int32_t high_seed = q_compact[1 + (high_lane & 255)] & 31;
        const int32_t low = ((low_lane * 5 + low_seed) % 31) - 15;
        const int32_t high = ((high_lane * 5 + high_seed) % 31) - 15;
        q_payload[idx] = pack_s16_pair(low, high);
    }
}

void currentkv_postprocess_payload(
    int32_t *q_compact,
    int32_t *k_compact,
    int32_t *v_compact,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    const int32_t current_token = current_token_buf[0];
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        const int32_t low_lane = idx * 2;
        const int32_t high_lane = low_lane + 1;
        const int32_t low_seed = q_compact[1 + (low_lane & 255)] & 31;
        const int32_t high_seed = q_compact[1 + (high_lane & 255)] & 31;
        const int32_t low = ((low_lane * 5 + low_seed) % 31) - 15;
        const int32_t high = ((high_lane * 5 + high_seed) % 31) - 15;
        q_payload[idx] = pack_s16_pair(low, high);
    }
    write_current_even_odd_from_compact(
        k_compact,
        v_compact,
        current_k,
        current_v,
        current_dwords,
        current_token
    );
}

void currentkv_postprocess_body_payload(
    int32_t *q_body,
    int32_t *k_body,
    int32_t *v_body,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        q_payload[idx] = q_body[idx];
    }
    write_current_even_odd_from_body(k_body, v_body, current_k, current_v, current_dwords, false);
    (void)current_token_buf;
}

void currentkv_postprocess_q4nx_body_payload(
    int32_t *q_body,
    int32_t *k_body,
    int32_t *v_body,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    for (int32_t idx = 0; idx < q_dwords; idx++) {
        q_payload[idx] = q4nx_body_attention_word_at(q_body, idx);
    }
    write_current_even_odd_from_body(k_body, v_body, current_k, current_v, current_dwords, true);
    (void)current_token_buf;
}

void qwen3_postprocess_q4nx_body_payload(
    int32_t *q_body,
    int32_t *k_body,
    int32_t *v_body,
    int32_t *qk_rope_side,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    constexpr int32_t q_heads = 32;
    constexpr int32_t head_dim = 128;
    bfloat16 *q_values = reinterpret_cast<bfloat16 *>(q_body);
    bfloat16 *q_output = reinterpret_cast<bfloat16 *>(q_payload);
    bfloat16 *side = reinterpret_cast<bfloat16 *>(qk_rope_side);
    bfloat16 *q_norm = side;
    bfloat16 *rope_cos = side + head_dim * 2;
    bfloat16 *rope_sin = rope_cos + head_dim / 2;

    for (int32_t head = 0; head < q_heads; head++) {
        const float scale = head_rms_scale(q_values, head, head_dim);
        for (int32_t dim = 0; dim < head_dim; dim += 2) {
            write_rope_pair(q_values, q_norm, rope_cos, rope_sin, q_output, head, dim, scale);
        }
    }

    write_qwen3_current_even_odd(k_body, v_body, qk_rope_side, current_k, current_v, current_dwords);
    (void)current_token_buf;
    (void)q_dwords;
}

} // extern "C"
