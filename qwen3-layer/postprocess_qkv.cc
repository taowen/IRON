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

} // extern "C"
