#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static bfloat16 compact_lane(int32_t *compact, int32_t lane) {
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(compact + 1);
    return payload[lane];
}

static bfloat16 *as_bf16(int32_t *words) {
    return reinterpret_cast<bfloat16 *>(words);
}

static uint16_t bf16_bits(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits = {value};
    const uint32_t lsb = (bits.u32 >> 16) & 1u;
    bits.u32 += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits.u32 >> 16);
}

static bfloat16 bf16_rne(float value) {
    bfloat16 output;
    reinterpret_cast<uint16_t *>(&output)[0] = bf16_bits(value);
    return output;
}

static void store_bf16_rne(bfloat16 *values, int32_t lane, float value) {
    reinterpret_cast<uint16_t *>(values)[lane] = bf16_bits(value);
}

static float fast_rsqrt(float value) {
    if (value <= 0.0f) {
        return 1.0f;
    }
    float y = 1.0f;
    if (value < 0.000244140625f) {
        y = 64.0f;
    } else if (value < 0.0009765625f) {
        y = 32.0f;
    } else if (value < 0.00390625f) {
        y = 16.0f;
    } else if (value < 0.015625f) {
        y = 8.0f;
    } else if (value < 0.0625f) {
        y = 4.0f;
    } else if (value < 0.25f) {
        y = 2.0f;
    } else if (value > 1.0f) {
        y = 0.5f;
    }
    if (value > 4.0f) {
        y = 0.25f;
    }
    if (value > 16.0f) {
        y = 0.125f;
    }
    const float half = value * 0.5f;
    for (int32_t iter = 0; iter < 10; iter++) {
        y = y * (1.5f - half * y * y);
    }
    return y;
}

static float rms_scale(bfloat16 *values, int32_t lanes) {
    float sum_sq = 0.0f;
    for (int32_t lane = 0; lane < lanes; lane++) {
        const float value = static_cast<float>(values[lane]);
        sum_sq += value * value;
    }
    constexpr float eps = 0.000001f;
    return fast_rsqrt(sum_sq / static_cast<float>(lanes) + eps);
}

static void write_weighted_rms_values(
    bfloat16 *values,
    bfloat16 *weight,
    bfloat16 *payload,
    int32_t lanes
) {
    const float scale = rms_scale(values, lanes);
    for (int32_t lane = 0; lane < lanes; lane++) {
        store_bf16_rne(
            payload,
            lane,
            static_cast<float>(values[lane]) * scale * static_cast<float>(weight[lane])
        );
    }
}

} // namespace

extern "C" {

void qkv_c1r2_summarize_compact(int32_t *compact, int32_t *summary, int32_t dwords) {
    uint32_t sum = 0;
    uint32_t hash = 0;
    for (int32_t idx = 0; idx < dwords; idx++) {
        const uint32_t value = static_cast<uint32_t>(compact[idx]);
        sum += value;
        hash = (hash * 16777619u) ^ (value + static_cast<uint32_t>(idx));
    }
    summary[0] = 0x51564F43;
    summary[1] = dwords;
    summary[2] = compact[0];
    summary[3] = compact[dwords - 1];
    summary[4] = static_cast<int32_t>(sum);
    summary[5] = static_cast<int32_t>(hash);
    summary[6] = compact[1];
    summary[7] = compact[dwords - 2];
}

void full_c1r2_make_input_norm_replay(
    int32_t *hidden,
    int32_t *norm_weight,
    int32_t *replay,
    int32_t payload_dwords
) {
    replay[0] = 0xC1000000;
    write_weighted_rms_values(
        as_bf16(hidden),
        as_bf16(norm_weight),
        reinterpret_cast<bfloat16 *>(replay + 1),
        payload_dwords * 2
    );
}

void full_c1r2_make_input_norm_payload(
    int32_t *hidden,
    int32_t *norm_weight,
    int32_t *payload,
    int32_t payload_dwords
) {
    write_weighted_rms_values(as_bf16(hidden), as_bf16(norm_weight), as_bf16(payload), payload_dwords * 2);
}

void full_c1r2_add_o_compact_to_residual(int32_t *hidden, int32_t *compact, int32_t block) {
    constexpr int32_t lanes_per_block = 512;
    bfloat16 *residual = as_bf16(hidden);
    const int32_t base = block * lanes_per_block;
    for (int32_t lane = 0; lane < lanes_per_block; lane++) {
        const int32_t global_lane = base + lane;
        residual[global_lane] = bf16_rne(
            static_cast<float>(residual[global_lane]) + static_cast<float>(compact_lane(compact, lane))
        );
    }
}

void full_c1r2_make_post_norm_replay(
    int32_t *residual,
    int32_t *norm_weight,
    int32_t *replay,
    int32_t payload_dwords
) {
    replay[0] = 0xC1000000;
    write_weighted_rms_values(
        as_bf16(residual),
        as_bf16(norm_weight),
        reinterpret_cast<bfloat16 *>(replay + 1),
        payload_dwords * 2
    );
}

void full_c1r2_make_post_norm_payload(
    int32_t *residual,
    int32_t *norm_weight,
    int32_t *payload,
    int32_t payload_dwords
) {
    write_weighted_rms_values(as_bf16(residual), as_bf16(norm_weight), as_bf16(payload), payload_dwords * 2);
}

void full_c1r2_write_down_block(int32_t *residual, int32_t *compact, int32_t *output, int32_t block) {
    constexpr int32_t lanes_per_block = 512;
    bfloat16 *residual_values = as_bf16(residual);
    bfloat16 *output_values = as_bf16(output);
    const int32_t base = block * lanes_per_block;
    for (int32_t lane = 0; lane < lanes_per_block; lane++) {
        const int32_t global_lane = base + lane;
        output_values[global_lane] = bf16_rne(
            static_cast<float>(residual_values[global_lane]) + static_cast<float>(compact_lane(compact, lane))
        );
    }
}

void full_c1r2_write_o_block(int32_t *compact, int32_t *output, int32_t block) {
    constexpr int32_t lanes_per_block = 512;
    bfloat16 *output_values = as_bf16(output);
    const int32_t base = block * lanes_per_block;
    for (int32_t lane = 0; lane < lanes_per_block; lane++) {
        output_values[base + lane] = compact_lane(compact, lane);
    }
}

} // extern "C"
