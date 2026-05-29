#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static int32_t unpack_s16(int32_t word, int32_t lane) {
    const uint32_t raw_word = static_cast<uint32_t>(word);
    const uint32_t raw = lane != 0 ? (raw_word >> 16) & 0xffff : raw_word & 0xffff;
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

static int32_t sigmoid_q15(int32_t gate) {
    constexpr int32_t one = 32768;
    constexpr int32_t half = one / 2;
    constexpr int32_t limit = 2048;
    constexpr int32_t slope = 8;
    if (gate <= -limit) {
        return 0;
    }
    if (gate >= limit) {
        return one - 1;
    }
    return half + gate * slope;
}

static int32_t swiglu_lane(int32_t up, int32_t gate) {
    constexpr int32_t sigmoid_scale = 32768;
    constexpr int32_t product_scale = 16384;
    const int32_t silu_gate = div_i32(gate * sigmoid_q15(gate), sigmoid_scale);
    return clamp_s16(div_i32(up * silu_gate, product_scale));
}

static int32_t slice_adjust(int32_t value, int32_t slice, int32_t lane) {
    if (slice == 0) {
        return value;
    }
    const int32_t scale = 256 + slice;
    const int32_t delta = ((slice * 13 + lane * 3) % 17) - 8;
    return clamp_s16(div_i32(value * scale, 256) + delta);
}

} // namespace

extern "C" {

void ffn_swiglu_slice_contract(int32_t *input, int32_t *output, int32_t dwords, int32_t slice) {
    const int32_t half = dwords / 2;
    for (int32_t idx = 0; idx < half; idx++) {
        const int32_t up_word = input[idx];
        const int32_t gate_word = input[half + idx];
        const int32_t low = slice_adjust(
            swiglu_lane(unpack_s16(up_word, 0), unpack_s16(gate_word, 0)),
            slice,
            idx * 2
        );
        const int32_t high = slice_adjust(
            swiglu_lane(unpack_s16(up_word, 1), unpack_s16(gate_word, 1)),
            slice,
            idx * 2 + 1
        );
        output[idx] = pack_s16_pair(low, high);
    }
}

void ffn_swiglu_slice_bf16_inputs_contract(
    int32_t *input,
    bfloat16 *output,
    int32_t dwords,
    int32_t slice
) {
    const int32_t half = dwords / 2;
    bfloat16 *values = reinterpret_cast<bfloat16 *>(input);
    for (int32_t idx = 0; idx < half * 2; idx++) {
        const float up = static_cast<float>(values[idx]);
        const float gate = static_cast<float>(values[half * 2 + idx]);
        float sigmoid = 0.5f + gate * 0.125f;
        sigmoid = sigmoid < 0.0f ? 0.0f : sigmoid;
        sigmoid = sigmoid > 1.0f ? 1.0f : sigmoid;
        const float slice_scale = 1.0f + static_cast<float>(slice) * 0.00390625f;
        output[idx] = static_cast<bfloat16>(up * gate * sigmoid * slice_scale);
    }
}

void ffn_swiglu_contract(int32_t *input, int32_t *output, int32_t dwords) {
    ffn_swiglu_slice_contract(input, output, dwords, 0);
}

} // extern "C"
