#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

constexpr int32_t sigmoid_table_last = 64;
constexpr float sigmoid_table_scale = 8.0f;
static const float sigmoid_table[sigmoid_table_last + 1] = {
    0.5000000000f, 0.5312093734f, 0.5621765009f, 0.5926666000f, 0.6224593312f,
    0.6513548647f, 0.6791786992f, 0.7057850278f, 0.7310585786f, 0.7549149869f,
    0.7772998612f, 0.7981867777f, 0.8175744762f, 0.8354835371f, 0.8519528020f,
    0.8670357598f, 0.8807970780f, 0.8933094061f, 0.9046505351f, 0.9149009550f,
    0.9241418200f, 0.9324533089f, 0.9399133498f, 0.9465966702f, 0.9525741268f,
    0.9579122721f, 0.9626731127f, 0.9669140216f, 0.9706877692f, 0.9740426428f,
    0.9770226301f, 0.9796676467f, 0.9820137900f, 0.9840936083f, 0.9859363730f,
    0.9875683491f, 0.9890130574f, 0.9902915235f, 0.9914225146f, 0.9924227587f,
    0.9933071491f, 0.9940889311f, 0.9947798743f, 0.9953904278f, 0.9959298623f,
    0.9964063974f, 0.9968273172f, 0.9971990730f, 0.9975273768f, 0.9978172836f,
    0.9980732653f, 0.9982992776f, 0.9984988177f, 0.9986749776f, 0.9988304897f,
    0.9989677690f, 0.9990889488f, 0.9991959141f, 0.9992903296f, 0.9993736658f,
    0.9994472214f, 0.9995121429f, 0.9995694429f, 0.9996200155f, 0.9996646499f
};

static float sigmoid_approx(float value) {
    if (value > 8.0f) {
        return 1.0f;
    }
    if (value < -8.0f) {
        return 0.0f;
    }
    const bool negative = value < 0.0f;
    const float scaled = abs_f32(value) * sigmoid_table_scale;
    int32_t index = static_cast<int32_t>(scaled);
    if (index >= sigmoid_table_last) {
        const float edge = sigmoid_table[sigmoid_table_last];
        return negative ? 1.0f - edge : edge;
    }
    const float fraction = scaled - static_cast<float>(index);
    const float low = sigmoid_table[index];
    const float high = sigmoid_table[index + 1];
    const float sigmoid = low + (high - low) * fraction;
    return negative ? 1.0f - sigmoid : sigmoid;
}

} // namespace

extern "C" {

void ffn_swiglu_slice_bf16_inputs(
    int32_t *input,
    bfloat16 *output,
    int32_t dwords,
    int32_t slice
) {
    (void)slice;
    const int32_t half = dwords / 2;
    bfloat16 *values = reinterpret_cast<bfloat16 *>(input);
    for (int32_t idx = 0; idx < half * 2; idx++) {
        const float up = static_cast<float>(values[idx]);
        const float gate = static_cast<float>(values[half * 2 + idx]);
        output[idx] = static_cast<bfloat16>(up * gate * sigmoid_approx(gate));
    }
}

} // extern "C"
