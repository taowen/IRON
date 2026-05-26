#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void swiglu_fused(bfloat16 *gate, bfloat16 *up, bfloat16 *output, int32_t n) {
    aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
    aie::vector<bfloat16, 16> one  = aie::broadcast<bfloat16, 16>(1.0f);

    for (int i = 0; i < n; i += 16) {
        auto g = aie::load_v<16>(gate + i);
        auto u = aie::load_v<16>(up + i);

        // sigmoid(x) approx 0.5 * (1 + tanh(x/2))
        auto half_g = aie::mul(g, half);
        auto tanh_hg = aie::tanh<bfloat16>(half_g.to_vector<float>());
        auto tanh_plus_one = aie::add(tanh_hg, one);
        aie::vector<bfloat16, 16> sigmoid_g = aie::mul(tanh_plus_one, half);

        // SiLU(gate) = gate * sigmoid(gate)
        auto silu_acc = aie::mul(g, sigmoid_g);

        // SwiGLU = SiLU(gate) * up
        auto result = aie::mul(silu_acc.to_vector<bfloat16>(), u);

        aie::store_v(output + i, result.to_vector<bfloat16>());
    }
}

} // extern "C"
