#include <aie_api/aie.hpp>
#include <aie2pintrin.h>
#include <stdint.h>

extern "C" void asm_scalar_store(int32_t *dst);
extern "C" void asm_vmac_bf16_store(const bfloat16 *src, bfloat16 *dst);
extern "C" void asm_vmac_float_store(float *dst);
extern "C" void asm_vector_clobber_probe(const bfloat16 *src, bfloat16 *dst);

extern "C" void cpp_call_asm_scalar_store(int32_t *dst) {
    asm_scalar_store(dst);
}

extern "C" void cpp_call_asm_vmac_bf16_store(const bfloat16 *src, bfloat16 *dst) {
    asm_vmac_bf16_store(src, dst);
}

extern "C" void cpp_call_asm_vmac_float_store(float *dst) {
    asm_vmac_float_store(dst);
}

extern "C" void cpp_call_asm_vector_clobber_probe(const bfloat16 *src, bfloat16 *dst) {
    aie::vector<bfloat16, 16> before = aie::load_v<16>(src);
    asm_vector_clobber_probe(src, dst);
    aie::vector<bfloat16, 16> after = aie::load_v<16>(src + 16);
    aie::store_v(dst + 16, aie::add(before, after));
}

extern "C" void cpp_accum_float_store(const bfloat16 *src, float *dst) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    aie::vector<bfloat16, 16> values = aie::load_v<16>(src);
    aie::vector<bfloat16, 16> one =
        aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f));
    acc = aie::mac(acc, values, one);
    aie::store_v(dst, acc.to_vector<float>());
}

extern "C" void cpp_accum_bf16_store(const bfloat16 *src, bfloat16 *dst) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    aie::vector<bfloat16, 16> values = aie::load_v<16>(src);
    aie::vector<bfloat16, 16> one =
        aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f));
    acc = aie::mac(acc, values, one);
    aie::store_v(dst, acc.to_vector<bfloat16>());
}
