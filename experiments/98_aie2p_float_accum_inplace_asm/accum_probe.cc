#include <stdint.h>

extern "C" void asm_float_accum_inplace(float *dst);

extern "C" void cpp_call_asm_float_accum_inplace(float *dst) {
    asm_float_accum_inplace(dst);
}
