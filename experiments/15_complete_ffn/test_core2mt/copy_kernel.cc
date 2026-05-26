#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void copy_bf16(bfloat16 *src, bfloat16 *dst, int32_t n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

} // extern "C"
