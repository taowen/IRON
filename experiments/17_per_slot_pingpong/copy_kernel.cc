#include <aie_api/aie.hpp>
extern "C" void copy_section(bfloat16* src, bfloat16* dst, int offset) {
    for (int i = 0; i < 32; i++) {
        dst[i] = src[offset + i];
    }
}
extern "C" void copy_gathered(bfloat16* src, bfloat16* dst, int offset) {
    for (int i = 0; i < 32; i++) {
        dst[i] = src[offset + i];
    }
}
