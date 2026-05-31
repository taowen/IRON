#include <stdint.h>

extern "C" void probe_asm_store_word(int32_t *dst);
extern "C" void probe_asm_vector_mac_smoke(
    const int32_t *lhs,
    const int32_t *rhs,
    int32_t *dst
);

extern "C" void probe_asm_store_word_wrapper(int32_t *dst) {
    probe_asm_store_word(dst);
}

extern "C" void probe_asm_vector_mac_smoke_wrapper(
    const int32_t *lhs,
    const int32_t *rhs,
    int32_t *dst
) {
    probe_asm_vector_mac_smoke(lhs, rhs, dst);
}

extern "C" void probe_cpp_copy_word(const int32_t *src, int32_t *dst) {
    dst[0] = src[0];
}

extern "C" void probe_asm_vector_mac_local_wrapper(
    int32_t *lhs,
    int32_t *rhs,
    int32_t *dst
) {
    for (int idx = 0; idx < 16; idx++) {
        lhs[idx] = 0x1000 + idx;
        rhs[idx] = 0x2000 + idx;
    }
    probe_asm_vector_mac_smoke(lhs, rhs, dst);
    volatile int32_t *observed = dst;
    observed[15] = observed[15];
}
