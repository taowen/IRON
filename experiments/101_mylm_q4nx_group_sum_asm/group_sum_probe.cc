#include <stdint.h>

extern "C" void asm_mylm_q4_group_sum_body(int32_t *workspace);

extern "C" void cpp_call_mylm_q4_group_sum_body(int32_t *workspace) {
    asm_mylm_q4_group_sum_body(workspace);
}
