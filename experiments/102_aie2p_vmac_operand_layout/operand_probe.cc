#include <stdint.h>

extern "C" void asm_vmac_operand_layout(int32_t *workspace);

extern "C" void cpp_call_vmac_operand_layout(int32_t *workspace) {
    asm_vmac_operand_layout(workspace);
}
