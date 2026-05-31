#include <stdint.h>

extern "C" void asm_aie2p_latency_probe(int32_t *workspace);

extern "C" void cpp_call_aie2p_latency_probe(int32_t *workspace) {
    asm_aie2p_latency_probe(workspace);
}
