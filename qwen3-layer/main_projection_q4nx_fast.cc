#include <aie_api/aie.hpp>
#include <stdint.h>

#include "qwen3_constants.h"

extern "C" void q4nx_chunk_accum_asm_zol(
    float *target,
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice
);

extern "C" {

alignas(64) float q4nx_main16_accum[qwen3::kMainRowsPerTile];

void q4nx_fill_perf_inputs(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t weight_bf16,
    int32_t activation_dwords
) {
    for (int32_t idx = 0; idx < weight_bf16; idx++) {
        packed_chunk[idx] = static_cast<bfloat16>(1.0f);
    }
    for (int32_t idx = 0; idx < activation_dwords; idx++) {
        activation_words[idx] = 0x3f803f80;
    }
}

void q4nx_clear_accum_fast(int32_t num_rows) {
    for (int32_t idx = 0; idx < qwen3::kMainRowsPerTile; idx++) {
        if (idx < num_rows) {
            q4nx_main16_accum[idx] = 0.0f;
        }
    }
}

void q4nx_chunk_accum_slice_i32_fast(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t num_rows
) {
    (void)num_rows;
    q4nx_chunk_accum_asm_zol(
        q4nx_main16_accum,
        packed_chunk,
        reinterpret_cast<bfloat16 *>(activation_words)
    );
}

} // extern "C"
