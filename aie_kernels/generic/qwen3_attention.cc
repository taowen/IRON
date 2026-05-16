// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 256
#endif

#ifndef CACHE_BLOCK
#define CACHE_BLOCK 64
#endif

#ifndef ATTN_SCALE
#define ATTN_SCALE 0.08838834764831845f
#endif

extern "C" {

void qwen3_pack_qk_pair_bf16(const bfloat16 *__restrict q,
                             const bfloat16 *__restrict current_k,
                             bfloat16 *__restrict qk_pair,
                             int32_t q_select)
{
    event0();

    for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
        qk_pair[q_select * HEAD_DIM + dim] = q[dim];
        if (q_select == 0) {
            qk_pair[2 * HEAD_DIM + dim] = current_k[dim];
        }
    }

    event1();
}

void qwen3_attention_scores_bf16(const bfloat16 *__restrict qk_pair,
                                 const bfloat16 *__restrict k_cache,
                                 bfloat16 *__restrict debug_scores,
                                 bfloat16 *__restrict softmax_scores,
                                 int32_t position,
                                 int32_t row_base,
                                 int32_t q_select)
{
    event0();

    if (row_base == 0) {
        for (int32_t row = 0; row < MAX_SEQ_LEN; row++) {
            debug_scores[row] = static_cast<bfloat16>(0.0f);
            softmax_scores[row] = static_cast<bfloat16>(0.0f);
        }
    }

    const bfloat16 *__restrict q = qk_pair + q_select * HEAD_DIM;
    const bfloat16 *__restrict current_k = qk_pair + 2 * HEAD_DIM;
    for (int32_t row = 0; row < CACHE_BLOCK; row++) {
        int32_t global_row = row_base + row;
        float score = 0.0f;
        if (global_row <= position) {
            const bfloat16 *__restrict k = (global_row == position) ? current_k : k_cache + row * HEAD_DIM;
            float acc = 0.0f;
            for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
                acc += static_cast<float>(q[dim]) * static_cast<float>(k[dim]);
            }
            score = acc * ATTN_SCALE;
        }
        bfloat16 out = static_cast<bfloat16>(score);
        debug_scores[global_row] = out;
        softmax_scores[global_row] = out;
    }

    event1();
}

} // extern "C"
