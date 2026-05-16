// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reuse the current MHA online-softmax and PV kernels. SageAttention replaces
// the QK stage with int8 x int8 -> int32. Runtime dequant scales are streamed
// to the softmax worker, where int32 scores are converted to bf16 logits before
// the existing partial softmax runs. The QK output stays in the blocked layout
// used by the MHA dataflow; the ObjectFIFO forward between QK and softmax
// performs the layout conversion.
#include "mha.cc"

extern "C" {

void matmul_i8_i32_wrapper(int8 *q_in, int8 *k_in, int32 *scores_out, int32_t *idx_buffer)
{
    ::aie::set_rounding(ROUNDING_MODE);

    if (idx_buffer[0] > idx_buffer[1]) {
        return;
    }

    zero_vectorized<int32, DIM_M, DIM_N>(scores_out);
    matmul_vectorized_8x8x8_i8_i32<DIM_M, DIM_K, DIM_N>(q_in, k_in, scores_out);
}

void partial_softmax_i32_dequant(int32 *scores_in,
                                 bfloat16 *logits,
                                 bfloat16 *P,
                                 bfloat16 *scale_buffer,
                                 int32_t *idx_buffer,
                                 float *dequant_scales,
                                 bfloat16 inv_scale,
                                 int32_t B_q,
                                 int32_t B_kv,
                                 int32_t S_q_eff,
                                 int32_t S_kv_eff)
{
    ::aie::set_rounding(ROUNDING_MODE);

    if (idx_buffer[0] > idx_buffer[1]) {
        zero_bf16(P);
        return;
    }

    constexpr int elements = DIM_M * DIM_N;
    constexpr int vec_len = 16;
    const int32_t scale_idx = idx_buffer[0];
    const auto scale = aie::broadcast<float, vec_len>(dequant_scales[scale_idx]);
    for (int i = 0; i < elements; i += vec_len) {
        auto qk_i32 = aie::load_v<vec_len>(scores_in + i);
        auto qk_f32 = aie::to_float<float>(qk_i32, 0);
        auto scaled = aie::mul(qk_f32, scale);
        aie::store_v(logits + i, scaled.to_vector<bfloat16>());
    }

    partial_softmax(
        logits, P, scale_buffer, idx_buffer, inv_scale, B_q, B_kv, S_q_eff, S_kv_eff);
}

} // extern "C"
