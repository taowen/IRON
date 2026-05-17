// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 256
#endif

#ifndef CACHE_BLOCK
#define CACHE_BLOCK 64
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 1024
#endif

#ifndef RMS_NORM_EPSILON
#define RMS_NORM_EPSILON 1e-6f
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
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr int vec_len = 32;

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
            aie::accum acc = aie::zeros<accfloat, vec_len>();
            for (int32_t dim = 0; dim < HEAD_DIM; dim += vec_len) {
                aie::vector<bfloat16, vec_len> q_vec = aie::load_v<vec_len>(q + dim);
                aie::vector<bfloat16, vec_len> k_vec = aie::load_v<vec_len>(k + dim);
                acc = aie::mac(acc, q_vec, k_vec);
            }
            score = aie::reduce_add(acc.template to_vector<float>()) * ATTN_SCALE;
        }
        bfloat16 out = static_cast<bfloat16>(score);
        debug_scores[global_row] = out;
        softmax_scores[global_row] = out;
    }

    event1();
}

void qwen3_merge_current_v_bf16(const bfloat16 *__restrict v_cache,
                                const bfloat16 *__restrict current_v,
                                bfloat16 *__restrict merged_v,
                                int32_t position,
                                int32_t row_base)
{
    event0();

    for (int32_t row = 0; row < CACHE_BLOCK; row++) {
        int32_t global_row = row_base + row;
        const bfloat16 *__restrict src = (global_row == position) ? current_v : v_cache + row * HEAD_DIM;
        for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
            merged_v[row * HEAD_DIM + dim] = src[dim];
        }
    }

    event1();
}

void qwen3_attention_context_bf16(const bfloat16 *__restrict weights,
                                  const bfloat16 *__restrict v_block,
                                  bfloat16 *__restrict context,
                                  int32_t position,
                                  int32_t row_base)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    float accum[HEAD_DIM];
    for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
        accum[dim] = (row_base == 0) ? 0.0f : static_cast<float>(context[dim]);
    }

    for (int32_t row = 0; row < CACHE_BLOCK; row++) {
        int32_t global_row = row_base + row;
        if (global_row <= position) {
            float weight = static_cast<float>(weights[global_row]);
            const bfloat16 *__restrict v = v_block + row * HEAD_DIM;
            for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
                accum[dim] += weight * static_cast<float>(v[dim]);
            }
        }
    }

    for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
        context[dim] = static_cast<bfloat16>(accum[dim]);
    }

    event1();
}

void qwen3_pack_context_head_bf16(const bfloat16 *__restrict context_head,
                                  bfloat16 *__restrict context_flat,
                                  int32_t head_idx)
{
    event0();

    bfloat16 *__restrict dst = context_flat + head_idx * HEAD_DIM;
    for (int32_t dim = 0; dim < HEAD_DIM; dim++) {
        dst[dim] = context_head[dim];
    }

    event1();
}

void qwen3_add_hidden_tile_to_full_bf16(const bfloat16 *__restrict hidden_tile,
                                        const bfloat16 *__restrict update_full,
                                        bfloat16 *__restrict output_full,
                                        int32_t row_offset,
                                        int32_t size)
{
    event0();

    for (int32_t i = 0; i < size; i++) {
        output_full[row_offset + i] =
            static_cast<bfloat16>(static_cast<float>(hidden_tile[i]) + static_cast<float>(update_full[row_offset + i]));
    }

    event1();
}

void qwen3_add_full_slice_bf16(const bfloat16 *__restrict lhs_full,
                               const bfloat16 *__restrict rhs_full,
                               bfloat16 *__restrict output_full,
                               int32_t row_offset,
                               int32_t size)
{
    event0();

    for (int32_t i = 0; i < size; i++) {
        int32_t idx = row_offset + i;
        output_full[idx] = static_cast<bfloat16>(static_cast<float>(lhs_full[idx]) + static_cast<float>(rhs_full[idx]));
    }

    event1();
}

void qwen3_weighted_rms_norm_bf16(const bfloat16 *__restrict input,
                                  const bfloat16 *__restrict weight,
                                  bfloat16 *__restrict output,
                                  int32_t size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr int vec_len = 16;
    constexpr float epsilon = RMS_NORM_EPSILON;

    float sum_sq = 0.0f;
    for (int32_t i = 0; i < size; i += vec_len) {
        aie::vector<bfloat16, vec_len> x_vec = aie::load_v<vec_len>(input + i);
        aie::vector<float, vec_len> sq = aie::mul_square(x_vec);
        sum_sq += aie::reduce_add(sq);
    }

    float inv_rms = aie::invsqrt(sum_sq / static_cast<float>(size) + epsilon);
    aie::vector<bfloat16, vec_len> inv_v = aie::broadcast<bfloat16, vec_len>(static_cast<bfloat16>(inv_rms));
    for (int32_t i = 0; i < size; i += vec_len) {
        aie::vector<bfloat16, vec_len> x_vec = aie::load_v<vec_len>(input + i);
        aie::vector<bfloat16, vec_len> w_vec = aie::load_v<vec_len>(weight + i);
        auto norm = aie::mul(x_vec, inv_v).template to_vector<bfloat16>();
        auto out = aie::mul(norm, w_vec).template to_vector<bfloat16>();
        aie::store_v(output + i, out);
    }

    event1();
}

void qwen3_mlp_matvec4_rows_bf16(int32_t m,
                                 int32_t row_offset,
                                 const bfloat16 *__restrict row0,
                                 const bfloat16 *__restrict row1,
                                 const bfloat16 *__restrict row2,
                                 const bfloat16 *__restrict row3,
                                 const bfloat16 *__restrict input,
                                 bfloat16 *__restrict output)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr int vec_len = 64;

#define QWEN3_DOT_ROW(row_ptr, local_row)                                                                              \
    if (m > local_row) {                                                                                               \
        aie::accum acc = aie::zeros<accfloat, vec_len>();                                                              \
        for (int32_t i = 0; i < HIDDEN_SIZE; i += vec_len) {                                                           \
            aie::vector<bfloat16, vec_len> w_vec = aie::load_v<vec_len>((row_ptr) + i);                                \
            aie::vector<bfloat16, vec_len> x_vec = aie::load_v<vec_len>(input + i);                                    \
            acc = aie::mac(acc, w_vec, x_vec);                                                                         \
        }                                                                                                              \
        output[row_offset + local_row] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));      \
    }

    QWEN3_DOT_ROW(row0, 0)
    QWEN3_DOT_ROW(row1, 1)
    QWEN3_DOT_ROW(row2, 2)
    QWEN3_DOT_ROW(row3, 3)

#undef QWEN3_DOT_ROW

    event1();
}

void qwen3_norm_rope_with_weight_offset_bf16(const bfloat16 *__restrict input,
                                             const bfloat16 *__restrict weights,
                                             const bfloat16 *__restrict lut,
                                             bfloat16 *__restrict norm_output,
                                             bfloat16 *__restrict rope_output,
                                             int32_t weight_offset,
                                             int32_t size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr int vec_len = 16;
    constexpr float epsilon = RMS_NORM_EPSILON;

    float sum_sq = 0.0f;
    for (int32_t i = 0; i < size; i += vec_len) {
        aie::vector<bfloat16, vec_len> x_vec = aie::load_v<vec_len>(input + i);
        aie::vector<float, vec_len> sq = aie::mul_square(x_vec);
        sum_sq += aie::reduce_add(sq);
    }

    float inv_rms = aie::invsqrt(sum_sq / static_cast<float>(size) + epsilon);
    aie::vector<bfloat16, vec_len> inv_v = aie::broadcast<bfloat16, vec_len>(static_cast<bfloat16>(inv_rms));
    for (int32_t i = 0; i < size; i += vec_len) {
        aie::vector<bfloat16, vec_len> x_vec = aie::load_v<vec_len>(input + i);
        aie::vector<bfloat16, vec_len> w_vec = aie::load_v<vec_len>(weights + weight_offset + i);
        auto norm = aie::mul(x_vec, inv_v).template to_vector<bfloat16>();
        auto out = aie::mul(norm, w_vec).template to_vector<bfloat16>();
        aie::store_v(norm_output + i, out);
    }

    int32_t half = size / 2;
    for (int32_t v = 0, lut_idx = 0; v < half; v += vec_len, lut_idx += 2 * vec_len) {
        aie::vector<bfloat16, vec_len> x1 = aie::load_v<vec_len>(norm_output + v);
        aie::vector<bfloat16, vec_len> x2 = aie::load_v<vec_len>(norm_output + v + half);
        aie::vector<bfloat16, 2 * vec_len> cache = aie::load_v<2 * vec_len>(lut + lut_idx);
        aie::vector<bfloat16, vec_len> cos_val = aie::filter_even(cache, 1);
        aie::vector<bfloat16, vec_len> sin_val = aie::filter_odd(cache, 1);

        aie::vector<bfloat16, vec_len> y_first_half =
            aie::sub(aie::mul(x1, cos_val), aie::mul(x2, sin_val)).template to_vector<bfloat16>();
        aie::vector<bfloat16, vec_len> y_second_half =
            aie::add(aie::mul(x2, cos_val), aie::mul(x1, sin_val)).template to_vector<bfloat16>();
        aie::store_v(rope_output + v, y_first_half);
        aie::store_v(rope_output + v + half, y_second_half);
    }

    event1();
}

void qwen3_pack_qk_rope_metadata_bf16(const bfloat16 *__restrict qk_norm_weights,
                                      const bfloat16 *__restrict lut,
                                      bfloat16 *__restrict metadata,
                                      int32_t size)
{
    event0();

    for (int32_t i = 0; i < 2 * size; i++) {
        metadata[i] = qk_norm_weights[i];
    }
    for (int32_t i = 0; i < size; i++) {
        metadata[2 * size + i] = lut[i];
    }

    event1();
}

void qwen3_norm_rope_with_metadata_bf16(const bfloat16 *__restrict input,
                                        const bfloat16 *__restrict metadata,
                                        bfloat16 *__restrict norm_output,
                                        bfloat16 *__restrict rope_output,
                                        int32_t weight_offset,
                                        int32_t size)
{
    qwen3_norm_rope_with_weight_offset_bf16(
        input, metadata, metadata + 2 * size, norm_output, rope_output, weight_offset, size);
}

void qwen3_silu_mul_bf16(const bfloat16 *__restrict gate,
                         const bfloat16 *__restrict up,
                         bfloat16 *__restrict hidden,
                         int32_t size)
{
    event0();
    constexpr int vec_len = 16;
    aie::vector<bfloat16, vec_len> half = aie::broadcast<bfloat16, vec_len>(0.5f);
    aie::vector<bfloat16, vec_len> one = aie::broadcast<bfloat16, vec_len>(1.0f);

    for (int32_t i = 0; i < size; i += vec_len) {
        aie::vector<bfloat16, vec_len> gate_vec = aie::load_v<vec_len>(gate + i);
        aie::vector<bfloat16, vec_len> up_vec = aie::load_v<vec_len>(up + i);
        auto half_gate = aie::mul(gate_vec, half);
        auto tanh_half_gate = aie::tanh<bfloat16>(half_gate.to_vector<float>());
        auto tanh_half_gate_approx = aie::add(tanh_half_gate, one);
        aie::vector<bfloat16, vec_len> sigmoid_approx = aie::mul(tanh_half_gate_approx, half);
        auto silu = aie::mul(gate_vec, sigmoid_approx);
        auto out = aie::mul(silu.to_vector<bfloat16>(), up_vec).template to_vector<bfloat16>();
        aie::store_v(hidden + i, out);
    }

    event1();
}

} // extern "C"
