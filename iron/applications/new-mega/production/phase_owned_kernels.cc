// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef NEW_MEGA_ATTN_SCALE
#define NEW_MEGA_ATTN_SCALE 0.08838834764831845f
#endif

#define NEW_MEGA_LOG2E_F 1.4426950408889634f

static inline float new_mega_exp_approx(float x)
{
    aie::vector<float, 8> input = aie::broadcast<float, 8>(x * NEW_MEGA_LOG2E_F);
    aie::vector<bfloat16, 8> output = aie::exp2<bfloat16>(input);
    return static_cast<float>(output.get(0));
}

static inline float new_mega_silu_approx(float x)
{
    return x / (1.0f + new_mega_exp_approx(-x));
}

extern "C" {

void new_mega_phase_state_init_f32(float *__restrict state)
{
    event0();
    state[0] = 0.0f;
    event1();
}

void new_mega_phase0_q_shard_bf16(const bfloat16 *__restrict shared_packet,
                                  const bfloat16 *__restrict lane_packet,
                                  bfloat16 *__restrict hidden_state,
                                  float *__restrict state,
                                  bfloat16 *__restrict lane_output,
                                  int32_t packet_size,
                                  int32_t hidden_size,
                                  int32_t q_rows_per_packet,
                                  int32_t q_output_values_per_lane,
                                  int32_t layer_id)
{
    event0();
    (void)packet_size;

    if (layer_id == 0) {
        for (int32_t i = 0; i < hidden_size; i++) {
            hidden_state[i] = shared_packet[i];
        }
    }

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        const float x = static_cast<float>(hidden_state[i]);
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    const bfloat16 *weight = shared_packet + hidden_size;
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *q_row = lane_packet + row * hidden_size;
        float q_acc = 0.0f;
        for (int32_t i = 0; i < hidden_size; i++) {
            const float xnorm = static_cast<float>(hidden_state[i]) * inv_rms * static_cast<float>(weight[i]);
            if (row == 0) {
                checksum += xnorm;
            }
            q_acc += xnorm * static_cast<float>(q_row[i]);
        }
        checksum += q_acc;
        lane_output[row] = static_cast<bfloat16>(q_acc);
    }

    for (int32_t row = q_rows_per_packet; row < q_output_values_per_lane; row++) {
        lane_output[row] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_gate_up_shard_bf16(const bfloat16 *__restrict packet,
                                       float *__restrict state,
                                       bfloat16 *__restrict lane_output,
                                       int32_t packet_size,
                                       int32_t hidden_size,
                                       int32_t q_rows_per_packet,
                                       int32_t attention_output_base,
                                       int32_t gate_output_base,
                                       int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;

    const int32_t residual_row_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const bfloat16 *attn_residual = packet + 1;
    const bfloat16 *post_norm_weight = attn_residual + hidden_size;
    const bfloat16 *gate_block = post_norm_weight + hidden_size;
    const bfloat16 *up_block = gate_block + q_rows_per_packet * hidden_size;

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        float x = static_cast<float>(attn_residual[i]);
        if (i >= residual_row_base && i < residual_row_base + q_rows_per_packet) {
            x = static_cast<float>(lane_output[attention_output_base + i - residual_row_base]);
        }
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *gate_row = gate_block + row * hidden_size;
        const bfloat16 *up_row = up_block + row * hidden_size;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int32_t i = 0; i < hidden_size; i++) {
            float x = static_cast<float>(attn_residual[i]);
            if (i >= residual_row_base && i < residual_row_base + q_rows_per_packet) {
                x = static_cast<float>(lane_output[attention_output_base + i - residual_row_base]);
            }
            const float xnorm = x * inv_rms * static_cast<float>(post_norm_weight[i]);
            gate_acc += xnorm * static_cast<float>(gate_row[i]);
            up_acc += xnorm * static_cast<float>(up_row[i]);
        }
        checksum += gate_acc + up_acc;
        lane_output[gate_output_base + row] = static_cast<bfloat16>(gate_acc);
        lane_output[gate_output_base + q_rows_per_packet + row] = static_cast<bfloat16>(up_acc);
    }

    for (int32_t i = gate_output_base + 2 * q_rows_per_packet; i < output_values_per_lane; i++) {
        lane_output[i] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_projection_shard_bf16(const bfloat16 *__restrict shared_packet,
                                          const bfloat16 *__restrict lane_packet,
                                          const bfloat16 *__restrict hidden_state,
                                          float *__restrict state,
                                          bfloat16 *__restrict lane_output,
                                          int32_t packet_size,
                                          int32_t hidden_size,
                                          int32_t q_rows_per_packet,
                                          int32_t output_base,
                                          int32_t output_values)
{
    event0();
    (void)packet_size;

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        const float x = static_cast<float>(hidden_state[i]);
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    const bfloat16 *weight = shared_packet + hidden_size;
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *proj_row = lane_packet + row * hidden_size;
        float proj_acc = 0.0f;
        for (int32_t i = 0; i < hidden_size; i++) {
            const float xnorm = static_cast<float>(hidden_state[i]) * inv_rms * static_cast<float>(weight[i]);
            proj_acc += xnorm * static_cast<float>(proj_row[i]);
        }
        checksum += proj_acc;
        lane_output[output_base + row] = static_cast<bfloat16>(proj_acc);
    }

    for (int32_t row = q_rows_per_packet; row < output_values; row++) {
        lane_output[output_base + row] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_norm_rope_shard_bf16(const bfloat16 *__restrict packet,
                                         float *__restrict state,
                                         bfloat16 *__restrict lane_output,
                                         int32_t packet_size,
                                         int32_t head_dim,
                                         int32_t q_rows_per_packet,
                                         int32_t output_base,
                                         int32_t output_values)
{
    event0();
    (void)packet_size;

    const int32_t row_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const bfloat16 *raw_head = packet + 1;
    const bfloat16 *norm_weight = raw_head + head_dim;
    const bfloat16 *cos_values = norm_weight + head_dim;
    const bfloat16 *sin_values = cos_values + head_dim;
    const int32_t half_dim = head_dim / 2;

    float mean_square = 0.0f;
    for (int32_t i = 0; i < head_dim; i++) {
        const float x = static_cast<float>(raw_head[i]);
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(head_dim);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const int32_t idx = row_base + row;
        const int32_t pair_idx = idx < half_dim ? idx + half_dim : idx - half_dim;
        const float x = static_cast<float>(raw_head[idx]) * inv_rms * static_cast<float>(norm_weight[idx]);
        const float pair_x =
            static_cast<float>(raw_head[pair_idx]) * inv_rms * static_cast<float>(norm_weight[pair_idx]);
        const float rotated = idx < half_dim ? -pair_x : pair_x;
        const float out = x * static_cast<float>(cos_values[idx]) + rotated * static_cast<float>(sin_values[idx]);
        checksum += out;
        lane_output[output_base + row] = static_cast<bfloat16>(out);
    }

    for (int32_t row = q_rows_per_packet; row < output_values; row++) {
        lane_output[output_base + row] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_attention_init_f32(float *__restrict attention_state,
                                       float *__restrict attention_acc,
                                       int32_t head_dim)
{
    event0();

    attention_state[0] = -__builtin_inff();
    attention_state[1] = 0.0f;
    for (int32_t dim = 0; dim < head_dim; dim++) {
        attention_acc[dim] = 0.0f;
    }

    event1();
}

void new_mega_phase_attention_update_packed_bf16(const bfloat16 *__restrict packet,
                                                 float *__restrict attention_state,
                                                 float *__restrict attention_acc,
                                                 int32_t chunk_size,
                                                 int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = 32;
    const bfloat16 *__restrict q = packet;
    const bfloat16 *__restrict k = q + head_dim;
    const bfloat16 *__restrict v = k + chunk_size * head_dim;
    const bfloat16 *__restrict mask = v + chunk_size * head_dim;
    float chunk_max = -__builtin_inff();
    bool has_valid = false;

    for (int32_t row = 0; row < chunk_size; row++) {
        if (static_cast<float>(mask[row]) <= 0.5f) {
            continue;
        }

        aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
        const bfloat16 *__restrict k_row = k + row * head_dim;
        for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
            aie::vector<bfloat16, vec_len> q_vec = aie::load_v<vec_len>(q + dim);
            aie::vector<bfloat16, vec_len> k_vec = aie::load_v<vec_len>(k_row + dim);
            dot = aie::mac(dot, q_vec, k_vec);
        }

        float score = aie::reduce_add(dot.template to_vector<float>()) * NEW_MEGA_ATTN_SCALE;
        if (!has_valid || score > chunk_max) {
            chunk_max = score;
            has_valid = true;
        }
    }

    if (!has_valid) {
        event1();
        return;
    }

    float old_max = attention_state[0];
    float old_sum = attention_state[1];
    float new_max = old_max > chunk_max ? old_max : chunk_max;
    float correction = old_sum > 0.0f ? new_mega_exp_approx(old_max - new_max) : 0.0f;
    float chunk_sum = 0.0f;

    for (int32_t dim = 0; dim < head_dim; dim++) {
        attention_acc[dim] *= correction;
    }

    for (int32_t row = 0; row < chunk_size; row++) {
        if (static_cast<float>(mask[row]) <= 0.5f) {
            continue;
        }

        aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
        const bfloat16 *__restrict k_row = k + row * head_dim;
        for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
            aie::vector<bfloat16, vec_len> q_vec = aie::load_v<vec_len>(q + dim);
            aie::vector<bfloat16, vec_len> k_vec = aie::load_v<vec_len>(k_row + dim);
            dot = aie::mac(dot, q_vec, k_vec);
        }

        float score = aie::reduce_add(dot.template to_vector<float>()) * NEW_MEGA_ATTN_SCALE;
        float weight = new_mega_exp_approx(score - new_max);
        chunk_sum += weight;

        const bfloat16 *__restrict v_row = v + row * head_dim;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            attention_acc[dim] += weight * static_cast<float>(v_row[dim]);
        }
    }

    attention_state[0] = new_max;
    attention_state[1] = old_sum * correction + chunk_sum;

    event1();
}

void new_mega_phase_attention_finalize_bf16(const float *__restrict attention_state,
                                            const float *__restrict attention_acc,
                                            float *__restrict state,
                                            bfloat16 *__restrict lane_output,
                                            int32_t context_output_base,
                                            int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    float checksum = state[0];
    float denom = attention_state[1];
    if (denom <= 0.0f) {
        for (int32_t dim = 0; dim < head_dim; dim++) {
            lane_output[context_output_base + dim] = static_cast<bfloat16>(0.0f);
        }
    } else {
        float inv = 1.0f / denom;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            const float out = attention_acc[dim] * inv;
            checksum += out;
            lane_output[context_output_base + dim] = static_cast<bfloat16>(out);
        }
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_o_residual_shard_bf16(const bfloat16 *__restrict packet,
                                          float *__restrict state,
                                          bfloat16 *__restrict lane_output,
                                          int32_t packet_size,
                                          int32_t attention_size,
                                          int32_t q_rows_per_packet,
                                          int32_t context_output_base,
                                          int32_t head_dim,
                                          int32_t attention_output_base,
                                          int32_t attention_output_values_per_lane,
                                          int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;
    (void)output_values_per_lane;

    const int32_t context_head_index0 = static_cast<int32_t>(static_cast<float>(packet[0]));
    const int32_t context_head_index1 = static_cast<int32_t>(static_cast<float>(packet[1]));
    const int32_t context_head_start0 = context_head_index0 * head_dim;
    const int32_t context_head_end0 = context_head_start0 + head_dim;
    const int32_t context_head_start1 = context_head_index1 * head_dim;
    const int32_t context_head_end1 = context_head_start1 + head_dim;
    const bfloat16 *attention_context = packet + 2;
    const bfloat16 *residual_shard = attention_context + attention_size;
    const bfloat16 *o_block = residual_shard + q_rows_per_packet;

    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *o_row = o_block + row * attention_size;
        float o_acc = 0.0f;
        for (int32_t i = 0; i < attention_size; i++) {
            float context_value = static_cast<float>(attention_context[i]);
            if (i >= context_head_start0 && i < context_head_end0) {
                context_value = static_cast<float>(lane_output[context_output_base + i - context_head_start0]);
            } else if (i >= context_head_start1 && i < context_head_end1) {
                context_value =
                    static_cast<float>(lane_output[context_output_base + head_dim + i - context_head_start1]);
            }
            o_acc += context_value * static_cast<float>(o_row[i]);
        }
        const float residual = o_acc + static_cast<float>(residual_shard[row]);
        checksum += residual;
        lane_output[attention_output_base + row] = static_cast<bfloat16>(residual);
    }

    for (int32_t i = attention_output_base + q_rows_per_packet;
         i < attention_output_base + attention_output_values_per_lane;
         i++) {
        lane_output[i] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_down_residual_shard_bf16(const bfloat16 *__restrict packet,
                                             float *__restrict state,
                                             bfloat16 *__restrict lane_output,
                                             int32_t packet_size,
                                             int32_t hidden_size,
                                             int32_t intermediate_size,
                                             int32_t q_rows_per_packet,
                                             int32_t gate_output_base,
                                             int32_t residual_output_base,
                                             int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;
    (void)hidden_size;

    const int32_t ffn_row_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const bfloat16 *ffn_hidden = packet + 1;
    const bfloat16 *residual_shard = ffn_hidden + intermediate_size;
    const bfloat16 *down_block = residual_shard + q_rows_per_packet;

    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    float local_ffn[16];
    for (int32_t local = 0; local < 16; local++) {
        local_ffn[local] = 0.0f;
    }
    for (int32_t local = 0; local < q_rows_per_packet && local < 16; local++) {
        const float gate = static_cast<float>(lane_output[gate_output_base + local]);
        const float up = static_cast<float>(lane_output[gate_output_base + q_rows_per_packet + local]);
        local_ffn[local] = new_mega_silu_approx(gate) * up;
    }

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *down_row = down_block + row * intermediate_size;
        float down_acc = 0.0f;
        for (int32_t i = 0; i < intermediate_size; i++) {
            float ffn_value = static_cast<float>(ffn_hidden[i]);
            if (i >= ffn_row_base && i < ffn_row_base + q_rows_per_packet && i - ffn_row_base < 16) {
                const int32_t local = i - ffn_row_base;
                ffn_value = local_ffn[local];
            }
            down_acc += ffn_value * static_cast<float>(down_row[i]);
        }
        const float residual = down_acc + static_cast<float>(residual_shard[row]);
        checksum += residual;
        lane_output[residual_output_base + row] = static_cast<bfloat16>(residual);
    }

    for (int32_t i = residual_output_base + q_rows_per_packet; i < output_values_per_lane; i++) {
        lane_output[i] = static_cast<bfloat16>(0.0f);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_next_hidden_bf16(const bfloat16 *__restrict packet,
                                     bfloat16 *__restrict hidden_state,
                                     float *__restrict state,
                                     int32_t packet_size,
                                     int32_t hidden_size,
                                     int32_t layer_id)
{
    event0();
    (void)packet_size;
    (void)layer_id;

    float checksum = state[0];
    for (int32_t i = 0; i < hidden_size; i++) {
        hidden_state[i] = packet[i];
        checksum += static_cast<float>(packet[i]);
    }
    state[0] = checksum;
    event1();
}

} // extern "C"
