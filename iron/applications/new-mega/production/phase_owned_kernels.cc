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

void new_mega_phase0_q_shard_bf16(const bfloat16 *__restrict lane_packet,
                                  bfloat16 *__restrict hidden_state,
                                  bfloat16 *__restrict norm_weight_state,
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
            hidden_state[i] = lane_packet[i];
        }
    }
    for (int32_t i = 0; i < hidden_size; i++) {
        norm_weight_state[i] = lane_packet[hidden_size + i];
    }

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        const float x = static_cast<float>(hidden_state[i]);
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    const bfloat16 *weight = norm_weight_state;
    const bfloat16 *q_block = lane_packet + 2 * hidden_size;
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *q_row = q_block + row * hidden_size;
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
                                       float *__restrict ffn_partial,
                                       int32_t packet_size,
                                       int32_t hidden_size,
                                       int32_t q_rows_per_packet,
                                       int32_t residual_group_size,
                                       int32_t ffn_group_size,
                                       int32_t attention_output_base)
{
    event0();
    const int32_t residual_row_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const int32_t ffn_row_base = static_cast<int32_t>(static_cast<float>(packet[packet_size - 1]));
    const bfloat16 *attn_residual = packet + 1;
    const bfloat16 *post_norm_weight = attn_residual + hidden_size;
    const bfloat16 *gate_block = post_norm_weight + hidden_size;
    const bfloat16 *up_block = gate_block + q_rows_per_packet * hidden_size;

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        float x = static_cast<float>(attn_residual[i]);
        if (i >= residual_row_base && i < residual_row_base + residual_group_size) {
            x = static_cast<float>(lane_output[attention_output_base + i - residual_row_base]);
        }
        mean_square += x * x;
    }
    mean_square /= static_cast<float>(hidden_size);

    const float inv_rms = aie::invsqrt(mean_square + 0.000001f);
    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < ffn_group_size; row++) {
        ffn_partial[row] = 0.0f;
    }

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *gate_row = gate_block + row * hidden_size;
        const bfloat16 *up_row = up_block + row * hidden_size;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int32_t i = 0; i < hidden_size; i++) {
            float x = static_cast<float>(attn_residual[i]);
            if (i >= residual_row_base && i < residual_row_base + residual_group_size) {
                x = static_cast<float>(lane_output[attention_output_base + i - residual_row_base]);
            }
            const float xnorm = x * inv_rms * static_cast<float>(post_norm_weight[i]);
            gate_acc += xnorm * static_cast<float>(gate_row[i]);
            up_acc += xnorm * static_cast<float>(up_row[i]);
        }
        checksum += gate_acc + up_acc;

        const float gate_bf16 = static_cast<float>(static_cast<bfloat16>(gate_acc));
        const float up_bf16 = static_cast<float>(static_cast<bfloat16>(up_acc));
        ffn_partial[ffn_row_base + row] = new_mega_silu_approx(gate_bf16) * up_bf16;
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_projection_shard_bf16(const bfloat16 *__restrict lane_packet,
                                          const bfloat16 *__restrict hidden_state,
                                          const bfloat16 *__restrict norm_weight_state,
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
    const bfloat16 *weight = norm_weight_state;
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

void new_mega_phase_o_partial_shard_bf16(const bfloat16 *__restrict packet,
                                         float *__restrict state,
                                         const bfloat16 *__restrict lane_output,
                                         float *__restrict partial_output,
                                         int32_t packet_size,
                                         int32_t head_dim,
                                         int32_t q_rows_per_packet,
                                         int32_t residual_lane_count,
                                         int32_t target_lane_count,
                                         int32_t context_output_base)
{
    event0();
    (void)packet_size;

    const int32_t group_rows = target_lane_count * q_rows_per_packet;
    const int32_t residual_group_rows = residual_lane_count * q_rows_per_packet;
    const bfloat16 *o_local_weight_block = packet + 2 + residual_group_rows;

    float checksum = state[0];

    for (int32_t row = 0; row < group_rows; row++) {
        const bfloat16 *o_row = o_local_weight_block + row * 2 * head_dim;
        float partial = 0.0f;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            partial += static_cast<float>(lane_output[context_output_base + dim]) * static_cast<float>(o_row[dim]);
        }
        for (int32_t dim = 0; dim < head_dim; dim++) {
            partial += static_cast<float>(lane_output[context_output_base + head_dim + dim]) *
                       static_cast<float>(o_row[head_dim + dim]);
        }
        partial_output[row] = partial;
        checksum += partial;
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_o_source_reduce_targets_f32(const float *__restrict group_partials,
                                                float *__restrict target0_reduced,
                                                float *__restrict target1_reduced,
                                                int32_t fabric_group_size,
                                                int32_t target_rows)
{
    event0();

    for (int32_t row = 0; row < target_rows; row++) {
        float acc = 0.0f;
        for (int32_t lane = 0; lane < fabric_group_size; lane++) {
            const int32_t base = lane * target_rows;
            acc += group_partials[base + row];
        }
        target0_reduced[row] = acc;
        target1_reduced[row] = acc;
    }

    event1();
}

void new_mega_phase_o_reduce_two_sources_f32(const float *__restrict source0,
                                             const float *__restrict source1,
                                             float *__restrict group_reduced,
                                             int32_t target_rows)
{
    event0();

    for (int32_t row = 0; row < target_rows; row++) {
        group_reduced[row] = source0[row] + source1[row];
    }

    event1();
}

void new_mega_phase_o_finalize_reduced_bf16(const bfloat16 *__restrict packet,
                                            const float *__restrict reduced_rows,
                                            float *__restrict state,
                                            bfloat16 *__restrict lane_output,
                                            int32_t packet_size,
                                            int32_t q_rows_per_packet,
                                            int32_t target_lane_count,
                                            int32_t attention_output_base,
                                            int32_t attention_output_values_per_lane)
{
    event0();
    (void)packet_size;
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const int32_t chunk_row_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const int32_t group_rows = target_lane_count * q_rows_per_packet;
    const bfloat16 *residual_group = packet + 2;
    float checksum = state[0];

    for (int32_t row = 0; row < group_rows; row++) {
        const float residual = static_cast<float>(residual_group[row]) + reduced_rows[row];
        checksum += residual;
        lane_output[attention_output_base + chunk_row_base + row] = static_cast<bfloat16>(residual);
    }

    state[0] = checksum;
    event1();
}

void new_mega_phase_down_residual_shard_bf16(const bfloat16 *__restrict packet,
                                             const float *__restrict ffn_reduced,
                                             float *__restrict down_acc,
                                             float *__restrict state,
                                             bfloat16 *__restrict lane_output,
                                             int32_t intermediate_size,
                                             int32_t q_rows_per_packet,
                                             int32_t ffn_group_size,
                                             int32_t ffn_npu_rows,
                                             int32_t residual_output_base,
                                             int32_t output_values_per_lane)
{
    event0();

    const int32_t ffn_chunk_base = static_cast<int32_t>(static_cast<float>(packet[0]));
    const bfloat16 *ffn_hidden = packet + 1;
    const bfloat16 *residual_shard = ffn_hidden + intermediate_size;
    const bfloat16 *down_block = residual_shard + q_rows_per_packet;

    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    if (ffn_chunk_base == 0) {
        for (int32_t row = 0; row < q_rows_per_packet; row++) {
            down_acc[row] = 0.0f;
        }
    }

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *down_row = down_block + row * intermediate_size;
        float partial_acc = down_acc[row];
        for (int32_t i = 0; i < ffn_group_size; i++) {
            const int32_t ffn_idx = ffn_chunk_base + i;
            partial_acc += ffn_reduced[i] * static_cast<float>(down_row[ffn_idx]);
        }
        down_acc[row] = partial_acc;
    }

    if (ffn_chunk_base + ffn_group_size >= ffn_npu_rows) {
        for (int32_t row = 0; row < q_rows_per_packet; row++) {
            const bfloat16 *down_row = down_block + row * intermediate_size;
            float final_acc = down_acc[row];
            for (int32_t i = ffn_npu_rows; i < intermediate_size; i++) {
                final_acc += static_cast<float>(ffn_hidden[i]) * static_cast<float>(down_row[i]);
            }
            const float residual = final_acc + static_cast<float>(residual_shard[row]);
            checksum += residual;
            lane_output[residual_output_base + row] = static_cast<bfloat16>(residual);
        }

        for (int32_t i = residual_output_base + q_rows_per_packet; i < output_values_per_lane; i++) {
            lane_output[i] = static_cast<bfloat16>(0.0f);
        }
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
