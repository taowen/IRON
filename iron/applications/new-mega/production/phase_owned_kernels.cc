// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

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
                                       int32_t gate_output_base,
                                       int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;

    const bfloat16 *attn_residual = packet;
    const bfloat16 *post_norm_weight = packet + hidden_size;
    const bfloat16 *gate_block = packet + 2 * hidden_size;
    const bfloat16 *up_block = gate_block + q_rows_per_packet * hidden_size;

    float mean_square = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        const float x = static_cast<float>(attn_residual[i]);
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
            const float xnorm =
                static_cast<float>(attn_residual[i]) * inv_rms * static_cast<float>(post_norm_weight[i]);
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

void new_mega_phase_o_residual_shard_bf16(const bfloat16 *__restrict packet,
                                          float *__restrict state,
                                          bfloat16 *__restrict lane_output,
                                          int32_t packet_size,
                                          int32_t attention_size,
                                          int32_t q_rows_per_packet,
                                          int32_t attention_output_base,
                                          int32_t attention_output_values_per_lane,
                                          int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;
    (void)output_values_per_lane;

    const bfloat16 *attention_context = packet;
    const bfloat16 *residual_shard = packet + attention_size;
    const bfloat16 *o_block = residual_shard + q_rows_per_packet;

    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *o_row = o_block + row * attention_size;
        float o_acc = 0.0f;
        for (int32_t i = 0; i < attention_size; i++) {
            o_acc += static_cast<float>(attention_context[i]) * static_cast<float>(o_row[i]);
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
                                             int32_t residual_output_base,
                                             int32_t output_values_per_lane)
{
    event0();
    (void)packet_size;
    (void)hidden_size;

    const bfloat16 *ffn_hidden = packet;
    const bfloat16 *residual_shard = packet + intermediate_size;
    const bfloat16 *down_block = residual_shard + q_rows_per_packet;

    float checksum = state[0];
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t row = 0; row < q_rows_per_packet; row++) {
        const bfloat16 *down_row = down_block + row * intermediate_size;
        float down_acc = 0.0f;
        for (int32_t i = 0; i < intermediate_size; i++) {
            down_acc += static_cast<float>(ffn_hidden[i]) * static_cast<float>(down_row[i]);
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

void new_mega_phase_packet_accum_bf16(const bfloat16 *__restrict packet,
                                      float *__restrict state,
                                      int32_t packet_size,
                                      int32_t layer_id,
                                      int32_t phase_id)
{
    event0();

    float acc = state[0];
    const float marker = static_cast<float>((layer_id + 1) * 17 + (phase_id + 1));
    for (int32_t i = 0; i < packet_size; i++) {
        acc += static_cast<float>(packet[i]) + marker * 0.000001f;
    }

    state[0] = acc;

    event1();
}

} // extern "C"
