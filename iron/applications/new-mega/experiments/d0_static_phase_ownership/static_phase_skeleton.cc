// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void d0_phase_state_init_f32(float *__restrict state)
{
    event0();
    state[0] = 0.0f;
    event1();
}

void d0_phase_packet_accum_bf16(const bfloat16 *__restrict packet, float *__restrict state, int32_t packet_size)
{
    event0();

    float acc = state[0];
    for (int32_t i = 0; i < packet_size; i++) {
        acc += static_cast<float>(packet[i]);
    }
    state[0] = acc;

    event1();
}

void d0_phase_state_finalize_bf16(const float *__restrict state, bfloat16 *__restrict out)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    out[0] = static_cast<bfloat16>(state[0]);
    for (int32_t i = 1; i < 8; i++) {
        out[i] = static_cast<bfloat16>(0.0f);
    }
    event1();
}

} // extern "C"
