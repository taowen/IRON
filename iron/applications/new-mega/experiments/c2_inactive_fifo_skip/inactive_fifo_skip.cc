// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void c2_phase0_bf16(const bfloat16 *__restrict in, bfloat16 *__restrict state, int32_t size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t i = 0; i < size; i++) {
        float x = static_cast<float>(in[i]);
        state[i] = static_cast<bfloat16>(2.0f * x + 1.0f);
    }

    event1();
}

void c2_finalize_skip_bf16(const bfloat16 *__restrict state, bfloat16 *__restrict out, int32_t size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t i = 0; i < size; i++) {
        float x = static_cast<float>(state[i]);
        out[i] = static_cast<bfloat16>(x + 7.0f);
    }

    event1();
}

void c2_finalize_active_bf16(const bfloat16 *__restrict state,
                             const bfloat16 *__restrict optional,
                             bfloat16 *__restrict out,
                             int32_t size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t i = 0; i < size; i++) {
        float x = static_cast<float>(state[i]);
        float y = static_cast<float>(optional[i]);
        out[i] = static_cast<bfloat16>(x + 3.0f * y + 7.0f);
    }

    event1();
}

} // extern "C"
