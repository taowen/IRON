// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void host_kv_writeback_acc_init_f32(float *__restrict acc, int32_t head_dim)
{
    event0();

    for (int32_t dim = 0; dim < head_dim; dim++) {
        acc[dim] = 0.0f;
    }

    event1();
}

void host_kv_writeback_acc_update_bf16(const bfloat16 *__restrict packed_chunk,
                                       float *__restrict acc,
                                       int32_t chunk_size,
                                       int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const bfloat16 *__restrict k = packed_chunk;
    const bfloat16 *__restrict v = packed_chunk + chunk_size * head_dim;
    const bfloat16 *__restrict mask = packed_chunk + 2 * chunk_size * head_dim;

    for (int32_t row = 0; row < chunk_size; row++) {
        if (static_cast<float>(mask[row]) <= 0.5f) {
            continue;
        }

        const bfloat16 *__restrict k_row = k + row * head_dim;
        const bfloat16 *__restrict v_row = v + row * head_dim;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            acc[dim] += static_cast<float>(k_row[dim]) + 0.5f * static_cast<float>(v_row[dim]);
        }
    }

    event1();
}

void host_kv_writeback_finalize_bf16(const bfloat16 *__restrict current,
                                     const float *__restrict acc,
                                     bfloat16 *__restrict out,
                                     int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    bfloat16 *__restrict present_k = out;
    bfloat16 *__restrict present_v = out + head_dim;
    bfloat16 *__restrict cache_summary = out + 2 * head_dim;

    for (int32_t dim = 0; dim < head_dim; dim++) {
        float x = static_cast<float>(current[dim]);
        present_k[dim] = static_cast<bfloat16>(x + 0.25f);
        present_v[dim] = static_cast<bfloat16>(x - 0.50f);
        cache_summary[dim] = static_cast<bfloat16>(acc[dim]);
    }

    event1();
}

} // extern "C"
