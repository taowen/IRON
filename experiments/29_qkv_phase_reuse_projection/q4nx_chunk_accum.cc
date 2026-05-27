#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef Q4_M
#define Q4_M 32
#endif
#ifndef Q4_K_CHUNK
#define Q4_K_CHUNK 256
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

static float accum[Q4_M];

extern "C" {

void q4nx_chunk_accum_offset(
    bfloat16 *packed_chunk,
    bfloat16 *activation,
    int32_t act_offset,
    int32_t num_rows
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = Q4_M * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros  = packed_chunk + num_scales;
    uint4 *data      = (uint4 *)(packed_chunk + 2 * num_scales);

    bfloat16 *act_slice = activation + act_offset;

    for (int row = 0; row < num_rows; row++) {
        float row_acc = 0.0f;

        for (int g = 0; g < groups_per_row; g++) {
            bfloat16 s = scales[row * groups_per_row + g];
            bfloat16 z = zeros[row * groups_per_row + g];

            uint4 *group_ptr = data + row * (Q4_K_CHUNK / 2) + g * (GROUP_SIZE / 2);
            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);

            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> z_vec = aie::broadcast<bfloat16, GROUP_SIZE>(z);
            aie::vector<bfloat16, GROUP_SIZE> s_vec = aie::broadcast<bfloat16, GROUP_SIZE>(s);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, z_vec);
            aie::accum<accfloat, GROUP_SIZE> dq_acc = aie::mul(shifted, s_vec);
            aie::vector<bfloat16, GROUP_SIZE> dq = dq_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(act_slice + g * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dq, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();

            row_acc += aie::reduce_add(mac_f32);
        }

        accum[row] += row_acc;
    }
}

void q4nx_flush_output(
    bfloat16 *output,
    int32_t num_rows
) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

} // extern "C"
