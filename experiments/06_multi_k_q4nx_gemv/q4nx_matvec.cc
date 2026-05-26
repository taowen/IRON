#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef Q4_M
#define Q4_M 32
#endif
#ifndef Q4_K
#define Q4_K 256
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

extern "C" {

void q4nx_zero_f32(float *acc, int32_t n) {
    for (int i = 0; i < n; i++) {
        acc[i] = 0.0f;
    }
}

void q4nx_mac_f32(
    bfloat16 *packed_bf16,
    bfloat16 *activation,
    float *acc_output,
    int32_t num_rows
) {
    constexpr int groups_per_row = Q4_K / GROUP_SIZE;
    constexpr int num_scales = Q4_M * groups_per_row;

    bfloat16 *scales = packed_bf16;
    bfloat16 *zeros  = packed_bf16 + num_scales;
    uint4 *data      = (uint4 *)(packed_bf16 + 2 * num_scales);

    for (int row = 0; row < num_rows; row++) {
        float row_acc = acc_output[row];

        for (int g = 0; g < groups_per_row; g++) {
            bfloat16 s = scales[row * groups_per_row + g];
            bfloat16 z = zeros[row * groups_per_row + g];

            uint4 *group_ptr = data + row * (Q4_K / 2) + g * (GROUP_SIZE / 2);
            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);

            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> z_vec = aie::broadcast<bfloat16, GROUP_SIZE>(z);
            aie::vector<bfloat16, GROUP_SIZE> s_vec = aie::broadcast<bfloat16, GROUP_SIZE>(s);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, z_vec);
            aie::accum<accfloat, GROUP_SIZE> dq_acc = aie::mul(shifted, s_vec);
            aie::vector<bfloat16, GROUP_SIZE> dq = dq_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(activation + g * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dq, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();
            row_acc += aie::reduce_add(mac_f32);
        }

        acc_output[row] = row_acc;
    }
}

void q4nx_convert_f32_bf16(float *input, bfloat16 *output, int32_t n) {
    for (int i = 0; i < n; i++) {
        output[i] = static_cast<bfloat16>(input[i]);
    }
}

} // extern "C"
