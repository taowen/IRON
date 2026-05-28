#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int M_PER_TILE = 32;
constexpr int ACT_SLICE_BF16 = 256;
constexpr int Q4_K_CHUNK = 256;
constexpr int GROUP_SIZE = 32;
constexpr int OUT_RECORD_BF16 = M_PER_TILE + 2;

static float accum[M_PER_TILE];

static inline bfloat16 activation_value(int32_t group, int32_t row, int32_t chunk, int32_t idx) {
    int32_t raw = (group * 17 + row * 19 + chunk * 23 + idx * 7 + 5) % 127;
    return static_cast<bfloat16>(static_cast<float>(raw - 63) / 64.0f);
}

} // namespace

extern "C" {

void edge_make_activation_slice(
    bfloat16 *activation_slice,
    int32_t group,
    int32_t row,
    int32_t chunk
) {
    for (int idx = 0; idx < ACT_SLICE_BF16; idx++) {
        activation_slice[idx] = activation_value(group, row, chunk, idx);
    }
}

void q4nx_chunk_accum_slice(
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice,
    int32_t num_rows
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = M_PER_TILE * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);

    for (int row = 0; row < num_rows; row++) {
        float row_acc = 0.0f;

        for (int group = 0; group < groups_per_row; group++) {
            bfloat16 scale = scales[row * groups_per_row + group];
            bfloat16 zero = zeros[row * groups_per_row + group];
            uint4 *group_ptr = data + row * (Q4_K_CHUNK / 2) + group * (GROUP_SIZE / 2);

            aie::vector<uint4, GROUP_SIZE> packed = aie::load_v<GROUP_SIZE>(group_ptr);
            aie::vector<uint8, GROUP_SIZE> as_u8 = aie::unpack(packed);
            aie::vector<uint16, GROUP_SIZE> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, GROUP_SIZE> as_bf16 = aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, GROUP_SIZE> zero_vec = aie::broadcast<bfloat16, GROUP_SIZE>(zero);
            aie::vector<bfloat16, GROUP_SIZE> scale_vec = aie::broadcast<bfloat16, GROUP_SIZE>(scale);
            aie::vector<bfloat16, GROUP_SIZE> shifted = aie::sub(as_bf16, zero_vec);
            aie::accum<accfloat, GROUP_SIZE> dequant_acc = aie::mul(shifted, scale_vec);
            aie::vector<bfloat16, GROUP_SIZE> dequant = dequant_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(activation_slice + group * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dequant, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();

            row_acc += aie::reduce_add(mac_f32);
        }

        accum[row] += row_acc;
    }
}

void flush_projection_output(
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    output[0] = static_cast<bfloat16>(static_cast<float>(group));
    output[1] = static_cast<bfloat16>(static_cast<float>(row));
    for (int idx = 0; idx < num_rows; idx++) {
        output[2 + idx] = static_cast<bfloat16>(accum[idx]);
        accum[idx] = 0.0f;
    }
    for (int idx = 2 + num_rows; idx < OUT_RECORD_BF16; idx++) {
        output[idx] = static_cast<bfloat16>(0.0f);
    }
}

} // extern "C"
