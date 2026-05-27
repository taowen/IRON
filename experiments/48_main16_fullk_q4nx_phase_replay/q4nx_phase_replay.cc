#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int M_PER_TILE = 32;
constexpr int HIDDEN_DIM = 4096;
constexpr int Q4_K_CHUNK = 256;
constexpr int GROUP_SIZE = 32;
constexpr int NUM_PHASES = 4;

static float accum[M_PER_TILE];

static inline bfloat16 activation_value(int32_t group, int32_t row, int32_t idx) {
    int32_t raw = ((group * 17 + row * 13 + idx * 5) % 31) - 15;
    return static_cast<bfloat16>(static_cast<float>(raw) / 32.0f);
}

} // namespace

extern "C" {

void init_activation(bfloat16 *activation, int32_t group, int32_t row) {
    for (int idx = 0; idx < HIDDEN_DIM; idx++) {
        activation[idx] = activation_value(group, row, idx);
    }
}

void q4nx_chunk_accum_offset(
    bfloat16 *packed_chunk,
    bfloat16 *activation,
    int32_t act_offset,
    int32_t num_rows
) {
    constexpr int groups_per_row = Q4_K_CHUNK / GROUP_SIZE;
    constexpr int num_scales = M_PER_TILE * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);

    bfloat16 *act_slice = activation + act_offset;

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

            aie::vector<bfloat16, GROUP_SIZE> act = aie::load_v<GROUP_SIZE>(act_slice + group * GROUP_SIZE);
            aie::accum<accfloat, GROUP_SIZE> mac_acc = aie::mul(dequant, act);
            aie::vector<float, GROUP_SIZE> mac_f32 = mac_acc.to_vector<float>();

            row_acc += aie::reduce_add(mac_f32);
        }

        accum[row] += row_acc;
    }
}

void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

void combine_q4nx_phases(
    bfloat16 *o_phase,
    bfloat16 *gate_phase,
    bfloat16 *up_phase,
    bfloat16 *down_phase,
    bfloat16 *output,
    int32_t group,
    int32_t row
) {
    for (int lane = 0; lane < M_PER_TILE; lane++) {
        float o = static_cast<float>(o_phase[lane]);
        float gate = static_cast<float>(gate_phase[lane]);
        float up = static_cast<float>(up_phase[lane]);
        float down = static_cast<float>(down_phase[lane]);
        float swiglu = (gate * up) / 128.0f;
        output[lane] = static_cast<bfloat16>(o + swiglu + down * 0.25f + group * 0.5f + row * 0.125f);
    }
}

} // extern "C"
