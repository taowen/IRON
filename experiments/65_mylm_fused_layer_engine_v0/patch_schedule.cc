#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int M_PER_TILE = 32;
constexpr int ACT_SLICE_BF16 = 256;
constexpr int Q4_K_CHUNK = 256;
constexpr int GROUP_SIZE = 32;
constexpr int RECORD_DWORDS = 17;
constexpr int RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1;
constexpr int OUT_RECORD_BF16 = M_PER_TILE + 2;
constexpr int O_PHASE = 3;

static float accum[M_PER_TILE];

static inline int32_t record_header(int32_t tag, int32_t group, int32_t row) {
    return 0x54000000 | (tag << 16) | (group << 12) | (row << 4) | 0xA;
}

static inline float summary_scale(int32_t phase, int32_t block) {
    return 0.0078125f * static_cast<float>((phase + 1) * ((block & 3) + 1));
}

static inline void add_to_summary(
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t num_rows
) {
    float scale = summary_scale(phase, block);
    for (int idx = 0; idx < num_rows; idx++) {
        float next = static_cast<float>(summary[idx]) + static_cast<float>(phase_out[idx]) * scale;
        summary[idx] = static_cast<bfloat16>(next);
    }
}

static inline bfloat16 record_value(
    int32_t *record,
    int32_t phase,
    int32_t block,
    int32_t chunk_idx,
    int32_t group,
    int32_t row,
    int32_t global_idx
) {
    int32_t lane = global_idx & (RECORD_PAYLOAD_DWORDS - 1);
    int32_t a = record[1 + lane];
    int32_t b = record[1 + ((lane + phase + block + 5) & (RECORD_PAYLOAD_DWORDS - 1))];
    int32_t raw = (
        a * (phase + 3) + b * 2 + global_idx * 7 + chunk_idx * 11 +
        block * 13 + group * 17 + row * 19 + phase * 23
    ) % 127;
    return static_cast<bfloat16>(static_cast<float>(raw - 63) / 64.0f);
}

static inline bfloat16 attention_output_value(int32_t global_idx) {
    int32_t head = global_idx / 128;
    int32_t dim = global_idx - head * 128;
    int32_t raw = (head * 11 + dim * 7 + global_idx * 3 + 19) % 127;
    return static_cast<bfloat16>(static_cast<float>(raw - 63) / 64.0f);
}

} // namespace

extern "C" {

void clear_summary(bfloat16 *summary, int32_t num_rows) {
    for (int idx = 0; idx < num_rows; idx++) {
        summary[idx] = static_cast<bfloat16>(0.0f);
        accum[idx] = 0.0f;
    }
}

void emit_seed_sideband(int32_t *record, int32_t group, int32_t row) {
    record[0] = record_header(0, group, row);
    for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
        record[1 + lane] = group * 37 + row * 11 + lane * 3 + 1;
    }
}

void emit_next_block_sideband(
    int32_t *record,
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    add_to_summary(phase_out, summary, phase, block, num_rows);

    int32_t tag = phase * 32 + block + 1;
    record[0] = record_header(tag, group, row);
    for (int lane = 0; lane < RECORD_PAYLOAD_DWORDS; lane++) {
        record[1 + lane] = group * 37 + row * 11 + (phase + 1) * 101 + (block + 1) * 17 + lane * 3 + 1;
    }
}

void accumulate_block_summary(
    bfloat16 *phase_out,
    bfloat16 *summary,
    int32_t phase,
    int32_t block,
    int32_t num_rows
) {
    add_to_summary(phase_out, summary, phase, block, num_rows);
}

void edge_make_block_slice(
    int32_t *record,
    bfloat16 *activation_slice,
    int32_t phase,
    int32_t block,
    int32_t chunk_idx,
    int32_t group,
    int32_t row
) {
    int32_t base = chunk_idx * ACT_SLICE_BF16;
    for (int idx = 0; idx < ACT_SLICE_BF16; idx++) {
        if (phase == O_PHASE) {
            activation_slice[idx] = attention_output_value(base + idx);
        } else {
            activation_slice[idx] = record_value(record, phase, block, chunk_idx, group, row, base + idx);
        }
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

void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

void flush_schedule_output_with_header(
    bfloat16 *output,
    bfloat16 *summary,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    output[0] = static_cast<bfloat16>(static_cast<float>(group));
    output[1] = static_cast<bfloat16>(static_cast<float>(row));
    for (int idx = 0; idx < num_rows; idx++) {
        output[2 + idx] = summary[idx];
    }
    for (int idx = 2 + num_rows; idx < OUT_RECORD_BF16; idx++) {
        output[idx] = static_cast<bfloat16>(0.0f);
    }
}

} // extern "C"
