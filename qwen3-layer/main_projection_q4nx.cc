#include <aie_api/aie.hpp>
#include <stdint.h>

#include "record_format.h"

namespace {

static float accum[qwen3::kMainRowsPerTile];

static inline void q4nx_chunk_accum_slice(
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice,
    int32_t num_rows
) {
    constexpr int groups_per_row = qwen3::kQ4KChunk / qwen3::kQ4GroupSize;
    constexpr int num_scales = qwen3::kMainRowsPerTile * groups_per_row;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + num_scales;
    uint4 *data = reinterpret_cast<uint4 *>(packed_chunk + 2 * num_scales);

    for (int row = 0; row < num_rows; row++) {
        float row_acc = 0.0f;

        for (int group = 0; group < groups_per_row; group++) {
            bfloat16 scale = scales[row * groups_per_row + group];
            bfloat16 zero = zeros[row * groups_per_row + group];
            uint4 *group_ptr =
                data + row * (qwen3::kQ4KChunk / 2) + group * (qwen3::kQ4GroupSize / 2);

            aie::vector<uint4, qwen3::kQ4GroupSize> packed =
                aie::load_v<qwen3::kQ4GroupSize>(group_ptr);
            aie::vector<uint8, qwen3::kQ4GroupSize> as_u8 = aie::unpack(packed);
            aie::vector<uint16, qwen3::kQ4GroupSize> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, qwen3::kQ4GroupSize> as_bf16 =
                aie::to_float<bfloat16>(as_u16, 0);

            aie::vector<bfloat16, qwen3::kQ4GroupSize> zero_vec =
                aie::broadcast<bfloat16, qwen3::kQ4GroupSize>(zero);
            aie::vector<bfloat16, qwen3::kQ4GroupSize> scale_vec =
                aie::broadcast<bfloat16, qwen3::kQ4GroupSize>(scale);
            aie::vector<bfloat16, qwen3::kQ4GroupSize> shifted = aie::sub(as_bf16, zero_vec);
            aie::accum<accfloat, qwen3::kQ4GroupSize> dequant_acc =
                aie::mul(shifted, scale_vec);
            aie::vector<bfloat16, qwen3::kQ4GroupSize> dequant =
                dequant_acc.to_vector<bfloat16>();

            aie::vector<bfloat16, qwen3::kQ4GroupSize> act =
                aie::load_v<qwen3::kQ4GroupSize>(
                    activation_slice + group * qwen3::kQ4GroupSize
                );
            aie::accum<accfloat, qwen3::kQ4GroupSize> mac_acc = aie::mul(dequant, act);
            aie::vector<float, qwen3::kQ4GroupSize> mac_f32 = mac_acc.to_vector<float>();

            row_acc += aie::reduce_add(mac_f32);
        }

        accum[row] += row_acc;
    }
}

static inline void write_record_payload(bfloat16 *payload, bfloat16 *output, int32_t num_rows) {
    for (int32_t idx = 0; idx < qwen3::kRecordPayloadBf16; idx++) {
        payload[idx] = idx < num_rows ? output[idx] : static_cast<bfloat16>(0.0f);
    }
}

static inline void emit_projection_record(
    int32_t *records,
    bfloat16 *output,
    int32_t phase,
    int32_t offset,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    records[offset] = qwen3::projection_record_header(phase, group, row);
    write_record_payload(qwen3::record_payload_bf16(records + offset), output, num_rows);
}

static inline void emit_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t phase,
    int32_t block,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    const int32_t offset = block * qwen3::kRecordDwords;
    records[offset] = qwen3::body_record_header(phase, block, group, row);
    write_record_payload(qwen3::record_payload_bf16(records + offset), output, num_rows);
}

} // namespace

extern "C" {

void clear_summary(bfloat16 *summary, int32_t num_rows) {
    for (int idx = 0; idx < num_rows; idx++) {
        summary[idx] = static_cast<bfloat16>(0.0f);
        accum[idx] = 0.0f;
    }
}

void q4nx_chunk_accum_slice_i32(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t num_rows
) {
    q4nx_chunk_accum_slice(packed_chunk, reinterpret_cast<bfloat16 *>(activation_words), num_rows);
}

void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

void q4nx_emit_down_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    emit_projection_record(
        records,
        output,
        qwen3::kDownPhase,
        qwen3::kDownPhase * qwen3::kRecordDwords,
        group,
        row,
        num_rows
    );
}

void q4nx_emit_o_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    emit_projection_record(
        records,
        output,
        qwen3::kOPhase,
        qwen3::kOPhase * qwen3::kRecordDwords,
        group,
        row,
        num_rows
    );
}

void q4nx_emit_q_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t block,
    int32_t num_rows
) {
    emit_body_record(records, output, qwen3::kQPhase, block, group, row, num_rows);
}

void q4nx_emit_k_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t block,
    int32_t num_rows
) {
    emit_body_record(records, output, qwen3::kKPhase, block, group, row, num_rows);
}

void q4nx_emit_v_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t block,
    int32_t num_rows
) {
    emit_body_record(records, output, qwen3::kVPhase, block, group, row, num_rows);
}

void q4nx_emit_upgate_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t replay,
    int32_t num_rows
) {
    const int32_t phase = (replay & 1) == 0 ? qwen3::kUpPhase : qwen3::kGatePhase;
    emit_projection_record(
        records,
        output,
        phase,
        replay * qwen3::kRecordDwords,
        group,
        row,
        num_rows
    );
}

} // extern "C"
