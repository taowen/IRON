#include <aie_api/aie.hpp>
#include <stdint.h>

#include "record_format.h"

namespace {

constexpr int32_t kOutputBlocks = 8;

static float accum[qwen3::kMainRowsPerTile];
static float block_accum[kOutputBlocks][qwen3::kMainRowsPerTile];

static inline void q4nx_chunk_accum_slice(
    float *target,
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice,
    int32_t num_rows
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr int groups_per_row = qwen3::kQ4KChunk / qwen3::kQ4GroupSize;
    constexpr int rows_per_lane = qwen3::kMainRowsPerTile / 2;
    constexpr int row_pair_bytes = rows_per_lane / 2;
    constexpr int bytes_per_lane = qwen3::kQ4KChunk * row_pair_bytes;

    bfloat16 *scales = packed_chunk;
    bfloat16 *zeros = packed_chunk + qwen3::kMainRowsPerTile * groups_per_row;
    uint8_t *data = reinterpret_cast<uint8_t *>(
        packed_chunk + 2 * qwen3::kMainRowsPerTile * groups_per_row
    );

    for (int lane = 0; lane < 2; lane++) {
        const int row_base = lane * rows_per_lane;
        aie::accum<accfloat, rows_per_lane> row_acc = aie::zeros<accfloat, rows_per_lane>();
        for (int group = 0; group < groups_per_row; group++) {
            aie::vector<bfloat16, rows_per_lane> scale_vec =
                aie::load_v<rows_per_lane>(scales + group * qwen3::kMainRowsPerTile + row_base);
            aie::vector<bfloat16, rows_per_lane> zero_vec =
                aie::load_v<rows_per_lane>(zeros + group * qwen3::kMainRowsPerTile + row_base);
            for (int dim = 0; dim < qwen3::kQ4GroupSize; dim += 2) {
                const int col = group * qwen3::kQ4GroupSize + dim;
                uint4 *column_nibbles = reinterpret_cast<uint4 *>(
                    data + lane * bytes_per_lane + col * row_pair_bytes
                );
                aie::vector<uint4, qwen3::kMainRowsPerTile> packed =
                    aie::load_v<qwen3::kMainRowsPerTile>(column_nibbles);
                aie::vector<uint8, qwen3::kMainRowsPerTile> as_u8 = aie::unpack(packed);
                aie::vector<uint16, qwen3::kMainRowsPerTile> as_u16 = aie::unpack(as_u8);
                aie::vector<bfloat16, qwen3::kMainRowsPerTile> q_values =
                    aie::to_float<bfloat16>(as_u16, 0);

                aie::vector<bfloat16, rows_per_lane> q0 = q_values.extract<rows_per_lane>(0);
                aie::vector<bfloat16, rows_per_lane> shifted0 = aie::sub(q0, zero_vec);
                aie::accum<accfloat, rows_per_lane> dequant0_acc = aie::mul(shifted0, scale_vec);
                aie::vector<bfloat16, rows_per_lane> dequant0 = dequant0_acc.to_vector<bfloat16>();
                aie::vector<bfloat16, rows_per_lane> activation0 =
                    aie::broadcast<bfloat16, rows_per_lane>(activation_slice[col]);
                row_acc = aie::mac(row_acc, dequant0, activation0);

                aie::vector<bfloat16, rows_per_lane> q1 = q_values.extract<rows_per_lane>(1);
                aie::vector<bfloat16, rows_per_lane> shifted1 = aie::sub(q1, zero_vec);
                aie::accum<accfloat, rows_per_lane> dequant1_acc = aie::mul(shifted1, scale_vec);
                aie::vector<bfloat16, rows_per_lane> dequant1 = dequant1_acc.to_vector<bfloat16>();
                aie::vector<bfloat16, rows_per_lane> activation1 =
                    aie::broadcast<bfloat16, rows_per_lane>(activation_slice[col + 1]);
                row_acc = aie::mac(row_acc, dequant1, activation1);
            }
        }

        aie::accum<accfloat, rows_per_lane> merged;
        merged.from_vector(aie::load_v<rows_per_lane>(target + row_base), 0);
        merged = aie::add(merged, row_acc.to_vector<float>());
        aie::store_v(target + row_base, merged.to_vector<float>());
    }

    (void)num_rows;
}

static inline void q4nx_chunk_accum_single(
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice,
    int32_t num_rows
) {
    q4nx_chunk_accum_slice(accum, packed_chunk, activation_slice, num_rows);
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
    q4nx_chunk_accum_single(packed_chunk, reinterpret_cast<bfloat16 *>(activation_words), num_rows);
}

void q4nx_clear_block_summaries(int32_t blocks, int32_t num_rows) {
    for (int32_t block = 0; block < blocks && block < kOutputBlocks; block++) {
        for (int32_t row = 0; row < num_rows; row++) {
            block_accum[block][row] = 0.0f;
        }
    }
}

void q4nx_chunk_accum_block_slice_i32(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t block,
    int32_t num_rows
) {
    if (block < 0 || block >= kOutputBlocks) {
        return;
    }
    q4nx_chunk_accum_slice(
        block_accum[block],
        packed_chunk,
        reinterpret_cast<bfloat16 *>(activation_words),
        num_rows
    );
}

void q4nx_flush_output(bfloat16 *output, int32_t num_rows) {
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(accum[row]);
        accum[row] = 0.0f;
    }
}

void q4nx_flush_block_output(bfloat16 *output, int32_t block, int32_t num_rows) {
    if (block < 0 || block >= kOutputBlocks) {
        return;
    }
    for (int row = 0; row < num_rows; row++) {
        output[row] = static_cast<bfloat16>(block_accum[block][row]);
        block_accum[block][row] = 0.0f;
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

void q4nx_emit_o_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t block,
    int32_t num_rows
) {
    emit_body_record(records, output, qwen3::kOPhase, block, group, row, num_rows);
}

void q4nx_emit_down_body_record(
    int32_t *records,
    bfloat16 *output,
    int32_t group,
    int32_t row,
    int32_t block,
    int32_t num_rows
) {
    emit_body_record(records, output, qwen3::kDownPhase, block, group, row, num_rows);
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
