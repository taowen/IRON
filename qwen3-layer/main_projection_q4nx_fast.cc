#include <aie_api/aie.hpp>
#include <stdint.h>

#include "record_format.h"

namespace {

constexpr int32_t kGroupsPerRow = qwen3::kQ4KChunk / qwen3::kQ4GroupSize;
constexpr int32_t kRowsPerLane = qwen3::kMainRowsPerTile / 2;
constexpr int32_t kRowPairBytes = kRowsPerLane / 2;
constexpr int32_t kBytesPerLane = qwen3::kQ4KChunk * kRowPairBytes;
#ifndef QWEN3_MAIN16_UNROLLED_GROUP_DIMS
#define QWEN3_MAIN16_UNROLLED_GROUP_DIMS 18
#endif
constexpr int32_t kUnrolledGroupDims = QWEN3_MAIN16_UNROLLED_GROUP_DIMS;

static float accum[qwen3::kMainRowsPerTile];

static inline bfloat16 *select_weight_buffer(
    int32_t weight_chunk,
    bfloat16 *wt_ping,
    bfloat16 *wt_pong
) {
    return (weight_chunk & 1) == 0 ? wt_ping : wt_pong;
}

static inline int32_t *select_activation_buffer(
    int32_t activation_chunk,
    int32_t *chunk_ping,
    int32_t *chunk_pong
) {
    return (activation_chunk & 1) == 0 ? chunk_ping : chunk_pong;
}

static inline int32_t *select_record_buffer(
    int32_t block,
    int32_t *record_ping,
    int32_t *record_pong
) {
    return (block & 1) == 0 ? record_ping : record_pong;
}

static inline void acquire_activation_and_weight() {
    acquire_greater_equal(qwen3::kMainActivationFullCoreLock, 1);
    acquire_greater_equal(qwen3::kMainWeightFullCoreLock, 1);
}

static inline void release_activation_and_weight() {
    release(qwen3::kMainActivationEmptyCoreLock, 1);
    release(qwen3::kMainWeightEmptyCoreLock, 1);
}

static inline void acquire_record_buffer() {
    acquire_greater_equal(qwen3::kMainRecordEmptyCoreLock, 1);
}

static inline void release_record_buffer() {
    release(qwen3::kMainRecordFullCoreLock, 1);
}

__attribute__((noinline)) static void write_accum_record_payload(
    bfloat16 *payload,
    float *target,
    int32_t num_rows
) {
    for (int32_t idx = 0; idx < qwen3::kRecordPayloadBf16; idx++) {
        payload[idx] = idx < num_rows ? static_cast<bfloat16>(target[idx]) : static_cast<bfloat16>(0.0f);
        if (idx < qwen3::kMainRowsPerTile) {
            target[idx] = 0.0f;
        }
    }
}

__attribute__((noinline)) static void emit_accum_projection_record(
    int32_t *records,
    float *target,
    int32_t phase,
    int32_t replay,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    (void)replay;
    records[0] = qwen3::projection_record_header(phase, group, row);
    write_accum_record_payload(qwen3::record_payload_bf16(records), target, num_rows);
}

__attribute__((noinline)) static void emit_accum_body_record(
    int32_t *records,
    float *target,
    int32_t phase,
    int32_t block,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    records[0] = qwen3::body_record_header(phase, block, group, row);
    write_accum_record_payload(qwen3::record_payload_bf16(records), target, num_rows);
}

template <int32_t Lane, int32_t Dim>
__attribute__((always_inline)) static inline void q4nx_accum_pair(
    aie::accum<accfloat, kRowsPerLane> &row_acc,
    aie::vector<bfloat16, kRowsPerLane> scale_vec,
    aie::vector<bfloat16, kRowsPerLane> offset_vec,
    uint8_t *data,
    bfloat16 *activation_slice,
    int32_t group
) {
    const int32_t col = group * qwen3::kQ4GroupSize + Dim;
    uint4 *column_nibbles = reinterpret_cast<uint4 *>(
        data + Lane * kBytesPerLane + col * kRowPairBytes
    );
    aie::vector<uint4, qwen3::kMainRowsPerTile> packed =
        aie::load_v<qwen3::kMainRowsPerTile>(column_nibbles);
    aie::vector<uint8, qwen3::kMainRowsPerTile> as_u8 = aie::unpack(packed);
    aie::vector<uint16, qwen3::kMainRowsPerTile> as_u16 = aie::unpack(as_u8);
    aie::vector<bfloat16, qwen3::kMainRowsPerTile> q_values =
        aie::to_float<bfloat16>(as_u16, 0);

    aie::vector<bfloat16, kRowsPerLane> q0 = q_values.extract<kRowsPerLane>(0);
    aie::accum<accfloat, kRowsPerLane> dequant0_acc = aie::mul(q0, scale_vec);
    aie::vector<bfloat16, kRowsPerLane> dequant0 =
        aie::add(dequant0_acc.to_vector<bfloat16>(), offset_vec);
    aie::vector<bfloat16, kRowsPerLane> activation0 =
        aie::broadcast<bfloat16, kRowsPerLane>(activation_slice[col]);
    row_acc = aie::mac(row_acc, dequant0, activation0);

    aie::vector<bfloat16, kRowsPerLane> q1 = q_values.extract<kRowsPerLane>(1);
    aie::accum<accfloat, kRowsPerLane> dequant1_acc = aie::mul(q1, scale_vec);
    aie::vector<bfloat16, kRowsPerLane> dequant1 =
        aie::add(dequant1_acc.to_vector<bfloat16>(), offset_vec);
    aie::vector<bfloat16, kRowsPerLane> activation1 =
        aie::broadcast<bfloat16, kRowsPerLane>(activation_slice[col + 1]);
    row_acc = aie::mac(row_acc, dequant1, activation1);
}

template <int32_t Lane, int32_t Dim>
__attribute__((always_inline)) static inline void q4nx_accum_group_dims(
    aie::accum<accfloat, kRowsPerLane> &row_acc,
    aie::vector<bfloat16, kRowsPerLane> scale_vec,
    aie::vector<bfloat16, kRowsPerLane> offset_vec,
    uint8_t *data,
    bfloat16 *activation_slice,
    int32_t group
) {
    if constexpr (Dim < kUnrolledGroupDims) {
        q4nx_accum_pair<Lane, Dim>(
            row_acc,
            scale_vec,
            offset_vec,
            data,
            activation_slice,
            group
        );
        q4nx_accum_group_dims<Lane, Dim + 2>(
            row_acc,
            scale_vec,
            offset_vec,
            data,
            activation_slice,
            group
        );
    }
}

template <int32_t Lane>
__attribute__((always_inline)) static inline void q4nx_accum_group_tail(
    aie::accum<accfloat, kRowsPerLane> &row_acc,
    aie::vector<bfloat16, kRowsPerLane> scale_vec,
    aie::vector<bfloat16, kRowsPerLane> offset_vec,
    uint8_t *data,
    bfloat16 *activation_slice,
    int32_t group
) {
#pragma clang loop unroll(disable)
    for (int32_t dim = kUnrolledGroupDims; dim < qwen3::kQ4GroupSize; dim += 2) {
        const int32_t col = group * qwen3::kQ4GroupSize + dim;
        uint4 *column_nibbles = reinterpret_cast<uint4 *>(
            data + Lane * kBytesPerLane + col * kRowPairBytes
        );
        aie::vector<uint4, qwen3::kMainRowsPerTile> packed =
            aie::load_v<qwen3::kMainRowsPerTile>(column_nibbles);
        aie::vector<uint8, qwen3::kMainRowsPerTile> as_u8 = aie::unpack(packed);
        aie::vector<uint16, qwen3::kMainRowsPerTile> as_u16 = aie::unpack(as_u8);
        aie::vector<bfloat16, qwen3::kMainRowsPerTile> q_values =
            aie::to_float<bfloat16>(as_u16, 0);

        aie::vector<bfloat16, kRowsPerLane> q0 = q_values.extract<kRowsPerLane>(0);
        aie::accum<accfloat, kRowsPerLane> dequant0_acc = aie::mul(q0, scale_vec);
        aie::vector<bfloat16, kRowsPerLane> dequant0 =
            aie::add(dequant0_acc.to_vector<bfloat16>(), offset_vec);
        aie::vector<bfloat16, kRowsPerLane> activation0 =
            aie::broadcast<bfloat16, kRowsPerLane>(activation_slice[col]);
        row_acc = aie::mac(row_acc, dequant0, activation0);

        aie::vector<bfloat16, kRowsPerLane> q1 = q_values.extract<kRowsPerLane>(1);
        aie::accum<accfloat, kRowsPerLane> dequant1_acc = aie::mul(q1, scale_vec);
        aie::vector<bfloat16, kRowsPerLane> dequant1 =
            aie::add(dequant1_acc.to_vector<bfloat16>(), offset_vec);
        aie::vector<bfloat16, kRowsPerLane> activation1 =
            aie::broadcast<bfloat16, kRowsPerLane>(activation_slice[col + 1]);
        row_acc = aie::mac(row_acc, dequant1, activation1);
    }
}

template <int32_t Lane>
__attribute__((always_inline)) static inline void q4nx_accum_lane(
    float *target,
    bfloat16 *scales,
    bfloat16 *offsets,
    uint8_t *data,
    bfloat16 *activation_slice
) {
    constexpr int32_t row_base = Lane * kRowsPerLane;
    aie::accum<accfloat, kRowsPerLane> row_acc = aie::zeros<accfloat, kRowsPerLane>();
    for (int32_t group = 0; group < kGroupsPerRow; group++) {
        aie::vector<bfloat16, kRowsPerLane> scale_vec =
            aie::load_v<kRowsPerLane>(
                scales + group * qwen3::kMainRowsPerTile + row_base
            );
        aie::vector<bfloat16, kRowsPerLane> offset_vec =
            aie::load_v<kRowsPerLane>(
                offsets + group * qwen3::kMainRowsPerTile + row_base
            );
        q4nx_accum_group_dims<Lane, 0>(
            row_acc,
            scale_vec,
            offset_vec,
            data,
            activation_slice,
            group
        );
        q4nx_accum_group_tail<Lane>(
            row_acc,
            scale_vec,
            offset_vec,
            data,
            activation_slice,
            group
        );
    }

    aie::accum<accfloat, kRowsPerLane> merged;
    merged.from_vector(aie::load_v<kRowsPerLane>(target + row_base), 0);
    merged = aie::add(merged, row_acc.to_vector<float>());
    aie::store_v(target + row_base, merged.to_vector<float>());
}

__attribute__((noinline)) static void q4nx_chunk_accum_fast(
    float *target,
    bfloat16 *packed_chunk,
    bfloat16 *activation_slice
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *scales = packed_chunk;
    bfloat16 *offsets = packed_chunk + qwen3::kMainRowsPerTile * kGroupsPerRow;
    uint8_t *data = reinterpret_cast<uint8_t *>(
        packed_chunk + 2 * qwen3::kMainRowsPerTile * kGroupsPerRow
    );
    q4nx_accum_lane<0>(target, scales, offsets, data, activation_slice);
    q4nx_accum_lane<1>(target, scales, offsets, data, activation_slice);
}

static void run_single_accum_phase(
    int32_t records,
    int32_t chunks_per_record,
    int32_t weight_base,
    int32_t phase,
    int32_t group,
    int32_t row,
    int32_t num_rows,
    bfloat16 *wt_ping,
    bfloat16 *wt_pong,
    int32_t *chunk_ping,
    int32_t *chunk_pong,
    int32_t *record_ping,
    int32_t *record_pong
) {
#pragma clang loop unroll(disable)
    for (int32_t block = 0; block < records; block++) {
#pragma clang loop unroll(disable)
        for (int32_t chunk = 0; chunk < chunks_per_record; chunk++) {
            const int32_t local_chunk = block * chunks_per_record + chunk;
            const int32_t weight_chunk = weight_base + local_chunk;
            acquire_activation_and_weight();
            q4nx_chunk_accum_fast(
                accum,
                select_weight_buffer(weight_chunk, wt_ping, wt_pong),
                reinterpret_cast<bfloat16 *>(
                    select_activation_buffer(weight_chunk, chunk_ping, chunk_pong)
                )
            );
            release_activation_and_weight();
        }
        acquire_record_buffer();
        emit_accum_body_record(
            select_record_buffer(block, record_ping, record_pong),
            accum,
            phase,
            block,
            group,
            row,
            num_rows
        );
        release_record_buffer();
    }
}

static void run_upgate_phase(
    int32_t group,
    int32_t row,
    int32_t num_rows,
    bfloat16 *wt_ping,
    bfloat16 *wt_pong,
    int32_t *chunk_ping,
    int32_t *chunk_pong,
    int32_t *record_ping,
    int32_t *record_pong
) {
#pragma clang loop unroll(disable)
    for (int32_t replay = 0; replay < qwen3::kUpGateReplays; replay++) {
#pragma clang loop unroll(disable)
        for (int32_t chunk = 0; chunk < qwen3::kUpGateChunksPerReplay; chunk++) {
            const int32_t global_chunk = replay * qwen3::kUpGateChunksPerReplay + chunk;
            const int32_t weight_chunk = qwen3::kFullLayerUpGateWeightChunkBase + global_chunk;
            acquire_activation_and_weight();
            q4nx_chunk_accum_fast(
                accum,
                select_weight_buffer(weight_chunk, wt_ping, wt_pong),
                reinterpret_cast<bfloat16 *>(
                    select_activation_buffer(weight_chunk, chunk_ping, chunk_pong)
                )
            );
            release_activation_and_weight();
        }
        const int32_t phase = (replay & 1) == 0 ? qwen3::kUpPhase : qwen3::kGatePhase;
        acquire_record_buffer();
        emit_accum_projection_record(
            select_record_buffer(replay, record_ping, record_pong),
            accum,
            phase,
            replay,
            group,
            row,
            num_rows
        );
        release_record_buffer();
    }
}

} // namespace

extern "C" {

void q4nx_fill_perf_inputs(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t weight_bf16,
    int32_t activation_dwords
) {
    for (int32_t idx = 0; idx < weight_bf16; idx++) {
        packed_chunk[idx] = static_cast<bfloat16>(1.0f);
    }
    for (int32_t idx = 0; idx < activation_dwords; idx++) {
        activation_words[idx] = 0x3f803f80;
    }
}

void q4nx_clear_accum_fast(int32_t num_rows) {
    for (int32_t idx = 0; idx < qwen3::kMainRowsPerTile; idx++) {
        if (idx < num_rows) {
            accum[idx] = 0.0f;
        }
    }
}

void q4nx_chunk_accum_slice_i32_fast(
    bfloat16 *packed_chunk,
    int32_t *activation_words,
    int32_t num_rows
) {
    (void)num_rows;
    q4nx_chunk_accum_fast(
        accum,
        packed_chunk,
        reinterpret_cast<bfloat16 *>(activation_words)
    );
}

void q4nx_main16_full_scheduler(
    bfloat16 *wt_ping,
    bfloat16 *wt_pong,
    int32_t *chunk_ping,
    int32_t *chunk_pong,
    int32_t *record_ping,
    int32_t *record_pong,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    run_single_accum_phase(
        qwen3::kQBodyRecords,
        qwen3::kQChunksPerRecord,
        qwen3::kQWeightChunkBase,
        qwen3::kQPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kKvBodyRecords,
        qwen3::kKvChunksPerRecord,
        qwen3::kKWeightChunkBase,
        qwen3::kKPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kKvBodyRecords,
        qwen3::kKvChunksPerRecord,
        qwen3::kVWeightChunkBase,
        qwen3::kVPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kOBodyRecords,
        qwen3::kOChunksPerRecord,
        qwen3::kFullLayerOWeightChunkBase,
        qwen3::kOPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_upgate_phase(
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kDownBodyRecords,
        qwen3::kDownChunksPerRecord,
        qwen3::kFullLayerDownWeightChunkBase,
        qwen3::kDownPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
}

void q4nx_main16_qkv_scheduler(
    bfloat16 *wt_ping,
    bfloat16 *wt_pong,
    int32_t *chunk_ping,
    int32_t *chunk_pong,
    int32_t *record_ping,
    int32_t *record_pong,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    run_single_accum_phase(
        qwen3::kQBodyRecords,
        qwen3::kQChunksPerRecord,
        qwen3::kQWeightChunkBase,
        qwen3::kQPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kKvBodyRecords,
        qwen3::kKvChunksPerRecord,
        qwen3::kKWeightChunkBase,
        qwen3::kKPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
    run_single_accum_phase(
        qwen3::kKvBodyRecords,
        qwen3::kKvChunksPerRecord,
        qwen3::kVWeightChunkBase,
        qwen3::kVPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
}

void q4nx_main16_qkvo_scheduler(
    bfloat16 *wt_ping,
    bfloat16 *wt_pong,
    int32_t *chunk_ping,
    int32_t *chunk_pong,
    int32_t *record_ping,
    int32_t *record_pong,
    int32_t group,
    int32_t row,
    int32_t num_rows
) {
    q4nx_main16_qkv_scheduler(
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong,
        group,
        row,
        num_rows
    );
    run_single_accum_phase(
        qwen3::kOBodyRecords,
        qwen3::kOChunksPerRecord,
        qwen3::kFullLayerOWeightChunkBase,
        qwen3::kOPhase,
        group,
        row,
        num_rows,
        wt_ping,
        wt_pong,
        chunk_ping,
        chunk_pong,
        record_ping,
        record_pong
    );
}

} // extern "C"
