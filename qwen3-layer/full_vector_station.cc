#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

static int32_t clamp_s16(int32_t value) {
    if (value < -32768) {
        return -32768;
    }
    if (value > 32767) {
        return 32767;
    }
    return value;
}

static int32_t pack_s16_pair(int32_t low, int32_t high) {
    const uint32_t low_u16 = static_cast<uint32_t>(clamp_s16(low)) & 0xffff;
    const uint32_t high_u16 = static_cast<uint32_t>(clamp_s16(high)) & 0xffff;
    return static_cast<int32_t>(low_u16 | (high_u16 << 16));
}

static int32_t div_i32(int32_t numerator, int32_t denominator) {
    if (denominator == 0) {
        return 0;
    }
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -((-numerator) / denominator);
}

static int32_t isqrt_i32(int32_t value) {
    int32_t result = 0;
    for (int32_t candidate = 1; candidate <= 512; candidate++) {
        if (candidate * candidate <= value) {
            result = candidate;
        }
    }
    return result;
}

static int32_t isqrt_i64(int64_t value) {
    int32_t result = 0;
    for (int32_t candidate = 1; candidate <= 4096; candidate++) {
        const int64_t square = static_cast<int64_t>(candidate) * static_cast<int64_t>(candidate);
        if (square <= value) {
            result = candidate;
        }
    }
    return result;
}

static int32_t compact_raw_lane(int32_t *compact, int32_t lane) {
    const int32_t seed = compact[1 + ((lane * 13 + (lane >> 5)) & 255)];
    const int32_t mixed = seed + lane * 17 + (lane >> 3) * 31 + (lane >> 7) * 127;
    return (mixed & 1023) - 512;
}

static int32_t compact_numeric_lane(int32_t *compact, int32_t lane) {
    constexpr int32_t compact_lanes = 512;
    constexpr float fixed_scale = 256.0f;
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(compact + 1);
    return static_cast<int32_t>(static_cast<float>(payload[lane & (compact_lanes - 1)]) * fixed_scale);
}

static bfloat16 compact_block_lane(int32_t *compacts, int32_t block, int32_t lane) {
    constexpr int32_t compact_dwords = 257;
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(compacts + block * compact_dwords + 1);
    return payload[lane];
}

} // namespace

extern "C" {

void c1r2_make_replay(
    int32_t *compact,
    int32_t *replay,
    int32_t replay_idx,
    int32_t payload_dwords
) {
    replay[0] = 0xC1000000 | replay_idx;
    int64_t sum_sq = 0;
    for (int32_t lane = 0; lane < payload_dwords * 2; lane++) {
        const int32_t raw = compact_raw_lane(compact, lane);
        sum_sq += raw * raw;
    }
    const int32_t denominator = isqrt_i32(sum_sq / (payload_dwords * 2) + 1);
    for (int32_t idx = 0; idx < payload_dwords; idx++) {
        const int32_t low = div_i32(compact_raw_lane(compact, idx * 2) * 1024, denominator);
        const int32_t high = div_i32(compact_raw_lane(compact, idx * 2 + 1) * 1024, denominator);
        replay[1 + idx] = pack_s16_pair(low, high);
    }
}

void c1r2_make_replay_bf16(
    int32_t *compact,
    int32_t *replay,
    int32_t replay_idx,
    int32_t payload_dwords
) {
    constexpr int32_t compact_lanes = 512;
    replay[0] = 0xC1000000 | replay_idx;
    int32_t sum_sq = 0;
    for (int32_t lane = 0; lane < compact_lanes; lane++) {
        const int32_t raw = compact_numeric_lane(compact, lane);
        sum_sq += raw * raw;
    }
    const int32_t denominator = isqrt_i32(sum_sq / compact_lanes + 1);
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(replay + 1);
    for (int32_t lane = 0; lane < payload_dwords * 2; lane++) {
        const int32_t scaled = div_i32(compact_numeric_lane(compact, lane) * 1024, denominator);
        payload[lane] = static_cast<bfloat16>(static_cast<float>(scaled) / 1024.0f);
    }
}

void qkv_c1r2_summarize_compact(int32_t *compact, int32_t *summary, int32_t dwords) {
    uint32_t sum = 0;
    uint32_t hash = 0;
    for (int32_t idx = 0; idx < dwords; idx++) {
        const uint32_t value = static_cast<uint32_t>(compact[idx]);
        sum += value;
        hash = (hash * 16777619u) ^ (value + static_cast<uint32_t>(idx));
    }
    summary[0] = 0x51564F43;
    summary[1] = dwords;
    summary[2] = compact[0];
    summary[3] = compact[dwords - 1];
    summary[4] = static_cast<int32_t>(sum);
    summary[5] = static_cast<int32_t>(hash);
    summary[6] = compact[1];
    summary[7] = compact[dwords - 2];
}

void full_c1r2_copy_compact(int32_t *compact, int32_t *output, int32_t dwords) {
    for (int32_t idx = 0; idx < dwords; idx++) {
        output[idx] = compact[idx];
    }
}

void full_c1r2_make_replay_from_o_compacts(
    int32_t *compacts,
    int32_t *replay,
    int32_t replay_idx,
    int32_t payload_dwords,
    int32_t blocks
) {
    constexpr float fixed_scale = 256.0f;
    replay[0] = 0xC1000000 | replay_idx;
    int32_t sum_sq = 0;
    for (int32_t block = 0; block < blocks; block++) {
        for (int32_t lane = 0; lane < 512; lane++) {
            const int32_t raw = static_cast<int32_t>(
                static_cast<float>(compact_block_lane(compacts, block, lane)) * fixed_scale
            );
            sum_sq += static_cast<int64_t>(raw) * static_cast<int64_t>(raw);
        }
    }
    const int32_t denominator = isqrt_i64(sum_sq / (payload_dwords * 2) + 1);
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(replay + 1);
    for (int32_t lane = 0; lane < payload_dwords * 2; lane++) {
        const int32_t block = lane / 512;
        const int32_t local_lane = lane - block * 512;
        const int32_t raw = static_cast<int32_t>(
            static_cast<float>(compact_block_lane(compacts, block, local_lane)) * fixed_scale
        );
        const int32_t scaled = div_i32(raw * 1024, denominator);
        payload[lane] = static_cast<bfloat16>(static_cast<float>(scaled) / 1024.0f);
    }
}

void full_c1r2_output_from_down_compacts(
    int32_t *compacts,
    int32_t *output,
    int32_t dwords,
    int32_t blocks
) {
    bfloat16 *out = reinterpret_cast<bfloat16 *>(output);
    const int32_t lanes = dwords * 2;
    for (int32_t lane = 0; lane < lanes; lane++) {
        const int32_t block = lane / 512;
        const int32_t local_lane = lane - block * 512;
        if (block < blocks) {
            out[lane] = compact_block_lane(compacts, block, local_lane);
        } else {
            out[lane] = static_cast<bfloat16>(0.0f);
        }
    }
}

void full_c1r2_copy_hidden_replay(int32_t *hidden, int32_t *replay, int32_t payload_dwords) {
    replay[0] = 0xC1000000;
    for (int32_t idx = 0; idx < payload_dwords; idx++) {
        replay[1 + idx] = hidden[idx];
    }
}

} // extern "C"
