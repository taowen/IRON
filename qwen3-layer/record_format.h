#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "qwen3_constants.h"

namespace qwen3 {

static inline bfloat16 *record_payload_bf16(int32_t *record) {
    return reinterpret_cast<bfloat16 *>(record + 1);
}

static inline int32_t projection_record_header(int32_t phase, int32_t group, int32_t row) {
    return (phase << 24) | (group << 16) | (row << 8) | 0xD0;
}

static inline int32_t body_record_header(int32_t phase, int32_t block, int32_t group, int32_t row) {
    return (phase << 24) | (block << 20) | (group << 16) | (row << 8) | 0xD0;
}

} // namespace qwen3
