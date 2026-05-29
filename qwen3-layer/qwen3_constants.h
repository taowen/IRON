#pragma once

#include <stdint.h>

namespace qwen3 {

constexpr int32_t kMainRowsPerTile = 32;
constexpr int32_t kQ4KChunk = 256;
constexpr int32_t kQ4GroupSize = 32;
constexpr int32_t kRecordDwords = 17;
constexpr int32_t kRecordPayloadDwords = kRecordDwords - 1;
constexpr int32_t kRecordPayloadBf16 = kRecordPayloadDwords * 2;

constexpr int32_t kQPhase = 0;
constexpr int32_t kKPhase = 1;
constexpr int32_t kVPhase = 2;
constexpr int32_t kOPhase = 3;
constexpr int32_t kUpPhase = 4;
constexpr int32_t kGatePhase = 5;
constexpr int32_t kDownPhase = 6;

} // namespace qwen3
