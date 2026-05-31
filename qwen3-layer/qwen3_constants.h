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

constexpr int32_t kMain16PhaseLimitQkv = kVPhase + 1;
constexpr int32_t kMain16PhaseLimitQkvo = kOPhase + 1;
constexpr int32_t kMain16PhaseLimitUpGate = kGatePhase + 1;
constexpr int32_t kMain16PhaseLimitFull = kDownPhase + 1;

constexpr int32_t kQCompactPacketId = 10;
constexpr int32_t kKCompactPacketId = 11;
constexpr int32_t kVCompactPacketId = 12;
constexpr int32_t kOCompactPacketId = 13;
constexpr int32_t kFfnCompactPacketId = 14;
constexpr int32_t kDownCompactPacketId = 15;

constexpr int32_t kMainActivationEmptyLock = 0;
constexpr int32_t kMainActivationFullLock = 1;
constexpr int32_t kMainWeightEmptyLock = 2;
constexpr int32_t kMainWeightFullLock = 3;
constexpr int32_t kMainRecordEmptyLock = 4;
constexpr int32_t kMainRecordFullLock = 5;
constexpr int32_t kCoreLocalLockBase = 0x30;
constexpr int32_t kMainActivationEmptyCoreLock = kCoreLocalLockBase + kMainActivationEmptyLock;
constexpr int32_t kMainActivationFullCoreLock = kCoreLocalLockBase + kMainActivationFullLock;
constexpr int32_t kMainWeightEmptyCoreLock = kCoreLocalLockBase + kMainWeightEmptyLock;
constexpr int32_t kMainWeightFullCoreLock = kCoreLocalLockBase + kMainWeightFullLock;
constexpr int32_t kMainRecordEmptyCoreLock = kCoreLocalLockBase + kMainRecordEmptyLock;
constexpr int32_t kMainRecordFullCoreLock = kCoreLocalLockBase + kMainRecordFullLock;

constexpr int32_t kQBodyRecords = 8;
constexpr int32_t kKvBodyRecords = 2;
constexpr int32_t kOBodyRecords = 8;
constexpr int32_t kUpGateReplays = 48;
constexpr int32_t kDownBodyRecords = 8;

constexpr int32_t kQChunksPerRecord = 16;
constexpr int32_t kKvChunksPerRecord = 16;
constexpr int32_t kOChunksPerRecord = 16;
constexpr int32_t kUpGateChunksPerReplay = 16;
constexpr int32_t kDownChunksPerRecord = 48;

constexpr int32_t kQWeightChunkBase = 0;
constexpr int32_t kKWeightChunkBase = 128;
constexpr int32_t kVWeightChunkBase = 160;
constexpr int32_t kFullLayerOWeightChunkBase = 192;
constexpr int32_t kFullLayerUpGateWeightChunkBase = 320;
constexpr int32_t kFullLayerDownWeightChunkBase = 1088;

} // namespace qwen3
