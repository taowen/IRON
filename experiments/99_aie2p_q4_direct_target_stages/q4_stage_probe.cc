#include <stdint.h>

extern "C" void asm_q4_direct_exact_group1(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_hwloop_smoke_lc2(float *dst);

extern "C" void asm_hwloop_smoke_lc2_aligned(float *dst);

extern "C" void asm_q4_direct_exact_group8(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group8_hwloop_addnc_aligned(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group8_unrolled(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group4_unrolled(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group4x2_asm_call(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_hwloop(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_hwloop_addnc(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_hwloop_addnc_gap(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_hwloop_addnc_lc3(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_hwloop_addnc_aligned(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

extern "C" void asm_q4_direct_exact_group2_unrolled(
    uint8_t *packed,
    uint16_t *scales,
    uint16_t *offsets,
    uint16_t *activation,
    float *dst);

namespace {
constexpr int32_t kDstDwordOffset = 0;
constexpr int32_t kPackedDwordOffset = 16;
constexpr int32_t kScaleDwordOffset = 528;
constexpr int32_t kOffsetDwordOffset = 656;
constexpr int32_t kActivationDwordOffset = 784;
constexpr int32_t kPackedFourGroupBytes = 0x400;
constexpr int32_t kBf16FourGroupElements = 0x100 / sizeof(uint16_t);

static inline uint8_t *workspace_bytes(int32_t *workspace, int32_t dword_offset) {
    return reinterpret_cast<uint8_t *>(workspace + dword_offset);
}

static inline uint16_t *workspace_bf16(int32_t *workspace, int32_t dword_offset) {
    return reinterpret_cast<uint16_t *>(workspace + dword_offset);
}

static inline float *workspace_float(int32_t *workspace, int32_t dword_offset) {
    return reinterpret_cast<float *>(workspace + dword_offset);
}
}  // namespace

extern "C" void cpp_call_q4_direct_exact_group1(int32_t *workspace) {
    asm_q4_direct_exact_group1(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_hwloop_smoke_lc2(int32_t *workspace) {
    asm_hwloop_smoke_lc2(workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_hwloop_smoke_lc2_aligned(int32_t *workspace) {
    asm_hwloop_smoke_lc2_aligned(workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group8(int32_t *workspace) {
    asm_q4_direct_exact_group8(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group8_hwloop_addnc_aligned(int32_t *workspace) {
    asm_q4_direct_exact_group8_hwloop_addnc_aligned(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group8_unrolled(int32_t *workspace) {
    asm_q4_direct_exact_group8_unrolled(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group4_unrolled(int32_t *workspace) {
    asm_q4_direct_exact_group4_unrolled(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group4x2_call(int32_t *workspace) {
    uint8_t *packed = workspace_bytes(workspace, kPackedDwordOffset);
    uint16_t *scales = workspace_bf16(workspace, kScaleDwordOffset);
    uint16_t *offsets = workspace_bf16(workspace, kOffsetDwordOffset);
    uint16_t *activation = workspace_bf16(workspace, kActivationDwordOffset);
    float *dst = workspace_float(workspace, kDstDwordOffset);
    asm_q4_direct_exact_group4_unrolled(packed, scales, offsets, activation, dst);
    asm_q4_direct_exact_group4_unrolled(
        packed + kPackedFourGroupBytes,
        scales + kBf16FourGroupElements,
        offsets + kBf16FourGroupElements,
        activation + kBf16FourGroupElements,
        dst);
}

extern "C" void cpp_call_q4_direct_exact_group4x2_asm_call(int32_t *workspace) {
    asm_q4_direct_exact_group4x2_asm_call(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_hwloop(int32_t *workspace) {
    asm_q4_direct_exact_group2_hwloop(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_hwloop_addnc(int32_t *workspace) {
    asm_q4_direct_exact_group2_hwloop_addnc(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_hwloop_addnc_gap(int32_t *workspace) {
    asm_q4_direct_exact_group2_hwloop_addnc_gap(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_hwloop_addnc_lc3(int32_t *workspace) {
    asm_q4_direct_exact_group2_hwloop_addnc_lc3(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_hwloop_addnc_aligned(int32_t *workspace) {
    asm_q4_direct_exact_group2_hwloop_addnc_aligned(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2(int32_t *workspace) {
    asm_q4_direct_exact_group2(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}

extern "C" void cpp_call_q4_direct_exact_group2_unrolled(int32_t *workspace) {
    asm_q4_direct_exact_group2_unrolled(
        workspace_bytes(workspace, kPackedDwordOffset),
        workspace_bf16(workspace, kScaleDwordOffset),
        workspace_bf16(workspace, kOffsetDwordOffset),
        workspace_bf16(workspace, kActivationDwordOffset),
        workspace_float(workspace, kDstDwordOffset));
}
