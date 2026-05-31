#include <aie_api/aie.hpp>
#include <aie2pintrin.h>
#include <stdint.h>

namespace {

constexpr int32_t kVec16 = 16;
constexpr int32_t kVec32 = 32;
constexpr int32_t kQ4Rows = 32;
constexpr int32_t kRowsPerLane = 16;
constexpr int32_t kQ4Columns = 256;
constexpr int32_t kQ4GroupSize = 32;
constexpr int32_t kQ4Groups = kQ4Columns / kQ4GroupSize;
constexpr int32_t kLaneNibbles = kQ4Columns * kRowsPerLane;
constexpr int32_t kGroupNibblesPerLane = kQ4GroupSize * kRowsPerLane;

template <int32_t Dim>
__attribute__((always_inline)) static inline v16accfloat q4_scaled_pair_mac_signed(
    v16accfloat acc,
    const uint4 *__restrict packed,
    v16bfloat16 scale_low,
    v32int16 activation_bits
) {
    aie::vector<uint4, kQ4Rows> q4_vec = aie::load_v<kQ4Rows>(packed);
    v64uint4 q4 = set_v64uint4(0, (v32uint4)q4_vec);
    v64uint8 q8 = unpack(q4);
    v64uint16 q16 = unpack(q8);
    aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
    aie::vector<bfloat16, kVec32> q_bf16_vec =
        aie::to_float<bfloat16>(q16_vec, 0);

    v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
    v16bfloat16 scaled0 = to_v16bfloat16(
        mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low))
    );
    acc = mac_elem_16_conf(
        set_v32bfloat16(0, scaled0),
        __SIGN_SIGNED,
        (v32bfloat16)broadcast_elem(activation_bits, Dim),
        __SIGN_SIGNED,
        acc,
        0,
        0,
        0
    );

    v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
    v16bfloat16 scaled1 = to_v16bfloat16(
        mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low))
    );
    return mac_elem_16_conf(
        set_v32bfloat16(0, scaled1),
        __SIGN_SIGNED,
        (v32bfloat16)broadcast_elem(activation_bits, Dim + 1),
        __SIGN_SIGNED,
        acc,
        0,
        0,
        0
    );
}

template <int32_t Dim, int32_t Limit>
__attribute__((always_inline)) static inline v16accfloat q4_scaled_group_unroll_signed(
    v16accfloat acc,
    const uint4 *__restrict packed,
    v16bfloat16 scale_low,
    v32int16 activation_bits
) {
    if constexpr (Dim < Limit) {
        acc = q4_scaled_pair_mac_signed<Dim>(
            acc,
            packed + Dim * kRowsPerLane,
            scale_low,
            activation_bits
        );
        return q4_scaled_group_unroll_signed<Dim + 2, Limit>(
            acc,
            packed,
            scale_low,
            activation_bits
        );
    }
    return acc;
}

template <int32_t Dim>
__attribute__((always_inline)) static inline v16accfloat q4_exact_pair_mac_signed(
    v16accfloat acc,
    const uint4 *__restrict packed,
    v16bfloat16 scale_low,
    v16bfloat16 offset_low,
    v32int16 activation_bits
) {
    aie::vector<uint4, kQ4Rows> q4_vec = aie::load_v<kQ4Rows>(packed);
    v64uint4 q4 = set_v64uint4(0, (v32uint4)q4_vec);
    v64uint8 q8 = unpack(q4);
    v64uint16 q16 = unpack(q8);
    aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
    aie::vector<bfloat16, kVec32> q_bf16_vec =
        aie::to_float<bfloat16>(q16_vec, 0);

    v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
    v16bfloat16 scaled0 = to_v16bfloat16(
        mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low))
    );
    v16bfloat16 dequant0 = to_v16bfloat16(
        add(ups_to_v16accfloat(scaled0), ups_to_v16accfloat(offset_low))
    );
    acc = mac_elem_16_conf(
        set_v32bfloat16(0, dequant0),
        __SIGN_SIGNED,
        (v32bfloat16)broadcast_elem(activation_bits, Dim),
        __SIGN_SIGNED,
        acc,
        0,
        0,
        0
    );

    v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
    v16bfloat16 scaled1 = to_v16bfloat16(
        mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low))
    );
    v16bfloat16 dequant1 = to_v16bfloat16(
        add(ups_to_v16accfloat(scaled1), ups_to_v16accfloat(offset_low))
    );
    return mac_elem_16_conf(
        set_v32bfloat16(0, dequant1),
        __SIGN_SIGNED,
        (v32bfloat16)broadcast_elem(activation_bits, Dim + 1),
        __SIGN_SIGNED,
        acc,
        0,
        0,
        0
    );
}

template <int32_t Dim, int32_t Limit>
__attribute__((always_inline)) static inline v16accfloat q4_exact_group_unroll_signed(
    v16accfloat acc,
    const uint4 *__restrict packed,
    v16bfloat16 scale_low,
    v16bfloat16 offset_low,
    v32int16 activation_bits
) {
    if constexpr (Dim < Limit) {
        acc = q4_exact_pair_mac_signed<Dim>(
            acc,
            packed + Dim * kRowsPerLane,
            scale_low,
            offset_low,
            activation_bits
        );
        return q4_exact_group_unroll_signed<Dim + 2, Limit>(
            acc,
            packed,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    return acc;
}

template <int32_t Group, int32_t Limit>
__attribute__((always_inline)) static inline v16accfloat q4_scaled_chunk_lane_unroll_signed(
    v16accfloat acc,
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits
) {
    if constexpr (Group < Limit) {
        v16bfloat16 scale_vec =
            *reinterpret_cast<const v16bfloat16 *>(scale_lane + Group * kQ4Rows);
        v16bfloat16 offset_vec =
            *reinterpret_cast<const v16bfloat16 *>(offset_lane + Group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + Group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_scaled_group_unroll_signed<0, kQ4GroupSize>(
            acc,
            packed_lane + Group * kGroupNibblesPerLane,
            scale_vec,
            activation_bits
        );

        v32int16 group_sum = broadcast_s16(activation_group_sum_bits[Group]);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, offset_vec),
            __SIGN_SIGNED,
            (v32bfloat16)group_sum,
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );
        return q4_scaled_chunk_lane_unroll_signed<Group + 1, Limit>(
            acc,
            packed_lane,
            scale_lane,
            offset_lane,
            activation,
            activation_group_sum_bits
        );
    }
    return acc;
}

template <int32_t Group, int32_t Limit>
__attribute__((always_inline)) static inline v16accfloat q4_exact_chunk_lane_unroll_signed(
    v16accfloat acc,
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation
) {
    if constexpr (Group < Limit) {
        v16bfloat16 scale_vec =
            *reinterpret_cast<const v16bfloat16 *>(scale_lane + Group * kQ4Rows);
        v16bfloat16 offset_vec =
            *reinterpret_cast<const v16bfloat16 *>(offset_lane + Group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + Group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_exact_group_unroll_signed<0, kQ4GroupSize>(
            acc,
            packed_lane + Group * kGroupNibblesPerLane,
            scale_vec,
            offset_vec,
            activation_bits
        );
        return q4_exact_chunk_lane_unroll_signed<Group + 1, Limit>(
            acc,
            packed_lane,
            scale_lane,
            offset_lane,
            activation
        );
    }
    return acc;
}

} // namespace

extern "C" {

void probe_bf16_load_store(
    const bfloat16 *__restrict input,
    bfloat16 *__restrict output,
    int32_t chunks
) {
#pragma clang loop unroll(disable)
    for (int32_t chunk = 0; chunk < chunks; chunk++) {
        const bfloat16 *src = input + chunk * kVec16;
        bfloat16 *dst = output + chunk * kVec16;
        aie::vector<bfloat16, kVec16> values = aie::load_v<kVec16>(src);
        aie::store_v(dst, values);
    }
}

void probe_bf16_broadcast_extract_mac(
    const bfloat16 *__restrict lhs,
    const bfloat16 *__restrict rhs,
    float *__restrict output,
    int32_t chunks
) {
    aie::accum<accfloat, 8> acc = aie::zeros<accfloat, 8>();
#pragma clang loop unroll(disable)
    for (int32_t chunk = 0; chunk < chunks; chunk++) {
        aie::vector<bfloat16, kVec16> lhs_vec =
            aie::load_v<kVec16>(lhs + chunk * kVec16);
        aie::vector<bfloat16, 8> lhs_low = lhs_vec.extract<8>(0);
        aie::vector<bfloat16, 8> rhs_scalar =
            aie::broadcast<bfloat16, 8>(rhs[chunk]);
        acc = aie::mac(acc, lhs_low, rhs_scalar);
    }
    aie::store_v(output, acc.to_vector<float>());
}

void probe_q4_unpack_broadcast_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    float *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 8> acc = aie::zeros<accfloat, 8>();
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        aie::vector<uint4, kQ4Rows> q4 =
            aie::load_v<kQ4Rows>(packed + group * kQ4Rows);
        aie::vector<uint8, kQ4Rows> q8 = aie::unpack(q4);
        aie::vector<uint16, kQ4Rows> q16 = aie::unpack(q8);
        aie::vector<bfloat16, kQ4Rows> q_bf16 =
            aie::to_float<bfloat16>(q16, 0);

        aie::vector<bfloat16, 8> q_low = q_bf16.extract<8>(0);
        aie::vector<bfloat16, 8> scale_vec =
            aie::broadcast<bfloat16, 8>(scale[group]);
        aie::vector<bfloat16, 8> offset_vec =
            aie::broadcast<bfloat16, 8>(offset[group]);
        aie::accum<accfloat, 8> dequant_acc = aie::mul(q_low, scale_vec);
        aie::vector<bfloat16, 8> dequant =
            aie::add(dequant_acc.to_vector<bfloat16>(), offset_vec);
        aie::vector<bfloat16, 8> act_vec =
            aie::broadcast<bfloat16, 8>(activation[group * kQ4GroupSize]);
        acc = aie::mac(acc, dequant, act_vec);
    }
    aie::store_v(output, acc.to_vector<float>());
}

void probe_native_bf16_vextbcst_mac(
    const bfloat16 *__restrict lhs,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v32accfloat acc = broadcast_zero_to_v32accfloat();
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v32bfloat16 lhs_vec =
            *reinterpret_cast<const v32bfloat16 *>(lhs + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        acc = mac_elem_32(lhs_vec, broadcast_elem(activation_group, 0), acc);
    }
    *reinterpret_cast<v32bfloat16 *>(output) = to_v32bfloat16(acc);
}

void probe_native_bf16_acc16_vextbcst_mac(
    const bfloat16 *__restrict lhs,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v32bfloat16 lhs_vec =
            *reinterpret_cast<const v32bfloat16 *>(lhs + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        acc = mac_elem_16_conf(
            lhs_vec,
            broadcast_elem(activation_group, 0),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_bf16_i16_view_acc16_mac(
    const bfloat16 *__restrict lhs,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v32bfloat16 lhs_vec =
            *reinterpret_cast<const v32bfloat16 *>(lhs + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;
        v32int16 activation_bits_broadcast = broadcast_elem(activation_bits, 0);
        acc = mac_elem_16_conf(
            lhs_vec,
            (v32bfloat16)activation_bits_broadcast,
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_arg_i16_vextbcst(
    v32int16 input,
    int32_t idx,
    int16_t *__restrict output
) {
    *reinterpret_cast<v32int16 *>(output) = broadcast_elem(input, idx);
}

void probe_arg_i16_view_bf16_acc16_mac(
    v32bfloat16 lhs,
    v32int16 activation_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc = mac_elem_16_conf(
        lhs,
        (v32bfloat16)broadcast_elem(activation_bits, 0),
        acc,
        0,
        0,
        0
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_arg_i16_view_bf16_acc16_mac_signed(
    v32bfloat16 lhs,
    v32int16 activation_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc = mac_elem_16_conf(
        lhs,
        __SIGN_SIGNED,
        (v32bfloat16)broadcast_elem(activation_bits, 0),
        __SIGN_SIGNED,
        acc,
        0,
        0,
        0
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_unpack_ups(
    const uint4 *__restrict packed,
    uint16_t *__restrict output,
    int32_t groups
) {
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        v32uint16 q16_low = extract_v32uint16(q16, 0);
        v32acc32 q_acc = ups_to_v32acc32(q16_low, 0);
        *reinterpret_cast<v32uint16 *>(output + group * kVec32) =
            to_v32uint16(q_acc, 0);
    }
}

void probe_native_q4_unpack_bridge_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v32bfloat16 q_bf16 = q_bf16_vec;
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;
        acc = mac_elem_16_conf(
            q_bf16,
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );
        acc = mac_elem_16_conf(
            q_bf16,
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_dequant_i16_activation_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);

        v16accfloat scaled_acc =
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled_bf16 = to_v16bfloat16(scaled_acc);
        v16accfloat dequant_acc =
            add(ups_to_v16accfloat(scaled_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant = to_v16bfloat16(dequant_acc);

        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant),
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_dequant_i16_activation_pair_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16accfloat scaled0_acc =
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled0_bf16 = to_v16bfloat16(scaled0_acc);
        v16accfloat dequant0_acc =
            add(ups_to_v16accfloat(scaled0_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant0 = to_v16bfloat16(dequant0_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant0),
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16accfloat scaled1_acc =
            mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled1_bf16 = to_v16bfloat16(scaled1_acc);
        v16accfloat dequant1_acc =
            add(ups_to_v16accfloat(scaled1_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant1 = to_v16bfloat16(dequant1_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant1),
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_dequant_i16_activation_pair_mac_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16accfloat scaled0_acc =
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled0_bf16 = to_v16bfloat16(scaled0_acc);
        v16accfloat dequant0_acc =
            add(ups_to_v16accfloat(scaled0_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant0 = to_v16bfloat16(dequant0_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant0),
            __SIGN_SIGNED,
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16accfloat scaled1_acc =
            mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled1_bf16 = to_v16bfloat16(scaled1_acc);
        v16accfloat dequant1_acc =
            add(ups_to_v16accfloat(scaled1_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant1 = to_v16bfloat16(dequant1_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant1),
            __SIGN_SIGNED,
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_group_sum_correction_pair_mac_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        aie::vector<uint4, kQ4Rows> q4_vec =
            aie::load_v<kQ4Rows>(packed + group * kQ4Rows);
        v64uint4 q4 = set_v64uint4(0, (v32uint4)q4_vec);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16bfloat16 scaled0 = to_v16bfloat16(
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low))
        );
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, scaled0),
            __SIGN_SIGNED,
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16bfloat16 scaled1 = to_v16bfloat16(
            mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low))
        );
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, scaled1),
            __SIGN_SIGNED,
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );

        v32int16 group_sum = broadcast_s16(activation_group_sum_bits[group]);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, offset_low),
            __SIGN_SIGNED,
            (v32bfloat16)group_sum,
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_group_sum_correction_unroll8_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_scaled_group_unroll_signed<0, 8>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            activation_bits
        );

        v32int16 group_sum = broadcast_s16(activation_group_sum_bits[group]);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, offset_low),
            __SIGN_SIGNED,
            (v32bfloat16)group_sum,
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_group_sum_correction_unroll32_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_scaled_group_unroll_signed<0, kQ4GroupSize>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            activation_bits
        );

        v32int16 group_sum = broadcast_s16(activation_group_sum_bits[group]);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, offset_low),
            __SIGN_SIGNED,
            (v32bfloat16)group_sum,
            __SIGN_SIGNED,
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_unroll4_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_exact_group_unroll_signed<0, 4>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_unroll8_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_exact_group_unroll_signed<0, 8>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_unroll16_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_exact_group_unroll_signed<0, 16>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

#define DEFINE_EXACT_GROUP4_KERNEL(NAME, DIM) \
__attribute__((noinline)) v16accfloat NAME( \
    v16accfloat acc, \
    const uint4 *__restrict packed, \
    v16bfloat16 scale_low, \
    v16bfloat16 offset_low, \
    v32int16 activation_bits \
) { \
    return q4_exact_group_unroll_signed<DIM, DIM + 4>( \
        acc, \
        packed, \
        scale_low, \
        offset_low, \
        activation_bits \
    ); \
}

DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim0_kernel_signed, 0)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim4_kernel_signed, 4)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim8_kernel_signed, 8)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim12_kernel_signed, 12)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim16_kernel_signed, 16)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim20_kernel_signed, 20)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim24_kernel_signed, 24)
DEFINE_EXACT_GROUP4_KERNEL(probe_native_q4_exact_rounding_group4_dim28_kernel_signed, 28)

#undef DEFINE_EXACT_GROUP4_KERNEL

#define DEFINE_EXACT_GROUP8_KERNEL(NAME, DIM) \
__attribute__((noinline)) v16accfloat NAME( \
    v16accfloat acc, \
    const uint4 *__restrict packed, \
    v16bfloat16 scale_low, \
    v16bfloat16 offset_low, \
    v32int16 activation_bits \
) { \
    return q4_exact_group_unroll_signed<DIM, DIM + 8>( \
        acc, \
        packed, \
        scale_low, \
        offset_low, \
        activation_bits \
    ); \
}

DEFINE_EXACT_GROUP8_KERNEL(probe_native_q4_exact_rounding_group8_dim0_kernel_signed, 0)
DEFINE_EXACT_GROUP8_KERNEL(probe_native_q4_exact_rounding_group8_dim8_kernel_signed, 8)
DEFINE_EXACT_GROUP8_KERNEL(probe_native_q4_exact_rounding_group8_dim16_kernel_signed, 16)
DEFINE_EXACT_GROUP8_KERNEL(probe_native_q4_exact_rounding_group8_dim24_kernel_signed, 24)

#undef DEFINE_EXACT_GROUP8_KERNEL

#define DEFINE_EXACT_GROUP16_KERNEL(NAME, DIM) \
__attribute__((noinline)) v16accfloat NAME( \
    v16accfloat acc, \
    const uint4 *__restrict packed, \
    v16bfloat16 scale_low, \
    v16bfloat16 offset_low, \
    v32int16 activation_bits \
) { \
    return q4_exact_group_unroll_signed<DIM, DIM + 16>( \
        acc, \
        packed, \
        scale_low, \
        offset_low, \
        activation_bits \
    ); \
}

DEFINE_EXACT_GROUP16_KERNEL(probe_native_q4_exact_rounding_group16_dim0_kernel_signed, 0)
DEFINE_EXACT_GROUP16_KERNEL(probe_native_q4_exact_rounding_group16_dim16_kernel_signed, 16)

#undef DEFINE_EXACT_GROUP16_KERNEL

void probe_native_q4_exact_rounding_group4_call_chain_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;
        const uint4 *group_packed = packed + group * kQ4GroupSize * kQ4Rows;

        acc = probe_native_q4_exact_rounding_group4_dim0_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim4_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim8_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim12_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim16_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim20_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim24_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group4_dim28_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_group8_call_chain_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;
        const uint4 *group_packed = packed + group * kQ4GroupSize * kQ4Rows;

        acc = probe_native_q4_exact_rounding_group8_dim0_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group8_dim8_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group8_dim16_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group8_dim24_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_group16_call_chain_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;
        const uint4 *group_packed = packed + group * kQ4GroupSize * kQ4Rows;

        acc = probe_native_q4_exact_rounding_group16_dim0_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
        acc = probe_native_q4_exact_rounding_group16_dim16_kernel_signed(
            acc,
            group_packed,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_unroll32_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kQ4Rows);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kQ4Rows);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kQ4GroupSize);
        v32int16 activation_bits = (v32int16)activation_group;

        acc = q4_exact_group_unroll_signed<0, kQ4GroupSize>(
            acc,
            packed + group * kQ4GroupSize * kQ4Rows,
            scale_low,
            offset_low,
            activation_bits
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_group_sum_correction_chunk_lane_signed(
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
        acc,
        packed_lane,
        scale_lane,
        offset_lane,
        activation,
        activation_group_sum_bits
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

__attribute__((noinline)) void probe_native_q4_exact_rounding_chunk_lane_kernel_signed(
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc = q4_exact_chunk_lane_unroll_signed<0, kQ4Groups>(
        acc,
        packed_lane,
        scale_lane,
        offset_lane,
        activation
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_exact_rounding_chunk_two_lane_calls_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output
) {
    probe_native_q4_exact_rounding_chunk_lane_kernel_signed(
        packed,
        scale,
        offset,
        activation,
        output
    );
    probe_native_q4_exact_rounding_chunk_lane_kernel_signed(
        packed + kLaneNibbles,
        scale + kRowsPerLane,
        offset + kRowsPerLane,
        activation,
        output + kRowsPerLane
    );
}

void probe_native_q4_group_sum_correction_chunk_two_lanes_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc0 = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    v16accfloat acc1 = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc0 = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
        acc0,
        packed,
        scale,
        offset,
        activation,
        activation_group_sum_bits
    );
    acc1 = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
        acc1,
        packed + kLaneNibbles,
        scale + kRowsPerLane,
        offset + kRowsPerLane,
        activation,
        activation_group_sum_bits
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc0);
    *reinterpret_cast<v16bfloat16 *>(output + kRowsPerLane) = to_v16bfloat16(acc1);
}

void probe_native_q4_group_sum_correction_chunk_lane_loop_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
    for (int32_t lane = 0; lane < 2; lane++) {
        v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
        acc = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
            acc,
            packed + lane * kLaneNibbles,
            scale + lane * kRowsPerLane,
            offset + lane * kRowsPerLane,
            activation,
            activation_group_sum_bits
        );
        *reinterpret_cast<v16bfloat16 *>(output + lane * kRowsPerLane) =
            to_v16bfloat16(acc);
    }
}

__attribute__((noinline)) void probe_native_q4_group_sum_correction_chunk_lane_kernel_signed(
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    acc = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
        acc,
        packed_lane,
        scale_lane,
        offset_lane,
        activation,
        activation_group_sum_bits
    );
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_group_sum_correction_chunk_two_lane_calls_signed(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output
) {
    probe_native_q4_group_sum_correction_chunk_lane_kernel_signed(
        packed,
        scale,
        offset,
        activation,
        activation_group_sum_bits,
        output
    );
    probe_native_q4_group_sum_correction_chunk_lane_kernel_signed(
        packed + kLaneNibbles,
        scale + kRowsPerLane,
        offset + kRowsPerLane,
        activation,
        activation_group_sum_bits,
        output + kRowsPerLane
    );
}

void probe_native_q4_dequant_i16_activation_pair_mac_pipelined(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
    for (int32_t group = 0; group < groups; group++)
    chess_prepare_for_pipelining
    chess_loop_range(4, )
    {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16accfloat scaled0_acc =
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled0_bf16 = to_v16bfloat16(scaled0_acc);
        v16accfloat dequant0_acc =
            add(ups_to_v16accfloat(scaled0_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant0 = to_v16bfloat16(dequant0_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant0),
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16accfloat scaled1_acc =
            mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled1_bf16 = to_v16bfloat16(scaled1_acc);
        v16accfloat dequant1_acc =
            add(ups_to_v16accfloat(scaled1_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant1 = to_v16bfloat16(dequant1_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant1),
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_dequant_accfloat_pair_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        v64uint4 q4 =
            *reinterpret_cast<const v64uint4 *>(packed + group * 64);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v16accfloat offset_acc = ups_to_v16accfloat(offset_low);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16accfloat dequant0_acc =
            add(mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low)), offset_acc);
        v16bfloat16 dequant0 = to_v16bfloat16(dequant0_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant0),
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16accfloat dequant1_acc =
            add(mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low)), offset_acc);
        v16bfloat16 dequant1 = to_v16bfloat16(dequant1_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant1),
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_native_q4_v32load_dequant_pair_mac(
    const uint4 *__restrict packed,
    const bfloat16 *__restrict scale,
    const bfloat16 *__restrict offset,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        aie::vector<uint4, kQ4Rows> q4_vec =
            aie::load_v<kQ4Rows>(packed + group * kQ4Rows);
        v64uint4 q4 = set_v64uint4(0, (v32uint4)q4_vec);
        v64uint8 q8 = unpack(q4);
        v64uint16 q16 = unpack(q8);
        aie::vector<uint16, kVec32> q16_vec = extract_v32uint16(q16, 0);
        aie::vector<bfloat16, kVec32> q_bf16_vec =
            aie::to_float<bfloat16>(q16_vec, 0);
        v16bfloat16 scale_low =
            *reinterpret_cast<const v16bfloat16 *>(scale + group * kVec32);
        v16bfloat16 offset_low =
            *reinterpret_cast<const v16bfloat16 *>(offset + group * kVec32);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;

        v16bfloat16 q_low = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 0);
        v16accfloat scaled0_acc =
            mul_elem_16(set_v32bfloat16(0, q_low), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled0_bf16 = to_v16bfloat16(scaled0_acc);
        v16accfloat dequant0_acc =
            add(ups_to_v16accfloat(scaled0_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant0 = to_v16bfloat16(dequant0_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant0),
            (v32bfloat16)broadcast_elem(activation_bits, 0),
            acc,
            0,
            0,
            0
        );

        v16bfloat16 q_high = extract_v16bfloat16((v32bfloat16)q_bf16_vec, 1);
        v16accfloat scaled1_acc =
            mul_elem_16(set_v32bfloat16(0, q_high), set_v32bfloat16(0, scale_low));
        v16bfloat16 scaled1_bf16 = to_v16bfloat16(scaled1_acc);
        v16accfloat dequant1_acc =
            add(ups_to_v16accfloat(scaled1_bf16), ups_to_v16accfloat(offset_low));
        v16bfloat16 dequant1 = to_v16bfloat16(dequant1_acc);
        acc = mac_elem_16_conf(
            set_v32bfloat16(0, dequant1),
            (v32bfloat16)broadcast_elem(activation_bits, 1),
            acc,
            0,
            0,
            0
        );
    }
    *reinterpret_cast<v16bfloat16 *>(output) = to_v16bfloat16(acc);
}

void probe_aie_mac_native_activation_view(
    const bfloat16 *__restrict lhs,
    const bfloat16 *__restrict activation,
    float *__restrict output,
    int32_t groups
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, kVec16> acc = aie::zeros<accfloat, kVec16>();
#pragma clang loop unroll(disable)
    for (int32_t group = 0; group < groups; group++) {
        aie::vector<bfloat16, kVec16> lhs_vec =
            aie::load_v<kVec16>(lhs + group * kVec16);
        v32bfloat16 activation_group =
            *reinterpret_cast<const v32bfloat16 *>(activation + group * kVec32);
        v32int16 activation_bits = (v32int16)activation_group;
        aie::vector<bfloat16, kVec16> activation0 =
            extract_v16bfloat16(
                (v32bfloat16)broadcast_elem(activation_bits, 0),
                0
            );
        aie::vector<bfloat16, kVec16> activation1 =
            extract_v16bfloat16(
                (v32bfloat16)broadcast_elem(activation_bits, 1),
                0
            );
        acc = aie::mac(acc, lhs_vec, activation0);
        acc = aie::mac(acc, lhs_vec, activation1);
    }
    aie::store_v(output, acc.to_vector<float>());
}

void probe_lock_counted_loop(int32_t iterations) {
#pragma clang loop unroll(disable)
    for (int32_t i = 0; i < iterations; i++) {
        acquire_greater_equal(3, 1);
        release(2, 1);
    }
}

void probe_builtin_broadcast_elem_i16(
    const int16_t *__restrict input,
    int16_t *__restrict output,
    int32_t idx
) {
    v32int16 values = *reinterpret_cast<const v32int16 *>(input);
    v32int16 broadcast = broadcast_elem(values, idx);
    *reinterpret_cast<v32int16 *>(output) = broadcast;
}

void probe_builtin_broadcast_elem_bf16(
    const bfloat16 *__restrict input,
    bfloat16 *__restrict output,
    int32_t idx
) {
    v32bfloat16 values = *reinterpret_cast<const v32bfloat16 *>(input);
    v32bfloat16 broadcast = broadcast_elem(values, idx);
    *reinterpret_cast<v32bfloat16 *>(output) = broadcast;
}

void probe_builtin_shuffle_bf16(
    bfloat16 value,
    bfloat16 *__restrict output,
    uint32_t mode
) {
    v32bfloat16 shuffled = shuffle_bfloat16(value, mode);
    *reinterpret_cast<v32bfloat16 *>(output) = shuffled;
}

} // extern "C"
