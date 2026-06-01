#!/usr/bin/env python3
"""Compile narrow Q4NX candidate kernels with Peano and score the assembly."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PEANO_DIR = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
MLIR_AIE_DIR = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "peano_q4nx_compiler_route.json"
REPORT = EXPERIMENT_DIR / "peano_q4nx_compiler_route.md"

SOURCE = r"""
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
    aie::vector<bfloat16, kVec32> q_bf16_vec = aie::to_float<bfloat16>(q16_vec, 0);

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
    aie::vector<bfloat16, kVec32> q_bf16_vec = aie::to_float<bfloat16>(q16_vec, 0);

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

void peano_mac_signed_probe(
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

__attribute__((noinline)) void peano_q4_group_sum_chunk_lane_unrolled(
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

__attribute__((noinline)) void peano_q4_group_sum_chunk_lane_loop(
    const uint4 *__restrict packed_lane,
    const bfloat16 *__restrict scale_lane,
    const bfloat16 *__restrict offset_lane,
    const bfloat16 *__restrict activation,
    const int16_t *__restrict activation_group_sum_bits,
    bfloat16 *__restrict output,
    int32_t chunks
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
#pragma clang loop min_iteration_count(2)
    for (int32_t chunk = 0; chunk < chunks; chunk++) {
        v16accfloat acc = extract_v16accfloat(broadcast_zero_to_v32accfloat(), 0);
        acc = q4_scaled_chunk_lane_unroll_signed<0, kQ4Groups>(
            acc,
            packed_lane + chunk * kQ4Groups * kGroupNibblesPerLane,
            scale_lane + chunk * kQ4Groups * kQ4Rows,
            offset_lane + chunk * kQ4Groups * kQ4Rows,
            activation + chunk * kQ4Columns,
            activation_group_sum_bits + chunk * kQ4Groups
        );
        *reinterpret_cast<v16bfloat16 *>(output + chunk * kVec16) = to_v16bfloat16(acc);
    }
}

__attribute__((noinline)) void peano_q4_exact_chunk_lane_unrolled(
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

} // extern "C"
"""


PATTERNS = {
    "vmac_f": re.compile(r"\bvmac\.f\b"),
    "vmul_f": re.compile(r"\bvmul\.f\b"),
    "vadd": re.compile(r"\bvadd\b"),
    "vsub_f": re.compile(r"\bvsub\.f\b"),
    "vbcst_16": re.compile(r"\bvbcst\.16\b"),
    "vextbcst_16": re.compile(r"\bvextbcst\.16\b"),
    "vextbcst_32": re.compile(r"\bvextbcst\.32\b"),
    "vunpack": re.compile(r"\bvunpack\b"),
    "vups": re.compile(r"\bvups\b"),
    "vups_2x": re.compile(r"\bvups\.2x\b"),
    "vups_4x": re.compile(r"\bvups\.4x\b"),
    "vconv_bf16_fp32": re.compile(r"\bvconv\.bf16\.fp32\b"),
    "vconv_fp32_bf16": re.compile(r"\bvconv\.fp32\.bf16\b"),
    "vldb": re.compile(r"\bvldb\b"),
    "vlda": re.compile(r"\bvlda\b"),
    "vst": re.compile(r"\bvst\b"),
    "control_33c": re.compile(r"#0x33c\b"),
    "control_03c": re.compile(r"#0x0?3c\b"),
    "hardware_loop_regs": re.compile(r"\b(?:lc|ls|le)\b"),
}

FUNCTIONS = (
    "peano_mac_signed_probe",
    "peano_q4_group_sum_chunk_lane_unrolled",
    "peano_q4_group_sum_chunk_lane_loop",
    "peano_q4_exact_chunk_lane_unrolled",
)


@dataclass(frozen=True)
class BuildArtifacts:
    source: Path
    llvm_ir: Path
    obj: Path
    disasm: Path


def run_cmd(cmd: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def reset_build_dir(force: bool) -> None:
    if BUILD_DIR.exists():
        if not force:
            raise FileExistsError(f"build directory exists: {BUILD_DIR}; pass --force")
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)


def peano_tools(peano_dir: Path) -> tuple[Path, Path]:
    clang = peano_dir / "bin/clang++"
    objdump = peano_dir / "bin/llvm-objdump"
    if not clang.exists() or not objdump.exists():
        raise FileNotFoundError(f"missing Peano tools under {peano_dir}")
    return clang, objdump


def compile_source(peano_dir: Path, mlir_aie_dir: Path) -> tuple[BuildArtifacts, str]:
    clang, objdump = peano_tools(peano_dir)
    source = BUILD_DIR / "peano_q4nx_compiler_route.cc"
    llvm_ir = BUILD_DIR / "peano_q4nx_compiler_route.ll"
    obj = BUILD_DIR / "peano_q4nx_compiler_route.o"
    disasm = BUILD_DIR / "peano_q4nx_compiler_route.s"
    source.write_text(SOURCE)
    include_dir = mlir_aie_dir / "include"
    runtime_include_dir = mlir_aie_dir / "aie_runtime_lib/AIE2P"
    base_cmd = (
        str(clang),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_dir}",
        f"-I{runtime_include_dir}",
    )
    run_cmd(base_cmd + ("-S", "-emit-llvm", str(source), "-o", str(llvm_ir)))
    compile_result = run_cmd(
        base_cmd
        + (
            "-Rpass=postpipeliner",
            "-Rpass-missed=postpipeliner",
            "-Rpass-analysis=aie-hardware-loops",
            "-c",
            str(source),
            "-o",
            str(obj),
        )
    )
    disasm_text = run_cmd((str(objdump), "-d", "--no-show-raw-insn", str(obj))).stdout
    disasm.write_text(disasm_text)
    return BuildArtifacts(source=source, llvm_ir=llvm_ir, obj=obj, disasm=disasm), compile_result.stderr


def function_body(disasm_text: str, function_name: str) -> str:
    start = re.search(rf"^[0-9a-fA-F]+ <{re.escape(function_name)}>:\n", disasm_text, re.MULTILINE)
    if start is None:
        raise ValueError(f"function not found in disassembly: {function_name}")
    next_func = re.search(r"^[0-9a-fA-F]+ <(?!\.)[^>]+>:\n", disasm_text[start.end():], re.MULTILINE)
    if next_func is None:
        return disasm_text[start.end():]
    return disasm_text[start.end(): start.end() + next_func.start()]


def count_patterns(body: str) -> dict[str, int]:
    counts = {name: len(pattern.findall(body)) for name, pattern in PATTERNS.items()}
    counts["instruction_lines"] = sum(1 for line in body.splitlines() if line.strip() and ":" in line)
    return counts


def classify(function_name: str, counts: dict[str, int]) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if function_name == "peano_mac_signed_probe":
        if counts["vextbcst_16"] != 1 or counts["vmac_f"] != 1 or counts["control_33c"] != 1:
            reasons.append("signed mac probe did not preserve vextbcst.16/vmac.f/#0x33c")
    if function_name == "peano_q4_group_sum_chunk_lane_unrolled":
        expected = {"vmac_f": 264, "vextbcst_16": 256, "vextbcst_32": 0}
        for key, value in expected.items():
            if counts[key] != value:
                reasons.append(f"{key}={counts[key]} expected {value}")
        if counts["vst"] > 8:
            reasons.append(f"spill/store pressure too high: vst={counts['vst']} > 8")
        if counts["vmul_f"] > 32:
            reasons.append(f"still scale-multiplies per dim: vmul.f={counts['vmul_f']} > 32")
        if counts["vconv_bf16_fp32"] > 160:
            reasons.append(f"conversion pressure too high: vconv.bf16.fp32={counts['vconv_bf16_fp32']} > 160")
    if function_name == "peano_q4_group_sum_chunk_lane_loop":
        if counts["hardware_loop_regs"] == 0:
            reasons.append("loop candidate did not expose hardware-loop registers")
        if counts["vmul_f"] > 32:
            reasons.append(f"still scale-multiplies per dim: vmul.f={counts['vmul_f']} > 32")
        if counts["vconv_bf16_fp32"] > 160:
            reasons.append(f"conversion pressure too high: vconv.bf16.fp32={counts['vconv_bf16_fp32']} > 160")
        if counts["vlda"] > 64:
            reasons.append(f"loop schedule uses too many vector local loads: vlda={counts['vlda']} > 64")
        if counts["vst"] > 32:
            reasons.append(f"loop form spills too much: vst={counts['vst']} > 32")
    if function_name == "peano_q4_exact_chunk_lane_unrolled":
        if counts["vst"] > 128:
            reasons.append(f"exact rounding form is not viable as compiler route: vst={counts['vst']} > 128")
    if reasons:
        return "fail", tuple(reasons)
    return "pass", ()


def build_manifest(artifacts: BuildArtifacts, remarks: str) -> dict[str, object]:
    disasm_text = artifacts.disasm.read_text()
    functions = {}
    for name in FUNCTIONS:
        body = function_body(disasm_text, name)
        counts = count_patterns(body)
        status, reasons = classify(name, counts)
        functions[name] = {
            "status": status,
            "reasons": list(reasons),
            "counts": counts,
        }
    return {
        "experiment": "020_peano_q4nx_compiler_route",
        "status": "partial",
        "artifacts": {
            "source": str(artifacts.source),
            "llvm_ir": str(artifacts.llvm_ir),
            "object": str(artifacts.obj),
            "disasm": str(artifacts.disasm),
        },
        "remarks_stderr": remarks.strip().splitlines(),
        "mylm_static_target": {
            "vmac_f": 264,
            "vextbcst_16": 256,
            "vextbcst_32": 0,
            "vups_4x": 64,
            "vunpack": 64,
            "vst_hot_loop": 0,
            "vconv_bf16_fp32": 136,
            "vmul_f": 8,
        },
        "functions": functions,
        "postpipeliner_missed": "No schedule found" in remarks,
        "conclusion": (
            "Peano can generate the signed vextbcst.16/vmac.f primitive and the "
            "264/256 group-sum macro shape, but the current C++ intrinsic form "
            "still has too much scale multiplication/conversion and does not match "
            "MyLM's register-resident vups.4x schedule."
        ),
        "next_step": (
            "Keep Peano as backend, but generate a narrower DSL/MIR-level Q4NX "
            "body that exposes MyLM's vups.4x dequant pipeline instead of emitting "
            "per-dim C++ scale/dequant operations."
        ),
    }


def write_report(manifest: dict[str, object]) -> None:
    lines = [
        "# Peano Q4NX Compiler Route",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "This experiment tries the compiler route for the main16 Q4NX hot body.",
        "It uses Peano/llvm-aie to compile narrow C++ intrinsic candidates and scores the resulting AIE2P assembly.",
        "",
        "## MyLM Target",
        "",
    ]
    target = manifest["mylm_static_target"]
    if not isinstance(target, dict):
        raise TypeError("manifest target must be a dict")
    for key, value in target.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Candidate Results", ""])
    functions = manifest["functions"]
    if not isinstance(functions, dict):
        raise TypeError("manifest functions must be a dict")
    for name, raw_info in functions.items():
        if not isinstance(raw_info, dict):
            raise TypeError("function info must be a dict")
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"Status: `{raw_info['status']}`")
        reasons = raw_info["reasons"]
        if isinstance(reasons, list) and reasons:
            lines.append("")
            lines.append("Reasons:")
            for reason in reasons:
                lines.append(f"- {reason}")
        counts = raw_info["counts"]
        if not isinstance(counts, dict):
            raise TypeError("counts must be a dict")
        lines.append("")
        lines.append("Key counts:")
        for key in (
            "instruction_lines",
            "vmac_f",
            "vextbcst_16",
            "vextbcst_32",
            "vunpack",
            "vups",
            "vups_2x",
            "vups_4x",
            "vmul_f",
            "vconv_bf16_fp32",
            "vconv_fp32_bf16",
            "vldb",
            "vst",
            "control_33c",
            "hardware_loop_regs",
        ):
            lines.append(f"- `{key}`: `{counts.get(key, 0)}`")
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            str(manifest["conclusion"]),
            "",
            "## Next Step",
            "",
            str(manifest["next_step"]),
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))


def run(force: bool, peano_dir: Path, mlir_aie_dir: Path) -> dict[str, object]:
    reset_build_dir(force)
    artifacts, remarks = compile_source(peano_dir, mlir_aie_dir)
    manifest = build_manifest(artifacts, remarks)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--peano-dir", type=Path, default=PEANO_DIR)
    parser.add_argument("--mlir-aie-dir", type=Path, default=MLIR_AIE_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        manifest = run(args.force, args.peano_dir, args.mlir_aie_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
