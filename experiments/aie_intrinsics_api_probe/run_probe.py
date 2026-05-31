#!/usr/bin/env python3
"""Compile small AIE C++ API kernels and summarize the generated AIE2P code."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PEANO = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
DEFAULT_MLIR_AIE_INSTALL = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "build"
SOURCE = Path(__file__).resolve().parent / "intrinsic_probe.cc"
FUNCTIONS = (
    "probe_bf16_load_store",
    "probe_bf16_broadcast_extract_mac",
    "probe_q4_unpack_broadcast_mac",
    "probe_native_bf16_vextbcst_mac",
    "probe_native_bf16_acc16_vextbcst_mac",
    "probe_native_bf16_i16_view_acc16_mac",
    "probe_arg_i16_vextbcst",
    "probe_arg_i16_view_bf16_acc16_mac",
    "probe_arg_i16_view_bf16_acc16_mac_signed",
    "probe_native_q4_unpack_ups",
    "probe_native_q4_unpack_bridge_mac",
    "probe_native_q4_dequant_i16_activation_mac",
    "probe_native_q4_dequant_i16_activation_pair_mac",
    "probe_native_q4_dequant_i16_activation_pair_mac_signed",
    "probe_native_q4_group_sum_correction_pair_mac_signed",
    "probe_native_q4_group_sum_correction_unroll8_signed",
    "probe_native_q4_group_sum_correction_unroll32_signed",
    "probe_native_q4_exact_rounding_unroll4_signed",
    "probe_native_q4_exact_rounding_unroll8_signed",
    "probe_native_q4_exact_rounding_unroll16_signed",
    "probe_native_q4_exact_rounding_group4_dim0_kernel_signed",
    "probe_native_q4_exact_rounding_group4_call_chain_signed",
    "probe_native_q4_exact_rounding_group8_dim0_kernel_signed",
    "probe_native_q4_exact_rounding_group8_call_chain_signed",
    "probe_native_q4_exact_rounding_group16_dim0_kernel_signed",
    "probe_native_q4_exact_rounding_group16_call_chain_signed",
    "probe_native_q4_exact_rounding_unroll32_signed",
    "probe_native_q4_group_sum_correction_chunk_lane_signed",
    "probe_native_q4_exact_rounding_chunk_lane_kernel_signed",
    "probe_native_q4_exact_rounding_chunk_two_lane_calls_signed",
    "probe_native_q4_group_sum_correction_chunk_two_lanes_signed",
    "probe_native_q4_group_sum_correction_chunk_lane_loop_signed",
    "probe_native_q4_group_sum_correction_chunk_lane_kernel_signed",
    "probe_native_q4_group_sum_correction_chunk_two_lane_calls_signed",
    "probe_native_q4_dequant_i16_activation_pair_mac_pipelined",
    "probe_native_q4_dequant_accfloat_pair_mac",
    "probe_native_q4_v32load_dequant_pair_mac",
    "probe_aie_mac_native_activation_view",
    "probe_lock_counted_loop",
    "probe_builtin_broadcast_elem_i16",
    "probe_builtin_broadcast_elem_bf16",
    "probe_builtin_shuffle_bf16",
)
PATTERNS = {
    "vmac_f": re.compile(r"\bvmac\.f\b"),
    "vmul_f": re.compile(r"\bvmul\.f\b"),
    "vadd": re.compile(r"\bvadd\b"),
    "vbcst_16": re.compile(r"\bvbcst\.16\b"),
    "vbcstshfl_16": re.compile(r"\bvbcstshfl\.16\b"),
    "vbcstshfl_32": re.compile(r"\bvbcstshfl\.32\b"),
    "vextbcst": re.compile(r"\bvextbcst\.(?:8|16|32|64|128)\b"),
    "vextbcst_16": re.compile(r"\bvextbcst\.16\b"),
    "vextbcst_32": re.compile(r"\bvextbcst\.32\b"),
    "vextract_16": re.compile(r"\bvextract\.16\b"),
    "vunpack": re.compile(r"\bvunpack\b"),
    "vups": re.compile(r"\bvups\b"),
    "vsrs": re.compile(r"\bvsrs\b"),
    "vconv_bf16_fp32": re.compile(r"\bvconv\.bf16\.fp32\b"),
    "vconv_fp32_bf16": re.compile(r"\bvconv\.fp32\.bf16\b"),
    "scalar_load": re.compile(r"\blda\.[su]16\b"),
    "vldb": re.compile(r"\bvldb\b"),
    "vst": re.compile(r"\bvst\b"),
    "control_33c": re.compile(r"#0x33c\b"),
    "control_03c": re.compile(r"#0x0?3c\b"),
    "jl": re.compile(r"\bjl\b"),
    "acq": re.compile(r"\bacq\b"),
    "rel": re.compile(r"\brel\b"),
    "hardware_loop": re.compile(r"\b(?:lc|ls|le)\b"),
}


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    objdump: Path


@dataclass(frozen=True)
class ProbeArtifacts:
    source: Path
    llvm_ir: Path
    object_file: Path
    disasm: Path


def run_command(cmd: tuple[str, ...]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed.stdout


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory exists: {output_dir}; pass --force")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def toolchain(peano: Path) -> Toolchain:
    return Toolchain(
        clang=peano / "bin/clang++",
        objdump=peano / "bin/llvm-objdump",
    )


def compile_probe(
    tools: Toolchain,
    mlir_aie_install: Path,
    output_dir: Path,
) -> ProbeArtifacts:
    llvm_ir = output_dir / "intrinsic_probe.ll"
    object_file = output_dir / "intrinsic_probe.o"
    disasm = output_dir / "intrinsic_probe.s"
    include_dir = mlir_aie_install / "include"
    runtime_include_dir = mlir_aie_install / "aie_runtime_lib/AIE2P"
    base_cmd = (
        str(tools.clang),
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
    run_command(base_cmd + ("-S", "-emit-llvm", str(SOURCE), "-o", str(llvm_ir)))
    run_command(base_cmd + ("-c", str(SOURCE), "-o", str(object_file)))
    disasm_text = run_command((str(tools.objdump), "-d", "--no-show-raw-insn", str(object_file)))
    disasm.write_text(disasm_text)
    return ProbeArtifacts(
        source=SOURCE,
        llvm_ir=llvm_ir,
        object_file=object_file,
        disasm=disasm,
    )


def count_patterns(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}


def function_body(disasm_text: str, function_name: str) -> str:
    start = re.search(rf"^[0-9a-fA-F]+ <{re.escape(function_name)}>:\n", disasm_text, re.MULTILINE)
    if start is None:
        raise ValueError(f"function not found in disassembly: {function_name}")
    next_func = re.search(
        r"^[0-9a-fA-F]+ <(?!\.)[^>]+>:\n",
        disasm_text[start.end():],
        re.MULTILINE,
    )
    if next_func is None:
        return disasm_text[start.end():]
    return disasm_text[start.end(): start.end() + next_func.start()]


def summarize(artifacts: ProbeArtifacts, mlir_aie_source: Path, llvm_aie_source: Path) -> str:
    disasm_text = artifacts.disasm.read_text()
    lines = [
        "aie_intrinsics_api_probe:",
        f"  source={artifacts.source}",
        f"  llvm_ir={artifacts.llvm_ir}",
        f"  object={artifacts.object_file}",
        f"  disasm={artifacts.disasm}",
        f"  mlir_aie_source={mlir_aie_source}",
        f"  llvm_aie_source={llvm_aie_source}",
        "  functions:",
    ]
    for function_name in FUNCTIONS:
        body = function_body(disasm_text, function_name)
        counts = count_patterns(body)
        lines.append(f"    {function_name}:")
        for name, count in counts.items():
            lines.append(f"      {name}={count}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--mlir-aie-install", type=Path, default=DEFAULT_MLIR_AIE_INSTALL)
    parser.add_argument("--mlir-aie-source", type=Path, default=Path("/var/home/taowen/projects/mlir-aie"))
    parser.add_argument("--llvm-aie-source", type=Path, default=Path("/var/home/taowen/projects/llvm-aie"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.force)
    artifacts = compile_probe(
        tools=toolchain(args.peano),
        mlir_aie_install=args.mlir_aie_install,
        output_dir=output_dir,
    )
    print(summarize(artifacts, args.mlir_aie_source, args.llvm_aie_source), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
