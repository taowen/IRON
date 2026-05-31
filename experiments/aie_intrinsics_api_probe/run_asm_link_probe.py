#!/usr/bin/env python3
"""Prove that AIE C++ wrappers and source assembly can form one role object."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PEANO = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
DEFAULT_MLIR_AIE_INSTALL = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "build"
CPP_SOURCE = Path(__file__).resolve().parent / "asm_link_probe.cc"
ASM_SOURCE = Path(__file__).resolve().parent / "asm_link_probe.s"
SYMBOLS = (
    "probe_asm_store_word",
    "probe_asm_store_word_wrapper",
    "probe_asm_vector_mac_smoke",
    "probe_asm_vector_mac_smoke_wrapper",
    "probe_cpp_copy_word",
    "probe_asm_vector_mac_local_wrapper",
    "probe_asm_vector_mac_local_asm",
    "probe_asm_vector_copy_release",
    "probe_asm_bf16_mac_result_release",
    "probe_asm_q4_exact_unroll4_release",
    "probe_asm_q4_exact_group8_release",
    "probe_asm_q4_exact_group32_release",
    "probe_asm_q4_exact_lane8_loop_release",
)
PATTERNS = (
    "vlda",
    "vldb",
    "vmac.f",
    "vextbcst.16",
    "vunpack",
    "vups",
    "vst.conv.bf16.fp32",
    "vst",
    "j\t#0",
    "jl",
    "R_AIE_1",
)


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    clangxx: Path
    linker: Path
    nm: Path
    objdump: Path


@dataclass(frozen=True)
class LinkArtifacts:
    cpp_object: Path
    asm_object: Path
    combined_object: Path
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


def toolchain(peano: Path) -> Toolchain:
    return Toolchain(
        clang=peano / "bin/clang",
        clangxx=peano / "bin/clang++",
        linker=peano / "bin/ld.lld",
        nm=peano / "bin/llvm-nm",
        objdump=peano / "bin/llvm-objdump",
    )


def compile_and_link(tools: Toolchain, mlir_aie_install: Path, output_dir: Path) -> LinkArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    cpp_object = output_dir / "asm_link_probe.cc.o"
    asm_object = output_dir / "asm_link_probe.s.o"
    combined_object = output_dir / "asm_link_probe.combined.o"
    disasm = output_dir / "asm_link_probe.combined.s"
    include_dir = mlir_aie_install / "include"
    runtime_include_dir = mlir_aie_install / "aie_runtime_lib/AIE2P"
    run_command(
        (
            str(tools.clangxx),
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{include_dir}",
            f"-I{runtime_include_dir}",
            "-c",
            str(CPP_SOURCE),
            "-o",
            str(cpp_object),
        )
    )
    run_command(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-c",
            str(ASM_SOURCE),
            "-o",
            str(asm_object),
        )
    )
    run_command(
        (
            str(tools.linker),
            "-r",
            str(cpp_object),
            str(asm_object),
            "-o",
            str(combined_object),
        )
    )
    disasm.write_text(
        run_command(
            (
                str(tools.objdump),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                str(combined_object),
            )
        )
    )
    return LinkArtifacts(
        cpp_object=cpp_object,
        asm_object=asm_object,
        combined_object=combined_object,
        disasm=disasm,
    )


def summarize(artifacts: LinkArtifacts, tools: Toolchain) -> str:
    nm_text = run_command((str(tools.nm), str(artifacts.combined_object)))
    disasm_text = artifacts.disasm.read_text()
    lines = [
        "aie_asm_link_probe:",
        f"  cpp_object={artifacts.cpp_object}",
        f"  asm_object={artifacts.asm_object}",
        f"  combined_object={artifacts.combined_object}",
        f"  disasm={artifacts.disasm}",
        "  symbols:",
    ]
    for symbol in SYMBOLS:
        lines.append(f"    {symbol}={nm_text.count(symbol)}")
    lines.append(f"  unresolved={nm_text.count(' U ')}")
    lines.append("  instruction_counts:")
    for pattern in PATTERNS:
        lines.append(f"    {pattern}={disasm_text.count(pattern)}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--mlir-aie-install", type=Path, default=DEFAULT_MLIR_AIE_INSTALL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tools = toolchain(args.peano)
    artifacts = compile_and_link(
        tools=tools,
        mlir_aie_install=args.mlir_aie_install,
        output_dir=args.output_dir.resolve(),
    )
    print(summarize(artifacts, tools), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
