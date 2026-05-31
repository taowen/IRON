#!/usr/bin/env python3
"""Compile AIE2P source assembly probes and summarize their instruction shape."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PEANO = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "build"
SOURCE = Path(__file__).resolve().parent / "asm_probe.s"
PATTERNS = (
    "vmac.f",
    "vextbcst.16",
    "vextbcst.32",
    "vbcst.16",
    "lda.s16",
    "vlda",
    "vldb",
    "vunpack",
    "vups",
    "vconv.bf16.fp32",
    "vmul.f",
    "vst",
)
FUNCTIONS = (
    "probe_asm_q4_group_shape",
    "probe_asm_q4_group_with_prep_shape",
)


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    objdump: Path
    size: Path


@dataclass(frozen=True)
class AsmArtifacts:
    source: Path
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


def toolchain(peano: Path) -> Toolchain:
    return Toolchain(
        clang=peano / "bin/clang",
        objdump=peano / "bin/llvm-objdump",
        size=peano / "bin/llvm-size",
    )


def compile_probe(tools: Toolchain, output_dir: Path) -> AsmArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_file = output_dir / "asm_probe.o"
    disasm = output_dir / "asm_probe.s"
    run_command(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-c",
            str(SOURCE),
            "-o",
            str(object_file),
        )
    )
    disasm_text = run_command(
        (
            str(tools.objdump),
            "--triple=aie2p",
            "-dr",
            "--no-print-imm-hex",
            str(object_file),
        )
    )
    disasm.write_text(disasm_text)
    return AsmArtifacts(source=SOURCE, object_file=object_file, disasm=disasm)


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


def summarize(artifacts: AsmArtifacts, tools: Toolchain) -> str:
    text = artifacts.disasm.read_text()
    size_text = run_command((str(tools.size), str(artifacts.object_file))).strip()
    lines = [
        "aie_asm_probe:",
        f"  source={artifacts.source}",
        f"  object={artifacts.object_file}",
        f"  disasm={artifacts.disasm}",
        "  functions:",
    ]
    for function_name in FUNCTIONS:
        body = function_body(text, function_name)
        lines.append(f"    {function_name}:")
        for pattern in PATTERNS:
            lines.append(f"      {pattern}={body.count(pattern)}")
    lines.extend(
        [
            "  size:",
            "    " + size_text.replace("\n", "\n    "),
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tools = toolchain(args.peano)
    artifacts = compile_probe(tools, args.output_dir.resolve())
    print(summarize(artifacts, tools), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
