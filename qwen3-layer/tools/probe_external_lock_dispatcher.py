#!/usr/bin/env python3
"""Probe whether linked C++ AIE core code can own locks and loops."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PEANO = Path(".venv/lib/python3.12/site-packages/llvm-aie")
DEFAULT_OUTPUT_DIR = Path("/tmp/iron_external_lock_dispatcher_probe")
SOURCE = r'''
extern "C" void qwen3_external_lock_dispatcher_probe(int limit) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < limit; ++i) {
    acquire_greater_equal(3, 1);
    release(2, 1);
  }
}
'''
PATTERNS = {
    "acq": re.compile(r"\bacq\b"),
    "rel": re.compile(r"\brel\b"),
    "lc_ls_le": re.compile(r"\b(?:lc|ls|le)\b"),
    "jnz": re.compile(r"\bjnz\b"),
    "jl": re.compile(r"\bjl\b"),
}


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    llvm_objdump: Path


@dataclass(frozen=True)
class ProbeOutput:
    source: Path
    llvm_ir: Path
    object_file: Path
    disasm: Path
    acq: int
    rel: int
    lc_ls_le: int
    jnz: int
    jl: int


def _toolchain(peano: Path) -> Toolchain:
    bin_dir = peano / "bin"
    return Toolchain(
        clang=bin_dir / "clang++",
        llvm_objdump=bin_dir / "llvm-objdump",
    )


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory exists, pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _run(cmd: tuple[str, ...]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def _counts(disasm: str) -> dict[str, int]:
    return {name: len(pattern.findall(disasm)) for name, pattern in PATTERNS.items()}


def _require_shape(output: ProbeOutput) -> None:
    if output.acq != 1 or output.rel != 1:
        raise ValueError(f"expected one lock acquire/release in loop body, got acq={output.acq} rel={output.rel}")
    if output.lc_ls_le < 3:
        raise ValueError(f"expected an AIE hardware loop with lc/ls/le, got lc_ls_le={output.lc_ls_le}")


def probe_external_lock_dispatcher(peano: Path, output_dir: Path, force: bool) -> str:
    output_dir = output_dir.resolve()
    _prepare_output_dir(output_dir, force)
    tools = _toolchain(peano)

    source = output_dir / "external_lock_dispatcher_probe.cc"
    llvm_ir = output_dir / "external_lock_dispatcher_probe.ll"
    object_file = output_dir / "external_lock_dispatcher_probe.o"
    disasm_file = output_dir / "external_lock_dispatcher_probe.s"
    source.write_text(SOURCE)

    _run(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-S",
            "-emit-llvm",
            str(source),
            "-o",
            str(llvm_ir),
        )
    )
    _run(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
            str(source),
            "-o",
            str(object_file),
        )
    )
    disasm = _run((str(tools.llvm_objdump), "-d", "--no-show-raw-insn", str(object_file)))
    disasm_file.write_text(disasm)
    counts = _counts(disasm)
    output = ProbeOutput(
        source=source,
        llvm_ir=llvm_ir,
        object_file=object_file,
        disasm=disasm_file,
        acq=counts["acq"],
        rel=counts["rel"],
        lc_ls_le=counts["lc_ls_le"],
        jnz=counts["jnz"],
        jl=counts["jl"],
    )
    _require_shape(output)
    return (
        "external_lock_dispatcher_probe:\n"
        f"  source={output.source}\n"
        f"  llvm_ir={output.llvm_ir}\n"
        f"  object={output.object_file}\n"
        f"  disasm={output.disasm}\n"
        "  compile_path=Peano clang++ --target=aie2p-none-unknown-elf -O2\n"
        f"  disasm_acq={output.acq}\n"
        f"  disasm_rel={output.rel}\n"
        f"  disasm_lc_ls_le={output.lc_ls_le}\n"
        f"  disasm_jnz={output.jnz}\n"
        f"  disasm_jl={output.jl}\n"
        "  conclusion=linked C++ AIE core code can compile AIE2P lock builtins and generate a hardware loop; final linked core kernels still need packet/BD/lock contract checks and hardware validation.\n"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(
        probe_external_lock_dispatcher(
            peano=args.peano,
            output_dir=args.output_dir,
            force=args.force,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
