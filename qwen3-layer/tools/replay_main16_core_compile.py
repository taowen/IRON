#!/usr/bin/env python3
"""Replay main16 core compilation with phase-loop unrolling disabled."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_DIR = Path("design.mlir.prj")
DEFAULT_OUTPUT_DIR = Path("/tmp/iron_main16_disable_unroll_elfs")
DEFAULT_PEANO = Path(".venv/lib/python3.12/site-packages/llvm-aie")
MAIN16_TILES = tuple((col, row) for col in range(2, 6) for row in range(2, 6))
Q4_CALL_SYMBOLS = (
    "q4nx_chunk_accum_slice_i32_fast",
    "q4nx_chunk_accum_block_slice_i32_fast",
    "q4nx_chunk_accum_slice_i32",
    "q4nx_chunk_accum_block_slice_i32",
)
OPT_FLAGS = ("--passes=default<O1>", "-inline-threshold=10", "-disable-loop-unrolling")
DISASM_PATTERNS = {
    "jl": re.compile(r"\bjl\b"),
    "jnz": re.compile(r"\bjnz\b"),
    "acq": re.compile(r"\bacq\b"),
    "rel": re.compile(r"\brel\b"),
    "lc_ls_le": re.compile(r"\b(?:lc|ls|le)\b"),
}


@dataclass(frozen=True)
class Toolchain:
    opt: Path
    llc: Path
    clang: Path
    ld_lld: Path
    llvm_objdump: Path
    llvm_size: Path
    llvm_nm: Path


@dataclass(frozen=True)
class CoreInput:
    core_name: str
    llvm_ir: Path
    peanohack_ir: Path
    linker_script: Path


@dataclass(frozen=True)
class CoreOutput:
    core_name: str
    opt_ir: Path
    object_file: Path
    elf: Path
    llvm_q4_refs: int
    opt_q4_refs: int
    text_bytes: int
    jl: int
    jnz: int
    acq: int
    rel: int
    lc_ls_le: int


def _run(cmd: tuple[str, ...]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return result.stdout


def _toolchain(peano: Path) -> Toolchain:
    bin_dir = peano / "bin"
    return Toolchain(
        opt=bin_dir / "opt",
        llc=bin_dir / "llc",
        clang=bin_dir / "clang",
        ld_lld=bin_dir / "ld.lld",
        llvm_objdump=bin_dir / "llvm-objdump",
        llvm_size=bin_dir / "llvm-size",
        llvm_nm=bin_dir / "llvm-nm",
    )


def _core_input(project_dir: Path, col: int, row: int) -> CoreInput:
    core_name = f"main_core_{col}_{row}"
    item = CoreInput(
        core_name=core_name,
        llvm_ir=project_dir / f"{core_name}.ll",
        peanohack_ir=project_dir / f"{core_name}.peanohack.ll",
        linker_script=project_dir / f"{core_name}.ld.script",
    )
    for path in (item.llvm_ir, item.peanohack_ir, item.linker_script):
        if not path.exists():
            raise FileNotFoundError(f"missing generated core artifact: {path}")
    return item


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory exists, pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _count(text: str, needle: str) -> int:
    return len(re.findall(re.escape(needle), text))


def _count_q4_refs(text: str) -> int:
    return sum(_count(text, symbol) for symbol in Q4_CALL_SYMBOLS)


def _text_size(tools: Toolchain, elf: Path) -> int:
    output = _run((str(tools.llvm_size), "-A", str(elf)))
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text size in {elf}")


def _disasm_counts(tools: Toolchain, elf: Path) -> dict[str, int]:
    disasm = _run((str(tools.llvm_objdump), "-d", "--no-show-raw-insn", str(elf)))
    return {name: len(pattern.findall(disasm)) for name, pattern in DISASM_PATTERNS.items()}


def _require_no_undefined_symbols(tools: Toolchain, elf: Path) -> None:
    output = _run((str(tools.llvm_nm), "-C", str(elf)))
    undefined = tuple(line.strip() for line in output.splitlines() if " U " in f" {line} ")
    if undefined:
        joined = "\n".join(undefined)
        raise ValueError(f"replacement ELF has undefined symbols: {elf}\n{joined}")


def _compile_core(core: CoreInput, output_dir: Path, tools: Toolchain) -> CoreOutput:
    work_dir = output_dir / "work"
    work_dir.mkdir(exist_ok=True)
    opt_ir = work_dir / f"{core.core_name}.disable_unroll.opt.ll"
    object_file = work_dir / f"{core.core_name}.disable_unroll.o"
    elf = output_dir / f"{core.core_name}.elf"
    _run(
        (
            str(tools.opt),
            *OPT_FLAGS,
            "-S",
            str(core.peanohack_ir),
            "-o",
            str(opt_ir),
        )
    )
    _run(
        (
            str(tools.llc),
            str(opt_ir),
            "-O2",
            "--march=aie2p",
            "--function-sections",
            "--filetype=obj",
            "-o",
            str(object_file),
        )
    )
    _run(
        (
            str(tools.clang),
            "-O2",
            "--target=aie2p-none-unknown-elf",
            f"-fuse-ld={tools.ld_lld.resolve()}",
            str(object_file),
            "-Wl,--gc-sections",
            "-Wl,--orphan-handling=error",
            f"-Wl,-T,{core.linker_script.resolve()}",
            "-o",
            str(elf),
        )
    )
    _require_no_undefined_symbols(tools, elf)
    counts = _disasm_counts(tools, elf)
    return CoreOutput(
        core_name=core.core_name,
        opt_ir=opt_ir,
        object_file=object_file,
        elf=elf,
        llvm_q4_refs=_count_q4_refs(core.llvm_ir.read_text()),
        opt_q4_refs=_count_q4_refs(opt_ir.read_text()),
        text_bytes=_text_size(tools, elf),
        jl=counts["jl"],
        jnz=counts["jnz"],
        acq=counts["acq"],
        rel=counts["rel"],
        lc_ls_le=counts["lc_ls_le"],
    )


def _format_core(output: CoreOutput) -> str:
    return (
        f"    {output.core_name}: text_bytes={output.text_bytes} "
        f"llvm_q4_refs={output.llvm_q4_refs} opt_q4_refs={output.opt_q4_refs} "
        f"jl={output.jl} jnz={output.jnz} acq={output.acq} rel={output.rel} "
        f"lc_ls_le={output.lc_ls_le}"
    )


def replay_main16_cores(project_dir: Path, output_dir: Path, peano: Path, force: bool) -> str:
    project_dir = project_dir.resolve()
    output_dir = output_dir.resolve()
    _prepare_output_dir(output_dir, force)
    tools = _toolchain(peano)
    outputs = tuple(
        _compile_core(_core_input(project_dir, col, row), output_dir, tools)
        for col, row in MAIN16_TILES
    )
    text_sizes = tuple(sorted({item.text_bytes for item in outputs}))
    opt_q4_refs = tuple(sorted({item.opt_q4_refs for item in outputs}))
    acq_counts = tuple(sorted({item.acq for item in outputs}))
    rel_counts = tuple(sorted({item.rel for item in outputs}))
    lines = [
        "main16_core_replay:",
        f"  project_dir={project_dir}",
        f"  output_dir={output_dir}",
        "  compile_path=opt --passes=default<O1> -inline-threshold=10 -disable-loop-unrolling -> llc -> clang",
        f"  cores={len(outputs)}",
        f"  text_bytes={','.join(str(value) for value in text_sizes)}",
        f"  opt_q4_refs={','.join(str(value) for value in opt_q4_refs)}",
        f"  acq_counts={','.join(str(value) for value in acq_counts)}",
        f"  rel_counts={','.join(str(value) for value in rel_counts)}",
        "  per_core:",
    ]
    lines.extend(_format_core(item) for item in outputs)
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(
        replay_main16_cores(
            project_dir=args.project_dir,
            output_dir=args.output_dir,
            peano=args.peano,
            force=args.force,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
