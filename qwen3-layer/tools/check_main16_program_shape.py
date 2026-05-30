#!/usr/bin/env python3
"""Check whether a main16 ELF has a MyLM-style raw phase-control shape."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CANDIDATE_ELF = Path("design.mlir.prj/main_core_2_2.elf")
DEFAULT_REFERENCE_ELF = Path("/tmp/mylm_solidify_L31/programs/elf/c2r2.elf")
DEFAULT_REFERENCE_DISASM = Path("/tmp/mylm_solidify_L31/disasm/c2r2.s")
DEFAULT_LLVM_OBJDUMP = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump")
DEFAULT_LLVM_SIZE = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-size")
RAW_MIN_TEXT_BYTES = 4096
RAW_MAX_TEXT_BYTES = 20000
MAX_JL = 32
MAX_ACQ = 64
MAX_REL = 64
MAX_JNZ = 32
MIN_LC_LS_LE = 2
Q4_HELPER_SYMBOLS = (
    "q4nx_chunk_accum_slice_i32_fast",
    "q4nx_chunk_accum_block_slice_i32_fast",
    "q4nx_chunk_accum_slice_i32",
    "q4nx_chunk_accum_block_slice_i32",
)
PATTERNS = {
    "jl": re.compile(r"\bjl\b"),
    "jnz": re.compile(r"\bjnz\b"),
    "acq": re.compile(r"\bacq\b"),
    "rel": re.compile(r"\brel\b"),
    "lc_ls_le": re.compile(r"\b(?:lc|ls|le)\b"),
    "mylm_q4_call": re.compile(r"\bjl\s+#0x1f0\b"),
}


@dataclass(frozen=True)
class ProgramShape:
    label: str
    path: Path
    text_bytes: int
    jl: int
    jnz: int
    acq: int
    rel: int
    lc_ls_le: int
    mylm_q4_calls: int
    q4_helper_refs: int


def _run(cmd: tuple[str, ...]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _text_size(llvm_size: Path, elf: Path) -> int:
    output = _run((str(llvm_size), "-A", str(elf)))
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text section in {elf}")


def _disassemble(llvm_objdump: Path, elf: Path) -> str:
    return _run((str(llvm_objdump), "-d", "--no-show-raw-insn", str(elf)))


def _q4_helper_refs(disasm: str) -> int:
    return sum(disasm.count(symbol) for symbol in Q4_HELPER_SYMBOLS)


def _shape(
    label: str,
    elf: Path,
    disasm_path: Path | None,
    llvm_objdump: Path,
    llvm_size: Path,
) -> ProgramShape:
    if not elf.exists():
        raise FileNotFoundError(elf)
    if disasm_path is not None:
        if not disasm_path.exists():
            raise FileNotFoundError(disasm_path)
        disasm = disasm_path.read_text()
    else:
        disasm = _disassemble(llvm_objdump, elf)
    counts = {name: len(pattern.findall(disasm)) for name, pattern in PATTERNS.items()}
    return ProgramShape(
        label=label,
        path=elf,
        text_bytes=_text_size(llvm_size, elf),
        jl=counts["jl"],
        jnz=counts["jnz"],
        acq=counts["acq"],
        rel=counts["rel"],
        lc_ls_le=counts["lc_ls_le"],
        mylm_q4_calls=counts["mylm_q4_call"],
        q4_helper_refs=_q4_helper_refs(disasm),
    )


def _shape_lines(shape: ProgramShape) -> list[str]:
    return [
        f"  {shape.label}:",
        f"    elf={shape.path}",
        f"    text_bytes={shape.text_bytes}",
        f"    jl={shape.jl}",
        f"    jnz={shape.jnz}",
        f"    acq={shape.acq}",
        f"    rel={shape.rel}",
        f"    lc_ls_le={shape.lc_ls_le}",
        f"    mylm_q4_calls_to_0x1f0={shape.mylm_q4_calls}",
        f"    q4_helper_symbol_refs={shape.q4_helper_refs}",
    ]


def _candidate_errors(candidate: ProgramShape) -> list[str]:
    errors: list[str] = []
    if candidate.text_bytes < RAW_MIN_TEXT_BYTES:
        errors.append(f"text_bytes {candidate.text_bytes} < {RAW_MIN_TEXT_BYTES}")
    if candidate.text_bytes > RAW_MAX_TEXT_BYTES:
        errors.append(f"text_bytes {candidate.text_bytes} > {RAW_MAX_TEXT_BYTES}")
    if candidate.jl > MAX_JL:
        errors.append(f"jl {candidate.jl} > {MAX_JL}")
    if candidate.acq > MAX_ACQ:
        errors.append(f"acq {candidate.acq} > {MAX_ACQ}")
    if candidate.rel > MAX_REL:
        errors.append(f"rel {candidate.rel} > {MAX_REL}")
    if candidate.jnz > MAX_JNZ:
        errors.append(f"jnz {candidate.jnz} > {MAX_JNZ}")
    if candidate.lc_ls_le < MIN_LC_LS_LE:
        errors.append(f"lc/ls/le evidence {candidate.lc_ls_le} < {MIN_LC_LS_LE}")
    if candidate.q4_helper_refs:
        errors.append(f"candidate still references C++ Q4 helper symbols: {candidate.q4_helper_refs}")
    return errors


def check_program_shape(
    candidate_elf: Path,
    candidate_disasm: Path | None,
    reference_elf: Path,
    reference_disasm: Path | None,
    llvm_objdump: Path,
    llvm_size: Path,
) -> tuple[str, bool]:
    reference = _shape("mylm_reference", reference_elf, reference_disasm, llvm_objdump, llvm_size)
    candidate = _shape("candidate", candidate_elf, candidate_disasm, llvm_objdump, llvm_size)
    errors = _candidate_errors(candidate)
    lines = [
        "main16_program_shape_check:",
        "  rule:",
        "    A candidate raw main16 program must look like a small fixed phase dispatcher,",
        "    not like MLIR-expanded per-chunk C++ helper control.",
        "  thresholds:",
        f"    text_bytes={RAW_MIN_TEXT_BYTES}..{RAW_MAX_TEXT_BYTES}",
        f"    jl<={MAX_JL} acq<={MAX_ACQ} rel<={MAX_REL} jnz<={MAX_JNZ} lc_ls_le>={MIN_LC_LS_LE}",
    ]
    lines.extend(_shape_lines(reference))
    lines.extend(_shape_lines(candidate))
    if errors:
        lines.append("  verdict=FAIL")
        lines.append("  reasons:")
        lines.extend(f"    - {error}" for error in errors)
        lines.append("  next_step=replace the generated main16 phase-control body, not only the ELF wrapper")
    else:
        lines.append("  verdict=PASS")
        lines.append("  next_step=package this ELF through package_externalized_design.py and run full-layer numeric gates")
    return "\n".join(lines) + "\n", not errors


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-elf", type=Path, default=DEFAULT_CANDIDATE_ELF)
    parser.add_argument("--candidate-disasm", type=Path, default=None)
    parser.add_argument("--reference-elf", type=Path, default=DEFAULT_REFERENCE_ELF)
    parser.add_argument("--reference-disasm", type=Path, default=DEFAULT_REFERENCE_DISASM)
    parser.add_argument("--llvm-objdump", type=Path, default=DEFAULT_LLVM_OBJDUMP)
    parser.add_argument("--llvm-size", type=Path, default=DEFAULT_LLVM_SIZE)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    text, passed = check_program_shape(
        candidate_elf=args.candidate_elf,
        candidate_disasm=args.candidate_disasm,
        reference_elf=args.reference_elf,
        reference_disasm=args.reference_disasm,
        llvm_objdump=args.llvm_objdump,
        llvm_size=args.llvm_size,
    )
    print(text, end="")
    return 0 if passed or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
