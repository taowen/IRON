#!/usr/bin/env python3
"""Check qwen3 main16 source-assembly integration invariants."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from generate_main16_q4nx_asm import generate_assembly


DEFAULT_ROLE_OBJECT = Path("qwen3-layer/main_projection_q4nx_fast.o")
DEFAULT_CORE_ELF = Path("design.mlir.prj/main_core_2_2.elf")
DEFAULT_ASM_SOURCE = Path("qwen3-layer/main_projection_q4nx_asm.s")
DEFAULT_LLVM_NM = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-nm")
DEFAULT_LLVM_OBJDUMP = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump")
ASM_GROUP_SYMBOL = "q4nx_accum_lane_asm_group_shape"
ASM_EXACT_LANE_SYMBOL = "q4nx_accum_lane_exact_body_shape"
REJECTED_LANE_SYMBOL = "q4nx_accum_lane_asm"
NATIVE_SYMBOL = "q4nx_accum_lane_native"
PRODUCTION_Q4_SYMBOL = "_ZN12_GLOBAL__N_121q4nx_chunk_accum_fastEPfP8bfloat16S2_"
EXPECTED_GROUP_COUNTS = {
    "vmac.f": 33,
    "vextbcst.16": 32,
    "vextbcst.32": 0,
    "vbcst.16": 1,
    "lda.s16": 1,
    "vlda": 1,
    "vldb": 3,
    "vunpack": 8,
    "vups.4x": 8,
    "vst": 0,
}
EXPECTED_EXACT_LANE_COUNTS = {
    "vmac.f": 32,
    "vextbcst.16": 32,
    "vextbcst.32": 0,
    "vbcst.16": 8,
    "vbcst.32": 1,
    "vst.conv.bf16.fp32": 1,
    "rel\t": 0,
    "jnz": 1,
    "vunpack": 32,
    "vups.2x": 16,
    "vups.4x": 0,
    "vmul.f": 32,
    "vadd.f": 32,
    "vconv.bf16.fp32": 80,
    "vconv.fp32.bf16": 48,
    "vldb": 40,
    "vlda": 0,
    "vst": 1,
}
PRODUCTION_REQUIRED_COUNTS = {
    "vmac.f": 64,
    "vextbcst.16": 64,
    "vextbcst.32": 0,
}
PRODUCTION_REPORT_COUNTS = (
    "vmac.f",
    "vextbcst.16",
    "vextbcst.32",
    "vunpack",
    "vups.4x",
    "vmul.f",
    "vadd.f",
    "vconv.bf16.fp32",
    "vlda",
    "vldb",
    "vst",
)


@dataclass(frozen=True)
class AsmIntegrationReport:
    role_object: Path
    core_elf: Path
    asm_source: Path
    source_matches_generator: bool
    role_has_asm_group: bool
    role_has_exact_lane: bool
    role_has_rejected_lane: bool
    core_has_asm_group: bool
    core_has_exact_lane: bool
    has_native_symbol: bool
    has_bad_symbol_jump: bool
    group_counts: dict[str, int]
    exact_lane_counts: dict[str, int]
    role_has_production_q4: bool
    production_counts: dict[str, int]
    errors: tuple[str, ...]


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


def disassemble(llvm_objdump: Path, path: Path) -> str:
    return run_command(
        (
            str(llvm_objdump),
            "--triple=aie2p",
            "-dr",
            "--no-print-imm-hex",
            str(path),
        )
    )


def symbol_table(llvm_nm: Path, path: Path) -> str:
    return run_command((str(llvm_nm), str(path)))


def symbol_names(symbol_text: str) -> set[str]:
    return {
        line.split()[-1]
        for line in symbol_text.splitlines()
        if line.split()
    }


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


def check_asm_integration(
    role_object: Path,
    core_elf: Path,
    asm_source: Path,
    llvm_nm: Path,
    llvm_objdump: Path,
) -> AsmIntegrationReport:
    if not role_object.exists():
        raise FileNotFoundError(role_object)
    if not core_elf.exists():
        raise FileNotFoundError(core_elf)
    if not asm_source.exists():
        raise FileNotFoundError(asm_source)

    source_matches_generator = asm_source.read_text() == generate_assembly()
    role_symbols = symbol_table(llvm_nm, role_object)
    core_symbols = symbol_table(llvm_nm, core_elf)
    role_names = symbol_names(role_symbols)
    core_names = symbol_names(core_symbols)
    role_has_asm_group = ASM_GROUP_SYMBOL in role_names
    role_has_exact_lane = ASM_EXACT_LANE_SYMBOL in role_names
    role_has_rejected_lane = REJECTED_LANE_SYMBOL in role_names
    role_has_production_q4 = PRODUCTION_Q4_SYMBOL in role_names
    core_has_asm_group = ASM_GROUP_SYMBOL in core_names
    core_has_exact_lane = ASM_EXACT_LANE_SYMBOL in core_names
    has_native_symbol = NATIVE_SYMBOL in role_names or NATIVE_SYMBOL in core_names
    role_disasm = disassemble(llvm_objdump, role_object)
    body = function_body(role_disasm, ASM_GROUP_SYMBOL)
    group_counts = {pattern: body.count(pattern) for pattern in EXPECTED_GROUP_COUNTS}
    exact_lane_body = function_body(role_disasm, ASM_EXACT_LANE_SYMBOL)
    exact_lane_counts = {
        pattern: exact_lane_body.count(pattern)
        for pattern in EXPECTED_EXACT_LANE_COUNTS
    }
    if role_has_production_q4:
        production_body = function_body(role_disasm, PRODUCTION_Q4_SYMBOL)
        production_counts = {
            pattern: production_body.count(pattern)
            for pattern in PRODUCTION_REPORT_COUNTS
        }
    else:
        production_counts = {pattern: 0 for pattern in PRODUCTION_REPORT_COUNTS}
    has_bad_symbol_jump = bool(
        re.search(r"\bj\s+#0\b", role_disasm)
        and re.search(r"R_AIE_1\s+\*ABS\*", role_disasm)
    )

    errors: list[str] = []
    if not role_has_asm_group:
        errors.append(f"role object missing {ASM_GROUP_SYMBOL}")
    if not role_has_exact_lane:
        errors.append(f"role object missing {ASM_EXACT_LANE_SYMBOL}")
    if not source_matches_generator:
        errors.append(f"main16 source assembly drifted from generator: {asm_source}")
    if role_has_rejected_lane:
        errors.append(f"rejected approximate lane body is still present: {REJECTED_LANE_SYMBOL}")
    if core_has_asm_group:
        errors.append(f"unreferenced {ASM_GROUP_SYMBOL} survived into final core ELF")
    if core_has_exact_lane:
        errors.append(f"unreferenced {ASM_EXACT_LANE_SYMBOL} survived into final core ELF")
    if has_native_symbol:
        errors.append(f"old C++ lane symbol is still present: {NATIVE_SYMBOL}")
    if has_bad_symbol_jump:
        errors.append("source assembly contains a symbol jump that resolved as j #0")
    for pattern, expected in EXPECTED_GROUP_COUNTS.items():
        got = group_counts[pattern]
        if got != expected:
            errors.append(f"{ASM_GROUP_SYMBOL}: {pattern} expected={expected} got={got}")
    for pattern, expected in EXPECTED_EXACT_LANE_COUNTS.items():
        got = exact_lane_counts[pattern]
        if got != expected:
            errors.append(f"{ASM_EXACT_LANE_SYMBOL}: {pattern} expected={expected} got={got}")
    if not role_has_production_q4:
        errors.append(f"role object missing production Q4 function: {PRODUCTION_Q4_SYMBOL}")
    for pattern, expected in PRODUCTION_REQUIRED_COUNTS.items():
        got = production_counts[pattern]
        if got != expected:
            errors.append(f"{PRODUCTION_Q4_SYMBOL}: {pattern} expected={expected} got={got}")

    return AsmIntegrationReport(
        role_object=role_object,
        core_elf=core_elf,
        asm_source=asm_source,
        source_matches_generator=source_matches_generator,
        role_has_asm_group=role_has_asm_group,
        role_has_exact_lane=role_has_exact_lane,
        role_has_rejected_lane=role_has_rejected_lane,
        core_has_asm_group=core_has_asm_group,
        core_has_exact_lane=core_has_exact_lane,
        has_native_symbol=has_native_symbol,
        has_bad_symbol_jump=has_bad_symbol_jump,
        group_counts=group_counts,
        exact_lane_counts=exact_lane_counts,
        role_has_production_q4=role_has_production_q4,
        production_counts=production_counts,
        errors=tuple(errors),
    )


def render_report(report: AsmIntegrationReport) -> str:
    lines = [
        "main16_asm_integration_check:",
        f"  role_object={report.role_object}",
        f"  core_elf={report.core_elf}",
        f"  asm_source={report.asm_source}",
        f"  source_matches_generator={str(report.source_matches_generator).lower()}",
        f"  role_has_asm_group={str(report.role_has_asm_group).lower()}",
        f"  role_has_exact_lane={str(report.role_has_exact_lane).lower()}",
        f"  role_has_rejected_lane={str(report.role_has_rejected_lane).lower()}",
        f"  core_has_asm_group={str(report.core_has_asm_group).lower()}",
        f"  core_has_exact_lane={str(report.core_has_exact_lane).lower()}",
        f"  has_native_symbol={str(report.has_native_symbol).lower()}",
        f"  has_bad_symbol_jump={str(report.has_bad_symbol_jump).lower()}",
        f"  role_has_production_q4={str(report.role_has_production_q4).lower()}",
    ]
    lines.extend([
        "  asm_group_counts:",
    ])
    for pattern in EXPECTED_GROUP_COUNTS:
        lines.append(f"    {pattern}={report.group_counts[pattern]}")
    lines.extend([
        "  exact_lane_counts:",
    ])
    for pattern in EXPECTED_EXACT_LANE_COUNTS:
        lines.append(f"    {pattern}={report.exact_lane_counts[pattern]}")
    lines.extend([
        "  production_q4_counts:",
    ])
    for pattern in PRODUCTION_REPORT_COUNTS:
        lines.append(f"    {pattern}={report.production_counts[pattern]}")
    if report.errors:
        lines.append("  verdict=FAIL")
        lines.append("  errors:")
        lines.extend(f"    - {error}" for error in report.errors)
    else:
        lines.append("  verdict=PASS")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-object", type=Path, default=DEFAULT_ROLE_OBJECT)
    parser.add_argument("--core-elf", type=Path, default=DEFAULT_CORE_ELF)
    parser.add_argument("--asm-source", type=Path, default=DEFAULT_ASM_SOURCE)
    parser.add_argument("--llvm-nm", type=Path, default=DEFAULT_LLVM_NM)
    parser.add_argument("--llvm-objdump", type=Path, default=DEFAULT_LLVM_OBJDUMP)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = check_asm_integration(
        role_object=args.role_object,
        core_elf=args.core_elf,
        asm_source=args.asm_source,
        llvm_nm=args.llvm_nm,
        llvm_objdump=args.llvm_objdump,
    )
    print(render_report(report), end="")
    return 1 if args.strict and report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
