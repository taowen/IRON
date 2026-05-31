#!/usr/bin/env python3
"""Check qwen3 main16 source-assembly integration invariants."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from generate_main16_q4nx_asm import generate_assembly
from main16_q4nx_asm_lib import MAIN16_ACCUM_ABS_ADDR, MAIN16_CONTROL_ABS_ADDR


DEFAULT_ROLE_OBJECT = Path("qwen3-layer/main_projection_q4nx_fast.o")
DEFAULT_CORE_ELF = Path("design.mlir.prj/main_core_2_2.elf")
DEFAULT_ASM_SOURCE = Path("qwen3-layer/main_projection_q4nx_asm.s")
DEFAULT_CPP_SOURCE = Path("qwen3-layer/main_projection_q4nx_fast.cc")
DEFAULT_LLVM_NM = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-nm")
DEFAULT_LLVM_OBJDUMP = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump")
REJECTED_LANE_SYMBOL = "q4nx_accum_lane_asm"
NATIVE_SYMBOL = "q4nx_accum_lane_native"
OLD_CPP_WRAPPER_SYMBOL = "_ZN12_GLOBAL__N_121q4nx_chunk_accum_fastEPfP8bfloat16S2_"
PRODUCTION_ASM_SYMBOL = "q4nx_chunk_accum_asm_zol"
LAYER_SCHEDULER_SYMBOL = "q4nx_main16_layer_scheduler"
LAYER_Q4_BODY_SYMBOL = "q4nx_main16_layer_q4_body"
LAYER_REQUIRED_SYMBOLS = (
    LAYER_SCHEDULER_SYMBOL,
    "q4nx_main16_layer_dispatcher",
    LAYER_Q4_BODY_SYMBOL,
    "q4nx_main16_layer_emit_record",
    "q4nx_main16_layer_body_qkv",
    "q4nx_main16_layer_body_o",
    "q4nx_main16_layer_body_upgate",
    "q4nx_main16_layer_body_down",
)
REJECTED_SCHEDULER_SYMBOLS = (
    "q4nx_main16_qkv_scheduler",
    "q4nx_main16_qkvo_scheduler",
    "q4nx_main16_full_scheduler",
)
REJECTED_ROLE_SYMBOLS = (
    "q4nx_emit_accum_body_record_fast",
)
REJECTED_CPP_DATAFLOW_PATTERNS = (
    "acquire_greater_equal(",
    "release(",
    "select_weight_buffer",
    "select_activation_buffer",
    "select_record_buffer",
    "body_record_header",
    "projection_record_header",
    "record_payload_bf16",
)
LAYER_ACCUM_ADDR_PATTERNS = (
    f"#0x{MAIN16_ACCUM_ABS_ADDR:x}",
    f"#{MAIN16_ACCUM_ABS_ADDR}",
)
LAYER_CONTROL_ADDR_PATTERNS = (
    f"#0x{MAIN16_CONTROL_ABS_ADDR:x}",
    f"#{MAIN16_CONTROL_ABS_ADDR}",
)
LAYER_CONTROL_LOAD_REGEXES = (
    r"lda\s+r13,\s+\[p6, #8\]",
    r"lda\s+r23,\s+\[p6, #12\]",
    r"lda\s+r24,\s+\[p6, #16\]",
    r"lda\s+r25,\s+\[p6, #20\]",
    r"lda\s+r12,\s+\[p6, #4\]",
)
QKV_CONTROL_CONFIG_REGEXES = (
    r"mova\s+r12,\s+#8",
    r"mova\s+r12,\s+#2",
    r"mova\s+r12,\s+#10",
    r"mova\s+r12,\s+#11",
    r"mova\s+r12,\s+#12",
    r"st\s+r12,\s+\[p6, #4\]",
    r"st\s+r12,\s+\[p6, #8\]",
    r"st\s+r12,\s+\[p6, #12\]",
    r"st\s+r12,\s+\[p6, #20\]",
)
PRODUCTION_ASM_REQUIRED_COUNTS = {
    "vmac.f": 64,
    "vextbcst.16": 64,
    "vextbcst.32": 0,
    "add.nc\tlc": 2,
    "movxm\tls": 2,
    "movxm\tle": 2,
    "vlda": 2,
    "vst": 2,
    "#19201": 16,
    "#828": 16,
    "#60": 16,
    "crupsmode": 16,
    "mov\ts0": 16,
}
LAYER_Q4_BODY_REQUIRED_COUNTS = PRODUCTION_ASM_REQUIRED_COUNTS
PRODUCTION_ASM_REPORT_COUNTS = (
    "vmac.f",
    "vextbcst.16",
    "vextbcst.32",
    "vunpack",
    "vups.4x",
    "vups.2x",
    "vmul.f",
    "vadd.f",
    "vconv.bf16.fp32",
    "vlda",
    "vldb",
    "vst",
    "crunpacksize",
    "crupsmode",
    "vbcst.16",
    "#19201",
    "#828",
    "#60",
    "mov\ts0",
    "add.nc\tlc",
    "movxm\tls",
    "movxm\tle",
)


@dataclass(frozen=True)
class AsmIntegrationReport:
    role_object: Path
    core_elf: Path
    asm_source: Path
    cpp_source: Path
    source_matches_generator: bool
    cpp_source_has_old_dataflow: bool
    role_has_rejected_lane: bool
    has_native_symbol: bool
    has_bad_symbol_jump: bool
    role_has_old_cpp_wrapper: bool
    role_has_production_asm: bool
    core_has_production_asm: bool
    missing_role_layer_symbols: tuple[str, ...]
    missing_core_layer_symbols: tuple[str, ...]
    rejected_role_symbols: tuple[str, ...]
    rejected_core_symbols: tuple[str, ...]
    rejected_role_scheduler_symbols: tuple[str, ...]
    rejected_core_scheduler_symbols: tuple[str, ...]
    core_layer_accum_addr_pinned: bool
    core_layer_control_addr_pinned: bool
    core_layer_q4_body_loads_control: bool
    core_layer_qkv_body_writes_control: bool
    layer_q4_body_calls_production_helper: bool
    layer_q4_body_counts: dict[str, int]
    production_asm_counts: dict[str, int]
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
    cpp_source: Path,
    llvm_nm: Path,
    llvm_objdump: Path,
) -> AsmIntegrationReport:
    if not role_object.exists():
        raise FileNotFoundError(role_object)
    if not core_elf.exists():
        raise FileNotFoundError(core_elf)
    if not asm_source.exists():
        raise FileNotFoundError(asm_source)
    if not cpp_source.exists():
        raise FileNotFoundError(cpp_source)

    source_matches_generator = asm_source.read_text() == generate_assembly()
    cpp_source_text = cpp_source.read_text()
    cpp_source_has_old_dataflow = any(
        pattern in cpp_source_text for pattern in REJECTED_CPP_DATAFLOW_PATTERNS
    )
    role_symbols = symbol_table(llvm_nm, role_object)
    core_symbols = symbol_table(llvm_nm, core_elf)
    role_names = symbol_names(role_symbols)
    core_names = symbol_names(core_symbols)
    role_has_rejected_lane = REJECTED_LANE_SYMBOL in role_names
    role_has_old_cpp_wrapper = OLD_CPP_WRAPPER_SYMBOL in role_names
    role_has_production_asm = PRODUCTION_ASM_SYMBOL in role_names
    core_has_production_asm = PRODUCTION_ASM_SYMBOL in core_names
    missing_role_layer_symbols = tuple(
        symbol for symbol in LAYER_REQUIRED_SYMBOLS if symbol not in role_names
    )
    missing_core_layer_symbols = tuple(
        symbol for symbol in LAYER_REQUIRED_SYMBOLS if symbol not in core_names
    )
    rejected_role_symbols = tuple(
        symbol for symbol in REJECTED_ROLE_SYMBOLS if symbol in role_names
    )
    rejected_core_symbols = tuple(
        symbol for symbol in REJECTED_ROLE_SYMBOLS if symbol in core_names
    )
    rejected_role_scheduler_symbols = tuple(
        symbol for symbol in REJECTED_SCHEDULER_SYMBOLS if symbol in role_names
    )
    rejected_core_scheduler_symbols = tuple(
        symbol for symbol in REJECTED_SCHEDULER_SYMBOLS if symbol in core_names
    )
    has_native_symbol = NATIVE_SYMBOL in role_names or NATIVE_SYMBOL in core_names
    role_disasm = disassemble(llvm_objdump, role_object)
    core_disasm = disassemble(llvm_objdump, core_elf)
    if role_has_production_asm:
        production_body = function_body(role_disasm, PRODUCTION_ASM_SYMBOL)
        production_asm_counts = {
            pattern: production_body.count(pattern)
            for pattern in PRODUCTION_ASM_REPORT_COUNTS
        }
    else:
        production_asm_counts = {pattern: 0 for pattern in PRODUCTION_ASM_REPORT_COUNTS}
    if LAYER_Q4_BODY_SYMBOL in role_names:
        layer_q4_body = function_body(role_disasm, LAYER_Q4_BODY_SYMBOL)
        layer_q4_body_calls_production_helper = PRODUCTION_ASM_SYMBOL in layer_q4_body
        layer_q4_body_counts = {
            pattern: layer_q4_body.count(pattern)
            for pattern in PRODUCTION_ASM_REPORT_COUNTS
        }
    else:
        layer_q4_body_calls_production_helper = False
        layer_q4_body_counts = {pattern: 0 for pattern in PRODUCTION_ASM_REPORT_COUNTS}
    if LAYER_SCHEDULER_SYMBOL in core_names:
        core_layer_scheduler_body = function_body(core_disasm, LAYER_SCHEDULER_SYMBOL)
        core_layer_accum_addr_pinned = any(
            f"movxm\tr30, {pattern}" in core_layer_scheduler_body
            for pattern in LAYER_ACCUM_ADDR_PATTERNS
        )
    else:
        core_layer_accum_addr_pinned = False
    if LAYER_Q4_BODY_SYMBOL in core_names:
        core_layer_q4_body = function_body(core_disasm, LAYER_Q4_BODY_SYMBOL)
        core_layer_control_addr_pinned = any(
            f"movxm\tp6, {pattern}" in core_layer_q4_body
            for pattern in LAYER_CONTROL_ADDR_PATTERNS
        )
        core_layer_q4_body_loads_control = all(
            re.search(pattern, core_layer_q4_body) for pattern in LAYER_CONTROL_LOAD_REGEXES
        )
    else:
        core_layer_control_addr_pinned = False
        core_layer_q4_body_loads_control = False
    if "q4nx_main16_layer_body_qkv" in core_names:
        core_layer_qkv_body = function_body(core_disasm, "q4nx_main16_layer_body_qkv")
        core_layer_qkv_body_writes_control = (
            any(f"movxm\tp6, {pattern}" in core_layer_qkv_body for pattern in LAYER_CONTROL_ADDR_PATTERNS)
            and all(re.search(pattern, core_layer_qkv_body) for pattern in QKV_CONTROL_CONFIG_REGEXES)
        )
    else:
        core_layer_qkv_body_writes_control = False
    has_bad_symbol_jump = bool(
        re.search(r"\bj\s+#0\b", role_disasm)
        and re.search(r"R_AIE_1\s+\*ABS\*", role_disasm)
    )

    errors: list[str] = []
    if not source_matches_generator:
        errors.append(f"main16 source assembly drifted from generator: {asm_source}")
    if cpp_source_has_old_dataflow:
        errors.append(f"old C++ main16 dataflow scheduler code is still present: {cpp_source}")
    if role_has_rejected_lane:
        errors.append(f"rejected approximate lane body is still present: {REJECTED_LANE_SYMBOL}")
    if has_native_symbol:
        errors.append(f"old C++ lane symbol is still present: {NATIVE_SYMBOL}")
    if has_bad_symbol_jump:
        errors.append("source assembly contains a symbol jump that resolved as j #0")
    if role_has_old_cpp_wrapper:
        errors.append(f"old extra Q4 wrapper is still present: {OLD_CPP_WRAPPER_SYMBOL}")
    if not role_has_production_asm:
        errors.append(f"role object missing production Q4 asm hot body: {PRODUCTION_ASM_SYMBOL}")
    if missing_role_layer_symbols:
        errors.append(
            "role object missing generated layer scheduler symbols: "
            + ", ".join(missing_role_layer_symbols)
        )
    if missing_core_layer_symbols:
        errors.append(
            "final core ELF missing generated layer scheduler symbols: "
            + ", ".join(missing_core_layer_symbols)
        )
    if rejected_role_symbols:
        errors.append(
            "role object still contains rejected exported helpers: "
            + ", ".join(rejected_role_symbols)
        )
    if rejected_core_symbols:
        errors.append(
            "final core ELF still contains rejected exported helpers: "
            + ", ".join(rejected_core_symbols)
        )
    if rejected_role_scheduler_symbols:
        errors.append(
            "role object still contains rejected scheduler variants: "
            + ", ".join(rejected_role_scheduler_symbols)
        )
    if rejected_core_scheduler_symbols:
        errors.append(
            "final core ELF still contains rejected scheduler variants: "
            + ", ".join(rejected_core_scheduler_symbols)
        )
    if layer_q4_body_calls_production_helper:
        errors.append(
            f"{LAYER_Q4_BODY_SYMBOL} must inline the hot body, but references {PRODUCTION_ASM_SYMBOL}"
        )
    if not core_layer_accum_addr_pinned:
        errors.append(
            f"{LAYER_SCHEDULER_SYMBOL} must pin accumulator scratch at 0x{MAIN16_ACCUM_ABS_ADDR:x}"
        )
    if not core_layer_control_addr_pinned:
        errors.append(
            f"{LAYER_Q4_BODY_SYMBOL} must use tile-local control words at 0x{MAIN16_CONTROL_ABS_ADDR:x}"
        )
    if not core_layer_q4_body_loads_control:
        errors.append(
            f"{LAYER_Q4_BODY_SYMBOL} must load records/chunks/packet/phase config from control words"
        )
    if not core_layer_qkv_body_writes_control:
        errors.append("q4nx_main16_layer_body_qkv must write Q/K/V phase config into control words")
    for pattern, expected in PRODUCTION_ASM_REQUIRED_COUNTS.items():
        got = production_asm_counts[pattern]
        if got != expected:
            errors.append(f"{PRODUCTION_ASM_SYMBOL}: {pattern} expected={expected} got={got}")
    for pattern, expected in LAYER_Q4_BODY_REQUIRED_COUNTS.items():
        layer_got = layer_q4_body_counts[pattern]
        if layer_got != expected:
            errors.append(f"{LAYER_Q4_BODY_SYMBOL}: {pattern} expected={expected} got={layer_got}")

    return AsmIntegrationReport(
        role_object=role_object,
        core_elf=core_elf,
        asm_source=asm_source,
        cpp_source=cpp_source,
        source_matches_generator=source_matches_generator,
        cpp_source_has_old_dataflow=cpp_source_has_old_dataflow,
        role_has_rejected_lane=role_has_rejected_lane,
        has_native_symbol=has_native_symbol,
        has_bad_symbol_jump=has_bad_symbol_jump,
        role_has_old_cpp_wrapper=role_has_old_cpp_wrapper,
        role_has_production_asm=role_has_production_asm,
        core_has_production_asm=core_has_production_asm,
        missing_role_layer_symbols=missing_role_layer_symbols,
        missing_core_layer_symbols=missing_core_layer_symbols,
        rejected_role_symbols=rejected_role_symbols,
        rejected_core_symbols=rejected_core_symbols,
        rejected_role_scheduler_symbols=rejected_role_scheduler_symbols,
        rejected_core_scheduler_symbols=rejected_core_scheduler_symbols,
        core_layer_accum_addr_pinned=core_layer_accum_addr_pinned,
        core_layer_control_addr_pinned=core_layer_control_addr_pinned,
        core_layer_q4_body_loads_control=core_layer_q4_body_loads_control,
        core_layer_qkv_body_writes_control=core_layer_qkv_body_writes_control,
        layer_q4_body_calls_production_helper=layer_q4_body_calls_production_helper,
        layer_q4_body_counts=layer_q4_body_counts,
        production_asm_counts=production_asm_counts,
        errors=tuple(errors),
    )


def render_report(report: AsmIntegrationReport) -> str:
    lines = [
        "main16_asm_integration_check:",
        f"  role_object={report.role_object}",
        f"  core_elf={report.core_elf}",
        f"  asm_source={report.asm_source}",
        f"  cpp_source={report.cpp_source}",
        f"  source_matches_generator={str(report.source_matches_generator).lower()}",
        f"  cpp_source_has_old_dataflow={str(report.cpp_source_has_old_dataflow).lower()}",
        f"  role_has_rejected_lane={str(report.role_has_rejected_lane).lower()}",
        f"  has_native_symbol={str(report.has_native_symbol).lower()}",
        f"  has_bad_symbol_jump={str(report.has_bad_symbol_jump).lower()}",
        f"  role_has_old_cpp_wrapper={str(report.role_has_old_cpp_wrapper).lower()}",
        f"  role_has_production_asm={str(report.role_has_production_asm).lower()}",
        f"  core_has_production_asm={str(report.core_has_production_asm).lower()}",
        "  missing_role_layer_symbols="
        + (",".join(report.missing_role_layer_symbols) if report.missing_role_layer_symbols else "none"),
        "  missing_core_layer_symbols="
        + (",".join(report.missing_core_layer_symbols) if report.missing_core_layer_symbols else "none"),
        "  rejected_role_symbols="
        + (",".join(report.rejected_role_symbols) if report.rejected_role_symbols else "none"),
        "  rejected_core_symbols="
        + (",".join(report.rejected_core_symbols) if report.rejected_core_symbols else "none"),
        "  rejected_role_scheduler_symbols="
        + (
            ",".join(report.rejected_role_scheduler_symbols)
            if report.rejected_role_scheduler_symbols
            else "none"
        ),
        "  rejected_core_scheduler_symbols="
        + (
            ",".join(report.rejected_core_scheduler_symbols)
            if report.rejected_core_scheduler_symbols
            else "none"
        ),
        f"  core_layer_accum_addr_pinned={str(report.core_layer_accum_addr_pinned).lower()}",
        f"  core_layer_control_addr_pinned={str(report.core_layer_control_addr_pinned).lower()}",
        f"  core_layer_q4_body_loads_control={str(report.core_layer_q4_body_loads_control).lower()}",
        f"  core_layer_qkv_body_writes_control={str(report.core_layer_qkv_body_writes_control).lower()}",
        f"  layer_q4_body_calls_production_helper={str(report.layer_q4_body_calls_production_helper).lower()}",
    ]
    lines.extend([
        "  production_asm_counts:",
    ])
    for pattern in PRODUCTION_ASM_REPORT_COUNTS:
        lines.append(f"    {pattern}={report.production_asm_counts[pattern]}")
    lines.extend([
        "  layer_q4_body_counts:",
    ])
    for pattern in PRODUCTION_ASM_REPORT_COUNTS:
        lines.append(f"    {pattern}={report.layer_q4_body_counts[pattern]}")
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
    parser.add_argument("--cpp-source", type=Path, default=DEFAULT_CPP_SOURCE)
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
        cpp_source=args.cpp_source,
        llvm_nm=args.llvm_nm,
        llvm_objdump=args.llvm_objdump,
    )
    print(render_report(report), end="")
    return 1 if args.strict and report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
