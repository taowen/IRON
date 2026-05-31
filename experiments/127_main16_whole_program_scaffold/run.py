#!/usr/bin/env python3
"""Generate and compile the main16 whole-program scaffold."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
QWEN3_TOOLS = REPO_ROOT / "qwen3-layer/tools"
sys.path.insert(0, str(QWEN3_TOOLS))

from main16_q4nx_asm_lib import Q4_EXACT_MACROS, q4_exact_chunk_body_asm

EXP126_JSON = REPO_ROOT / "experiments/126_mylm_main16_whole_core_contract/mylm_main16_whole_core_contract.json"
CLANG = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/clang"
LLVM_NM = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-nm"
LLVM_OBJDUMP = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump"
DEFAULT_ASM = EXPERIMENT_DIR / "main16_whole_program_scaffold.s"
DEFAULT_OBJECT = EXPERIMENT_DIR / "main16_whole_program_scaffold.o"
DEFAULT_REPORT = EXPERIMENT_DIR / "main16_whole_program_scaffold.md"
DEFAULT_JSON = EXPERIMENT_DIR / "main16_whole_program_scaffold.json"
ACTIVE_Q4_GENERATOR = REPO_ROOT / "qwen3-layer/tools/generate_main16_q4nx_asm.py"
ACTIVE_Q4_ASM = REPO_ROOT / "qwen3-layer/main_projection_q4nx_asm.s"

ENTRY_SYMBOL = "q4nx_main16_whole_program_entry"
DISPATCHER_SYMBOL = "q4nx_main16_whole_dispatcher"
Q4_SYMBOL = "q4nx_main16_whole_q4_body"
EMIT_RECORD_SYMBOL = "q4nx_main16_whole_emit_record"
FLOAT_ACCUM_SYMBOL = "q4nx_main16_whole_float_accum"
CHUNKS_PER_QKVO_RECORD = 16
CHUNKS_PER_DOWN_RECORD = 48
RECORD_PAYLOAD_DWORDS = 16
RECORD_PAYLOAD_BF16 = 32
FLOAT_ACCUM_BYTES = 128


@dataclass(frozen=True)
class AbiArgument:
    name: str
    register: str
    c_type: str
    semantic: str


@dataclass(frozen=True)
class LockContract:
    name: str
    mlir_lock: int
    core_lock: int
    initial_tokens: int
    acquire_or_release: str
    semantic: str


@dataclass(frozen=True)
class StateRegister:
    name: str
    register: str
    source: str
    semantic: str


@dataclass(frozen=True)
class HeaderRun:
    name: str
    phases: tuple[int, ...]
    packet_id: int
    records: int
    chunks_per_record: int
    weight_chunk_base: int
    header_kind: str


@dataclass(frozen=True)
class PhaseBody:
    name: str
    symbol: str
    body_offset: int
    mylm_header: str
    records: int
    header_runs: tuple[HeaderRun, ...]


LINKED_C_ABI = (
    AbiArgument("wt_ping", "p0", "bfloat16 *", "main16 DMA1 weight ping buffer"),
    AbiArgument("wt_pong", "p1", "bfloat16 *", "main16 DMA1 weight pong buffer"),
    AbiArgument("chunk_ping", "p2", "int32_t *", "main16 DMA0 activation ping buffer"),
    AbiArgument("chunk_pong", "p3", "int32_t *", "main16 DMA0 activation pong buffer"),
    AbiArgument("record_ping", "p4", "int32_t *", "main16 compact record ping buffer"),
    AbiArgument("record_pong", "p5", "int32_t *", "main16 compact record pong buffer"),
    AbiArgument("group", "r0", "int32_t", "main16 column group id"),
    AbiArgument("row", "r1", "int32_t", "intile row id"),
    AbiArgument("num_rows", "r2", "int32_t", "valid rows in this tile"),
    AbiArgument("phase_limit", "r3", "int32_t", "exclusive phase limit: 3=qkv, 4=qkvo, 6=upgate, 7=full"),
)

LOCKS = (
    LockContract("activation_empty", 0, 48, 2, "release", "DMA0 activation ping/pong reusable by producer"),
    LockContract("activation_full", 1, 49, 0, "acquire", "DMA0 activation ping/pong ready for core"),
    LockContract("weight_empty", 2, 50, 2, "release", "DMA1 weight ping/pong reusable by row1 fanout"),
    LockContract("weight_full", 3, 51, 0, "acquire", "DMA1 weight ping/pong ready for core"),
    LockContract("record_empty", 4, 52, 2, "acquire", "compact record ping/pong reusable by core"),
    LockContract("record_full", 5, 53, 0, "release", "compact record ping/pong ready for row1 gather"),
)

STATE_REGISTERS = (
    StateRegister("group", "r16", "r0", "main16 column group id"),
    StateRegister("row", "r17", "r1", "intile row id"),
    StateRegister("num_rows", "r18", "r2", "valid rows for future payload copy"),
    StateRegister("phase_limit", "r31", "r3", "exclusive phase limit for the single dispatcher entry"),
    StateRegister("wt_ping_ptr", "r19", "p0", "weight ping pointer for Q4 body calls"),
    StateRegister("wt_pong_ptr", "r20", "p1", "weight pong pointer for Q4 body calls"),
    StateRegister("chunk_ping_ptr", "r21", "p2", "activation ping pointer for Q4 body calls"),
    StateRegister("chunk_pong_ptr", "r22", "p3", "activation pong pointer for Q4 body calls"),
    StateRegister("float_accum_ptr", "r30", FLOAT_ACCUM_SYMBOL, "32-lane FP32 projection accumulator"),
    StateRegister("record_ping_ptr", "r26", "p4", "compact record ping pointer"),
    StateRegister("record_pong_ptr", "r28", "p5", "compact record pong pointer"),
    StateRegister("vector_insert_index", "r29", "constant zero", "assembler-mandated index register for vinsert.32"),
    StateRegister("select_predicate", "r27", "derived", "required predicate register for sel.eqz"),
)

PHASE_BODIES = (
    PhaseBody(
        "qkv",
        "q4nx_main16_whole_body_qkv",
        0x1870,
        "0x1",
        12,
        (
            HeaderRun("Q", (0,), 10, 8, CHUNKS_PER_QKVO_RECORD, 0, "body"),
            HeaderRun("K", (1,), 11, 2, CHUNKS_PER_QKVO_RECORD, 128, "body"),
            HeaderRun("V", (2,), 12, 2, CHUNKS_PER_QKVO_RECORD, 160, "body"),
        ),
    ),
    PhaseBody(
        "o",
        "q4nx_main16_whole_body_o",
        0x1E80,
        "0x4",
        8,
        (HeaderRun("O", (3,), 13, 8, CHUNKS_PER_QKVO_RECORD, 192, "body"),),
    ),
    PhaseBody(
        "upgate",
        "q4nx_main16_whole_body_upgate",
        0x2490,
        "0x8",
        48,
        (HeaderRun("upgate", (4, 5), 14, 48, CHUNKS_PER_QKVO_RECORD, 320, "projection"),),
    ),
    PhaseBody(
        "down",
        "q4nx_main16_whole_body_down",
        0x2AA0,
        "0x4",
        8,
        (HeaderRun("down", (6,), 15, 8, CHUNKS_PER_DOWN_RECORD, 1088, "body"),),
    ),
)


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


def call_with_delay(target: str) -> list[str]:
    return [
        f"\tjl\t#{target}",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
    ]


def save_lr_frame_lines() -> list[str]:
    return [
        "\tpaddxm\t[sp], #0x40",
        "\tst\tlr, [sp, #-0x40]",
    ]


def restore_lr_frame_lines() -> list[str]:
    return [
        "\tlda\tlr, [sp, #-0x40]",
        "\tpaddxm\t[sp], #-0x40",
    ]


def return_with_delay(restore_lr: bool = False) -> list[str]:
    lines: list[str] = []
    if restore_lr:
        lines.extend(restore_lr_frame_lines())
    lines.extend(
        [
        "\tret\tlr",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        ]
    )
    return lines


def entry_setup_lines() -> list[str]:
    return [
        "// Preserve tile metadata and record buffer pointers for all phase bodies.",
        "\tmov\tr16, r0",
        "\tmov\tr17, r1",
        "\tmov\tr18, r2",
        "\tmov\tr31, r3",
        "\tmov\tr19, p0",
        "\tmov\tr20, p1",
        "\tmov\tr21, p2",
        "\tmov\tr22, p3",
        f"\tmovxm\tr30, #{FLOAT_ACCUM_SYMBOL}",
        "\tmov\tr26, p4",
        "\tmov\tr28, p5",
        "\tmova\tr29, #0",
    ]


def phase_pattern_text(phases: tuple[int, ...]) -> str:
    if len(phases) == 1:
        return f"phase{phases[0]}"
    return "/".join(f"phase{phase}" for phase in phases) + " by record parity"


def record_schedule_text(header_run: HeaderRun) -> str:
    return (
        f"{header_run.name}:{phase_pattern_text(header_run.phases)}/"
        f"packet{header_run.packet_id}x{header_run.records}/"
        f"chunks{header_run.chunks_per_record}/"
        f"weight_base{header_run.weight_chunk_base}/"
        f"{header_run.header_kind}"
    )


def q4_call_lines() -> list[str]:
    return [
        "// Select DMA1 weight and DMA0 activation ping/pong for this chunk.",
        "\tmova\tr15, #1",
        "\tand\tr27, r6, r15",
        "\tsel.eqz\tr9, r19, r20, r27",
        "\tmovs\tp1, r9",
        "\tsel.eqz\tr10, r21, r22, r27",
        "\tmovs\tp2, r10",
        "// p0 is the 32-lane FP32 accumulator owned by the whole-main16 program.",
        "\tmovs\tp0, r30",
        *q4_exact_chunk_body_asm(Q4_SYMBOL).splitlines(),
    ]


def control_lines(phase: PhaseBody) -> list[str]:
    lines = [
        *save_lr_frame_lines(),
        f"// {phase.name} phase body: configure shared Q4 run body.",
    ]
    for header_run in phase.header_runs:
        lines.extend(
            [
                f"// run {record_schedule_text(header_run)}",
                f"\tmova\tr12, #{header_run.records}",
                f"\tmova\tr13, #{header_run.chunks_per_record}",
                f"\tmova\tr23, #{header_run.packet_id}",
                f"\tmova\tr24, #{1 if header_run.header_kind == 'projection' else 0}",
                f"\tmova\tr25, #{header_run.phases[0]}",
                *call_with_delay(Q4_SYMBOL),
            ]
        )
    lines.extend(return_with_delay(restore_lr=True))
    return lines


def q4_run_lines() -> list[str]:
    return [
        *save_lr_frame_lines(),
        "// Shared Q4 run body.",
        "// Inputs: r12=records, r13=chunks/record, r23=packet, r24=projection flag, r25=base phase.",
        "// Persistent state comes from entry: r16/r17/r18 metadata, r19/r20 weights, r21/r22 activations, r26/r28 records, r30 accumulator.",
        "\tmova\tr11, #0",
        ".Lwhole_q4_record_loop:",
        "\tmova\tr6, #0",
        ".Lwhole_q4_chunk_loop:",
        "\tmovx\tr14, #-1",
        "\tacq\t#49, r14",
        "\tacq\t#51, r14",
        *q4_call_lines(),
        "\tmova\tr15, #1",
        "\trel\t#48, r15",
        "\trel\t#50, r15",
        "\tadd\tr6, r6, #1",
        "\teq\tr0, r6, r13",
        "\tjz\tr0, #.Lwhole_q4_chunk_loop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tmovx\tr14, #-1",
        "\tacq\t#52, r14",
        "// Select record ping/pong from stable entry state and emit one 17-dword record.",
        "\tmova\tr15, #1",
        "\tand\tr27, r11, r15",
        "\tmov\tr6, r27",
        "\tsel.eqz\tr9, r26, r28, r27",
        "\tmovs\tp0, r9",
        "// Phase alternates only for the up/gate projection run, identified by packet14.",
        "\tmova\tr7, #14",
        "\teq\tr27, r23, r7",
        "\tadd\tr7, r25, r6",
        "\tsel.eqz\tr0, r25, r7, r27",
        "\tmova\tr7, #0",
        "\tmov\tr27, r24",
        "\tsel.eqz\tr1, r11, r7, r27",
        "\tmov\tr2, r23",
        "\tmov\tr3, r16",
        "\tmov\tr4, r17",
        "\tmov\tr5, r18",
        "\tmovs\tp6, r30",
        *call_with_delay(EMIT_RECORD_SYMBOL),
        "\tmova\tr15, #1",
        "\trel\t#53, r15",
        "\tadd\tr11, r11, #1",
        "\teq\tr0, r11, r12",
        "\tjz\tr0, #.Lwhole_q4_record_loop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        *return_with_delay(restore_lr=True),
    ]


def emit_symbol(symbol: str, lines: list[str]) -> list[str]:
    return [
        f"\t.section\t.text.{symbol},\"ax\",@progbits",
        f"\t.globl\t{symbol}",
        "\t.p2align\t4",
        f"\t.type\t{symbol},@function",
        f"{symbol}:",
        *lines,
        f"\t.size\t{symbol}, .-{symbol}",
        "",
    ]


def emit_bss_symbol(symbol: str, size_bytes: int, alignment_power: int) -> list[str]:
    return [
        f"\t.section\t.bss.{symbol},\"aw\",@nobits",
        f"\t.globl\t{symbol}",
        f"\t.p2align\t{alignment_power}",
        f"\t.type\t{symbol},@object",
        f"{symbol}:",
        f"\t.space\t{size_bytes}",
        f"\t.size\t{symbol}, .-{symbol}",
        "",
    ]


def generate_asm() -> str:
    lines: list[str] = [
        "// Generated by exp127. Compileable scaffold only; not active qwen3-layer code.",
        "// It keeps the active linked ABI and adds a single phase-limited dispatcher.",
        "",
    ]
    lines.extend(emit_bss_symbol(FLOAT_ACCUM_SYMBOL, FLOAT_ACCUM_BYTES, 6))
    lines.extend(
        emit_symbol(
            ENTRY_SYMBOL,
            [
                *save_lr_frame_lines(),
                *entry_setup_lines(),
                *call_with_delay(DISPATCHER_SYMBOL),
                *return_with_delay(restore_lr=True),
            ],
        )
    )
    lines.extend(
        emit_symbol(
            DISPATCHER_SYMBOL,
            [
                *save_lr_frame_lines(),
                "// MyLM normal phase order: Q/K/V -> O -> up/gate -> down.",
                "// r31 is the single-entry phase limit: 3=qkv, 4=qkvo, 6=upgate, 7=full.",
                *call_with_delay(PHASE_BODIES[0].symbol),
                "\tmova\tr15, #3",
                "\teq\tr27, r31, r15",
                "\tjnz\tr27, #.Lwhole_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(PHASE_BODIES[1].symbol),
                "\tmova\tr15, #4",
                "\teq\tr27, r31, r15",
                "\tjnz\tr27, #.Lwhole_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(PHASE_BODIES[2].symbol),
                "\tmova\tr15, #6",
                "\teq\tr27, r31, r15",
                "\tjnz\tr27, #.Lwhole_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(PHASE_BODIES[3].symbol),
                ".Lwhole_dispatch_done:",
                *return_with_delay(restore_lr=True),
            ],
        )
    )
    lines.extend(Q4_EXACT_MACROS.strip().splitlines())
    lines.append("")
    lines.extend(emit_symbol(Q4_SYMBOL, q4_run_lines()))
    lines.append("")
    lines.extend(
        emit_symbol(
            EMIT_RECORD_SYMBOL,
            [
                "// p0=selected record buffer, r0=phase, r1=block, r2=packet id, r3=group, r4=row.",
                "// p6=32-lane FP32 projection accumulator.",
                "// This scaffold writes the real IRON header, converts 32 FP32 lanes to BF16 payload, and clears the accumulator.",
                "\tmova\tr6, #24",
                "\tlshl\tr8, r0, r6",
                "\tmova\tr6, #20",
                "\tlshl\tr9, r1, r6",
                "\tor\tr8, r8, r9",
                "\tmova\tr6, #16",
                "\tlshl\tr9, r3, r6",
                "\tor\tr8, r8, r9",
                "\tmova\tr6, #8",
                "\tlshl\tr9, r4, r6",
                "\tor\tr8, r8, r9",
                "\tor\tr8, r8, r2",
                "\tst\tr8, [p0], #4",
                "\tmov\tp3, p6",
                "\tmov\tr10, r5",
                "\tmova\tr5, #0",
                f"\tmova\tr2, #{RECORD_PAYLOAD_BF16}",
                "\tmova\tr1, #2",
                "\tmova\tr3, #1",
                ".Lwhole_emit_payload_loop:",
                "\tge\tr6, r5, r10",
                "\tjnz\tr6, #.Lwhole_emit_payload_zero",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tlshl\tr6, r5, r1",
                "\tmov\tdj0, r6",
                "\tlda\tr7, [p3, dj0]",
                "\tvinsert.32\tx0, x0, r29, r7",
                "\tvmov\tbmll0, x0",
                "\tnop",
                "\tvconv.bf16.fp32\twl0, bmll0",
                "\tnop",
                "\tvextract.16\tr7, x0, #0, vaddsign1",
                "\tj\t#.Lwhole_emit_payload_store",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                ".Lwhole_emit_payload_zero:",
                "\tmova\tr7, #0",
                "\tlshl\tr6, r5, r1",
                "\tmov\tdj0, r6",
                ".Lwhole_emit_payload_store:",
                "\tmova\tr8, #0",
                "\tst\tr8, [p3, dj0]",
                "\tlshl\tr9, r5, r3",
                "\tmov\tdj0, r9",
                "\tst.s16\tr7, [p0, dj0]",
                "\tadd\tr5, r5, #1",
                "\teq\tr7, r5, r2",
                "\tjz\tr7, #.Lwhole_emit_payload_loop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *return_with_delay(),
            ],
        )
    )
    for phase in PHASE_BODIES:
        header_schedule = ", ".join(record_schedule_text(run) for run in phase.header_runs)
        body_lines = [
            f"// phase={phase.name} mylm_body=0x{phase.body_offset:x} "
            f"mylm_header={phase.mylm_header} records={phase.records}",
            f"// iron_record_schedule={header_schedule}",
        ]
        body_lines.extend(control_lines(phase))
        lines.extend(emit_symbol(phase.symbol, body_lines))
    return "\n".join(lines)


def compile_asm(asm_path: Path, object_path: Path) -> None:
    run_command(
        (
            str(CLANG),
            "-O2",
            "--target=aie2p-none-unknown-elf",
            "-c",
            str(asm_path),
            "-o",
            str(object_path),
        )
    )


def symbol_table(object_path: Path) -> set[str]:
    text = run_command((str(LLVM_NM), str(object_path)))
    return {line.split()[-1] for line in text.splitlines() if line.split()}


def disassembly(object_path: Path) -> str:
    return run_command((str(LLVM_OBJDUMP), "--triple=aie2p", "-dr", "--no-print-imm-hex", str(object_path)))


def source_symbol_body(source: str, symbol: str) -> str:
    start = source.index(f"{symbol}:")
    end = source.index(f"\t.size\t{symbol}", start)
    return source[start:end]


def disasm_symbol_body(disasm_text: str, symbol: str) -> str:
    start = disasm_text.index(f"<{symbol}>:")
    end = disasm_text.find("\nDisassembly of section", start + 1)
    if end == -1:
        return disasm_text[start:]
    return disasm_text[start:end]


def referenced_registers(source: str, symbol: str, registers: tuple[str, ...]) -> list[str]:
    body = source_symbol_body(source, symbol)
    referenced: list[str] = []
    for register in registers:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){register}(?![A-Za-z0-9_])")
        if any(pattern.search(line.split("//", 1)[0]) for line in body.splitlines()):
            referenced.append(register)
    return referenced


def phase_dynamic_counts(phase: PhaseBody) -> dict[str, int]:
    records = sum(run.records for run in phase.header_runs)
    chunks = sum(run.records * run.chunks_per_record for run in phase.header_runs)
    return {
        "records": records,
        "chunks": chunks,
        "q4_calls": chunks,
        "emit_record_calls": records,
        "activation_acquire": chunks,
        "weight_acquire": chunks,
        "activation_release": chunks,
        "weight_release": chunks,
        "record_acquire": records,
        "record_release": records,
    }


def expected_q4_run_call_sites() -> int:
    return sum(len(phase.header_runs) for phase in PHASE_BODIES)


def static_control_sites(source: str, symbol: str) -> dict[str, int]:
    body = source_symbol_body(source, symbol)
    return {
        "q4_calls": body.count(f"jl\t#{Q4_SYMBOL}"),
        "emit_record_calls": body.count(f"jl\t#{EMIT_RECORD_SYMBOL}"),
        "activation_acquire": body.count("acq\t#49"),
        "weight_acquire": body.count("acq\t#51"),
        "activation_release": body.count("rel\t#48"),
        "weight_release": body.count("rel\t#50"),
        "record_acquire": body.count("acq\t#52"),
        "record_release": body.count("rel\t#53"),
    }


def build_manifest(asm_path: Path, object_path: Path) -> dict[str, object]:
    exp126 = json.loads(EXP126_JSON.read_text())
    names = symbol_table(object_path)
    required = [
        ENTRY_SYMBOL,
        DISPATCHER_SYMBOL,
        Q4_SYMBOL,
        EMIT_RECORD_SYMBOL,
        FLOAT_ACCUM_SYMBOL,
        *(phase.symbol for phase in PHASE_BODIES),
    ]
    disasm = disassembly(object_path)
    q4_disasm = disasm_symbol_body(disasm, Q4_SYMBOL)
    source = asm_path.read_text()
    phase_static = {phase.name: static_control_sites(source, phase.symbol) for phase in PHASE_BODIES}
    phase_dynamic = {phase.name: phase_dynamic_counts(phase) for phase in PHASE_BODIES}
    q4_static = static_control_sites(source, Q4_SYMBOL)
    dispatcher_source = source_symbol_body(source, DISPATCHER_SYMBOL)
    nonleaf_symbols = (
        ENTRY_SYMBOL,
        DISPATCHER_SYMBOL,
        Q4_SYMBOL,
        *(phase.symbol for phase in PHASE_BODIES),
    )
    lr_frame_counts = {
        symbol: {
            "save": source_symbol_body(source, symbol).count("st\tlr, [sp, #-0x40]"),
            "restore": source_symbol_body(source, symbol).count("lda\tlr, [sp, #-0x40]"),
            "calls": source_symbol_body(source, symbol).count("jl\t#"),
        }
        for symbol in nonleaf_symbols
    }
    phase_q4_run_call_sites = sum(counts["q4_calls"] for counts in phase_static.values())
    emitter_forbidden_state_refs = referenced_registers(
        source,
        EMIT_RECORD_SYMBOL,
        (
            "r11",
            "r12",
            "r14",
            "r15",
            "r16",
            "r17",
            "r18",
            "r19",
            "r20",
            "r21",
            "r22",
            "r26",
            "r27",
            "r28",
            "r31",
        ),
    )
    total_dynamic = {
        key: sum(counts[key] for counts in phase_dynamic.values())
        for key in next(iter(phase_dynamic.values())).keys()
    }
    return {
        "status": "compileable_scaffold_not_active",
        "required_symbols": required,
        "missing_symbols": [name for name in required if name not in names],
        "entry_symbol": ENTRY_SYMBOL,
        "dispatcher_symbol": DISPATCHER_SYMBOL,
        "q4_symbol": Q4_SYMBOL,
        "emit_record_symbol": EMIT_RECORD_SYMBOL,
        "float_accum_symbol": FLOAT_ACCUM_SYMBOL,
        "linked_c_abi": [
            {
                "name": arg.name,
                "register": arg.register,
                "c_type": arg.c_type,
                "semantic": arg.semantic,
            }
            for arg in LINKED_C_ABI
        ],
        "state_registers": [
            {
                "name": state.name,
                "register": state.register,
                "source": state.source,
                "semantic": state.semantic,
            }
            for state in STATE_REGISTERS
        ],
        "locks": [
            {
                "name": lock.name,
                "mlir_lock": lock.mlir_lock,
                "core_lock": lock.core_lock,
                "initial_tokens": lock.initial_tokens,
                "acquire_or_release": lock.acquire_or_release,
                "semantic": lock.semantic,
            }
            for lock in LOCKS
        ],
        "record_header_contract": {
            "body": "(phase << 24) | (block << 20) | (group << 16) | (row << 8) | packet_id",
            "projection": "(phase << 24) | (group << 16) | (row << 8) | packet_id",
            "packet_ids": {
                run.name: run.packet_id
                for phase in PHASE_BODIES
                for run in phase.header_runs
            },
            "upgate_phase_policy": "phase 4 on even replay records, phase 5 on odd replay records, packet 14 for both",
        },
        "dispatcher_contract": {
            "entry_symbol": ENTRY_SYMBOL,
            "scheduler_symbol_target": "q4nx_main16_layer_scheduler",
            "phase_limit_register": "r31",
            "phase_limits": {
                "qkv": 3,
                "qkvo": 4,
                "upgate": 6,
                "full": 7,
            },
            "phase_body_call_sites": {
                phase.name: dispatcher_source.count(f"jl\t#{phase.symbol}")
                for phase in PHASE_BODIES
            },
            "phase_limit_compare_sites": dispatcher_source.count("eq\tr27, r31, r15"),
            "phase_limit_exit_sites": dispatcher_source.count("jnz\tr27, #.Lwhole_dispatch_done"),
            "done_label_sites": dispatcher_source.count(".Lwhole_dispatch_done:"),
        },
        "lr_frame_contract": {
            "frame_bytes": 64,
            "nonleaf_symbols": lr_frame_counts,
            "leaf_symbols": {
                EMIT_RECORD_SYMBOL: {
                    "save": source_symbol_body(source, EMIT_RECORD_SYMBOL).count("st\tlr, [sp, #-0x40]"),
                    "restore": source_symbol_body(source, EMIT_RECORD_SYMBOL).count("lda\tlr, [sp, #-0x40]"),
                    "calls": source_symbol_body(source, EMIT_RECORD_SYMBOL).count("jl\t#"),
                }
            },
        },
        "q4_body_call_contract": {
            "status": "shared_q4_run_body_owns_record_chunk_loops",
            "source_of_truth": str(ACTIVE_Q4_GENERATOR.relative_to(REPO_ROOT)),
            "active_asm": str(ACTIVE_Q4_ASM.relative_to(REPO_ROOT)),
            "uses_save_restore": False,
            "phase_q4_run_call_sites_static": phase_q4_run_call_sites,
            "phase_q4_run_call_sites_expected": expected_q4_run_call_sites(),
            "dynamic_chunk_executions": total_dynamic["q4_calls"],
            "chunk_body_call_sites_static": q4_static["q4_calls"],
            "emit_record_call_sites_static": q4_static["emit_record_calls"],
            "p1": "selected inside shared Q4 run body from weight ping/pong by chunk parity",
            "p2": "selected inside shared Q4 run body from activation ping/pong by chunk parity",
            "p0": "32-lane FP32 projection accumulator selected inside shared Q4 run body",
            "selection_predicate": "r27 = chunk & 1 inside shared Q4 run body",
            "static_counts": {
                "vmac.f": q4_disasm.count("vmac.f"),
                "vextbcst.16": q4_disasm.count("vextbcst.16"),
                "vst": q4_disasm.count("\tvst\t"),
                "save_slots": q4_disasm.count("\tst\t"),
                "restore_slots": q4_disasm.count("\tlda\t"),
            },
        },
        "phase_bodies": [
            {
                "name": phase.name,
                "symbol": phase.symbol,
                "mylm_body_offset": phase.body_offset,
                "mylm_header": phase.mylm_header,
                "records": phase.records,
                "iron_record_schedule": [
                    {
                        "name": run.name,
                        "phases": list(run.phases),
                        "phase_pattern": phase_pattern_text(run.phases),
                        "packet_id": run.packet_id,
                        "records": run.records,
                        "chunks_per_record": run.chunks_per_record,
                        "weight_chunk_base": run.weight_chunk_base,
                        "header_kind": run.header_kind,
                    }
                    for run in phase.header_runs
                ],
            }
            for phase in PHASE_BODIES
        ],
        "active_baseline": exp126["active"],
        "decision": exp126["decision"],
        "open_gaps": [
            "shared Q4 run body still uses the current exact chunk body shape",
            "whole-program register plan must replace current exact dequant body with MyLM-density scheduled body",
            "active qwen3-layer uses q4nx_main16_layer_scheduler, but this generated whole-program entry is not active yet",
        ],
        "record_emitter": {
            "status": "header_writer_with_fp32_accum_to_bf16_payload_clear",
            "abi": "p0=selected record buffer, p6=FP32 accumulator, r0=phase, r1=block, r2=packet, r3=group, r4=row, r5=num_rows",
            "record_store_static_sites": source_symbol_body(source, EMIT_RECORD_SYMBOL).count("st\tr8, [p0], #4")
            + source_symbol_body(source, EMIT_RECORD_SYMBOL).count("st.s16\tr7, [p0, dj0]"),
            "payload_copy_loop_static_sites": source_symbol_body(source, EMIT_RECORD_SYMBOL).count(
                ".Lwhole_emit_payload_loop:"
            ),
            "payload_clear_store_static_sites": source_symbol_body(source, EMIT_RECORD_SYMBOL).count(
                "st\tr8, [p3, dj0]"
            ),
            "forbidden_caller_state_register_references": emitter_forbidden_state_refs,
            "record_dwords_dynamic": 17,
            "payload_bf16_dynamic": RECORD_PAYLOAD_BF16,
            "accum_clear_float_dynamic": RECORD_PAYLOAD_BF16,
        },
        "call_relocations": disasm.count("R_AIE_1"),
        "q4_run_static_control_sites": q4_static,
        "phase_static_control_sites": phase_static,
        "phase_dynamic_control_counts": phase_dynamic,
        "total_dynamic_control_counts": total_dynamic,
        "qkv_static_control_sites": phase_static["qkv"],
        "qkv_dynamic_control_counts": phase_dynamic["qkv"],
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Main16 Whole-Program Scaffold",
        "",
        "This is a compileable scaffold for the MyLM-style main16 role program.",
        "It is intentionally not linked into the active qwen3-layer path.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Entry: `{manifest['entry_symbol']}`",
        f"- Dispatcher: `{manifest['dispatcher_symbol']}`",
        f"- Shared Q4 body: `{manifest['q4_symbol']}`",
        f"- Record emitter: `{manifest['emit_record_symbol']}`",
        f"- FP32 accumulator: `{manifest['float_accum_symbol']}`",
        f"- Missing symbols: `{','.join(manifest['missing_symbols'])}`",
        f"- Call relocations: `{manifest['call_relocations']}`",
        f"- Full dynamic Q4 calls: `{dict(manifest['total_dynamic_control_counts'])['q4_calls']}`",
        f"- Full dynamic records: `{dict(manifest['total_dynamic_control_counts'])['records']}`",
        f"- Open gaps: `{len(manifest['open_gaps'])}`",
        "",
        "## Linked C ABI",
        "",
        "The first whole-program replacement keeps the existing MLIR call ABI so the",
        "current topology, buffers, BD rings, locks, and downstream compact routing stay unchanged.",
        "",
        "| Name | Register | Type | Semantic |",
        "| --- | --- | --- | --- |",
    ]
    for arg in manifest["linked_c_abi"]:
        item = dict(arg)
        lines.append(
            f"| `{item['name']}` | `{item['register']}` | `{item['c_type']}` | {item['semantic']} |"
        )
    lines.extend(
        [
            "",
            "## State Registers",
            "",
            "These registers are initialized once at entry and are owned by the whole-main16 program.",
            "`r27` is intentionally reserved for `sel.eqz` predicates because the assembler rejects low-register predicates.",
            "",
            "| Name | Register | Source | Semantic |",
            "| --- | --- | --- | --- |",
        ]
    )
    for state in manifest["state_registers"]:
        item = dict(state)
        lines.append(
            f"| `{item['name']}` | `{item['register']}` | `{item['source']}` | {item['semantic']} |"
        )
    lines.extend(
        [
            "",
            "## Lock Contract",
            "",
            "| Name | MLIR lock | Core lock | Initial | Core action | Semantic |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for lock in manifest["locks"]:
        item = dict(lock)
        lines.append(
            f"| `{item['name']}` | {item['mlir_lock']} | {item['core_lock']} | "
            f"{item['initial_tokens']} | `{item['acquire_or_release']}` | {item['semantic']} |"
        )
    header_contract = dict(manifest["record_header_contract"])
    dispatcher = dict(manifest["dispatcher_contract"])
    lines.extend(
        [
            "",
            "## Record Header Contract",
            "",
            f"- Body record: `{header_contract['body']}`",
            f"- Projection record: `{header_contract['projection']}`",
            f"- Up/gate phase policy: {header_contract['upgate_phase_policy']}.",
            "",
            "## Dispatcher Contract",
            "",
            f"- Entry-compatible target: `{dispatcher['scheduler_symbol_target']}`",
            f"- Phase-limit register: `{dispatcher['phase_limit_register']}`",
            f"- Phase limits: `{dispatcher['phase_limits']}`",
            f"- Phase body call sites: `{dispatcher['phase_body_call_sites']}`",
            f"- Phase-limit compare sites: `{dispatcher['phase_limit_compare_sites']}`",
            f"- Phase-limit exit sites: `{dispatcher['phase_limit_exit_sites']}`",
            "",
            "## LR Frame Contract",
            "",
            "Every generated non-leaf function saves its caller return address once at entry",
            "and restores it immediately before `ret lr`; the record emitter remains a leaf.",
            "",
            f"- Frame bytes: `{dict(manifest['lr_frame_contract'])['frame_bytes']}`",
            f"- Non-leaf symbols: `{dict(manifest['lr_frame_contract'])['nonleaf_symbols']}`",
            f"- Leaf symbols: `{dict(manifest['lr_frame_contract'])['leaf_symbols']}`",
        ]
    )
    q4_contract = dict(manifest["q4_body_call_contract"])
    lines.extend(
        [
            "",
            "## Q4 Body Call Contract",
            "",
            f"- Status: `{q4_contract['status']}`",
            f"- Source of truth: `{q4_contract['source_of_truth']}`",
            f"- Active asm: `{q4_contract['active_asm']}`",
            f"- Uses SAVE/RESTORE: `{q4_contract['uses_save_restore']}`",
            f"- Static phase-to-Q4 run calls: `{q4_contract['phase_q4_run_call_sites_static']}`",
            f"- Dynamic chunk executions: `{q4_contract['dynamic_chunk_executions']}`",
            f"- Static chunk-body calls inside Q4 run: `{q4_contract['chunk_body_call_sites_static']}`",
            f"- Static emit-record calls inside Q4 run: `{q4_contract['emit_record_call_sites_static']}`",
            f"- `p1`: {q4_contract['p1']}",
            f"- `p2`: {q4_contract['p2']}",
            f"- `p0`: {q4_contract['p0']}",
            f"- Selection predicate: `{q4_contract['selection_predicate']}`",
            f"- Static counts: `{q4_contract['static_counts']}`",
        ]
    )
    emitter = dict(manifest["record_emitter"])
    lines.extend(
        [
            "",
            "## Q4 Run Boundary",
            "",
            "Phase bodies now call the shared Q4 run body once per header run.",
            "The shared Q4 run body owns the record loop, chunk loop, DMA lock protocol,",
            "ping/pong selection, exact chunk body execution, and record emission.",
            "",
            "| Item | Count |",
            "| --- | ---: |",
            f"| Static phase-to-Q4 run calls | {q4_contract['phase_q4_run_call_sites_static']} |",
            f"| Expected phase-to-Q4 run calls | {q4_contract['phase_q4_run_call_sites_expected']} |",
            f"| Dynamic chunk executions | {q4_contract['dynamic_chunk_executions']} |",
            f"| Static chunk-body calls inside Q4 run | {q4_contract['chunk_body_call_sites_static']} |",
            f"| Static emit-record calls inside Q4 run | {q4_contract['emit_record_call_sites_static']} |",
            "",
            "## Record Emitter Scaffold",
            "",
            f"- Status: `{emitter['status']}`",
            f"- ABI: `{emitter['abi']}`",
            f"- Static record store sites: `{emitter['record_store_static_sites']}`",
            f"- Dynamic record dwords: `{emitter['record_dwords_dynamic']}`",
            f"- Dynamic payload BF16 values: `{emitter['payload_bf16_dynamic']}`",
            f"- Dynamic accumulator clear floats: `{emitter['accum_clear_float_dynamic']}`",
            f"- Forbidden caller state register references: `{','.join(emitter['forbidden_caller_state_register_references'])}`",
            "",
            "The emitter now writes the real IRON header to the selected ping/pong record buffer.",
            "Payload is converted from the FP32 accumulator to BF16 and the accumulator is cleared.",
        ]
    )
    lines.extend(
        [
        "",
        "## Phase Bodies",
        "",
        "| Phase | Symbol | MyLM body | MyLM header | Records | Q4 calls | IRON record schedule |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    )
    for phase in manifest["phase_bodies"]:
        item = dict(phase)
        schedule = ", ".join(
            f"{run['name']}:{run['phase_pattern']}/packet{run['packet_id']}x{run['records']}"
            f"/chunks{run['chunks_per_record']}/base{run['weight_chunk_base']}/{run['header_kind']}"
            for run in item["iron_record_schedule"]
        )
        dynamic_counts = dict(manifest["phase_dynamic_control_counts"])[str(item["name"])]
        lines.append(
            f"| `{item['name']}` | `{item['symbol']}` | "
            f"`0x{int(item['mylm_body_offset']):04x}` | `{item['mylm_header']}` | "
            f"{item['records']} | {dynamic_counts['q4_calls']} | `{schedule}` |"
        )
    lines.extend(
        [
            "",
            "## Control Skeleton",
            "",
            "Static sites are intentionally looped. Dynamic counts are the expected full",
            "main16 phase behavior when those loops run.",
            "",
            "### Static Sites",
            "",
            "| Phase | Q4 call sites | Emit call sites | Activation acquire sites | Weight acquire sites | Record acquire sites |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase_name, counts in dict(manifest["phase_static_control_sites"]).items():
        item = dict(counts)
        lines.append(
            f"| `{phase_name}` | {item['q4_calls']} | {item['emit_record_calls']} | "
            f"{item['activation_acquire']} | {item['weight_acquire']} | {item['record_acquire']} |"
        )
    lines.extend(
        [
            "",
            "### Dynamic Counts",
            "",
            "| Phase | Records | Chunks/Q4 calls | Activation acquires | Weight acquires | Record acquires |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase_name, counts in dict(manifest["phase_dynamic_control_counts"]).items():
        item = dict(counts)
        lines.append(
            f"| `{phase_name}` | {item['records']} | {item['q4_calls']} | "
            f"{item['activation_acquire']} | {item['weight_acquire']} | {item['record_acquire']} |"
        )
    total_counts = dict(manifest["total_dynamic_control_counts"])
    lines.append(
        f"| `total` | {total_counts['records']} | {total_counts['q4_calls']} | "
        f"{total_counts['activation_acquire']} | {total_counts['weight_acquire']} | "
        f"{total_counts['record_acquire']} |"
    )
    lines.extend(
        [
            "",
            "## Next Implementation Work",
            "",
            "1. Replace the current exact chunk body shape with a MyLM-density scheduled body.",
            "2. Keep the shared Q4 body accumulating into the `p6` FP32 accumulator.",
            "3. Preserve the current FP32-to-BF16 record emitter contract while changing the Q4 body internals.",
            "4. Preserve IRON compact headers `10..15` until downstream routing is explicitly changed.",
            "5. Only switch active MLIR to this entry after `full-layer-qkv-prefix token31` passes.",
            "",
            "## Open Gaps",
            "",
        ]
    )
    for item in manifest["open_gaps"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def write_outputs(asm_path: Path, object_path: Path, report_path: Path, json_path: Path) -> None:
    asm_path.write_text(generate_asm() + "\n")
    compile_asm(asm_path, object_path)
    manifest = build_manifest(asm_path, object_path)
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path.write_text(render_report(manifest) + "\n")
    if manifest["missing_symbols"]:
        raise RuntimeError("missing scaffold symbols: " + ", ".join(manifest["missing_symbols"]))
    q4_contract = dict(manifest["q4_body_call_contract"])
    dispatcher = dict(manifest["dispatcher_contract"])
    for phase in PHASE_BODIES:
        phase_calls = dict(dispatcher["phase_body_call_sites"])[phase.name]
        if phase_calls != 1:
            raise RuntimeError(f"dispatcher must call {phase.name} phase body exactly once")
    if dispatcher["phase_limit_compare_sites"] != 3:
        raise RuntimeError("dispatcher must compare phase_limit after qkv, o, and upgate")
    if dispatcher["phase_limit_exit_sites"] != 3:
        raise RuntimeError("dispatcher must have one conditional exit after qkv, o, and upgate")
    if dispatcher["done_label_sites"] != 1:
        raise RuntimeError("dispatcher must have one done label")
    lr_contract = dict(manifest["lr_frame_contract"])
    for symbol, counts in dict(lr_contract["nonleaf_symbols"]).items():
        item = dict(counts)
        if item["calls"] < 1:
            raise RuntimeError(f"non-leaf symbol {symbol} must have at least one call")
        if item["save"] != 1 or item["restore"] != 1:
            raise RuntimeError(f"non-leaf symbol {symbol} must save/restore lr exactly once")
    for symbol, counts in dict(lr_contract["leaf_symbols"]).items():
        item = dict(counts)
        if item["calls"] != 0:
            raise RuntimeError(f"leaf symbol {symbol} must not call another function")
        if item["save"] != 0 or item["restore"] != 0:
            raise RuntimeError(f"leaf symbol {symbol} must not allocate an lr frame")
    if q4_contract["phase_q4_run_call_sites_static"] != q4_contract["phase_q4_run_call_sites_expected"]:
        raise RuntimeError("phase bodies must call the shared Q4 run body once per header run")
    if q4_contract["chunk_body_call_sites_static"] != 0:
        raise RuntimeError("shared Q4 run body must not call itself or a per-chunk Q4 helper")
    if q4_contract["emit_record_call_sites_static"] != 1:
        raise RuntimeError("shared Q4 run body must have exactly one static record-emitter call")
    if q4_contract["uses_save_restore"]:
        raise RuntimeError("whole-program scaffold Q4 body must not use callable SAVE/RESTORE")
    q4_static = dict(manifest["q4_run_static_control_sites"])
    expected_q4_static = {
        "activation_acquire": 1,
        "weight_acquire": 1,
        "activation_release": 1,
        "weight_release": 1,
        "record_acquire": 1,
        "record_release": 1,
    }
    for name, expected in expected_q4_static.items():
        if q4_static[name] != expected:
            raise RuntimeError(f"shared Q4 run body static {name} must be {expected}")
    emitter = dict(manifest["record_emitter"])
    if emitter["record_store_static_sites"] != 2:
        raise RuntimeError(
            "record emitter must have one header store and one looped payload store site"
        )
    if emitter["payload_copy_loop_static_sites"] != 1:
        raise RuntimeError("record emitter must have one payload copy loop")
    if emitter["payload_clear_store_static_sites"] != 1:
        raise RuntimeError("record emitter must have one looped payload clear store site")
    if emitter["forbidden_caller_state_register_references"]:
        raise RuntimeError(
            "record emitter clobbers caller state registers: "
            + ", ".join(emitter["forbidden_caller_state_register_references"])
        )
    if emitter["record_dwords_dynamic"] != 17:
        raise RuntimeError("record emitter must write 17 dwords dynamically")
    if emitter["payload_bf16_dynamic"] != RECORD_PAYLOAD_BF16:
        raise RuntimeError("record emitter payload conversion length mismatch")
    if emitter["accum_clear_float_dynamic"] != RECORD_PAYLOAD_BF16:
        raise RuntimeError("record emitter accumulator clear length mismatch")
    qkv_counts = dict(manifest["qkv_dynamic_control_counts"])
    expected_counts = {
        "q4_calls": 192,
        "emit_record_calls": 12,
        "activation_acquire": 192,
        "weight_acquire": 192,
        "activation_release": 192,
        "weight_release": 192,
        "record_acquire": 12,
        "record_release": 12,
    }
    for name, expected in expected_counts.items():
        if qkv_counts[name] != expected:
            raise RuntimeError(f"{name}: expected {expected} got {qkv_counts[name]}")
    total_counts = dict(manifest["total_dynamic_control_counts"])
    total_expected = {
        "q4_calls": 1472,
        "emit_record_calls": 76,
        "activation_acquire": 1472,
        "weight_acquire": 1472,
        "activation_release": 1472,
        "weight_release": 1472,
        "record_acquire": 76,
        "record_release": 76,
    }
    for name, expected in total_expected.items():
        if total_counts[name] != expected:
            raise RuntimeError(f"total {name}: expected {expected} got {total_counts[name]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asm", type=Path, default=DEFAULT_ASM)
    parser.add_argument("--object", type=Path, default=DEFAULT_OBJECT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    write_outputs(args.asm, args.object, args.report, args.json)
    print(f"wrote {args.asm}")
    print(f"wrote {args.object}")
    print(f"wrote {args.report}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
