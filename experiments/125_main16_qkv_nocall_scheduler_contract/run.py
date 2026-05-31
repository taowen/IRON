#!/usr/bin/env python3
"""Build the main16 QKV nocall scheduler contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
MAIN16_CC = REPO_ROOT / "qwen3-layer/main_projection_q4nx_fast.cc"
MAIN16_ASM = REPO_ROOT / "qwen3-layer/main_projection_q4nx_asm.s"
MAIN16_OBJECT = REPO_ROOT / "qwen3-layer/main_projection_q4nx_fast.o"
LLVM_OBJDUMP = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump"
DEFAULT_REPORT = EXPERIMENT_DIR / "main16_qkv_nocall_scheduler_contract.md"
DEFAULT_JSON = EXPERIMENT_DIR / "main16_qkv_nocall_scheduler_contract.json"
DEFAULT_ASM_INC = EXPERIMENT_DIR / "q4nx_main16_qkv_scheduler_nocall_contract.s.inc"

MAIN_TILES = 16
CHUNKS_PER_QKV_RECORD = 16


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    records: int
    chunks_per_record: int
    weight_base: int
    phase: int

    @property
    def chunks(self) -> int:
        return self.records * self.chunks_per_record


@dataclass(frozen=True)
class HelperOverhead:
    save_slots: int
    restore_slots: int
    helper_wrapper_slots: int
    helper_lane_invocations: int
    accum_loads_per_call: int
    accum_stores_per_call: int
    q4_exact32_groups_per_call: int


@dataclass(frozen=True)
class DynamicCost:
    qkv_calls_per_tile: int
    full_calls_per_tile: int
    qkv_calls_all_main16: int
    full_calls_all_main16: int
    qkv_accum_loads_per_tile: int
    qkv_accum_stores_per_tile: int
    full_accum_loads_per_tile: int
    full_accum_stores_per_tile: int
    qkv_save_restore_slots_per_tile: int
    full_save_restore_slots_per_tile: int


@dataclass(frozen=True)
class SourceCheck:
    cpp_helper_call_sites: int
    cpp_qkv_scheduler_symbols: int
    asm_helper_symbols: int
    asm_save_macros: int
    asm_restore_macros: int
    asm_q4_lane_macros: int
    asm_required_markers_present: bool


@dataclass(frozen=True)
class DisasmCheck:
    qkv_static_helper_calls: int
    qkv_static_record_emit_calls: int
    activation_full_acq: int
    weight_full_acq: int
    activation_empty_rel: int
    weight_empty_rel: int
    record_empty_acq: int
    record_full_rel: int
    lock_shape_valid: bool


QKV_PHASES = (
    PhaseSpec("Q", 8, 16, 0, 0),
    PhaseSpec("K", 2, 16, 128, 1),
    PhaseSpec("V", 2, 16, 160, 2),
)
FULL_PHASES = (
    *QKV_PHASES,
    PhaseSpec("O", 8, 16, 192, 3),
    PhaseSpec("up/gate", 48, 16, 320, 4),
    PhaseSpec("down", 8, 48, 1088, 6),
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


def disassemble_role_object() -> str:
    return run_command(
        (
            str(LLVM_OBJDUMP),
            "--triple=aie2p",
            "-dr",
            "--no-print-imm-hex",
            str(MAIN16_OBJECT),
        )
    )


def disasm_function_body(disasm: str, name: str) -> str:
    start = re.search(rf"^[0-9a-fA-F]+ <{re.escape(name)}>:\n", disasm, re.MULTILINE)
    if start is None:
        raise ValueError(f"function not found in disassembly: {name}")
    next_function = re.search(
        r"^[0-9a-fA-F]+ <(?!\.)[^>]+>:\n",
        disasm[start.end():],
        re.MULTILINE,
    )
    if next_function is None:
        return disasm[start.end():]
    return disasm[start.end(): start.end() + next_function.start()]


def instruction_slots(lines: tuple[str, ...]) -> int:
    count = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("."):
            continue
        count += sum(1 for part in line.split(";") if part.strip())
    return count


def macro_body(source: str, name: str) -> tuple[str, ...]:
    pattern = re.compile(rf"^\s*\.macro\s+{re.escape(name)}\b(?P<body>.*?)^\s*\.endm\b", re.MULTILINE | re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"macro not found: {name}")
    return tuple(match.group("body").splitlines())


def function_body(source: str, name: str) -> tuple[str, ...]:
    marker = f"{name}:"
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"function not found: {name}")
    end_marker = f".size\t{name}"
    end = source.find(end_marker, start)
    if end < 0:
        raise ValueError(f"function size marker not found: {name}")
    return tuple(source[start + len(marker):end].splitlines())


def helper_overhead(asm_source: str) -> HelperOverhead:
    save = macro_body(asm_source, "SAVE_Q4_CALL_STATE")
    restore = macro_body(asm_source, "RESTORE_Q4_CALL_STATE")
    helper = function_body(asm_source, "q4nx_chunk_accum_asm_zol")
    return HelperOverhead(
        save_slots=instruction_slots(save),
        restore_slots=instruction_slots(restore),
        helper_wrapper_slots=instruction_slots(helper),
        helper_lane_invocations=sum("Q4_EXACT_LANE_ZOL" in line for line in helper),
        accum_loads_per_call=sum("vlda\tbmll1" in line for line in macro_body(asm_source, "Q4_EXACT_LANE_ZOL")) * 2,
        accum_stores_per_call=sum("vst\tbmll1" in line for line in macro_body(asm_source, "Q4_EXACT_LANE_ZOL")) * 2,
        q4_exact32_groups_per_call=sum("Q4_EXACT32_GROUP_DIRECT" in line for line in macro_body(asm_source, "Q4_EXACT_LANE_ZOL")) * 2,
    )


def dynamic_cost(overhead: HelperOverhead) -> DynamicCost:
    qkv_calls = sum(phase.chunks for phase in QKV_PHASES)
    full_calls = sum(phase.chunks for phase in FULL_PHASES)
    save_restore = overhead.save_slots + overhead.restore_slots
    return DynamicCost(
        qkv_calls_per_tile=qkv_calls,
        full_calls_per_tile=full_calls,
        qkv_calls_all_main16=qkv_calls * MAIN_TILES,
        full_calls_all_main16=full_calls * MAIN_TILES,
        qkv_accum_loads_per_tile=qkv_calls * overhead.accum_loads_per_call,
        qkv_accum_stores_per_tile=qkv_calls * overhead.accum_stores_per_call,
        full_accum_loads_per_tile=full_calls * overhead.accum_loads_per_call,
        full_accum_stores_per_tile=full_calls * overhead.accum_stores_per_call,
        qkv_save_restore_slots_per_tile=qkv_calls * save_restore,
        full_save_restore_slots_per_tile=full_calls * save_restore,
    )


def source_check(cpp_source: str, asm_source: str) -> SourceCheck:
    markers = (
        "q4nx_main16_qkv_scheduler",
        "q4nx_chunk_accum_asm_zol",
        "SAVE_Q4_CALL_STATE",
        "RESTORE_Q4_CALL_STATE",
        "Q4_EXACT_LANE_ZOL",
    )
    return SourceCheck(
        cpp_helper_call_sites=cpp_source.count("q4nx_chunk_accum_asm_zol("),
        cpp_qkv_scheduler_symbols=cpp_source.count("q4nx_main16_qkv_scheduler"),
        asm_helper_symbols=asm_source.count("q4nx_chunk_accum_asm_zol:"),
        asm_save_macros=asm_source.count(".macro SAVE_Q4_CALL_STATE"),
        asm_restore_macros=asm_source.count(".macro RESTORE_Q4_CALL_STATE"),
        asm_q4_lane_macros=asm_source.count(".macro Q4_EXACT_LANE_ZOL"),
        asm_required_markers_present=all(marker in cpp_source + asm_source for marker in markers),
    )


def disasm_check(disasm: str) -> DisasmCheck:
    body = disasm_function_body(disasm, "q4nx_main16_qkv_scheduler")
    helper_calls = len(re.findall(r"R_AIE_1\s+q4nx_chunk_accum_asm_zol\b", body))
    record_calls = len(re.findall(r"R_AIE_1\s+_ZN12_GLOBAL__N_122emit_accum_body_record", body))
    activation_full = body.count("acq\t#49")
    weight_full = body.count("acq\t#51")
    activation_empty = body.count("rel\t#48")
    weight_empty = body.count("rel\t#50")
    record_empty = body.count("acq\t#52")
    record_full = body.count("rel\t#53")
    return DisasmCheck(
        qkv_static_helper_calls=helper_calls,
        qkv_static_record_emit_calls=record_calls,
        activation_full_acq=activation_full,
        weight_full_acq=weight_full,
        activation_empty_rel=activation_empty,
        weight_empty_rel=weight_empty,
        record_empty_acq=record_empty,
        record_full_rel=record_full,
        lock_shape_valid=(
            helper_calls == 3
            and record_calls == 3
            and activation_full == 3
            and weight_full == 3
            and activation_empty == 3
            and weight_empty == 3
            and record_empty == 3
            and record_full == 3
        ),
    )


def phase_to_json(phase: PhaseSpec):
    return {
        "name": phase.name,
        "records": phase.records,
        "chunks_per_record": phase.chunks_per_record,
        "chunks": phase.chunks,
        "weight_base": phase.weight_base,
        "phase": phase.phase,
    }


def render_asm_contract() -> str:
    lines = [
        "// Generated by exp125. This is a nocall scheduler contract outline, not production assembly.",
        "// Target symbol: q4nx_main16_qkv_scheduler_nocall",
        "// C ABI parameters match q4nx_main16_qkv_scheduler:",
        "//   p0/p1: wt_ping/wt_pong",
        "//   p2/p3: activation ping/pong",
        "//   p4/p5: record ping/pong",
        "//   r0/r1/r2: group/row/num_rows",
        "",
        ".macro Q4NX_NOCALL_ACQUIRE_ACT_WEIGHT",
        "\t// acquire L1 activation full",
        "\t// acquire L3 weight full",
        ".endm",
        "",
        ".macro Q4NX_NOCALL_RELEASE_ACT_WEIGHT",
        "\t// release L0 activation empty",
        "\t// release L2 weight empty",
        ".endm",
        "",
        ".macro Q4NX_NOCALL_EMIT_RECORD phase, block",
        "\t// acquire L4 record empty",
        "\t// write header = (phase << 24) | (block << 20) | (group << 16) | (row << 8) | packet_id",
        "\t// write 16-dword payload from register-resident accumulators",
        "\t// release L5 record full",
        ".endm",
        "",
        ".macro Q4NX_NOCALL_QKV_PHASE name, records, chunks_per_record, weight_base, phase",
        "\t// for block in records:",
        "\t//   zero/register-initialize accumulators once per record",
        "\t//   for chunk in chunks_per_record:",
        "\t//     Q4NX_NOCALL_ACQUIRE_ACT_WEIGHT",
        "\t//     inline exact Q4NX body without SAVE_Q4_CALL_STATE or RESTORE_Q4_CALL_STATE",
        "\t//     Q4NX_NOCALL_RELEASE_ACT_WEIGHT",
        "\t//   Q4NX_NOCALL_EMIT_RECORD phase, block",
        ".endm",
        "",
        ".macro Q4NX_MAIN16_QKV_SCHEDULER_NOCALL_BODY",
    ]
    for phase in QKV_PHASES:
        lines.append(
            f"\tQ4NX_NOCALL_QKV_PHASE {phase.name}, {phase.records}, {phase.chunks_per_record}, {phase.weight_base}, {phase.phase}"
        )
    lines.extend([".endm", ""])
    return "\n".join(lines)


def render_json(check: SourceCheck, disasm: DisasmCheck, overhead: HelperOverhead, cost: DynamicCost):
    return {
        "source_check": check.__dict__,
        "disasm_check": disasm.__dict__,
        "helper_overhead": overhead.__dict__,
        "dynamic_cost": cost.__dict__,
        "qkv_phases": [phase_to_json(phase) for phase in QKV_PHASES],
        "full_phases": [phase_to_json(phase) for phase in FULL_PHASES],
        "target_symbol": "q4nx_main16_qkv_scheduler_nocall",
        "target_invariants": [
            "no jl/call to q4nx_chunk_accum_asm_zol inside qkv scheduler",
            "no SAVE_Q4_CALL_STATE/RESTORE_Q4_CALL_STATE per chunk",
            "no per-chunk target accumulator load/store roundtrip",
            "same DMA0/DMA1/record lock ABI as q4nx_main16_qkv_scheduler",
            "same record headers and compact packet ids as current qkv prefix",
        ],
    }


def render_phase_table(phases: tuple[PhaseSpec, ...]) -> list[str]:
    lines = [
        "| Phase | Records | Chunks/Record | Calls/Tile | Weight Base |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase in phases:
        lines.append(
            f"| `{phase.name}` | {phase.records} | {phase.chunks_per_record} | "
            f"{phase.chunks} | {phase.weight_base} |"
        )
    return lines


def render_report(check: SourceCheck, disasm: DisasmCheck, overhead: HelperOverhead, cost: DynamicCost) -> str:
    lines = [
        "# Main16 QKV Nocall Scheduler Contract",
        "",
        "This experiment quantifies the current per-chunk helper-call boundary and",
        "defines the minimal contract for `q4nx_main16_qkv_scheduler_nocall`.",
        "",
        "## Source Checks",
        "",
        f"- C++ helper call sites: `{check.cpp_helper_call_sites}`",
        f"- C++ QKV scheduler symbol references: `{check.cpp_qkv_scheduler_symbols}`",
        f"- ASM helper symbols: `{check.asm_helper_symbols}`",
        f"- SAVE macros: `{check.asm_save_macros}`",
        f"- RESTORE macros: `{check.asm_restore_macros}`",
        f"- Q4 lane macros: `{check.asm_q4_lane_macros}`",
        f"- Required markers present: `{check.asm_required_markers_present}`",
        "",
        "## Current QKV Scheduler Disassembly",
        "",
        f"- Static helper call relocations: `{disasm.qkv_static_helper_calls}`",
        f"- Static record emit call relocations: `{disasm.qkv_static_record_emit_calls}`",
        f"- Activation full acquire shape: `acq #49` x `{disasm.activation_full_acq}`",
        f"- Weight full acquire shape: `acq #51` x `{disasm.weight_full_acq}`",
        f"- Activation empty release shape: `rel #48` x `{disasm.activation_empty_rel}`",
        f"- Weight empty release shape: `rel #50` x `{disasm.weight_empty_rel}`",
        f"- Record empty acquire shape: `acq #52` x `{disasm.record_empty_acq}`",
        f"- Record full release shape: `rel #53` x `{disasm.record_full_rel}`",
        f"- Lock shape valid: `{disasm.lock_shape_valid}`",
        "",
        "## Current Dynamic Helper Boundary",
        "",
        *render_phase_table(QKV_PHASES),
        "",
        f"- QKV helper calls per tile: `{cost.qkv_calls_per_tile}`",
        f"- QKV helper calls across main16: `{cost.qkv_calls_all_main16}`",
        f"- Full-layer helper calls per tile: `{cost.full_calls_per_tile}`",
        f"- Full-layer helper calls across main16: `{cost.full_calls_all_main16}`",
        "",
        "## Per-Call Boundary Shape",
        "",
        f"- SAVE slots per call: `{overhead.save_slots}`",
        f"- RESTORE slots per call: `{overhead.restore_slots}`",
        f"- Helper wrapper slots before macro expansion: `{overhead.helper_wrapper_slots}`",
        f"- `Q4_EXACT_LANE_ZOL` invocations per call: `{overhead.helper_lane_invocations}`",
        f"- Accumulator loads per call from local memory: `{overhead.accum_loads_per_call}`",
        f"- Accumulator stores per call to local memory: `{overhead.accum_stores_per_call}`",
        f"- Exact 32-dim groups per call: `{overhead.q4_exact32_groups_per_call}`",
        "",
        "## Dynamic Boundary Cost",
        "",
        f"- QKV save/restore instruction slots per tile: `{cost.qkv_save_restore_slots_per_tile}`",
        f"- Full-layer save/restore instruction slots per tile: `{cost.full_save_restore_slots_per_tile}`",
        f"- QKV accumulator load/store roundtrips per tile: `{cost.qkv_accum_loads_per_tile}` / `{cost.qkv_accum_stores_per_tile}`",
        f"- Full-layer accumulator load/store roundtrips per tile: `{cost.full_accum_loads_per_tile}` / `{cost.full_accum_stores_per_tile}`",
        "",
        "## Nocall Scheduler Contract",
        "",
        "- Keep the current MLIR topology, buffer addresses, DMA rings, and lock IDs.",
        "- Add `q4nx_main16_qkv_scheduler_nocall` with the same C ABI as `q4nx_main16_qkv_scheduler`.",
        "- Implement Q/K/V record-major loops inside assembly.",
        "- Acquire activation/weight locks per chunk and release them after the inline body.",
        "- Keep accumulators register-resident across the 16 chunks of one record.",
        "- Emit the same 17-dword record headers and payload layout as the current C++ scheduler.",
        "- The scheduler body must not call or jump to `q4nx_chunk_accum_asm_zol`.",
        "",
        "## Generated Contract Include",
        "",
        f"- `{DEFAULT_ASM_INC}`",
        "",
        "## Next Step",
        "",
        "Use this contract to write a QKV-only linked assembly scheduler and run",
        "`full-layer-qkv-prefix` before attempting the full-layer scheduler.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--asm-inc", type=Path, default=DEFAULT_ASM_INC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpp_source = MAIN16_CC.read_text()
    asm_source = MAIN16_ASM.read_text()
    check = source_check(cpp_source, asm_source)
    disasm = disasm_check(disassemble_role_object())
    overhead = helper_overhead(asm_source)
    cost = dynamic_cost(overhead)
    args.asm_inc.write_text(render_asm_contract())
    args.json_output.write_text(json.dumps(render_json(check, disasm, overhead, cost), indent=2) + "\n")
    args.report.write_text(render_report(check, disasm, overhead, cost))
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.asm_inc}")
    print(f"qkv_calls_per_tile={cost.qkv_calls_per_tile}")
    print(f"full_calls_per_tile={cost.full_calls_per_tile}")
    print(f"qkv_save_restore_slots_per_tile={cost.qkv_save_restore_slots_per_tile}")
    return 0 if cost.qkv_calls_per_tile == 192 and cost.full_calls_per_tile == 1472 and disasm.lock_shape_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
