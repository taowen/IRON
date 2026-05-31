#!/usr/bin/env python3
"""Build the MyLM-style whole-main16 migration contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
MYLM_ROOT = Path("/var/home/taowen/projects/MyLM")
MAIN16_OBJECT = REPO_ROOT / "qwen3-layer/main_projection_q4nx_fast.o"
MYLM_UNDERSTANDING = MYLM_ROOT / "tools/re/fused-layer-engine/current-understanding.md"
IRON_COMPARISON = REPO_ROOT / "qwen3-layer/main16_q4nx_mylm_compare.md"
MYLM_OPERAND_GRAPH = REPO_ROOT / "experiments/119_mylm_q4nx_full_operand_graph/mylm_q4nx_full_operand_graph.json"
MYLM_GENERATOR_CONTRACT = REPO_ROOT / "experiments/120_mylm_q4nx_generator_contract/mylm_q4nx_generator_contract.json"
LLVM_NM = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-nm"
LLVM_OBJDUMP = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump"
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_main16_whole_core_contract.md"
DEFAULT_JSON = EXPERIMENT_DIR / "mylm_main16_whole_core_contract.json"

ACTIVE_Q4_SYMBOL = "q4nx_chunk_accum_asm_zol"
ACTIVE_LAYER_SCHEDULER = "q4nx_main16_layer_scheduler"
REJECTED_NOCALL_SYMBOL = "q4nx_main16_qkv_scheduler_nocall"

PROGRAM_SEGMENTS = (
    ("entry/setup", 0x0000, 492),
    ("shared Q4NX microkernel", 0x01F0, 5760),
    ("Q/K/V body", 0x1870, 1544),
    ("O body", 0x1E80, 1544),
    ("up/gate body", 0x2490, 1544),
    ("down body", 0x2AA0, 1560),
    ("alternate body", 0x30C0, 1544),
    ("dispatcher", 0x36D0, 504),
    ("tail/helper", 0x38D0, 324),
)

PHASE_BODIES = (
    ("Q/K/V", 0x1870, "0x1", 12),
    ("O", 0x1E80, "0x4", 8),
    ("up/gate", 0x2490, "0x8", 48),
    ("down", 0x2AA0, "0x4", 8),
)

IRON_HEADERS = (
    ("Q", 10),
    ("K", 11),
    ("V", 12),
    ("O", 13),
    ("FFN", 14),
    ("down", 15),
)


@dataclass(frozen=True)
class ActiveObjectSummary:
    has_q4: bool
    has_layer_scheduler: bool
    has_rejected_nocall: bool
    static_helper_relocations: int
    q4_counts: dict[str, int]


@dataclass(frozen=True)
class MyLMQ4Contract:
    slot_count: int
    mac_count: int
    group_macs: dict[str, int]
    op_counts: dict[str, int]
    sections: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MigrationDecision:
    active_path: str
    rejected_path: str
    next_path: str
    header_policy: str
    first_gate: str


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def symbol_names(path: Path) -> set[str]:
    text = run_command((str(LLVM_NM), str(path)))
    return {line.split()[-1] for line in text.splitlines() if line.split()}


def disassemble(path: Path) -> str:
    return run_command(
        (
            str(LLVM_OBJDUMP),
            "--triple=aie2p",
            "-dr",
            "--no-print-imm-hex",
            str(path),
        )
    )


def function_body(disasm: str, name: str) -> str:
    start = re.search(rf"^[0-9a-fA-F]+ <{re.escape(name)}>:\n", disasm, re.MULTILINE)
    if start is None:
        return ""
    next_function = re.search(
        r"^[0-9a-fA-F]+ <(?!\.)[^>]+>:\n",
        disasm[start.end():],
        re.MULTILINE,
    )
    if next_function is None:
        return disasm[start.end():]
    return disasm[start.end(): start.end() + next_function.start()]


def op_counts(body: str) -> dict[str, int]:
    patterns = (
        "vmac.f",
        "vextbcst.16",
        "vups.4x",
        "vups.2x",
        "vunpack",
        "vmul.f",
        "vconv.bf16.fp32",
        "vlda",
        "vldb",
        "vst",
        "add.nc\tlc",
        "movxm\tls",
        "movxm\tle",
    )
    return {pattern: body.count(pattern) for pattern in patterns}


def summarize_active_object() -> ActiveObjectSummary:
    names = symbol_names(MAIN16_OBJECT)
    disasm = disassemble(MAIN16_OBJECT)
    q4_body = function_body(disasm, ACTIVE_Q4_SYMBOL)
    return ActiveObjectSummary(
        has_q4=ACTIVE_Q4_SYMBOL in names,
        has_layer_scheduler=ACTIVE_LAYER_SCHEDULER in names,
        has_rejected_nocall=REJECTED_NOCALL_SYMBOL in names,
        static_helper_relocations=disasm.count(f"R_AIE_1\t{ACTIVE_Q4_SYMBOL}"),
        q4_counts=op_counts(q4_body),
    )


def summarize_mylm_q4() -> MyLMQ4Contract:
    graph = read_json(MYLM_OPERAND_GRAPH)
    contract = read_json(MYLM_GENERATOR_CONTRACT)
    return MyLMQ4Contract(
        slot_count=int(graph["slot_count"]),
        mac_count=int(graph["mac_count"]),
        group_macs={str(key): int(value) for key, value in graph["group_macs"].items()},
        op_counts={str(key): int(value) for key, value in graph["op_counts"].items()},
        sections=tuple(contract["sections"]),
        boundaries=tuple(contract["boundaries"]),
    )


def build_decision(active: ActiveObjectSummary) -> MigrationDecision:
    active_path = (
        "keep the single q4nx_main16_layer_scheduler(..., phase_limit) entry "
        "calling q4nx_chunk_accum_asm_zol until the generated whole-main16 "
        "replacement passes the QKV prefix gate"
    )
    rejected_path = (
        "do not re-enable the partial q4nx_main16_qkv_scheduler_nocall path; "
        "it removed the helper statically but timed out on full-layer-qkv-prefix"
    )
    next_path = (
        "generate one main16 role program with MyLM-style sections: shared Q4NX "
        "body, Q/K/V body, O body, up/gate body, down body, and dispatcher"
    )
    header_policy = (
        "preserve IRON compact headers 10..15 at the row1/c1r1 boundary for the "
        "first replacement; MyLM's 0x1/0x4/0x8 headers are a raw-program clue, "
        "not a drop-in ABI for current IRON downstream routing"
    )
    first_gate = (
        "full-layer-qkv-prefix token31 must pass before enabling the generated "
        "main16 program in any full-decode path"
    )
    if active.has_rejected_nocall:
        rejected_path += "; current role object still contains the rejected symbol"
    return MigrationDecision(
        active_path=active_path,
        rejected_path=rejected_path,
        next_path=next_path,
        header_policy=header_policy,
        first_gate=first_gate,
    )


def render_markdown(
    active: ActiveObjectSummary,
    mylm: MyLMQ4Contract,
    decision: MigrationDecision,
) -> str:
    q4_counts = Counter(active.q4_counts)
    lines = [
        "# MyLM Main16 Whole-Core Contract",
        "",
        "This experiment defines the next direction after the failed partial",
        "QKV-only nocall attempt. The goal is a MyLM-style main16 role program,",
        "not another temporary scheduler variant.",
        "",
        "## Active IRON Path",
        "",
        f"- Has active Q4 asm body `{ACTIVE_Q4_SYMBOL}`: `{active.has_q4}`",
        f"- Has active layer scheduler `{ACTIVE_LAYER_SCHEDULER}`: `{active.has_layer_scheduler}`",
        f"- Contains rejected nocall symbol: `{active.has_rejected_nocall}`",
        f"- Static helper-call relocations to `{ACTIVE_Q4_SYMBOL}`: `{active.static_helper_relocations}`",
        "",
        "| Active Q4 op | Count |",
        "| --- | ---: |",
    ]
    for key in active.q4_counts:
        lines.append(f"| `{key}` | {q4_counts[key]} |")

    lines.extend(
        [
            "",
            "## MyLM Whole-Core Shape",
            "",
            "| Segment | Offset | Bytes |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, offset, size in PROGRAM_SEGMENTS:
        lines.append(f"| `{name}` | `0x{offset:04x}` | {size} |")

    lines.extend(
        [
            "",
            "## MyLM Phase Bodies",
            "",
            "| Phase | Body | Header | Records/tile |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for name, body, header, records in PHASE_BODIES:
        lines.append(f"| `{name}` | `0x{body:04x}` | `{header}` | {records} |")

    lines.extend(
        [
            "",
            "## MyLM Q4 Body Contract",
            "",
            f"- Hot-loop slots: `{mylm.slot_count}`",
            f"- Static `vmac.f`: `{mylm.mac_count}`",
            f"- Group MAC shape: `{','.join(str(mylm.group_macs[str(idx)]) for idx in range(8))}`",
            "",
            "| MyLM Q4 op | Count |",
            "| --- | ---: |",
        ]
    )
    for key in sorted(mylm.op_counts):
        lines.append(f"| `{key}` | {mylm.op_counts[key]} |")

    lines.extend(
        [
            "",
            "## Generator Sections",
            "",
            "| Section | Groups | Template | MACs/group | Live in | Live out | Signature |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for section in mylm.sections:
        live_in = "entry" if section["live_in_boundary"] is None else f"boundary{section['live_in_boundary']}"
        live_out = "exit" if section["live_out_boundary"] is None else f"boundary{section['live_out_boundary']}"
        groups = ",".join(str(item) for item in section["groups"])
        signature = section["signature_hash"] or ""
        lines.append(
            f"| `{section['name']}` | `{groups}` | {section['template_group']} | "
            f"{section['macs_per_group']} | `{live_in}` | `{live_out}` | `{signature}` |"
        )

    lines.extend(
        [
            "",
            "## Header Policy",
            "",
            "Current IRON downstream routing uses packet/header IDs:",
            "",
            "| Phase | IRON header packet id |",
            "| --- | ---: |",
        ]
    )
    for name, packet_id in IRON_HEADERS:
        lines.append(f"| `{name}` | {packet_id} |")

    lines.extend(
        [
            "",
            "MyLM's `0x1/0x4/0x8` phase headers are part of the raw program",
            "contract, but they are not a drop-in replacement for current IRON",
            "row1/c1r1/c1r3 routing. The first whole-core replacement should keep",
            "IRON headers while adopting the MyLM phase/body/register scheduling",
            "shape.",
            "",
            "## Decision",
            "",
            f"- Active path: {decision.active_path}.",
            f"- Rejected path: {decision.rejected_path}.",
            f"- Next path: {decision.next_path}.",
            f"- Header policy: {decision.header_policy}.",
            f"- First gate: {decision.first_gate}.",
            "",
            "## Direction",
            "",
            "The next useful implementation is a generated main16 role program with",
            "one shared Q4 body and phase bodies that own the lock, record, and",
            "phase-order protocol. A QKV-only linked asm scheduler is too small a",
            "slice: it changes the call boundary without giving the code generator",
            "the fixed register/control plan that makes MyLM fast.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: Path, json_output: Path) -> None:
    active = summarize_active_object()
    mylm = summarize_mylm_q4()
    decision = build_decision(active)
    payload = {
        "active": {
            "has_q4": active.has_q4,
            "has_layer_scheduler": active.has_layer_scheduler,
            "has_rejected_nocall": active.has_rejected_nocall,
            "static_helper_relocations": active.static_helper_relocations,
            "q4_counts": active.q4_counts,
        },
        "mylm": {
            "program_segments": [
                {"name": name, "offset": offset, "bytes": size}
                for name, offset, size in PROGRAM_SEGMENTS
            ],
            "phase_bodies": [
                {"name": name, "body": body, "header": header, "records": records}
                for name, body, header, records in PHASE_BODIES
            ],
            "q4": {
                "slot_count": mylm.slot_count,
                "mac_count": mylm.mac_count,
                "group_macs": mylm.group_macs,
                "op_counts": mylm.op_counts,
                "sections": list(mylm.sections),
                "boundaries": list(mylm.boundaries),
            },
        },
        "decision": {
            "active_path": decision.active_path,
            "rejected_path": decision.rejected_path,
            "next_path": decision.next_path,
            "header_policy": decision.header_policy,
            "first_gate": decision.first_gate,
        },
        "sources": {
            "mylm_understanding": str(MYLM_UNDERSTANDING),
            "iron_comparison": str(IRON_COMPARISON),
            "mylm_operand_graph": str(MYLM_OPERAND_GRAPH),
            "mylm_generator_contract": str(MYLM_GENERATOR_CONTRACT),
            "main16_object": str(MAIN16_OBJECT),
        },
    }
    report.write_text(render_markdown(active, mylm, decision) + "\n")
    json_output.write_text(json.dumps(payload, indent=2) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    write_outputs(args.report, args.json)
    print(f"wrote {args.report}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
