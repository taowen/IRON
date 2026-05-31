#!/usr/bin/env python3
"""Build a full-hot-loop operand graph for every MyLM Q4NX MAC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP115_RUN = REPO_ROOT / "experiments/115_mylm_q4nx_operand_graph/run.py"
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_q4nx_full_operand_graph.md"
DEFAULT_JSON = EXPERIMENT_DIR / "mylm_q4nx_full_operand_graph.json"
STEADY_GROUPS = (1, 2, 3, 4, 5)
EXPECTED_GROUP_MACS = {0: 28, 1: 33, 2: 33, 3: 33, 4: 33, 5: 33, 6: 33, 7: 38}


@dataclass(frozen=True)
class GraphCheck:
    slot_count: int
    mac_count: int
    group_macs: dict[int, int]
    op_counts: Counter[str]
    steady_text_hashes: dict[int, str]
    steady_signature_hashes: dict[int, str]
    steady_text_stable: bool
    all_steady_signature_stable: bool
    steady_to_steady_signature_stable: bool
    expected_group_counts: bool
    mixed_vector_operands: int
    cross_group_vector_operands: int
    cross_group_accumulators: int


def load_exp115() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp115_operand_graph", EXP115_RUN)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {EXP115_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def section_for_group(group: int) -> str:
    if group == 0:
        return "fill"
    if group in STEADY_GROUPS:
        return "steady"
    if group == 6:
        return "pre_drain"
    return "drain"


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def cell_to_json(exp115: ModuleType, cell, consumer_group: int):
    producer = cell.slot
    return {
        "cell": cell.cell,
        "producer": exp115.slot_label(producer),
        "producer_kind": cell.kind,
        "lane": cell.lane,
        "producer_group": None if producer is None else producer.group,
        "crosses_group": producer is not None and producer.group != consumer_group,
    }


def operand_kind(exp115: ModuleType, trace) -> str:
    return exp115.operand_kind(trace)


def operand_to_json(exp115: ModuleType, trace, consumer_group: int):
    cells = [cell_to_json(exp115, cell, consumer_group) for cell in trace.cells]
    return {
        "register": trace.register,
        "kind": operand_kind(exp115, trace),
        "mixed": trace.mixed,
        "crosses_group": any(cell["crosses_group"] for cell in cells),
        "cells": cells,
    }


def mac_signature(exp115: ModuleType, trace) -> str:
    operands = (trace.accumulator, trace.left, trace.right)
    pieces: list[str] = [trace.slot.op]
    for operand in operands:
        cell_pieces: list[str] = []
        for cell in operand.cells:
            producer = cell.slot
            group_delta = "entry" if producer is None else str(trace.slot.group - producer.group)
            producer_offset = "entry" if producer is None else exp115.slot_offset(producer)
            lane = "" if cell.lane is None else cell.lane
            cell_pieces.append(f"{cell.cell}:{cell.kind}:{lane}:dg{group_delta}:{producer_offset}")
        pieces.append("|".join(cell_pieces))
    return " || ".join(pieces)


def mac_to_json(exp115: ModuleType, trace):
    return {
        "slot": exp115.slot_label(trace.slot),
        "offset": exp115.slot_offset(trace.slot),
        "group": trace.slot.group,
        "section": section_for_group(trace.slot.group),
        "accumulator": operand_to_json(exp115, trace.accumulator, trace.slot.group),
        "left": operand_to_json(exp115, trace.left, trace.slot.group),
        "right": operand_to_json(exp115, trace.right, trace.slot.group),
        "signature": mac_signature(exp115, trace),
    }


def group_text_hashes(exp115: ModuleType, slots) -> dict[int, str]:
    return {
        group: digest([slot.text for slot in slots if slot.group == group])
        for group in STEADY_GROUPS
    }


def group_signature_hashes(exp115: ModuleType, traces) -> dict[int, str]:
    return {
        group: digest([mac_signature(exp115, trace) for trace in traces if trace.slot.group == group])
        for group in STEADY_GROUPS
    }


def operand_crosses_group(operand, consumer_group: int) -> bool:
    return any(cell.slot is not None and cell.slot.group != consumer_group for cell in operand.cells)


def check_graph(exp115: ModuleType, slots, traces) -> GraphCheck:
    group_macs = {group: sum(trace.slot.group == group for trace in traces) for group in range(8)}
    text_hashes = group_text_hashes(exp115, slots)
    signature_hashes = group_signature_hashes(exp115, traces)
    vector_operands = [operand for trace in traces for operand in (trace.left, trace.right)]
    accumulators = [trace.accumulator for trace in traces]
    return GraphCheck(
        slot_count=len(slots),
        mac_count=len(traces),
        group_macs=group_macs,
        op_counts=Counter(slot.op for slot in slots),
        steady_text_hashes=text_hashes,
        steady_signature_hashes=signature_hashes,
        steady_text_stable=len(set(text_hashes.values())) == 1,
        all_steady_signature_stable=len(set(signature_hashes.values())) == 1,
        steady_to_steady_signature_stable=len({signature_hashes[group] for group in (2, 3, 4, 5)}) == 1,
        expected_group_counts=group_macs == EXPECTED_GROUP_MACS,
        mixed_vector_operands=sum(operand.mixed for operand in vector_operands),
        cross_group_vector_operands=sum(
            operand_crosses_group(operand, trace.slot.group)
            for trace in traces
            for operand in (trace.left, trace.right)
        ),
        cross_group_accumulators=sum(
            any(cell.slot is not None and cell.slot.group != trace.slot.group for cell in trace.accumulator.cells)
            for trace in traces
        ),
    )


def render_report(exp115: ModuleType, check: GraphCheck, traces) -> str:
    vector_pair_counts = Counter(
        " + ".join(sorted((operand_kind(exp115, trace.left), operand_kind(exp115, trace.right))))
        for trace in traces
    )
    lines = [
        "# MyLM Q4NX Full Operand Graph",
        "",
        "This experiment emits operand graph records for every `vmac.f` in the",
        "full MyLM Q4NX hot loop, not only the canonical steady group.",
        "",
        "## Checks",
        "",
        f"- Parsed slots: `{check.slot_count}`",
        f"- `vmac.f` records: `{check.mac_count}`",
        f"- Expected group MAC counts: `{check.expected_group_counts}`",
        f"- Steady instruction text stable: `{check.steady_text_stable}`",
        f"- All steady operand signatures stable: `{check.all_steady_signature_stable}`",
        f"- Steady-to-steady operand signatures stable: `{check.steady_to_steady_signature_stable}`",
        f"- Mixed vector operands: `{check.mixed_vector_operands}` / `{check.mac_count * 2}`",
        f"- Cross-group vector operands: `{check.cross_group_vector_operands}` / `{check.mac_count * 2}`",
        f"- Cross-group accumulator operands: `{check.cross_group_accumulators}` / `{check.mac_count}`",
        "",
        "## Group MAC Counts",
        "",
        "| Group | Section | MACs |",
        "| ---: | --- | ---: |",
    ]
    for group, count in check.group_macs.items():
        lines.append(f"| {group} | `{section_for_group(group)}` | {count} |")

    lines.extend(
        [
            "",
            "## Steady Hashes",
            "",
            "| Group | Text Hash | Operand Signature Hash |",
            "| ---: | --- | --- |",
        ]
    )
    for group in STEADY_GROUPS:
        lines.append(
            f"| {group} | `{check.steady_text_hashes[group]}` | "
            f"`{check.steady_signature_hashes[group]}` |"
        )

    lines.extend(
        [
            "",
            "## Opcode Counts",
            "",
            "| Op | Count |",
            "| --- | ---: |",
        ]
    )
    for op in (
        "vmac.f",
        "vextbcst.16",
        "vunpack",
        "vups.4x",
        "vconv.bf16.fp32",
        "vlda",
        "vldb",
        "lda.s16",
        "vbcst.16",
        "vst",
    ):
        lines.append(f"| `{op}` | {check.op_counts[op]} |")

    lines.extend(
        [
            "",
            "## Vector Operand Producer Pairs",
            "",
            "| Producer Pair | MAC Count |",
            "| --- | ---: |",
        ]
    )
    for pair, count in vector_pair_counts.most_common():
        lines.append(f"| `{pair}` | {count} |")

    lines.extend(
        [
            "",
            "## First Fill MACs",
            "",
            "| MAC | Acc | Left | Right |",
            "| --- | --- | --- | --- |",
        ]
    )
    for trace in traces[:12]:
        lines.append(
            f"| `{exp115.slot_label(trace.slot)}` | "
            f"`{operand_kind(exp115, trace.accumulator)}` | "
            f"`{operand_kind(exp115, trace.left)}` | "
            f"`{operand_kind(exp115, trace.right)}` |"
        )

    lines.extend(
        [
            "",
            "## Generator Boundary",
            "",
            "- The JSON output is the first full-loop MAC operand graph: all fill, steady, pre-drain, and drain MACs are represented.",
            "- The steady instruction text is stable. The group1 operand signature differs because it is the fill-to-steady transition; groups2..5 share the steady-to-steady signature.",
            "- The next experiment should consume this graph plus exp118 liveness to emit a modified numeric body and run a synthetic gate before touching `qwen3-layer`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_json(exp115: ModuleType, check: GraphCheck, traces):
    return {
        "source": "MyLM c2r2 0x260..0x1850",
        "slot_count": check.slot_count,
        "mac_count": check.mac_count,
        "group_macs": check.group_macs,
        "steady_text_hashes": check.steady_text_hashes,
        "steady_signature_hashes": check.steady_signature_hashes,
        "steady_text_stable": check.steady_text_stable,
        "all_steady_signature_stable": check.all_steady_signature_stable,
        "steady_to_steady_signature_stable": check.steady_to_steady_signature_stable,
        "op_counts": dict(sorted(check.op_counts.items())),
        "macs": [mac_to_json(exp115, trace) for trace in traces],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp115 = load_exp115()
    slots = exp115.parse_slots(exp115.DEFAULT_DISASM)
    if not slots:
        raise ValueError(f"no slots parsed from {exp115.DEFAULT_DISASM}")
    traces = exp115.trace_macs(slots)
    check = check_graph(exp115, slots, traces)
    args.json_output.write_text(json.dumps(render_json(exp115, check, traces), indent=2) + "\n")
    args.report.write_text(render_report(exp115, check, traces))
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"slots={check.slot_count}")
    print(f"vmac.f={check.mac_count}")
    print(f"group_macs={check.group_macs}")
    print(f"steady_text_stable={check.steady_text_stable}")
    print(f"all_steady_signature_stable={check.all_steady_signature_stable}")
    print(f"steady_to_steady_signature_stable={check.steady_to_steady_signature_stable}")
    return 0 if check.mac_count == 264 and check.expected_group_counts else 1


if __name__ == "__main__":
    raise SystemExit(main())
