#!/usr/bin/env python3
"""Trace MyLM Q4NX half-register cell producers at a steady boundary."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP006_RUN = REPO_ROOT / "main16-exps/006_q4nx_alias_lifetime_graph/run.py"
MANIFEST = EXPERIMENT_DIR / "half_register_boundary_trace.json"
REPORT = EXPERIMENT_DIR / "half_register_boundary_trace.md"
STEADY_GROUP = 1


@dataclass(frozen=True)
class CellProducer:
    cell: str
    event_index: int | None
    group: int | None
    address: int | None
    op: str
    semantic: str
    lane: int | None


@dataclass(frozen=True)
class OperandTrace:
    register: str
    cells: tuple[CellProducer, ...]


@dataclass(frozen=True)
class MacTrace:
    index: int
    address: int
    accumulator: OperandTrace
    left: OperandTrace
    right: OperandTrace


@dataclass(frozen=True)
class BoundaryCell:
    cell: str
    producer: CellProducer
    first_use_index: int
    first_use_address: int
    first_use_op: str


def load_exp006() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp006_for_half_cells", EXP006_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp006 helper: {EXP006_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP006 = load_exp006()


def dedup(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def register_cells(register: str) -> tuple[str, ...]:
    match = re.fullmatch(r"x(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"vec{index}.lo", f"vec{index}.hi")
    match = re.fullmatch(r"wl(\d+)", register)
    if match is not None:
        return (f"vec{match.group(1)}.lo",)
    match = re.fullmatch(r"wh(\d+)", register)
    if match is not None:
        return (f"vec{match.group(1)}.hi",)
    match = re.fullmatch(r"dm(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (
            f"acc{index}.bmll",
            f"acc{index}.bmlh",
            f"acc{index}.bmhl",
            f"acc{index}.bmhh",
        )
    match = re.fullmatch(r"cml(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"acc{index}.bmll", f"acc{index}.bmlh")
    match = re.fullmatch(r"cmh(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"acc{index}.bmhl", f"acc{index}.bmhh")
    match = re.fullmatch(r"bm(ll|lh|hl|hh)(\d+)", register)
    if match is not None:
        return (f"acc{match.group(2)}.bm{match.group(1)}",)
    return ()


def event_producer(event) -> CellProducer:
    return CellProducer(
        cell="",
        event_index=event.index,
        group=event.group,
        address=event.address,
        op=event.op,
        semantic=event.semantic,
        lane=event.lane,
    )


def entry_producer(cell: str) -> CellProducer:
    return CellProducer(
        cell=cell,
        event_index=None,
        group=None,
        address=None,
        op="entry",
        semantic="entry",
        lane=None,
    )


def with_cell(producer: CellProducer, cell: str) -> CellProducer:
    return CellProducer(
        cell=cell,
        event_index=producer.event_index,
        group=producer.group,
        address=producer.address,
        op=producer.op,
        semantic=producer.semantic,
        lane=producer.lane,
    )


def event_label(producer: CellProducer) -> str:
    if producer.event_index is None or producer.address is None:
        return "entry"
    lane = "" if producer.lane is None else f" lane #{producer.lane:#x}"
    return f"g{producer.group}@0x{producer.address:x}:{producer.op}/{producer.semantic}{lane}"


def producer_key(producer: CellProducer) -> tuple[int | None, str, str, int | None]:
    return (producer.event_index, producer.op, producer.semantic, producer.lane)


def operand_trace(register: str, state: dict[str, CellProducer]) -> OperandTrace:
    cells = tuple(with_cell(state.get(cell, entry_producer(cell)), cell) for cell in register_cells(register))
    return OperandTrace(register=register, cells=cells)


def is_mixed(trace: OperandTrace) -> bool:
    if len(trace.cells) <= 1:
        return False
    return len({producer_key(cell) for cell in trace.cells}) > 1


def crosses_group(trace: OperandTrace, group: int) -> bool:
    return any(cell.group is not None and cell.group < group for cell in trace.cells)


def producer_shape(trace: OperandTrace) -> str:
    return "+".join(dedup([cell.semantic for cell in trace.cells]))


def trace_cells() -> tuple[tuple[BoundaryCell, ...], tuple[MacTrace, ...], dict[str, int]]:
    events = EXP006.build_events()
    state: dict[str, CellProducer] = {}
    boundary: dict[str, BoundaryCell] = {}
    macs: list[MacTrace] = []
    vector_shapes: Counter[str] = Counter()
    mixed_vector_operands = 0
    cross_group_vector_operands = 0
    cross_group_accumulators = 0

    for event in events:
        if event.group == STEADY_GROUP:
            for register in event.uses:
                for cell in register_cells(register):
                    producer = state.get(cell)
                    if producer is not None and producer.group == STEADY_GROUP - 1 and cell not in boundary:
                        boundary[cell] = BoundaryCell(
                            cell=cell,
                            producer=with_cell(producer, cell),
                            first_use_index=event.index,
                            first_use_address=event.address,
                            first_use_op=event.op,
                        )

        if event.group == STEADY_GROUP and event.semantic == "mac" and len(event.uses) >= 3:
            accumulator = operand_trace(event.uses[0], state)
            left = operand_trace(event.uses[1], state)
            right = operand_trace(event.uses[2], state)
            macs.append(
                MacTrace(
                    index=event.index,
                    address=event.address,
                    accumulator=accumulator,
                    left=left,
                    right=right,
                )
            )
            for trace in (left, right):
                vector_shapes[producer_shape(trace)] += 1
                if is_mixed(trace):
                    mixed_vector_operands += 1
                if crosses_group(trace, STEADY_GROUP):
                    cross_group_vector_operands += 1
            if crosses_group(accumulator, STEADY_GROUP):
                cross_group_accumulators += 1

        if event.defs:
            producer = event_producer(event)
            for register in event.defs:
                for cell in register_cells(register):
                    state[cell] = with_cell(producer, cell)

    metrics = {
        "group": STEADY_GROUP,
        "boundary_cells": len(boundary),
        "mac_count": len(macs),
        "mixed_vector_operands": mixed_vector_operands,
        "cross_group_vector_operands": cross_group_vector_operands,
        "cross_group_accumulators": cross_group_accumulators,
    }
    metrics.update({f"shape:{name}": count for name, count in sorted(vector_shapes.items())})
    return tuple(boundary[key] for key in sorted(boundary)), tuple(macs), metrics


def trace_row(trace: OperandTrace) -> dict[str, object]:
    return {
        "register": trace.register,
        "mixed": is_mixed(trace),
        "crosses_group": crosses_group(trace, STEADY_GROUP),
        "cells": [
            {
                "cell": cell.cell,
                "producer": event_label(cell),
                "semantic": cell.semantic,
                "lane": cell.lane,
            }
            for cell in trace.cells
        ],
    }


def build_manifest() -> dict[str, object]:
    boundary, macs, metrics = trace_cells()
    return {
        "status": "passed",
        "source": str(EXP006.EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "steady_group": STEADY_GROUP,
        "metrics": metrics,
        "boundary_cells": [
            {
                "cell": item.cell,
                "producer": event_label(item.producer),
                "first_use": f"g{STEADY_GROUP}@0x{item.first_use_address:x}:{item.first_use_op}",
            }
            for item in boundary
        ],
        "group_mac_traces": [
            {
                "index": item.index,
                "address": hex(item.address),
                "accumulator": trace_row(item.accumulator),
                "left": trace_row(item.left),
                "right": trace_row(item.right),
            }
            for item in macs
        ],
    }


def render_trace(trace: dict[str, object]) -> str:
    cells = trace["cells"]
    if not isinstance(cells, list):
        raise TypeError("trace cells must be a list")
    details = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise TypeError("trace cell must be a dict")
        details.append(f"{cell['cell']} <- {cell['producer']}")
    flags = []
    if trace["mixed"]:
        flags.append("mixed")
    if trace["crosses_group"]:
        flags.append("cross-group")
    suffix = "" if not flags else " [" + ", ".join(flags) + "]"
    return f"`{trace['register']}`{suffix}<br>" + "<br>".join(details)


def render_report(manifest: dict[str, object]) -> str:
    metrics = manifest["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    lines = [
        "# Half-Register Boundary Trace",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Source: `{manifest['source']}`",
        f"- Steady group: `{manifest['steady_group']}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for name, value in metrics.items():
        lines.append(f"| `{name}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Boundary Cells Into Group1",
            "",
            "| cell | producer | first group1 use |",
            "| --- | --- | --- |",
        ]
    )
    boundary_cells = manifest["boundary_cells"]
    if not isinstance(boundary_cells, list):
        raise TypeError("boundary_cells must be a list")
    for item in boundary_cells:
        if not isinstance(item, dict):
            raise TypeError("boundary cell must be a dict")
        lines.append(f"| `{item['cell']}` | `{item['producer']}` | `{item['first_use']}` |")

    lines.extend(
        [
            "",
            "## Group1 MAC Operand Table",
            "",
            "| MAC | Accumulator | Left vector | Right vector |",
            "| --- | --- | --- | --- |",
        ]
    )
    mac_traces = manifest["group_mac_traces"]
    if not isinstance(mac_traces, list):
        raise TypeError("group_mac_traces must be a list")
    for item in mac_traces:
        if not isinstance(item, dict):
            raise TypeError("mac trace must be a dict")
        lines.append(
            f"| `{item['address']}` | {render_trace(item['accumulator'])} | "
            f"{render_trace(item['left'])} | {render_trace(item['right'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Group1 is already in the steady state of the MyLM hot loop. Many vector "
            "operands are assembled from separately produced halves, and some of "
            "those halves cross the group0->group1 boundary. Treating an `xN` "
            "register as one indivisible value hides the actual software pipeline.",
            "",
            "The next experiment should connect these cell producers to the Q4NX "
            "payload formula: which cells are unpacked nibbles, which cells are "
            "bf16 coefficients after `vups.4x/vadd/vsub/vconv`, and which cells "
            "are activation broadcasts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        manifest = build_manifest()
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Half-Register Boundary Trace\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

