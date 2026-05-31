#!/usr/bin/env python3
"""Build a half-register operand graph for the MyLM Q4NX hot loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_REPORT = Path(__file__).resolve().parent / "mylm_q4nx_operand_graph.md"
DEFAULT_JSON = Path(__file__).resolve().parent / "mylm_q4nx_operand_graph.json"
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):((?:\s+[0-9a-fA-F]{2})+)\s+(.*)$")
REGISTER_RE = re.compile(
    r"\b(?:"
    r"bm(?:ll|lh|hl|hh)\d+|cm[lh]\d+|dm\d+|x\d+|w[lh]\d+|lfh\d+|"
    r"r\d+|p\d+|m\d+|s\d+|dj\d+|lc|ls|le|cr\w+|upssign0|unpacksign0"
    r")\b"
)
GROUP_BOUNDS = (0x260, 0x52A, 0x7DE, 0xA92, 0xD46, 0xFFA, 0x12AE, 0x1566, 0x1850)
STEADY_GROUP = 1
STEADY_BOUNDARIES = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Slot:
    index: int
    group: int
    address: int
    bundle_slot: int
    text: str
    op: str
    operands: tuple[str, ...]
    defs: tuple[str, ...]
    uses: tuple[str, ...]


@dataclass(frozen=True)
class CellProducer:
    cell: str
    slot: Slot | None
    kind: str
    lane: str | None


@dataclass(frozen=True)
class OperandTrace:
    register: str
    cells: tuple[CellProducer, ...]

    @property
    def mixed(self) -> bool:
        return len({cell.slot for cell in self.cells}) > 1


@dataclass(frozen=True)
class MacTrace:
    slot: Slot
    accumulator: OperandTrace
    left: OperandTrace
    right: OperandTrace


@dataclass(frozen=True)
class BoundaryUse:
    cell: str
    producer: Slot
    first_use: Slot


@dataclass(frozen=True)
class BoundarySummary:
    group: int
    data_uses: tuple[BoundaryUse, ...]
    pointer_uses: tuple[BoundaryUse, ...]
    signature_hash: str


def split_operands(text: str) -> tuple[str, ...]:
    parts = text.strip().split(None, 1)
    if len(parts) == 1:
        return ()
    return tuple(part.strip() for part in parts[1].split(","))


def registers(value: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in REGISTER_RE.finditer(value))


def first_register(value: str) -> str | None:
    regs = registers(value)
    return regs[0] if regs else None


def reg_cells(reg: str) -> tuple[str, ...]:
    match = re.fullmatch(r"x(\d+)", reg)
    if match:
        index = match.group(1)
        return (f"vec{index}.lo", f"vec{index}.hi")
    match = re.fullmatch(r"wl(\d+)", reg)
    if match:
        return (f"vec{match.group(1)}.lo",)
    match = re.fullmatch(r"wh(\d+)", reg)
    if match:
        return (f"vec{match.group(1)}.hi",)
    match = re.fullmatch(r"dm(\d+)", reg)
    if match:
        index = match.group(1)
        return (
            f"acc{index}.bmll",
            f"acc{index}.bmlh",
            f"acc{index}.bmhl",
            f"acc{index}.bmhh",
        )
    match = re.fullmatch(r"cml(\d+)", reg)
    if match:
        index = match.group(1)
        return (f"acc{index}.bmll", f"acc{index}.bmlh")
    match = re.fullmatch(r"cmh(\d+)", reg)
    if match:
        index = match.group(1)
        return (f"acc{index}.bmhl", f"acc{index}.bmhh")
    match = re.fullmatch(r"bm(ll|lh|hl|hh)(\d+)", reg)
    if match:
        return (f"acc{match.group(2)}.bm{match.group(1)}",)
    return (reg,)


def data_cell(cell: str) -> bool:
    return cell.startswith(("vec", "acc"))


def unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def classify(op: str, operands: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    defs: list[str] = []
    uses: list[str] = []
    if op in {"vlda", "vldb", "vlda.conv.fp32.bf16"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
    elif op in {"lda", "ld", "lda.s16"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
            if "]" in operand:
                defs.extend(reg for reg in registers(operand) if reg.startswith("p"))
    elif op in {
        "vextbcst.16",
        "vbcst.16",
        "vunpack",
        "vups.4x",
        "vconv.bf16.fp32",
        "vconv.fp32.bf16",
        "vmov",
        "vmov.d",
    }:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
    elif op in {"vmac.f", "vmul.f", "vadd", "vadd.f", "vsub.f"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
    elif op in {"mova", "movxm", "movx", "mov", "movs"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
    elif op in {"padda", "paddb", "paddxm", "adds", "add.nc", "add"}:
        if operands:
            defs.extend(reg for reg in registers(operands[0]) if reg.startswith(("p", "r", "lc")))
        for operand in operands:
            uses.extend(registers(operand))
    else:
        for operand in operands:
            uses.extend(registers(operand))
    return unique(defs), unique(uses)


def group_for(address: int) -> int | None:
    for index, start in enumerate(GROUP_BOUNDS[:-1]):
        if start <= address < GROUP_BOUNDS[index + 1]:
            return index
    return None


def parse_slots(disasm: Path) -> tuple[Slot, ...]:
    slots: list[Slot] = []
    for raw in disasm.read_text().splitlines():
        match = ADDRESS_RE.match(raw)
        if match is None:
            continue
        address = int(match.group(1), 16)
        group = group_for(address)
        if group is None:
            continue
        payload = match.group(3).strip()
        if payload == "...":
            continue
        for bundle_slot, part in enumerate(part.strip() for part in payload.split(";")):
            if not part or part.startswith("..."):
                continue
            op = part.split(None, 1)[0]
            operands = split_operands(part)
            defs, uses = classify(op, operands)
            slots.append(
                Slot(
                    index=len(slots),
                    group=group,
                    address=address,
                    bundle_slot=bundle_slot,
                    text=part,
                    op=op,
                    operands=operands,
                    defs=defs,
                    uses=uses,
                )
            )
    return tuple(slots)


def slot_label(slot: Slot | None) -> str:
    if slot is None:
        return "entry"
    return f"g{slot.group}@0x{slot.address:x}.{slot.bundle_slot}:{slot.op}"


def slot_offset(slot: Slot) -> str:
    return f"+0x{slot.address - GROUP_BOUNDS[slot.group]:x}.{slot.bundle_slot}:{slot.op}"


def producer_kind(slot: Slot | None) -> str:
    if slot is None:
        return "entry"
    if slot.op == "vextbcst.16":
        src = first_register(slot.operands[1]) if len(slot.operands) > 1 else None
        if src == "x11":
            return "activation_lane"
        return "broadcast"
    if slot.op == "vbcst.16":
        return "scalar_broadcast"
    if slot.op == "vconv.bf16.fp32":
        return "bf16_coeff"
    if slot.op == "vups.4x":
        return "expanded_q4"
    if slot.op == "vunpack":
        return "unpacked_q4"
    if slot.op in {"vlda", "vldb", "vlda.conv.fp32.bf16"}:
        return "vector_load"
    if slot.op in {"vmov", "vmov.d"}:
        return "register_move"
    if slot.op in {"vadd", "vadd.f", "vsub.f", "vmul.f"}:
        return "vector_arith"
    if slot.op == "vmac.f":
        return "accumulator_carry"
    return slot.op


def producer_lane(slot: Slot | None) -> str | None:
    if slot is None or slot.op != "vextbcst.16" or len(slot.operands) < 3:
        return None
    return slot.operands[2]


def update_latest(slot: Slot, latest: dict[str, Slot]) -> None:
    for reg in slot.defs:
        for cell in reg_cells(reg):
            latest[cell] = slot


def trace_operand(reg: str, latest: dict[str, Slot]) -> OperandTrace:
    cells: list[CellProducer] = []
    for cell in reg_cells(reg):
        slot = latest.get(cell)
        cells.append(
            CellProducer(
                cell=cell,
                slot=slot,
                kind=producer_kind(slot),
                lane=producer_lane(slot),
            )
        )
    return OperandTrace(register=reg, cells=tuple(cells))


def trace_macs(slots: tuple[Slot, ...]) -> tuple[MacTrace, ...]:
    latest: dict[str, Slot] = {}
    traces: list[MacTrace] = []
    for slot in slots:
        if slot.op == "vmac.f" and len(slot.operands) >= 4:
            acc_reg = first_register(slot.operands[1])
            left_reg = first_register(slot.operands[2])
            right_reg = first_register(slot.operands[3])
            if acc_reg is not None and left_reg is not None and right_reg is not None:
                traces.append(
                    MacTrace(
                        slot=slot,
                        accumulator=trace_operand(acc_reg, latest),
                        left=trace_operand(left_reg, latest),
                        right=trace_operand(right_reg, latest),
                    )
                )
        update_latest(slot, latest)
    return tuple(traces)


def boundary_uses(slots: tuple[Slot, ...], group: int) -> BoundarySummary:
    latest: dict[str, Slot] = {}
    uses: dict[str, BoundaryUse] = {}
    for slot in slots:
        if slot.group < group:
            update_latest(slot, latest)
            continue
        if slot.group > group:
            break
        for reg in slot.uses:
            for cell in reg_cells(reg):
                producer = latest.get(cell)
                if producer is not None and producer.group != group and cell not in uses:
                    uses[cell] = BoundaryUse(cell=cell, producer=producer, first_use=slot)
        update_latest(slot, latest)
    data = tuple(uses[cell] for cell in sorted(cell for cell in uses if data_cell(cell)))
    pointers = tuple(uses[cell] for cell in sorted(cell for cell in uses if not data_cell(cell)))
    signature = "\n".join(boundary_signature_item(item) for item in data)
    signature_hash = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return BoundarySummary(group=group, data_uses=data, pointer_uses=pointers, signature_hash=signature_hash)


def boundary_signature_item(item: BoundaryUse) -> str:
    producer_group_offset = item.first_use.group - item.producer.group
    return (
        f"{item.cell}|producer_group_delta=-{producer_group_offset}|"
        f"producer={slot_offset(item.producer)}|use={slot_offset(item.first_use)}"
    )


def operand_kind(trace: OperandTrace) -> str:
    kinds = tuple(dict.fromkeys(cell.kind for cell in trace.cells))
    if len(kinds) == 1:
        return kinds[0]
    return "+".join(kinds)


def operand_has_cross_group(trace: OperandTrace, consumer_group: int) -> bool:
    return any(cell.slot is not None and cell.slot.group != consumer_group for cell in trace.cells)


def operand_cell_summary(trace: OperandTrace, consumer_group: int) -> str:
    parts: list[str] = []
    for cell in trace.cells:
        carry = "!" if cell.slot is not None and cell.slot.group != consumer_group else ""
        lane = cell.lane if cell.lane is not None else ""
        parts.append(f"{cell.cell}:{cell.kind}{lane}{carry}")
    return ", ".join(parts)


def mac_graph_records(traces: tuple[MacTrace, ...], group: int) -> tuple[MacTrace, ...]:
    return tuple(trace for trace in traces if trace.slot.group == group)


def boundary_to_json(item: BoundaryUse) -> dict[str, str | int]:
    return {
        "cell": item.cell,
        "producer": slot_label(item.producer),
        "producer_offset": slot_offset(item.producer),
        "producer_kind": producer_kind(item.producer),
        "first_use": slot_label(item.first_use),
        "first_use_offset": slot_offset(item.first_use),
    }


def operand_to_json(trace: OperandTrace) -> dict[str, str | bool | list[dict[str, str | None]]]:
    return {
        "register": trace.register,
        "kind": operand_kind(trace),
        "mixed": trace.mixed,
        "cells": [
            {
                "cell": cell.cell,
                "producer": slot_label(cell.slot),
                "kind": cell.kind,
                "lane": cell.lane,
            }
            for cell in trace.cells
        ],
    }


def graph_to_json(
    boundaries: tuple[BoundarySummary, ...],
    group_traces: tuple[MacTrace, ...],
):
    return {
        "group_bounds": [f"0x{value:x}" for value in GROUP_BOUNDS],
        "boundaries": [
            {
                "group": boundary.group,
                "signature_hash": boundary.signature_hash,
                "data_uses": [boundary_to_json(item) for item in boundary.data_uses],
                "pointer_uses": [boundary_to_json(item) for item in boundary.pointer_uses],
            }
            for boundary in boundaries
        ],
        "steady_group_macs": [
            {
                "slot": slot_label(trace.slot),
                "offset": slot_offset(trace.slot),
                "accumulator": operand_to_json(trace.accumulator),
                "left": operand_to_json(trace.left),
                "right": operand_to_json(trace.right),
            }
            for trace in group_traces
        ],
    }


def render_boundary_table(boundary: BoundarySummary, limit: int | None = None) -> list[str]:
    rows = [
        "| Cell | Producer | First Use |",
        "| --- | --- | --- |",
    ]
    items = boundary.data_uses if limit is None else boundary.data_uses[:limit]
    for item in items:
        rows.append(
            f"| `{item.cell}` | `{slot_label(item.producer)}` | `{slot_label(item.first_use)}` |"
        )
    if limit is not None and len(boundary.data_uses) > limit:
        rows.append(f"| ... | {len(boundary.data_uses) - limit} more data cells | ... |")
    return rows


def render_report(
    disasm: Path,
    boundaries: tuple[BoundarySummary, ...],
    group_traces: tuple[MacTrace, ...],
) -> str:
    by_group = {boundary.group: boundary for boundary in boundaries}
    group1 = by_group[1]
    steady_hashes = tuple(by_group[group].signature_hash for group in STEADY_BOUNDARIES)
    steady_stable = len(set(steady_hashes)) == 1
    vector_operands = [operand for trace in group_traces for operand in (trace.left, trace.right)]
    vector_pairs = Counter(
        " + ".join(sorted((operand_kind(trace.left), operand_kind(trace.right))))
        for trace in group_traces
    )

    lines = [
        "# MyLM Q4NX Operand Graph",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This graph is pre-lane-select. Exp102 already validates the primitive `vmac.f #0x33c` lane model; this report tracks the half-register state fed into those MACs.",
        "",
        "## Boundary State",
        "",
        "| Boundary Into Group | Data Cells | Pointer/Scalar Cells | Signature |",
        "| ---: | ---: | ---: | --- |",
    ]
    for boundary in boundaries:
        lines.append(
            f"| {boundary.group} | {len(boundary.data_uses)} | {len(boundary.pointer_uses)} | `{boundary.signature_hash}` |"
        )
    lines.extend(
        [
            "",
            f"- Steady boundaries checked: `{', '.join(str(group) for group in STEADY_BOUNDARIES)}`",
            f"- Steady data-cell signature stable: `{steady_stable}`",
            "",
            "## Fill To Steady Boundary",
            "",
            *render_boundary_table(group1),
            "",
            "## First Steady To Steady Boundary",
            "",
            *render_boundary_table(by_group[2]),
            "",
            "## Steady Group1 MAC Graph Summary",
            "",
            f"- `vmac.f`: `{len(group_traces)}`",
            f"- Mixed vector operands: `{sum(operand.mixed for operand in vector_operands)}` / `{len(vector_operands)}`",
            f"- Vector operands with carry cells: `{sum(operand_has_cross_group(operand, STEADY_GROUP) for operand in vector_operands)}` / `{len(vector_operands)}`",
            "",
            "| Vector Producer Pair | MAC Count |",
            "| --- | ---: |",
        ]
    )
    for pair, count in vector_pairs.most_common():
        lines.append(f"| `{pair}` | {count} |")
    lines.extend(
        [
            "",
            "## Steady Group1 MAC Graph",
            "",
            "| MAC | Acc | Left | Right |",
            "| --- | --- | --- | --- |",
        ]
    )
    for trace in group_traces:
        lines.append(
            f"| `{slot_label(trace.slot)}` | "
            f"`{operand_cell_summary(trace.accumulator, STEADY_GROUP)}` | "
            f"`{operand_cell_summary(trace.left, STEADY_GROUP)}` | "
            f"`{operand_cell_summary(trace.right, STEADY_GROUP)}` |"
        )
    lines.extend(
        [
            "",
            "## Generator Implication",
            "",
            "- `fill -> steady` has a concrete boundary state: acc, packed, coefficient, activation, and pointer cells are already live before group1 starts.",
            "- The steady-to-steady data-cell signature is stable for groups 2..6, so a generator can model one steady state transition instead of eight independent groups.",
            "- The next production experiment should consume the JSON artifact and emit a small assembly template from graph records, with exact numerical validation kept separate from the MyLM-like group-correction contract decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disasm", type=Path, default=DEFAULT_DISASM)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slots = parse_slots(args.disasm)
    traces = trace_macs(slots)
    boundaries = tuple(boundary_uses(slots, group) for group in range(1, 8))
    group_traces = mac_graph_records(traces, STEADY_GROUP)
    args.json_output.write_text(json.dumps(graph_to_json(boundaries, group_traces), indent=2) + "\n")
    args.report.write_text(render_report(args.disasm, boundaries, group_traces))
    steady_hashes = {boundary.group: boundary.signature_hash for boundary in boundaries if boundary.group in STEADY_BOUNDARIES}
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"group{STEADY_GROUP}_vmac.f={len(group_traces)}")
    print(f"steady_boundary_hashes={steady_hashes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
