#!/usr/bin/env python3
"""Trace MyLM Q4NX producers at half-register cell granularity."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mylm_q4nx_half_register_alias.md"
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):((?:\s+[0-9a-fA-F]{2})+)\s+(.*)$")
REGISTER_RE = re.compile(
    r"\b(?:"
    r"bm(?:ll|lh|hl|hh)\d+|cm[lh]\d+|dm\d+|x\d+|w[lh]\d+|lfh\d+|"
    r"r\d+|p\d+|m\d+|s\d+|dj\d+|lc|ls|le|cr\w+|upssign0|unpacksign0"
    r")\b"
)
GROUP_BOUNDS = (0x260, 0x52A, 0x7DE, 0xA92, 0xD46, 0xFFA, 0x12AE, 0x1566, 0x1850)
STEADY_GROUP = 1


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
    mixed: bool
    crosses_group: bool


@dataclass(frozen=True)
class MacTrace:
    slot: Slot
    accumulator: OperandTrace
    left: OperandTrace
    right: OperandTrace


@dataclass(frozen=True)
class BoundaryCell:
    cell: str
    producer: Slot
    first_use: Slot


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


def trace_operand(reg: str, latest: dict[str, Slot], consumer: Slot) -> OperandTrace:
    producers: list[CellProducer] = []
    for cell in reg_cells(reg):
        slot = latest.get(cell)
        producers.append(
            CellProducer(
                cell=cell,
                slot=slot,
                kind=producer_kind(slot),
                lane=producer_lane(slot),
            )
        )
    producer_slots = {producer.slot for producer in producers}
    return OperandTrace(
        register=reg,
        cells=tuple(producers),
        mixed=len(producer_slots) > 1,
        crosses_group=any(producer.slot is not None and producer.slot.group != consumer.group for producer in producers),
    )


def update_latest(slot: Slot, latest: dict[str, Slot]) -> None:
    for reg in slot.defs:
        for cell in reg_cells(reg):
            latest[cell] = slot


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
                        accumulator=trace_operand(acc_reg, latest, slot),
                        left=trace_operand(left_reg, latest, slot),
                        right=trace_operand(right_reg, latest, slot),
                    )
                )
        update_latest(slot, latest)
    return tuple(traces)


def boundary_cells(slots: tuple[Slot, ...], group: int) -> tuple[BoundaryCell, ...]:
    latest: dict[str, Slot] = {}
    first_boundary_uses: dict[str, BoundaryCell] = {}
    for slot in slots:
        if slot.group < group:
            update_latest(slot, latest)
            continue
        if slot.group > group:
            break
        for reg in slot.uses:
            for cell in reg_cells(reg):
                producer = latest.get(cell)
                if producer is not None and producer.group != group and cell not in first_boundary_uses:
                    first_boundary_uses[cell] = BoundaryCell(
                        cell=cell,
                        producer=producer,
                        first_use=slot,
                    )
        update_latest(slot, latest)
    return tuple(first_boundary_uses[cell] for cell in sorted(first_boundary_uses))


def operand_kind(trace: OperandTrace) -> str:
    kinds = tuple(dict.fromkeys(cell.kind for cell in trace.cells))
    if len(kinds) == 1:
        return kinds[0]
    return "+".join(kinds)


def format_cell(cell: CellProducer, consumer: Slot) -> str:
    lane = f" {cell.lane}" if cell.lane is not None else ""
    distance = "entry" if cell.slot is None else f"-{consumer.index - cell.slot.index}"
    carry = " carry" if cell.slot is not None and cell.slot.group != consumer.group else ""
    return f"`{cell.cell}` <- {cell.kind}{lane}{carry} `{slot_label(cell.slot)}` ({distance})"


def format_operand(trace: OperandTrace, consumer: Slot) -> str:
    if trace.mixed:
        joined = "<br>".join(format_cell(cell, consumer) for cell in trace.cells)
        return f"`{trace.register}` mixed:<br>{joined}"
    cell_names = "+".join(cell.cell for cell in trace.cells)
    base = trace.cells[0]
    lane = f" {base.lane}" if base.lane is not None else ""
    distance = "entry" if base.slot is None else f"-{consumer.index - base.slot.index}"
    carry = " carry" if base.slot is not None and base.slot.group != consumer.group else ""
    return f"`{trace.register}` `{cell_names}` <- {base.kind}{lane}{carry} `{slot_label(base.slot)}` ({distance})"


def render(traces: tuple[MacTrace, ...], slots: tuple[Slot, ...], disasm: Path) -> str:
    group_traces = tuple(trace for trace in traces if trace.slot.group == STEADY_GROUP)
    vector_operands = [operand for trace in traces for operand in (trace.left, trace.right)]
    group_vectors = [operand for trace in group_traces for operand in (trace.left, trace.right)]
    group_accumulators = [trace.accumulator for trace in group_traces]
    vector_kinds = Counter(operand_kind(operand) for operand in vector_operands)
    group_kind_pairs = Counter(
        " + ".join(sorted((operand_kind(trace.left), operand_kind(trace.right))))
        for trace in group_traces
    )
    boundary = boundary_cells(slots, STEADY_GROUP)

    lines = [
        "# MyLM Q4NX Half-Register Alias Trace",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This report tracks producer state at vector-half and accumulator-quadrant granularity.",
        "",
        "## Whole Hot Loop",
        "",
        f"- `vmac.f`: `{len(traces)}`",
        f"- Vector operands: `{len(vector_operands)}`",
        f"- Mixed-half vector operands: `{sum(operand.mixed for operand in vector_operands)}`",
        f"- Vector operands with cross-group cells: `{sum(operand.crosses_group for operand in vector_operands)}`",
        "",
        "| Vector Operand Producer Shape | Count |",
        "| --- | ---: |",
    ]
    for kind, count in vector_kinds.most_common():
        lines.append(f"| `{kind}` | {count} |")

    lines.extend(
        [
            "",
            "## Relation To Exp102",
            "",
            "Exp102 already proved the primitive `vmac.f #0x33c` operand model on real NPU:",
            "",
            "- q operand registers `x2/x3/x5/x7/x9` all produced first lane `1` when multiplied by a broadcast one vector;",
            "- `vextbcst.16` lanes `0..31` produced first lanes `1..32`;",
            "- the group-sum correction produced `64`, and summing all 32 lanes produced `528`.",
            "",
            "So the remaining problem is not whether `#0x33c` is a valid BF16 MAC configuration. The remaining problem is preserving the scheduled half-register state that MyLM feeds into those MACs.",
            "",
            "## Steady Group1",
            "",
            f"- `vmac.f`: `{len(group_traces)}`",
            f"- Mixed-half vector operands: `{sum(operand.mixed for operand in group_vectors)}` / `{len(group_vectors)}`",
            f"- Vector operands with cross-group cells: `{sum(operand.crosses_group for operand in group_vectors)}` / `{len(group_vectors)}`",
            f"- Accumulator operands with cross-group cells: `{sum(operand.crosses_group for operand in group_accumulators)}` / `{len(group_accumulators)}`",
            "",
            "| Vector Producer Pair | MAC Count |",
            "| --- | ---: |",
        ]
    )
    for pair, count in group_kind_pairs.most_common():
        lines.append(f"| `{pair}` | {count} |")

    lines.extend(
        [
            "",
            "## Boundary Cells Into Group1",
            "",
            "| Cell | Producer | First Group1 Use |",
            "| --- | --- | --- |",
        ]
    )
    for item in boundary:
        lines.append(f"| `{item.cell}` | `{slot_label(item.producer)}` | `{slot_label(item.first_use)}` |")

    lines.extend(
        [
            "",
            "## Group1 MAC Operand Table",
            "",
            "| MAC | Accumulator Source | Left Vector | Right Vector |",
            "| --- | --- | --- | --- |",
        ]
    )
    for trace in group_traces:
        lines.append(
            f"| `{slot_label(trace.slot)}` | "
            f"{format_operand(trace.accumulator, trace.slot)} | "
            f"{format_operand(trace.left, trace.slot)} | "
            f"{format_operand(trace.right, trace.slot)} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Whole-vector producer tracking hides real composition: many `xN` operands are assembled from separately scheduled halves.",
            "- Group1 starts with live half-register cells from group0, including packed Q4 halves, coefficient halves, loaded vectors, and accumulator quadrants.",
            "- The next Q4NX generator should carry explicit cell state across `fill -> steady` and `steady -> steady` boundaries before trying to emit a new exact or MyLM-like body.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disasm", type=Path, default=DEFAULT_DISASM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slots = parse_slots(args.disasm)
    traces = trace_macs(slots)
    args.output.write_text(render(traces, slots, args.disasm))
    group_traces = [trace for trace in traces if trace.slot.group == STEADY_GROUP]
    group_vectors = [operand for trace in group_traces for operand in (trace.left, trace.right)]
    print(f"wrote {args.output}")
    print(f"vmac.f={len(traces)}")
    print(
        f"group{STEADY_GROUP}_mixed_vectors="
        f"{sum(operand.mixed for operand in group_vectors)}/{len(group_vectors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
