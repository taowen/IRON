#!/usr/bin/env python3
"""Trace MyLM Q4NX vmac operand producers across the hot loop."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mylm_q4nx_mac_operand_trace.md"
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):((?:\s+[0-9a-fA-F]{2})+)\s+(.*)$")
REGISTER_RE = re.compile(
    r"\b(?:"
    r"bm(?:ll|lh|hl|hh)\d+|cm[lh]\d+|dm\d+|x\d+|w[lh]\d+|"
    r"r\d+|p\d+|m\d+|s\d+|lc|ls|le|cr\w+|upssign0|unpacksign0"
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
class Producer:
    family: str
    register: str
    slot: Slot | None
    kind: str
    distance: int | None
    crosses_group: bool
    lane: str | None


@dataclass(frozen=True)
class MacTrace:
    slot: Slot
    accumulator: Producer
    left: Producer
    right: Producer


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


def alias_family(reg: str) -> str:
    match = re.fullmatch(r"dm(\d+)", reg)
    if match:
        return f"acc{match.group(1)}"
    match = re.fullmatch(r"bm(?:ll|lh|hl|hh)(\d+)", reg)
    if match:
        return f"acc{match.group(1)}"
    match = re.fullmatch(r"cm[lh](\d+)", reg)
    if match:
        return f"acc{match.group(1)}"
    match = re.fullmatch(r"x(\d+)", reg)
    if match:
        return f"vec{match.group(1)}"
    match = re.fullmatch(r"w[lh](\d+)", reg)
    if match:
        return f"vec{match.group(1)}"
    return reg


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


def make_producer(reg: str, latest: dict[str, Slot], consumer: Slot) -> Producer:
    family = alias_family(reg)
    slot = latest.get(family)
    return Producer(
        family=family,
        register=reg,
        slot=slot,
        kind=producer_kind(slot),
        distance=consumer.index - slot.index if slot is not None else None,
        crosses_group=slot is not None and slot.group != consumer.group,
        lane=producer_lane(slot),
    )


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
                        accumulator=make_producer(acc_reg, latest, slot),
                        left=make_producer(left_reg, latest, slot),
                        right=make_producer(right_reg, latest, slot),
                    )
                )
        for reg in slot.defs:
            latest[alias_family(reg)] = slot
    return tuple(traces)


def format_producer(producer: Producer) -> str:
    detail = producer.kind
    if producer.lane is not None:
        detail += f" lane {producer.lane}"
    distance = "entry" if producer.distance is None else f"-{producer.distance}"
    carry = " carry" if producer.crosses_group else ""
    return f"`{producer.register}` <- {detail}{carry} `{slot_label(producer.slot)}` ({distance})"


def count_vector_pair_categories(traces: tuple[MacTrace, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in traces:
        pair = " + ".join(sorted((trace.left.kind, trace.right.kind)))
        counts[pair] += 1
    return counts


def render(traces: tuple[MacTrace, ...], disasm: Path) -> str:
    group_traces = tuple(trace for trace in traces if trace.slot.group == STEADY_GROUP)
    all_vector_kinds = Counter()
    all_crossing_vectors = 0
    for trace in traces:
        all_vector_kinds.update((trace.left.kind, trace.right.kind))
        all_crossing_vectors += int(trace.left.crosses_group) + int(trace.right.crosses_group)

    group_vector_kinds = Counter()
    group_crossing_vectors = 0
    group_accumulator_carries = 0
    for trace in group_traces:
        group_vector_kinds.update((trace.left.kind, trace.right.kind))
        group_crossing_vectors += int(trace.left.crosses_group) + int(trace.right.crosses_group)
        group_accumulator_carries += int(trace.accumulator.crosses_group)

    lines = [
        "# MyLM Q4NX MAC Operand Trace",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This report traces the latest visible producer for each `vmac.f` vector operand.",
        "It is a conservative register-family analysis, not a full half-register alias decompiler.",
        "",
        "## Whole Hot Loop Summary",
        "",
        f"- Total `vmac.f`: `{len(traces)}`",
        f"- Vector operands crossing a group boundary: `{all_crossing_vectors}` / `{len(traces) * 2}`",
        "",
        "| Producer Kind | Vector Operands |",
        "| --- | ---: |",
    ]
    for kind, count in all_vector_kinds.most_common():
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(
        [
            "",
            "## Steady Group1 Summary",
            "",
            f"- Group1 `vmac.f`: `{len(group_traces)}`",
            f"- Vector operands crossing into group1: `{group_crossing_vectors}` / `{len(group_traces) * 2}`",
            f"- Accumulator source carries crossing into group1: `{group_accumulator_carries}` / `{len(group_traces)}`",
            "",
            "| Vector Producer Pair | MAC Count |",
            "| --- | ---: |",
        ]
    )
    for pair, count in count_vector_pair_categories(group_traces).most_common():
        lines.append(f"| `{pair}` | {count} |")
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
            f"{format_producer(trace.accumulator)} | "
            f"{format_producer(trace.left)} | "
            f"{format_producer(trace.right)} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The steady template is not self-contained: several vector operands and accumulator sources are live before group1 starts.",
            "- A production generator must model the boundary carry state before it emits even the first steady group.",
            "- The useful abstraction is a scheduled operand graph, not a macro that repeats 32 lane broadcasts and MACs in source order.",
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
    args.output.write_text(render(traces, args.disasm))
    print(f"wrote {args.output}")
    print(f"vmac.f={len(traces)}")
    steady = [trace for trace in traces if trace.slot.group == STEADY_GROUP]
    steady_crossing = sum(
        int(trace.left.crosses_group) + int(trace.right.crosses_group) + int(trace.accumulator.crosses_group)
        for trace in steady
    )
    print(f"group{STEADY_GROUP}_vmac.f={len(steady)} crossing_sources={steady_crossing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
