#!/usr/bin/env python3
"""Summarize MyLM Q4NX hot-loop software-pipeline dependencies."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mylm_q4nx_pipeline_schedule.md"
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):((?:\s+[0-9a-fA-F]{2})+)\s+(.*)$")
REGISTER_RE = re.compile(
    r"\b(?:"
    r"bm(?:ll|lh|hl|hh)\d+|cm[lh]\d+|dm\d+|x\d+|w[lh]\d+|"
    r"r\d+|p\d+|m\d+|s\d+|lc|ls|le|cr\w+|upssign0|unpacksign0"
    r")\b"
)
GROUP_BOUNDS = (0x260, 0x52A, 0x7DE, 0xA92, 0xD46, 0xFFA, 0x12AE, 0x1566, 0x1850)
KEY_OPS = (
    "vmac.f",
    "vextbcst.16",
    "vconv.bf16.fp32",
    "vconv.fp32.bf16",
    "vups.4x",
    "vunpack",
    "vldb",
    "vlda",
    "lda.s16",
    "vbcst.16",
    "vst",
)


@dataclass(frozen=True)
class Slot:
    index: int
    group: int
    address: int
    bundle_slot: int
    text: str
    op: str
    defs: tuple[str, ...]
    uses: tuple[str, ...]
    def_families: tuple[str, ...]
    use_families: tuple[str, ...]


@dataclass(frozen=True)
class GroupSummary:
    group: int
    start: int
    end: int
    slots: int
    op_counts: Counter[str]
    incoming_families: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryCarry:
    boundary: int
    family: str
    previous_def: str
    next_use: str


@dataclass(frozen=True)
class VextUse:
    group: int
    lane: str
    vector: str
    def_slot: str
    use_slot: str
    distance_slots: int


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
    elif op in {"vextbcst.16", "vbcst.16", "vunpack", "vups.4x", "vconv.bf16.fp32", "vconv.fp32.bf16", "vmov", "vmov.d"}:
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
            defs, uses = classify(op, split_operands(part))
            slots.append(
                Slot(
                    index=len(slots),
                    group=group,
                    address=address,
                    bundle_slot=bundle_slot,
                    text=part,
                    op=op,
                    defs=defs,
                    uses=uses,
                    def_families=unique([alias_family(reg) for reg in defs]),
                    use_families=unique([alias_family(reg) for reg in uses]),
                )
            )
    return tuple(slots)


def slot_label(slot: Slot) -> str:
    return f"g{slot.group}@0x{slot.address:x}.{slot.bundle_slot}:{slot.op}"


def group_summaries(slots: tuple[Slot, ...]) -> tuple[GroupSummary, ...]:
    summaries: list[GroupSummary] = []
    for group, start in enumerate(GROUP_BOUNDS[:-1]):
        group_slots = [slot for slot in slots if slot.group == group]
        defined: set[str] = set()
        incoming: list[str] = []
        for slot in group_slots:
            for family in slot.use_families:
                if family not in defined:
                    incoming.append(family)
            defined.update(slot.def_families)
        summaries.append(
            GroupSummary(
                group=group,
                start=start,
                end=GROUP_BOUNDS[group + 1],
                slots=len(group_slots),
                op_counts=Counter(slot.op for slot in group_slots),
                incoming_families=unique(incoming),
            )
        )
    return tuple(summaries)


def boundary_carries(slots: tuple[Slot, ...]) -> tuple[BoundaryCarry, ...]:
    incoming_uses: dict[tuple[int, str], Slot] = {}
    defined_in_group: dict[int, set[str]] = defaultdict(set)
    for slot in slots:
        group_defined = defined_in_group[slot.group]
        for family in slot.use_families:
            if family not in group_defined:
                incoming_uses.setdefault((slot.group, family), slot)
        group_defined.update(slot.def_families)
    carries: list[BoundaryCarry] = []
    for group in range(1, len(GROUP_BOUNDS) - 1):
        before = [slot for slot in slots if slot.group < group]
        previous_by_family: dict[str, Slot] = {}
        for slot in before:
            for family in slot.def_families:
                previous_by_family[family] = slot
        for family in sorted(family for (incoming_group, family) in incoming_uses if incoming_group == group):
            previous = previous_by_family.get(family)
            next_use = incoming_uses[(group, family)]
            if previous is not None:
                carries.append(
                    BoundaryCarry(
                        boundary=group,
                        family=family,
                        previous_def=slot_label(previous),
                        next_use=slot_label(next_use),
                    )
                )
    return tuple(carries)


def vext_consumers(slots: tuple[Slot, ...]) -> tuple[VextUse, ...]:
    pending: dict[str, tuple[Slot, str, str]] = {}
    uses: list[VextUse] = []
    lane_re = re.compile(r"#(0x[0-9a-fA-F]+|\d+)\b")
    for slot in slots:
        if slot.op == "vextbcst.16" and slot.defs:
            lane_match = lane_re.search(slot.text)
            lane = lane_match.group(1) if lane_match else "?"
            pending[alias_family(slot.defs[0])] = (slot, lane, slot.defs[0])
            continue
        if slot.op == "vmac.f":
            for family in slot.use_families:
                if family in pending:
                    producer, lane, vector = pending.pop(family)
                    uses.append(
                        VextUse(
                            group=producer.group,
                            lane=lane,
                            vector=vector,
                            def_slot=slot_label(producer),
                            use_slot=slot_label(slot),
                            distance_slots=slot.index - producer.index,
                        )
                    )
    return tuple(uses)


def render_counts(counts: Counter[str]) -> str:
    return ", ".join(f"`{op}={counts[op]}`" for op in KEY_OPS if counts[op])


def render(slots: tuple[Slot, ...], disasm: Path) -> str:
    summaries = group_summaries(slots)
    carries = boundary_carries(slots)
    vext_uses = vext_consumers(slots)
    distance_counts = Counter(use.distance_slots for use in vext_uses)
    lines = [
        "# MyLM Q4NX Pipeline Schedule",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This is a compact schedule artifact for writing the next Q4NX assembly",
        "generator. It intentionally avoids a full instruction listing and keeps",
        "only the facts that affect register lifetime and software pipelining.",
        "",
        "## Group Shape",
        "",
        "| Group | Range | Slots | Key Ops | Incoming Families |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for summary in summaries:
        incoming = ", ".join(f"`{family}`" for family in summary.incoming_families[:20])
        if len(summary.incoming_families) > 20:
            incoming += ", ..."
        lines.append(
            f"| {summary.group} | `0x{summary.start:x}..0x{summary.end:x}` | "
            f"{summary.slots} | {render_counts(summary.op_counts)} | {incoming} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Group Carry",
            "",
            "| Boundary Into Group | Family | Previous Def | First Use Before Local Def |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for carry in carries:
        lines.append(
            f"| {carry.boundary} | `{carry.family}` | `{carry.previous_def}` | `{carry.next_use}` |"
        )
    lines.extend(
        [
            "",
            "## Vext Consumer Distance",
            "",
            f"- Matched `vextbcst.16` consumers: `{len(vext_uses)}`",
            "- This is a conservative first-consumer heuristic. Unmatched",
            "  broadcasts are expected until the full alias/lane model is",
            "  decoded.",
            "- Distance is counted in parsed instruction slots between the broadcast",
            "  slot and the first later `vmac.f` slot that consumes the same vector",
            "  family.",
            "",
            "| Distance Slots | Count |",
            "| ---: | ---: |",
        ]
    )
    for distance, count in sorted(distance_counts.items()):
        lines.append(f"| {distance} | {count} |")
    lines.extend(
        [
            "",
            "## First 48 Vext Consumers",
            "",
            "| Group | Lane | Vector | Def | Use | Distance |",
            "| ---: | ---: | --- | --- | --- | ---: |",
        ]
    )
    for use in vext_uses[:48]:
        lines.append(
            f"| {use.group} | `{use.lane}` | `{use.vector}` | "
            f"`{use.def_slot}` | `{use.use_slot}` | {use.distance_slots} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Group0 and group7 have different counts from the steady-state middle",
            "  groups. Treating the loop as eight independent 33-MAC groups is wrong.",
            "- Several vector/accumulator families enter a group already live. A",
            "  production generator needs an explicit register-lifetime table, not",
            "  only an opcode template.",
            "- The `vextbcst.16` consumer distances are filled by real dequant and",
            "  accumulator movement work; replacing them with nop padding is only a",
            "  diagnostic crutch.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disasm", type=Path, default=DEFAULT_DISASM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slots = parse_slots(args.disasm)
    if not slots:
        raise ValueError(f"no slots parsed from {args.disasm}")
    args.output.write_text(render(slots, args.disasm))
    print(f"wrote {args.output}")
    print(f"slots={len(slots)}")
    for summary in group_summaries(slots):
        print(
            f"group{summary.group}: "
            f"vmac.f={summary.op_counts['vmac.f']} "
            f"vextbcst.16={summary.op_counts['vextbcst.16']} "
            f"vconv.bf16.fp32={summary.op_counts['vconv.bf16.fp32']} "
            f"vups.4x={summary.op_counts['vups.4x']} "
            f"vunpack={summary.op_counts['vunpack']} "
            f"incoming={len(summary.incoming_families)}"
        )
    print(f"vext_consumers={len(vext_consumers(slots))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
