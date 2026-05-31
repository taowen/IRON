#!/usr/bin/env python3
"""Annotate the first MyLM Q4NX hot-loop software-pipeline window."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mylm_q4nx_0x260_0x52a_register_flow.md"
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):((?:\s+[0-9a-fA-F]{2})+)\s+(.*)$")
REGISTER_RE = re.compile(
    r"\b(?:"
    r"bm(?:ll|lh|hl|hh)\d+|cm[lh]\d+|dm\d+|x\d+|w[lh]\d+|"
    r"r\d+|p\d+|m\d+|s\d+|lc|ls|le|cr\w+|upssign0|unpacksign0"
    r")\b"
)
START = 0x260
END = 0x52A


@dataclass(frozen=True)
class Slot:
    address: int
    slot: int
    text: str
    op: str
    defs: tuple[str, ...]
    uses: tuple[str, ...]
    aliases: tuple[str, ...]
    semantic: str


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


def alias_notes(defs: tuple[str, ...], uses: tuple[str, ...]) -> tuple[str, ...]:
    notes = []
    for reg in defs + uses:
        family = alias_family(reg)
        if family != reg:
            notes.append(f"{reg}->{family}")
    return tuple(dict.fromkeys(notes))


def unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def classify(op: str, operands: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    defs: list[str] = []
    uses: list[str] = []
    semantic = ""

    if op in {"vlda", "vldb", "vlda.conv.fp32.bf16"}:
        dest = first_register(operands[0]) if operands else None
        if dest:
            defs.append(dest)
        if len(operands) > 1:
            uses.extend(registers(operands[1]))
            if "]" not in operands[1]:
                defs.extend(reg for reg in registers(operands[1]) if reg.startswith("p"))
        if op == "vldb" and dest == "x11":
            semantic = "load 32-lane activation vector into x11"
        elif op == "vldb":
            semantic = "load packed/Q4 or scale vector"
        else:
            semantic = "load vector/accumulator from local memory"
    elif op == "lda.s16":
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        if len(operands) > 1:
            ptr_regs = registers(operands[1])
            uses.extend(ptr_regs)
            defs.extend(reg for reg in ptr_regs if reg.startswith("p"))
        semantic = "load one activation group-sum scratch scalar"
    elif op in {"lda", "ld"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        if len(operands) > 1:
            uses.extend(registers(operands[1]))
        semantic = "scalar/pointer load"
    elif op == "vextbcst.16":
        dest = first_register(operands[0])
        src = first_register(operands[1]) if len(operands) > 1 else None
        if dest:
            defs.append(dest)
        if src:
            uses.append(src)
        lane = operands[2] if len(operands) > 2 else "?"
        semantic = f"extract+broadcast 16-bit lane {lane}; consumer must respect latency"
    elif op == "vbcst.16":
        dest = first_register(operands[0])
        src = first_register(operands[1]) if len(operands) > 1 else None
        if dest:
            defs.append(dest)
        if src:
            uses.append(src)
        semantic = "scalar broadcast to BF16 vector"
    elif op == "vunpack":
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        if len(operands) > 1:
            uses.extend(registers(operands[1]))
        if len(operands) > 2:
            uses.extend(registers(operands[2]))
        semantic = "unpack packed integer lanes into vector register"
    elif op == "vups.4x":
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
        semantic = "upshift/expand unpacked Q4 lanes into FP32 accumulator family"
    elif op in {"vconv.bf16.fp32", "vconv.fp32.bf16"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        if len(operands) > 1:
            uses.extend(registers(operands[1]))
        semantic = "convert between accumulator segment and BF16 vector segment"
    elif op in {"vmac.f", "vmul.f", "vadd", "vadd.f", "vsub.f"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
        if op == "vmac.f":
            semantic = "FP MAC; first operand is written accumulator, second is source accumulator"
        else:
            semantic = "vector arithmetic"
    elif op in {"vmov", "vmov.d"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        if len(operands) > 1:
            uses.extend(registers(operands[1]))
        semantic = "move between aliasing vector/accumulator segments"
    elif op in {"mova", "movxm", "movx", "mov", "movs"}:
        if operands:
            dest = first_register(operands[0])
            if dest:
                defs.append(dest)
        for operand in operands[1:]:
            uses.extend(registers(operand))
        semantic = "scalar/control register setup"
    elif op in {"padda", "paddb", "paddxm", "adds", "add.nc", "add"}:
        if operands:
            defs.extend(reg for reg in registers(operands[0]) if reg.startswith(("p", "r", "lc")))
        for operand in operands:
            uses.extend(registers(operand))
        semantic = "pointer/scalar update"
    elif op.startswith("nop"):
        semantic = "bundle padding / hazard spacing"
    else:
        for operand in operands:
            uses.extend(registers(operand))
        semantic = "unclassified"
    return unique(defs), unique(uses), semantic


def parse_slots(disasm: Path, start: int, end: int) -> tuple[Slot, ...]:
    slots: list[Slot] = []
    for raw in disasm.read_text().splitlines():
        match = ADDRESS_RE.match(raw)
        if match is None:
            continue
        address = int(match.group(1), 16)
        if address < start or address >= end:
            continue
        payload = match.group(3).strip()
        if payload == "...":
            continue
        for slot_index, part in enumerate(part.strip() for part in payload.split(";")):
            if not part or part.startswith("..."):
                continue
            op = part.split(None, 1)[0]
            defs, uses, semantic = classify(op, split_operands(part))
            slots.append(
                Slot(
                    address=address,
                    slot=slot_index,
                    text=part,
                    op=op,
                    defs=defs,
                    uses=uses,
                    aliases=alias_notes(defs, uses),
                    semantic=semantic,
                )
            )
    return tuple(slots)


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def render(slots: tuple[Slot, ...], disasm: Path, start: int, end: int) -> str:
    op_counts = Counter(slot.op for slot in slots)
    lines = [
        "# MyLM Q4NX Register Flow 0x260..0x52a",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This table is intentionally register-level. It is the input for a real",
        "MyLM-style generator: every production instruction must be scheduled",
        "with its defs, uses, alias family, and latency in mind.",
        "",
        "## Summary",
        "",
        f"- Range: `0x{start:x}..0x{end:x}`",
        f"- Instruction slots: `{len(slots)}`",
        f"- `vmac.f`: `{op_counts['vmac.f']}`",
        f"- `vextbcst.16`: `{op_counts['vextbcst.16']}`",
        f"- `vconv.bf16.fp32`: `{op_counts['vconv.bf16.fp32']}`",
        f"- `vups.4x`: `{op_counts['vups.4x']}`",
        f"- `vunpack`: `{op_counts['vunpack']}`",
        f"- `lda.s16`: `{op_counts['lda.s16']}`",
        "",
        "## Key Rules Learned",
        "",
        "- `vextbcst.16` is not consume-immediate; exp102 proves a following",
        "  `vmac.f` reads the previous broadcast value unless latency is filled.",
        "- `dmN`, `bmllN/bmlhN/bmhlN/bmhhN`, and `cmlN/cmhN` are one accumulator",
        "  alias family. A `vups.4x` or `vconv` into that family can overwrite",
        "  values later consumed by `vmac.f`.",
        "- MyLM uses the software pipeline to fill those latency gaps with real",
        "  `vldb/vunpack/vups/vconv/vmov/vmac` work across group boundaries.",
        "",
        "## Table",
        "",
        "| Addr | Slot | Instruction | Defs | Uses | Alias Families | Semantics |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for slot in slots:
        lines.append(
            "| "
            f"`0x{slot.address:x}` | "
            f"{slot.slot} | "
            f"`{markdown_escape(slot.text)}` | "
            f"`{', '.join(slot.defs)}` | "
            f"`{', '.join(slot.uses)}` | "
            f"`{', '.join(slot.aliases)}` | "
            f"{markdown_escape(slot.semantic)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disasm", type=Path, default=DEFAULT_DISASM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=lambda value: int(value, 0), default=START)
    parser.add_argument("--end", type=lambda value: int(value, 0), default=END)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slots = parse_slots(args.disasm, args.start, args.end)
    if not slots:
        raise ValueError(f"no disasm slots found in 0x{args.start:x}..0x{args.end:x}")
    args.output.write_text(render(slots, args.disasm, args.start, args.end))
    print(f"wrote {args.output}")
    print(f"slots={len(slots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
