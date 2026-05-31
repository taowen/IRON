#!/usr/bin/env python3
"""Build a cell-liveness table for the full MyLM Q4NX hot loop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP115_RUN = REPO_ROOT / "experiments/115_mylm_q4nx_operand_graph/run.py"
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_q4nx_cell_liveness.md"
DEFAULT_JSON = EXPERIMENT_DIR / "mylm_q4nx_cell_liveness.json"


@dataclass
class LiveRange:
    cell: str
    producer: object | None
    producer_kind: str
    uses: list[object] = field(default_factory=list)


@dataclass(frozen=True)
class FrozenRange:
    cell: str
    producer: object | None
    producer_kind: str
    uses: tuple[object, ...]

    @property
    def first_use(self):
        return self.uses[0] if self.uses else None

    @property
    def last_use(self):
        return self.uses[-1] if self.uses else None

    @property
    def use_count(self) -> int:
        return len(self.uses)

    @property
    def start_group(self) -> int:
        return -1 if self.producer is None else self.producer.group

    @property
    def end_group(self) -> int:
        if self.uses:
            return max(slot.group for slot in self.uses)
        return self.start_group

    @property
    def crosses_group(self) -> bool:
        return self.end_group > self.start_group

    @property
    def span_slots(self) -> int:
        if not self.uses:
            return 0
        start = self.first_use.index if self.producer is None else self.producer.index
        return self.last_use.index - start


@dataclass(frozen=True)
class BoundaryLive:
    boundary: int
    data_ranges: tuple[FrozenRange, ...]
    control_ranges: tuple[FrozenRange, ...]


def load_exp115() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp115_operand_graph", EXP115_RUN)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {EXP115_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_data_cell(cell: str) -> bool:
    return cell.startswith(("vec", "acc"))


def freeze_ranges(ranges: list[LiveRange]) -> tuple[FrozenRange, ...]:
    return tuple(
        FrozenRange(
            cell=item.cell,
            producer=item.producer,
            producer_kind=item.producer_kind,
            uses=tuple(item.uses),
        )
        for item in ranges
    )


def build_ranges(exp115: ModuleType, slots: tuple[object, ...]) -> tuple[FrozenRange, ...]:
    active: dict[str, LiveRange] = {}
    ranges: list[LiveRange] = []
    for slot in slots:
        for reg in slot.uses:
            for cell in exp115.reg_cells(reg):
                live_range = active.get(cell)
                if live_range is None:
                    live_range = LiveRange(cell=cell, producer=None, producer_kind="entry")
                    active[cell] = live_range
                    ranges.append(live_range)
                live_range.uses.append(slot)
        for reg in slot.defs:
            for cell in exp115.reg_cells(reg):
                live_range = LiveRange(
                    cell=cell,
                    producer=slot,
                    producer_kind=exp115.producer_kind(slot),
                )
                active[cell] = live_range
                ranges.append(live_range)
    return freeze_ranges(ranges)


def boundary_live_ranges(ranges: tuple[FrozenRange, ...], boundary: int) -> BoundaryLive:
    crossing = tuple(
        item
        for item in ranges
        if item.use_count and item.start_group < boundary <= item.end_group
    )
    data = tuple(item for item in crossing if is_data_cell(item.cell))
    control = tuple(item for item in crossing if not is_data_cell(item.cell))
    return BoundaryLive(boundary=boundary, data_ranges=data, control_ranges=control)


def slot_label(exp115: ModuleType, slot: object | None) -> str:
    return exp115.slot_label(slot)


def range_to_json(exp115: ModuleType, item: FrozenRange) -> dict[str, object]:
    consumer_ops = Counter(slot.op for slot in item.uses)
    return {
        "cell": item.cell,
        "producer": slot_label(exp115, item.producer),
        "producer_kind": item.producer_kind,
        "first_use": slot_label(exp115, item.first_use),
        "last_use": slot_label(exp115, item.last_use),
        "use_count": item.use_count,
        "span_slots": item.span_slots,
        "start_group": item.start_group,
        "end_group": item.end_group,
        "crosses_group": item.crosses_group,
        "consumer_ops": dict(sorted(consumer_ops.items())),
    }


def render_range_row(exp115: ModuleType, item: FrozenRange) -> str:
    return (
        f"| `{item.cell}` | `{item.producer_kind}` | "
        f"`{slot_label(exp115, item.producer)}` | "
        f"`{slot_label(exp115, item.first_use)}` | "
        f"`{slot_label(exp115, item.last_use)}` | "
        f"{item.use_count} | {item.span_slots} |"
    )


def render_boundary_samples(exp115: ModuleType, boundary: BoundaryLive) -> list[str]:
    lines = [
        f"### Boundary Into Group {boundary.boundary}",
        "",
        f"- Data cells live across boundary: `{len(boundary.data_ranges)}`",
        f"- Pointer/scalar cells live across boundary: `{len(boundary.control_ranges)}`",
        "",
        "| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in sorted(boundary.data_ranges, key=lambda value: (-value.span_slots, value.cell))[:20]:
        lines.append(render_range_row(exp115, item))
    return lines


def render_report(
    exp115: ModuleType,
    disasm: Path,
    ranges: tuple[FrozenRange, ...],
    boundaries: tuple[BoundaryLive, ...],
    slots: tuple[object, ...],
) -> str:
    used_ranges = tuple(item for item in ranges if item.use_count)
    data_ranges = tuple(item for item in used_ranges if is_data_cell(item.cell))
    cross_group = tuple(item for item in used_ranges if item.crosses_group)
    producer_kinds = Counter(item.producer_kind for item in used_ranges)
    op_counts = Counter(slot.op for slot in slots)
    longest = sorted(data_ranges, key=lambda item: (-item.span_slots, item.cell))[:32]
    unused_defs = tuple(item for item in ranges if item.producer is not None and not item.uses)

    lines = [
        "# MyLM Q4NX Cell Liveness",
        "",
        f"Source disasm: `{disasm}`",
        "",
        "This report tracks the live ranges of vector-half, accumulator-quadrant,",
        "pointer, and scalar cells across the full `0x260..0x1850` MyLM Q4NX hot",
        "loop. It is a learning artifact for assembly codegen; it does not change",
        "the active IRON kernel.",
        "",
        "## Summary",
        "",
        f"- Parsed slots: `{len(slots)}`",
        f"- Live ranges with at least one use: `{len(used_ranges)}`",
        f"- Data live ranges: `{len(data_ranges)}`",
        f"- Cross-group live ranges: `{len(cross_group)}`",
        f"- Unused definitions: `{len(unused_defs)}`",
        "",
        "## Boundary Pressure",
        "",
        "| Boundary Into Group | Data Cells | Pointer/Scalar Cells |",
        "| ---: | ---: | ---: |",
    ]
    for boundary in boundaries:
        lines.append(
            f"| {boundary.boundary} | {len(boundary.data_ranges)} | {len(boundary.control_ranges)} |"
        )

    lines.extend(
        [
            "",
            "## Producer Kinds",
            "",
            "| Producer Kind | Live Ranges |",
            "| --- | ---: |",
        ]
    )
    for kind, count in producer_kinds.most_common():
        lines.append(f"| `{kind}` | {count} |")

    lines.extend(
        [
            "",
            "## Key Opcode Counts",
            "",
            "| Op | Count |",
            "| --- | ---: |",
        ]
    )
    for op in (
        "vmac.f",
        "vextbcst.16",
        "vups.4x",
        "vunpack",
        "vconv.bf16.fp32",
        "vlda",
        "vldb",
        "lda.s16",
        "vbcst.16",
        "vst",
    ):
        lines.append(f"| `{op}` | {op_counts[op]} |")

    lines.extend(
        [
            "",
            "## Longest Data Live Ranges",
            "",
            "| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in longest:
        lines.append(render_range_row(exp115, item))

    lines.extend(["", "## Boundary Samples", ""])
    for boundary in boundaries:
        lines.extend(render_boundary_samples(exp115, boundary))
        lines.append("")

    lines.extend(
        [
            "## What This Teaches",
            "",
            "- MyLM's fast body is a register-residency schedule, not an opcode list.",
            "- The steady groups keep dozens of data cells live across group boundaries, so a generator must model live state explicitly.",
            "- A production rewrite should first reproduce these live ranges on a small numeric probe, then move the graph-derived body into main16.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_json(
    exp115: ModuleType,
    ranges: tuple[FrozenRange, ...],
    boundaries: tuple[BoundaryLive, ...],
    slots: tuple[object, ...],
) -> dict[str, object]:
    return {
        "source": "MyLM c2r2 0x260..0x1850",
        "slot_count": len(slots),
        "op_counts": dict(sorted(Counter(slot.op for slot in slots).items())),
        "boundaries": [
            {
                "boundary": boundary.boundary,
                "data_count": len(boundary.data_ranges),
                "control_count": len(boundary.control_ranges),
                "data_cells": [range_to_json(exp115, item) for item in boundary.data_ranges],
                "control_cells": [range_to_json(exp115, item) for item in boundary.control_ranges],
            }
            for boundary in boundaries
        ],
        "ranges": [range_to_json(exp115, item) for item in ranges if item.use_count],
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
    ranges = build_ranges(exp115, slots)
    boundaries = tuple(boundary_live_ranges(ranges, group) for group in range(1, 8))
    args.json_output.write_text(json.dumps(render_json(exp115, ranges, boundaries, slots), indent=2) + "\n")
    args.report.write_text(render_report(exp115, exp115.DEFAULT_DISASM, ranges, boundaries, slots))
    used_ranges = tuple(item for item in ranges if item.use_count)
    cross_group = tuple(item for item in used_ranges if item.crosses_group)
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"slots={len(slots)}")
    print(f"live_ranges={len(used_ranges)}")
    print(f"cross_group_live_ranges={len(cross_group)}")
    for boundary in boundaries:
        print(
            f"boundary{boundary.boundary}: "
            f"data={len(boundary.data_ranges)} "
            f"control={len(boundary.control_ranges)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
