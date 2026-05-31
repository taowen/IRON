#!/usr/bin/env python3
"""Annotate MyLM Q4NX group0 instruction semantics."""

from __future__ import annotations

import argparse
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
LIVENESS_JSON = REPO_ROOT / "experiments/118_mylm_q4nx_cell_liveness/mylm_q4nx_cell_liveness.json"
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_q4nx_group0_instruction_semantics.md"
DEFAULT_JSON = EXPERIMENT_DIR / "mylm_q4nx_group0_instruction_semantics.json"
DEFAULT_TSV = EXPERIMENT_DIR / "mylm_q4nx_group0_instruction_semantics.tsv"
GROUP = 0
GROUP_START = 0x260
GROUP_END = 0x52A


@dataclass(frozen=True)
class CellUse:
    cell: str
    producer: str
    producer_kind: str


@dataclass(frozen=True)
class SlotRecord:
    index: int
    address: int
    bundle_slot: int
    op: str
    role: str
    text: str
    defs: tuple[str, ...]
    uses: tuple[CellUse, ...]


@dataclass(frozen=True)
class BoundaryCell:
    cell: str
    producer: str
    producer_kind: str
    first_use: str
    last_use: str
    use_count: int
    span_slots: int


@dataclass(frozen=True)
class SemanticCheck:
    group: int
    start: int
    end: int
    slots: int
    macs: int
    vextbcst16: int
    vups: int
    vconv_bf16_fp32: int
    boundary1_data_cells: int
    boundary1_control_cells: int
    boundary1_group0_produced_cells: int
    boundary1_entry_cells: int
    boundary1_group0_data_cells: int
    boundary1_group0_control_cells: int
    boundary1_entry_data_cells: int
    boundary1_entry_control_cells: int


def load_exp115() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp115_operand_graph", EXP115_RUN)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {EXP115_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text())


def slot_cells(exp115: ModuleType, registers: tuple[str, ...]) -> tuple[str, ...]:
    cells: list[str] = []
    for reg in registers:
        cells.extend(exp115.reg_cells(reg))
    return tuple(dict.fromkeys(cells))


def role_for_slot(exp115: ModuleType, slot) -> str:
    if slot.op == "vmac.f":
        return "mac_accumulate"
    if slot.op == "vextbcst.16":
        source = exp115.first_register(slot.operands[1]) if len(slot.operands) > 1 else ""
        if source == "x11":
            return "activation_lane_broadcast"
        return "coefficient_lane_broadcast"
    return exp115.producer_kind(slot)


def use_records(exp115: ModuleType, latest: dict[str, str], latest_kind: dict[str, str], slot) -> tuple[CellUse, ...]:
    uses: list[CellUse] = []
    for cell in slot_cells(exp115, slot.uses):
        uses.append(
            CellUse(
                cell=cell,
                producer=latest.get(cell, "entry"),
                producer_kind=latest_kind.get(cell, "entry"),
            )
        )
    return tuple(uses)


def update_latest(exp115: ModuleType, latest: dict[str, str], latest_kind: dict[str, str], slot) -> None:
    label = exp115.slot_label(slot)
    role = role_for_slot(exp115, slot)
    for cell in slot_cells(exp115, slot.defs):
        latest[cell] = label
        latest_kind[cell] = role


def build_records(exp115: ModuleType, slots) -> tuple[SlotRecord, ...]:
    latest: dict[str, str] = {}
    latest_kind: dict[str, str] = {}
    records: list[SlotRecord] = []
    for slot in slots:
        records.append(
            SlotRecord(
                index=slot.index,
                address=slot.address,
                bundle_slot=slot.bundle_slot,
                op=slot.op,
                role=role_for_slot(exp115, slot),
                text=slot.text,
                defs=slot_cells(exp115, slot.defs),
                uses=use_records(exp115, latest, latest_kind, slot),
            )
        )
        update_latest(exp115, latest, latest_kind, slot)
    return tuple(records)


def boundary_cells(liveness) -> tuple[BoundaryCell, ...]:
    boundary = next(item for item in liveness["boundaries"] if int(item["boundary"]) == 1)
    cells: list[BoundaryCell] = []
    for item in boundary["data_cells"] + boundary["control_cells"]:
        cells.append(
            BoundaryCell(
                cell=str(item["cell"]),
                producer=str(item["producer"]),
                producer_kind=str(item["producer_kind"]),
                first_use=str(item["first_use"]),
                last_use=str(item["last_use"]),
                use_count=int(item["use_count"]),
                span_slots=int(item["span_slots"]),
            )
        )
    return tuple(cells)


def source_group(label: str) -> int | None:
    if not label.startswith("g"):
        return None
    group_text = label.split("@", 1)[0][1:]
    return int(group_text)


def check_records(records: tuple[SlotRecord, ...], cells: tuple[BoundaryCell, ...]) -> SemanticCheck:
    ops = Counter(record.op for record in records)
    data_cells = tuple(cell for cell in cells if cell.cell.startswith(("vec", "acc")))
    control_cells = tuple(cell for cell in cells if not cell.cell.startswith(("vec", "acc")))
    group0_cells = tuple(cell for cell in cells if source_group(cell.producer) == GROUP)
    entry_cells = tuple(cell for cell in cells if cell.producer == "entry")
    group0_data = tuple(cell for cell in data_cells if source_group(cell.producer) == GROUP)
    group0_control = tuple(cell for cell in control_cells if source_group(cell.producer) == GROUP)
    entry_data = tuple(cell for cell in data_cells if cell.producer == "entry")
    entry_control = tuple(cell for cell in control_cells if cell.producer == "entry")
    return SemanticCheck(
        group=GROUP,
        start=GROUP_START,
        end=GROUP_END,
        slots=len(records),
        macs=ops["vmac.f"],
        vextbcst16=ops["vextbcst.16"],
        vups=ops["vups.4x"],
        vconv_bf16_fp32=ops["vconv.bf16.fp32"],
        boundary1_data_cells=len(data_cells),
        boundary1_control_cells=len(control_cells),
        boundary1_group0_produced_cells=len(group0_cells),
        boundary1_entry_cells=len(entry_cells),
        boundary1_group0_data_cells=len(group0_data),
        boundary1_group0_control_cells=len(group0_control),
        boundary1_entry_data_cells=len(entry_data),
        boundary1_entry_control_cells=len(entry_control),
    )


def format_uses(uses: tuple[CellUse, ...]) -> str:
    return "; ".join(
        f"{item.cell}<-{item.producer_kind}:{item.producer}" for item in uses
    )


def format_cells(cells: tuple[str, ...]) -> str:
    return ", ".join(cells)


def record_to_json(record: SlotRecord):
    return {
        "index": record.index,
        "address": f"0x{record.address:x}",
        "bundle_slot": record.bundle_slot,
        "op": record.op,
        "role": record.role,
        "text": record.text,
        "defs": list(record.defs),
        "uses": [item.__dict__ for item in record.uses],
    }


def boundary_to_json(cell: BoundaryCell):
    return cell.__dict__


def check_to_json(check: SemanticCheck):
    return {
        "group": check.group,
        "range": f"0x{check.start:x}..0x{check.end:x}",
        "slots": check.slots,
        "macs": check.macs,
        "vextbcst16": check.vextbcst16,
        "vups": check.vups,
        "vconv_bf16_fp32": check.vconv_bf16_fp32,
        "boundary1_data_cells": check.boundary1_data_cells,
        "boundary1_control_cells": check.boundary1_control_cells,
        "boundary1_group0_produced_cells": check.boundary1_group0_produced_cells,
        "boundary1_entry_cells": check.boundary1_entry_cells,
        "boundary1_group0_data_cells": check.boundary1_group0_data_cells,
        "boundary1_group0_control_cells": check.boundary1_group0_control_cells,
        "boundary1_entry_data_cells": check.boundary1_entry_data_cells,
        "boundary1_entry_control_cells": check.boundary1_entry_control_cells,
    }


def render_tsv(records: tuple[SlotRecord, ...]) -> str:
    lines = ["address\tbundle\top\trole\tdefs\tuses\ttext"]
    for record in records:
        lines.append(
            "\t".join(
                (
                    f"0x{record.address:x}",
                    str(record.bundle_slot),
                    record.op,
                    record.role,
                    format_cells(record.defs),
                    format_uses(record.uses),
                    record.text,
                )
            )
        )
    return "\n".join(lines) + "\n"


def render_boundary_summary(cells: tuple[BoundaryCell, ...]) -> list[str]:
    producer_kinds = Counter(cell.producer_kind for cell in cells)
    group0_cells = tuple(cell for cell in cells if source_group(cell.producer) == GROUP)
    longest = sorted(cells, key=lambda item: (-item.span_slots, item.cell))[:16]
    lines = [
        "## Boundary1 Live State",
        "",
        "| Producer Kind | Cells |",
        "| --- | ---: |",
    ]
    for kind, count in producer_kinds.most_common():
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(
        [
            "",
            f"- Cells produced inside group0 and live into group1: `{len(group0_cells)}`",
            "",
            "| Cell | Producer | First Use | Last Use | Uses | Span Slots |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for cell in longest:
        lines.append(
            f"| `{cell.cell}` | `{cell.producer}` | `{cell.first_use}` | "
            f"`{cell.last_use}` | {cell.use_count} | {cell.span_slots} |"
        )
    return lines


def render_role_summary(records: tuple[SlotRecord, ...]) -> list[str]:
    roles = Counter(record.role for record in records)
    ops = Counter(record.op for record in records)
    lines = [
        "## Role Counts",
        "",
        "| Role | Slots |",
        "| --- | ---: |",
    ]
    for role, count in roles.most_common():
        lines.append(f"| `{role}` | {count} |")
    lines.extend(["", "## Key Opcode Counts", "", "| Op | Slots |", "| --- | ---: |"])
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
        lines.append(f"| `{op}` | {ops[op]} |")
    return lines


def render_instruction_table(records: tuple[SlotRecord, ...]) -> list[str]:
    lines = [
        "## Group0 Instruction Table",
        "",
        "| Address | Slot | Op | Role | Def Cells | Use Cells And Producers | Instruction |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| `0x{record.address:x}` | {record.bundle_slot} | `{record.op}` | "
            f"`{record.role}` | `{format_cells(record.defs)}` | "
            f"`{format_uses(record.uses)}` | `{record.text}` |"
        )
    return lines


def render_report(records: tuple[SlotRecord, ...], cells: tuple[BoundaryCell, ...], check: SemanticCheck) -> str:
    lines = [
        "# MyLM Q4NX Group0 Instruction Semantics",
        "",
        "This experiment annotates the first MyLM Q4NX hot-loop section at",
        "instruction-slot granularity. It is a learning artifact for writing a",
        "future generator; it does not change the active IRON kernel.",
        "",
        "## Checks",
        "",
        f"- Group range: `0x{check.start:x}..0x{check.end:x}`",
        f"- Parsed instruction slots: `{check.slots}`",
        f"- `vmac.f`: `{check.macs}`",
        f"- `vextbcst.16`: `{check.vextbcst16}`",
        f"- `vups.4x`: `{check.vups}`",
        f"- `vconv.bf16.fp32`: `{check.vconv_bf16_fp32}`",
        f"- Boundary1 data cells: `{check.boundary1_data_cells}`",
        f"- Boundary1 control cells: `{check.boundary1_control_cells}`",
        f"- Boundary1 cells produced by group0: `{check.boundary1_group0_produced_cells}`",
        f"- Boundary1 entry cells carried through: `{check.boundary1_entry_cells}`",
        f"- Boundary1 group0-produced data/control cells: `{check.boundary1_group0_data_cells}` / `{check.boundary1_group0_control_cells}`",
        f"- Boundary1 entry data/control cells: `{check.boundary1_entry_data_cells}` / `{check.boundary1_entry_control_cells}`",
        "",
        "## Interpretation",
        "",
        "- Group0 is pipeline fill. It does not close over a complete Q4NX group by itself.",
        "- The useful learning unit is the slot-level def/use trace plus the boundary1 live state.",
        "- A production generator must preserve the live cells crossing into group1 before changing the arithmetic body.",
        "",
    ]
    lines.extend(render_role_summary(records))
    lines.append("")
    lines.extend(render_boundary_summary(cells))
    lines.append("")
    lines.extend(render_instruction_table(records))
    lines.append("")
    return "\n".join(lines)


def render_json(records: tuple[SlotRecord, ...], cells: tuple[BoundaryCell, ...], check: SemanticCheck):
    return {
        "source": "MyLM c2r2 0x260..0x52a",
        "checks": check_to_json(check),
        "records": [record_to_json(record) for record in records],
        "boundary1_cells": [boundary_to_json(cell) for cell in cells],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tsv-output", type=Path, default=DEFAULT_TSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp115 = load_exp115()
    slots = tuple(slot for slot in exp115.parse_slots(exp115.DEFAULT_DISASM) if slot.group == GROUP)
    if not slots:
        raise ValueError(f"no group{GROUP} slots parsed from {exp115.DEFAULT_DISASM}")
    records = build_records(exp115, slots)
    cells = boundary_cells(load_json(LIVENESS_JSON))
    check = check_records(records, cells)
    args.tsv_output.write_text(render_tsv(records))
    args.json_output.write_text(json.dumps(render_json(records, cells, check), indent=2) + "\n")
    args.report.write_text(render_report(records, cells, check))
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.tsv_output}")
    print(f"group0_slots={check.slots}")
    print(f"group0_vmac.f={check.macs}")
    print(f"boundary1_group0_produced_cells={check.boundary1_group0_produced_cells}")
    return 0 if check.start == GROUP_START and check.end == GROUP_END and check.macs == 28 else 1


if __name__ == "__main__":
    raise SystemExit(main())
