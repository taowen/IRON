#!/usr/bin/env python3
"""Annotate MyLM Q4NX steady-to-steady instruction semantics."""

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
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_q4nx_steady_transition_semantics.md"
DEFAULT_JSON = EXPERIMENT_DIR / "mylm_q4nx_steady_transition_semantics.json"
DEFAULT_TSV = EXPERIMENT_DIR / "mylm_q4nx_steady_transition_semantics.tsv"
GROUP = 2
INPUT_BOUNDARY = GROUP
OUTPUT_BOUNDARY = GROUP + 1
GROUP_START = 0x7DE
GROUP_END = 0xA92


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
class BoundarySummary:
    boundary: int
    data_cells: int
    control_cells: int
    producer_classes: dict[str, int]


@dataclass(frozen=True)
class ProducerChange:
    cell: str
    before: str
    after: str
    before_kind: str
    after_kind: str


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
    vst: int
    input_data_cells: int
    input_control_cells: int
    output_data_cells: int
    output_control_cells: int
    boundary_cell_names_stable: bool
    relative_producer_classes_stable: bool
    changed_producer_cells: int
    changed_data_cells: int
    changed_control_cells: int


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
        if slot.group == GROUP:
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
        if slot.group > GROUP:
            break
    return tuple(records)


def boundary_cells(liveness, boundary: int) -> tuple[BoundaryCell, ...]:
    item = next(entry for entry in liveness["boundaries"] if int(entry["boundary"]) == boundary)
    cells: list[BoundaryCell] = []
    for cell in item["data_cells"] + item["control_cells"]:
        cells.append(
            BoundaryCell(
                cell=str(cell["cell"]),
                producer=str(cell["producer"]),
                producer_kind=str(cell["producer_kind"]),
                first_use=str(cell["first_use"]),
                last_use=str(cell["last_use"]),
                use_count=int(cell["use_count"]),
                span_slots=int(cell["span_slots"]),
            )
        )
    return tuple(cells)


def source_group(label: str) -> int | None:
    if not label.startswith("g"):
        return None
    return int(label.split("@", 1)[0][1:])


def is_data_cell(cell: str) -> bool:
    return cell.startswith(("vec", "acc"))


def producer_class(cell: BoundaryCell, boundary: int) -> str:
    group = source_group(cell.producer)
    if cell.producer == "entry":
        return "entry"
    if group == 0:
        return "persistent_group0"
    if group == boundary - 1:
        return "previous_group"
    return "other"


def boundary_summary(cells: tuple[BoundaryCell, ...], boundary: int) -> BoundarySummary:
    return BoundarySummary(
        boundary=boundary,
        data_cells=sum(is_data_cell(cell.cell) for cell in cells),
        control_cells=sum(not is_data_cell(cell.cell) for cell in cells),
        producer_classes=dict(sorted(Counter(producer_class(cell, boundary) for cell in cells).items())),
    )


def producer_changes(before: tuple[BoundaryCell, ...], after: tuple[BoundaryCell, ...]) -> tuple[ProducerChange, ...]:
    before_by_cell = {cell.cell: cell for cell in before}
    after_by_cell = {cell.cell: cell for cell in after}
    changes: list[ProducerChange] = []
    for cell_name in sorted(before_by_cell):
        before_cell = before_by_cell[cell_name]
        after_cell = after_by_cell[cell_name]
        if before_cell.producer == after_cell.producer:
            continue
        changes.append(
            ProducerChange(
                cell=cell_name,
                before=before_cell.producer,
                after=after_cell.producer,
                before_kind=before_cell.producer_kind,
                after_kind=after_cell.producer_kind,
            )
        )
    return tuple(changes)


def check_records(
    records: tuple[SlotRecord, ...],
    input_cells: tuple[BoundaryCell, ...],
    output_cells: tuple[BoundaryCell, ...],
) -> SemanticCheck:
    ops = Counter(record.op for record in records)
    input_summary = boundary_summary(input_cells, INPUT_BOUNDARY)
    output_summary = boundary_summary(output_cells, OUTPUT_BOUNDARY)
    changes = producer_changes(input_cells, output_cells)
    return SemanticCheck(
        group=GROUP,
        start=GROUP_START,
        end=GROUP_END,
        slots=len(records),
        macs=ops["vmac.f"],
        vextbcst16=ops["vextbcst.16"],
        vups=ops["vups.4x"],
        vconv_bf16_fp32=ops["vconv.bf16.fp32"],
        vst=ops["vst"],
        input_data_cells=input_summary.data_cells,
        input_control_cells=input_summary.control_cells,
        output_data_cells=output_summary.data_cells,
        output_control_cells=output_summary.control_cells,
        boundary_cell_names_stable={cell.cell for cell in input_cells} == {cell.cell for cell in output_cells},
        relative_producer_classes_stable=input_summary.producer_classes == output_summary.producer_classes,
        changed_producer_cells=len(changes),
        changed_data_cells=sum(is_data_cell(change.cell) for change in changes),
        changed_control_cells=sum(not is_data_cell(change.cell) for change in changes),
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


def boundary_to_json(cell: BoundaryCell, boundary: int):
    data = cell.__dict__.copy()
    data["relative_producer_class"] = producer_class(cell, boundary)
    return data


def change_to_json(change: ProducerChange):
    return change.__dict__


def check_to_json(check: SemanticCheck):
    return {
        "group": check.group,
        "range": f"0x{check.start:x}..0x{check.end:x}",
        "slots": check.slots,
        "macs": check.macs,
        "vextbcst16": check.vextbcst16,
        "vups": check.vups,
        "vconv_bf16_fp32": check.vconv_bf16_fp32,
        "vst": check.vst,
        "input_data_cells": check.input_data_cells,
        "input_control_cells": check.input_control_cells,
        "output_data_cells": check.output_data_cells,
        "output_control_cells": check.output_control_cells,
        "boundary_cell_names_stable": check.boundary_cell_names_stable,
        "relative_producer_classes_stable": check.relative_producer_classes_stable,
        "changed_producer_cells": check.changed_producer_cells,
        "changed_data_cells": check.changed_data_cells,
        "changed_control_cells": check.changed_control_cells,
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


def render_boundary_summary(input_cells: tuple[BoundaryCell, ...], output_cells: tuple[BoundaryCell, ...]) -> list[str]:
    input_summary = boundary_summary(input_cells, INPUT_BOUNDARY)
    output_summary = boundary_summary(output_cells, OUTPUT_BOUNDARY)
    changes = producer_changes(input_cells, output_cells)
    lines = [
        "## Steady Boundary Transition",
        "",
        "| Boundary | Data Cells | Control Cells | Producer Classes |",
        "| ---: | ---: | ---: | --- |",
        f"| {INPUT_BOUNDARY} | {input_summary.data_cells} | {input_summary.control_cells} | `{input_summary.producer_classes}` |",
        f"| {OUTPUT_BOUNDARY} | {output_summary.data_cells} | {output_summary.control_cells} | `{output_summary.producer_classes}` |",
        "",
        f"- Producer changes across group{GROUP}: `{len(changes)}`",
        f"- Changed data/control cells: `{sum(is_data_cell(change.cell) for change in changes)}` / `{sum(not is_data_cell(change.cell) for change in changes)}`",
        "",
        "| Cell | Before | After | Kind |",
        "| --- | --- | --- | --- |",
    ]
    for change in changes[:32]:
        lines.append(
            f"| `{change.cell}` | `{change.before}` | `{change.after}` | "
            f"`{change.before_kind} -> {change.after_kind}` |"
        )
    if len(changes) > 32:
        lines.append(f"| ... | {len(changes) - 32} more | ... | ... |")
    return lines


def render_instruction_table(records: tuple[SlotRecord, ...]) -> list[str]:
    lines = [
        "## Group2 Instruction Table",
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


def render_report(
    records: tuple[SlotRecord, ...],
    input_cells: tuple[BoundaryCell, ...],
    output_cells: tuple[BoundaryCell, ...],
    check: SemanticCheck,
) -> str:
    lines = [
        "# MyLM Q4NX Steady Transition Semantics",
        "",
        "This experiment annotates MyLM group2 with the group0-1 live state already",
        "applied. Group2 is the first true steady-to-steady section, so this is the",
        "best current learning artifact for a reusable assembly generator.",
        "",
        "## Checks",
        "",
        f"- Group range: `0x{check.start:x}..0x{check.end:x}`",
        f"- Parsed instruction slots: `{check.slots}`",
        f"- `vmac.f`: `{check.macs}`",
        f"- `vextbcst.16`: `{check.vextbcst16}`",
        f"- `vups.4x`: `{check.vups}`",
        f"- `vconv.bf16.fp32`: `{check.vconv_bf16_fp32}`",
        f"- `vst`: `{check.vst}`",
        f"- Input boundary data/control cells: `{check.input_data_cells}` / `{check.input_control_cells}`",
        f"- Output boundary data/control cells: `{check.output_data_cells}` / `{check.output_control_cells}`",
        f"- Boundary cell names stable: `{check.boundary_cell_names_stable}`",
        f"- Relative producer classes stable: `{check.relative_producer_classes_stable}`",
        f"- Changed producer cells: `{check.changed_producer_cells}`",
        f"- Changed data/control cells: `{check.changed_data_cells}` / `{check.changed_control_cells}`",
        "",
        "## Interpretation",
        "",
        "- Group2 is a state transformer: it keeps the same boundary cell set while replacing the previous group's live producers with group2 producers.",
        "- The persistent entry/group0 cells are part of the steady contract and must not be treated as local temporaries.",
        "- A production exact rewrite should preserve this boundary transition before replacing MyLM's arithmetic contract.",
        "",
    ]
    lines.extend(render_role_summary(records))
    lines.append("")
    lines.extend(render_boundary_summary(input_cells, output_cells))
    lines.append("")
    lines.extend(render_instruction_table(records))
    lines.append("")
    return "\n".join(lines)


def render_json(
    records: tuple[SlotRecord, ...],
    input_cells: tuple[BoundaryCell, ...],
    output_cells: tuple[BoundaryCell, ...],
    check: SemanticCheck,
):
    return {
        "source": "MyLM c2r2 group2 0x7de..0xa92",
        "checks": check_to_json(check),
        "records": [record_to_json(record) for record in records],
        "input_boundary": {
            "boundary": INPUT_BOUNDARY,
            "summary": boundary_summary(input_cells, INPUT_BOUNDARY).__dict__,
            "cells": [boundary_to_json(cell, INPUT_BOUNDARY) for cell in input_cells],
        },
        "output_boundary": {
            "boundary": OUTPUT_BOUNDARY,
            "summary": boundary_summary(output_cells, OUTPUT_BOUNDARY).__dict__,
            "cells": [boundary_to_json(cell, OUTPUT_BOUNDARY) for cell in output_cells],
        },
        "producer_changes": [change_to_json(change) for change in producer_changes(input_cells, output_cells)],
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
    slots = exp115.parse_slots(exp115.DEFAULT_DISASM)
    records = build_records(exp115, slots)
    if not records:
        raise ValueError(f"no group{GROUP} records parsed from {exp115.DEFAULT_DISASM}")
    liveness = load_json(LIVENESS_JSON)
    input_cells = boundary_cells(liveness, INPUT_BOUNDARY)
    output_cells = boundary_cells(liveness, OUTPUT_BOUNDARY)
    check = check_records(records, input_cells, output_cells)
    args.tsv_output.write_text(render_tsv(records))
    args.json_output.write_text(json.dumps(render_json(records, input_cells, output_cells, check), indent=2) + "\n")
    args.report.write_text(render_report(records, input_cells, output_cells, check))
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.tsv_output}")
    print(f"group{GROUP}_slots={check.slots}")
    print(f"group{GROUP}_vmac.f={check.macs}")
    print(f"changed_producer_cells={check.changed_producer_cells}")
    return 0 if check.slots == 189 and check.macs == 33 and check.relative_producer_classes_stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
