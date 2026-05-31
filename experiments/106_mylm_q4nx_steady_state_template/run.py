#!/usr/bin/env python3
"""Extract the repeatable steady-state template from MyLM's Q4NX loop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP104_RUN = REPO_ROOT / "experiments/104_mylm_q4nx_pipeline_schedule/run.py"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "mylm_q4nx_steady_state_template.md"


@dataclass(frozen=True)
class GroupIdentity:
    group: int
    slots: int
    text_hash: str
    op_hash: str
    matches_steady_text: bool
    matches_steady_ops: bool


def load_exp104() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp104_pipeline_schedule", EXP104_RUN)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {EXP104_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(values: list[str]) -> str:
    text = "\n".join(values).encode()
    return hashlib.sha256(text).hexdigest()[:16]


def lane_operand(text: str) -> str:
    match = re.search(r"#(0x[0-9a-fA-F]+|\d+)\b", text)
    return match.group(1) if match else ""


def group_identity(groups: dict[int, list[object]], steady_group: int) -> tuple[GroupIdentity, ...]:
    steady_text = [slot.text for slot in groups[steady_group]]
    steady_ops = [slot.op for slot in groups[steady_group]]
    identities: list[GroupIdentity] = []
    for group in range(8):
        text = [slot.text for slot in groups[group]]
        ops = [slot.op for slot in groups[group]]
        identities.append(
            GroupIdentity(
                group=group,
                slots=len(groups[group]),
                text_hash=digest(text),
                op_hash=digest(ops),
                matches_steady_text=text == steady_text,
                matches_steady_ops=ops == steady_ops,
            )
        )
    return tuple(identities)


def diff_table(base: list[object], candidate: list[object], base_name: str, candidate_name: str) -> list[str]:
    base_text = [slot.text for slot in base]
    candidate_text = [slot.text for slot in candidate]
    matcher = SequenceMatcher(None, base_text, candidate_text, autojunk=False)
    lines = [
        f"### `{candidate_name}` vs `{base_name}`",
        "",
        "| Kind | Base Range | Candidate Range | Base Text | Candidate Text |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        base_rows = "<br>".join(f"{idx}: `{base_text[idx]}`" for idx in range(i1, i2)) or "-"
        candidate_rows = "<br>".join(f"{idx}: `{candidate_text[idx]}`" for idx in range(j1, j2)) or "-"
        lines.append(f"| {tag} | `{i1}:{i2}` | `{j1}:{j2}` | {base_rows} | {candidate_rows} |")
    if len(lines) == 3:
        lines.append("| equal | - | - | no differences | no differences |")
    return lines


def render_template_table(group_slots: list[object]) -> list[str]:
    start = group_slots[0].address
    lines = [
        "## Steady Template Body",
        "",
        "Group1 is the canonical steady-state template. The same instruction text",
        "is repeated for groups 2, 3, 4, and 5.",
        "",
        "| Index | Offset | Bundle Slot | Instruction |",
        "| ---: | ---: | ---: | --- |",
    ]
    for index, slot in enumerate(group_slots):
        lines.append(
            f"| {index} | `+0x{slot.address - start:x}` | {slot.bundle_slot} | "
            f"`{slot.text.replace('|', '\\|')}` |"
        )
    return lines


def render(groups: dict[int, list[object]], output_source: Path) -> str:
    identities = group_identity(groups, steady_group=1)
    steady = groups[1]
    steady_lanes = [lane_operand(slot.text) for slot in steady if slot.op == "vextbcst.16"]
    lines = [
        "# MyLM Q4NX Steady-State Template",
        "",
        f"Parser source: `{output_source}`",
        "",
        "This report extracts the repeatable middle of the MyLM Q4NX software",
        "pipeline. It is intended as input for an assembly generator, not as a",
        "replacement implementation.",
        "",
        "## Group Identity",
        "",
        "| Group | Slots | Text Hash | Op Hash | Matches Group1 Text | Matches Group1 Ops |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in identities:
        lines.append(
            f"| {item.group} | {item.slots} | `{item.text_hash}` | `{item.op_hash}` | "
            f"{item.matches_steady_text} | {item.matches_steady_ops} |"
        )
    lines.extend(
        [
            "",
            "## Generator Shape",
            "",
            "```text",
            "fill(group0)",
            "steady_template(group1) * 5  # groups 1..5 are text-identical",
            "pre_drain(group6)",
            "drain(group7)",
            "```",
            "",
            "## Steady Vext Lane Order",
            "",
            "```text",
            ", ".join(steady_lanes),
            "```",
            "",
            "The lane order is not simply 0..31. Preserving this order matters",
            "because it is interleaved with unpack/upshift/convert/MAC work that",
            "fills vector-load and storeback latency.",
            "",
            "## Non-Steady Diffs",
            "",
        ]
    )
    lines.extend(diff_table(steady, groups[0], "group1", "group0-fill"))
    lines.append("")
    lines.extend(diff_table(steady, groups[6], "group1", "group6-pre-drain"))
    lines.append("")
    lines.extend(diff_table(steady, groups[7], "group1", "group7-drain"))
    lines.append("")
    lines.extend(render_template_table(steady))
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- group1..5 text identity is the only repeatable steady-state template",
            "  found in the shipped MyLM hot loop.",
            "- group6 must be generated as its own pre-drain variant because it",
            "  rewinds `p3` near the tail before entering group7.",
            "- group0 and group7 are not variants to guess from opcode counts; they",
            "  need explicit fill/drain templates.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp104 = load_exp104()
    slots = exp104.parse_slots(exp104.DEFAULT_DISASM)
    groups = {group: [slot for slot in slots if slot.group == group] for group in range(8)}
    args.output.write_text(render(groups, EXP104_RUN))
    identities = group_identity(groups, steady_group=1)
    print(f"wrote {args.output}")
    for item in identities:
        print(
            f"group{item.group}: slots={item.slots} "
            f"matches_text={item.matches_steady_text} matches_ops={item.matches_steady_ops}"
        )
    steady_ok = all(identities[group].matches_steady_text for group in range(1, 6))
    if not steady_ok:
        print("FAIL: group1..5 are not text-identical")
        return 1
    if identities[6].matches_steady_text or identities[7].matches_steady_text:
        print("FAIL: pre-drain/drain unexpectedly match steady template")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
