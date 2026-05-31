#!/usr/bin/env python3
"""Replay MyLM Q4NX hot-loop text from explicit software-pipeline templates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from types import ModuleType

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP104_RUN = REPO_ROOT / "experiments/104_mylm_q4nx_pipeline_schedule/run.py"
DEFAULT_REPORT = EXPERIMENT_DIR / "mylm_q4nx_template_codegen.md"
DEFAULT_ASM_INC = EXPERIMENT_DIR / "generated_mylm_q4nx_hot_loop.s.inc"


@dataclass(frozen=True)
class GeneratedSlot:
    template: str
    template_index: int
    text: str
    op: str


@dataclass(frozen=True)
class ReplayResult:
    original_slots: int
    generated_slots: int
    original_hash: str
    generated_hash: str
    exact_match: bool
    op_counts: Counter[str]


def load_exp104() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp104_pipeline_schedule", EXP104_RUN)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {EXP104_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def op_of(text: str) -> str:
    return text.split(None, 1)[0]


def template_program(groups: dict[int, list[object]]) -> list[GeneratedSlot]:
    program: list[GeneratedSlot] = []
    for slot in groups[0]:
        program.append(GeneratedSlot("fill", slot.index, slot.text, slot.op))
    for repeat in range(5):
        for index, slot in enumerate(groups[1]):
            program.append(GeneratedSlot(f"steady{repeat}", index, slot.text, slot.op))
    for index, slot in enumerate(groups[6]):
        program.append(GeneratedSlot("pre_drain", index, slot.text, slot.op))
    for index, slot in enumerate(groups[7]):
        program.append(GeneratedSlot("drain", index, slot.text, slot.op))
    return program


def replay_result(original_text: list[str], generated: list[GeneratedSlot]) -> ReplayResult:
    generated_text = [slot.text for slot in generated]
    return ReplayResult(
        original_slots=len(original_text),
        generated_slots=len(generated_text),
        original_hash=digest(original_text),
        generated_hash=digest(generated_text),
        exact_match=original_text == generated_text,
        op_counts=Counter(slot.op for slot in generated),
    )


def render_asm_inc(groups: dict[int, list[object]]) -> str:
    lines = [
        "// Generated from MyLM disasm templates by exp107.",
        "// This is a schedule artifact, not a standalone function.",
        "",
    ]
    templates = (
        ("MYLM_Q4NX_FILL", groups[0]),
        ("MYLM_Q4NX_STEADY", groups[1]),
        ("MYLM_Q4NX_PRE_DRAIN", groups[6]),
        ("MYLM_Q4NX_DRAIN", groups[7]),
    )
    for name, slots in templates:
        lines.append(f".macro {name}")
        for slot in slots:
            lines.append(f"\t{slot.text}")
        lines.append(".endm")
        lines.append("")
    lines.extend(
        [
            "// Replay shape:",
            "//   MYLM_Q4NX_FILL",
            "//   MYLM_Q4NX_STEADY x5",
            "//   MYLM_Q4NX_PRE_DRAIN",
            "//   MYLM_Q4NX_DRAIN",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(result: ReplayResult, generated: list[GeneratedSlot], original_text: list[str]) -> str:
    key_ops = (
        "vmac.f",
        "vextbcst.16",
        "vunpack",
        "vups.4x",
        "vconv.bf16.fp32",
        "vst",
        "vlda",
        "vldb",
        "lda.s16",
        "vbcst.16",
    )
    lines = [
        "# MyLM Q4NX Template Codegen",
        "",
        "This report proves that the MyLM hot loop can be represented as four",
        "explicit schedule templates rather than eight hand-copied groups.",
        "",
        "## Replay Check",
        "",
        f"- Original slots: `{result.original_slots}`",
        f"- Generated slots: `{result.generated_slots}`",
        f"- Original hash: `{result.original_hash}`",
        f"- Generated hash: `{result.generated_hash}`",
        f"- Exact text match: `{result.exact_match}`",
        "",
        "## Opcode Counts",
        "",
        "| Op | Count |",
        "| --- | ---: |",
    ]
    for op in key_ops:
        lines.append(f"| `{op}` | {result.op_counts[op]} |")
    lines.extend(
        [
            "",
            "## Template Boundaries",
            "",
            "| Generated Range | Template | Slots |",
            "| --- | --- | ---: |",
        ]
    )
    cursor = 0
    for name in ("fill", "steady0", "steady1", "steady2", "steady3", "steady4", "pre_drain", "drain"):
        count = sum(1 for slot in generated if slot.template == name)
        lines.append(f"| `{cursor}:{cursor + count}` | `{name}` | {count} |")
        cursor += count
    if not result.exact_match:
        lines.extend(
            [
                "",
                "## First Diff",
                "",
                "```diff",
            ]
        )
        generated_text = [slot.text for slot in generated]
        lines.extend(list(unified_diff(original_text, generated_text, fromfile="original", tofile="generated", n=4))[:120])
        lines.append("```")
    lines.extend(
        [
            "",
            "## Production Boundary",
            "",
            "The next production step is not to paste this MyLM body into IRON. The",
            "useful boundary is to port this generator shape to the IRON exact-Q4NX",
            "contract, then make each template pass the existing qwen3-layer numeric",
            "gates. MyLM's group-sum formula remains a separate numerical-contract",
            "decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--asm-inc", type=Path, default=DEFAULT_ASM_INC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp104 = load_exp104()
    slots = exp104.parse_slots(exp104.DEFAULT_DISASM)
    groups = {group: [slot for slot in slots if slot.group == group] for group in range(8)}
    original_text = [slot.text for group in range(8) for slot in groups[group]]
    generated = template_program(groups)
    result = replay_result(original_text, generated)
    args.report.write_text(render_report(result, generated, original_text))
    args.asm_inc.write_text(render_asm_inc(groups))
    print(f"wrote {args.report}")
    print(f"wrote {args.asm_inc}")
    print(f"original_slots={result.original_slots}")
    print(f"generated_slots={result.generated_slots}")
    print(f"exact_match={result.exact_match}")
    print(f"vmac.f={result.op_counts['vmac.f']}")
    print(f"vextbcst.16={result.op_counts['vextbcst.16']}")
    print(f"vst={result.op_counts['vst']}")
    return 0 if result.exact_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
