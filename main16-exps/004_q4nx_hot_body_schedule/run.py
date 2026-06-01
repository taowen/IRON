#!/usr/bin/env python3
"""Summarize the MyLM main16 Q4NX hot-body schedule."""

from __future__ import annotations

import json
import re
import subprocess
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_ELF = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf"
LLVM_OBJDUMP = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump"
MANIFEST = EXPERIMENT_DIR / "q4nx_hot_body_schedule.json"
REPORT = EXPERIMENT_DIR / "q4nx_hot_body_schedule.md"

MICROKERNEL_START = 0x1F0
MICROKERNEL_DISASM_END = 0x186C
HOT_START = 0x260
HOT_END = 0x1850
QKV_BODY_START = 0x1870
QKV_BODY_END = 0x1E80
FIRST_GROUP_LIMIT = 96

ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
LC_RE = re.compile(r"\bmova\s+lc,\s*#0x([0-9a-fA-F]+)\b")
REGISTER_RE = re.compile(
    r"\b(?:r\d+|p\d+|x\d+|wl\d+|wh\d+|dm\d+|cml\d+|bmll\d+|bmlh\d+|bmhl\d+|bmhh\d+|"
    r"lr|lc|ls|le|el0|crrnd|crupsmode|s0|vaddsign0)\b"
)
VFRAGMENT_RE = re.compile(r"^\s*([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)\s*(.*)$")
ACTIVATION_LOAD_RE = re.compile(r"\bvldb\s+x11,\s*\[p1\],\s*#0x40\b")
LANE_RE = re.compile(r"\bvextbcst\.16\s+\S+,\s*x11,\s*#0x([0-9a-fA-F]+)\b")
SCRATCH_LOAD_RE = re.compile(r"\blda\.s16\s+r7,\s*\[p3\],\s*#0x2\b")
SCRATCH_REWIND_RE = re.compile(r"\badd\.nc\s+p3,\s*r21,\s*#-0x10\b")
PHASE_ACTIVATION_LOAD_RE = re.compile(r"\bvlda\.conv\.fp32\.bf16\s+\w+,\s*\[p3")
PHASE_SCRATCH_STORE_RE = re.compile(r"\bst\.s16\s+\w+,\s*\[p2")
PHASE_SCALAR_EXTRACT_RE = re.compile(r"\bvextract\.16\s+\w+,\s*x\d+,\s*#0x0")


@dataclass(frozen=True)
class Instruction:
    address: int
    text: str
    fragments: tuple[str, ...]


@dataclass(frozen=True)
class LaneGroup:
    index: int
    start: int
    end: int
    lanes: tuple[int, ...]
    op_counts: Counter[str]


def run_cmd(cmd: tuple[str, ...]) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result.stdout


def disassemble() -> tuple[Instruction, ...]:
    # llvm-objdump currently asserts on the padding at 0x186e, so keep the
    # Q4 microkernel and phase-body ranges separate.
    text = "\n".join(
        (
            run_cmd(
                (
                    str(LLVM_OBJDUMP),
                    "-d",
                    f"--start-address=0x{MICROKERNEL_START:x}",
                    f"--stop-address=0x{MICROKERNEL_DISASM_END:x}",
                    str(DEFAULT_ELF),
                )
            ),
            run_cmd(
                (
                    str(LLVM_OBJDUMP),
                    "-d",
                    f"--start-address=0x{QKV_BODY_START:x}",
                    f"--stop-address=0x{QKV_BODY_END:x}",
                    str(DEFAULT_ELF),
                )
            ),
        )
    )
    instructions: list[Instruction] = []
    for line in text.splitlines():
        match = ADDRESS_RE.match(line)
        if match is None:
            continue
        payload = match.group(2).split("\t", 1)[1] if "\t" in match.group(2) else ""
        fragments = tuple(fragment.strip() for fragment in payload.split(";") if fragment.strip())
        instructions.append(Instruction(int(match.group(1), 16), line.strip(), fragments))
    return tuple(instructions)


def op_name(fragment: str) -> str:
    match = VFRAGMENT_RE.match(fragment)
    return match.group(1) if match is not None else ""


def op_counts(instructions: tuple[Instruction, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for instruction in instructions:
        for fragment in instruction.fragments:
            op = op_name(fragment)
            if op:
                counts[op] += 1
    return counts


def range_instructions(instructions: tuple[Instruction, ...], start: int, end: int) -> tuple[Instruction, ...]:
    return tuple(instruction for instruction in instructions if start <= instruction.address < end)


def loop_count(instructions: tuple[Instruction, ...]) -> int:
    for instruction in instructions:
        if instruction.address != MICROKERNEL_START:
            continue
        match = LC_RE.search(" ; ".join(instruction.fragments))
        if match is not None:
            return int(match.group(1), 16)
    raise ValueError("missing lc setup at microkernel entry")


def lane_groups(hot: tuple[Instruction, ...]) -> tuple[LaneGroup, ...]:
    starts = [instruction.address for instruction in hot if ACTIVATION_LOAD_RE.search(" ; ".join(instruction.fragments))]
    groups: list[LaneGroup] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else HOT_END
        body = range_instructions(hot, start, end)
        lanes: list[int] = []
        for instruction in body:
            for match in LANE_RE.finditer(" ; ".join(instruction.fragments)):
                lanes.append(int(match.group(1), 16))
        groups.append(LaneGroup(index, start, end, tuple(lanes), op_counts(body)))
    return tuple(groups)


def first_group_flow(groups: tuple[LaneGroup, ...], hot: tuple[Instruction, ...]) -> list[dict[str, object]]:
    if not groups:
        return []
    first = groups[0]
    rows: list[dict[str, object]] = []
    for instruction in range_instructions(hot, first.start, first.end)[:FIRST_GROUP_LIMIT]:
        for fragment in instruction.fragments:
            op = op_name(fragment)
            registers = REGISTER_RE.findall(fragment)
            defs: list[str] = []
            uses = registers[:]
            if op and registers and not op.startswith(("st", "vst", "rel", "acq", "jnz", "j", "ret")):
                defs = [registers[0]]
                uses = registers[1:]
            rows.append(
                {
                    "address": hex(instruction.address),
                    "op": op,
                    "fragment": fragment,
                    "defs": defs,
                    "uses": uses,
                }
            )
    return rows


def phase_group_sum_summary(instructions: tuple[Instruction, ...]) -> dict[str, object]:
    body = range_instructions(instructions, QKV_BODY_START, QKV_BODY_END)
    text_lines = [" ; ".join(instruction.fragments) for instruction in body]
    return {
        "activation_vector_loads": sum(1 for text in text_lines if PHASE_ACTIVATION_LOAD_RE.search(text)),
        "scratch_stores": sum(1 for text in text_lines if PHASE_SCRATCH_STORE_RE.search(text)),
        "scalar_extracts": sum(1 for text in text_lines if PHASE_SCALAR_EXTRACT_RE.search(text)),
        "op_counts": dict(op_counts(body)),
    }


def top_counts(counts: Counter[str], names: tuple[str, ...]) -> dict[str, int]:
    return {name: counts[name] for name in names}


def build_manifest() -> dict[str, object]:
    instructions = disassemble()
    hot = range_instructions(instructions, HOT_START, HOT_END)
    counts = op_counts(hot)
    lc = loop_count(instructions)
    groups = lane_groups(hot)
    group_rows = [
        {
            "index": group.index,
            "start": hex(group.start),
            "end": hex(group.end),
            "lane_count": len(group.lanes),
            "lanes_complete": tuple(sorted(group.lanes)) == tuple(range(32)),
            "first_lane": min(group.lanes) if group.lanes else None,
            "last_lane": max(group.lanes) if group.lanes else None,
            "counts": top_counts(
                group.op_counts,
                (
                    "vldb",
                    "lda.s16",
                    "vextbcst.16",
                    "vbcst.16",
                    "vmac.f",
                    "vunpack",
                    "vups.4x",
                    "vconv.bf16.fp32",
                    "vst",
                ),
            ),
        }
        for group in groups
    ]
    key_counts = top_counts(
        counts,
        (
            "vldb",
            "lda.s16",
            "vextbcst.16",
            "vbcst.16",
            "vmac.f",
            "vunpack",
            "vups.4x",
            "vconv.bf16.fp32",
            "vst",
            "vst.conv.bf16.fp32",
        ),
    )
    return {
        "status": "passed",
        "source_elf": str(DEFAULT_ELF.relative_to(REPO_ROOT)),
        "ranges": {
            "microkernel": [hex(MICROKERNEL_START), hex(HOT_END)],
            "hot_loop": [hex(HOT_START), hex(HOT_END)],
            "qkv_body": [hex(QKV_BODY_START), hex(QKV_BODY_END)],
        },
        "loop_count": lc,
        "hot_instruction_lines": len(hot),
        "hot_op_slots": sum(counts.values()),
        "key_static_counts": key_counts,
        "key_dynamic_counts": {name: value * lc for name, value in key_counts.items()},
        "scratch_load_addresses": [
            hex(instruction.address)
            for instruction in hot
            if SCRATCH_LOAD_RE.search(" ; ".join(instruction.fragments))
        ],
        "scratch_rewind_addresses": [
            hex(instruction.address)
            for instruction in hot
            if SCRATCH_REWIND_RE.search(" ; ".join(instruction.fragments))
        ],
        "lane_groups": group_rows,
        "phase_group_sum": phase_group_sum_summary(instructions),
        "first_group_flow": first_group_flow(groups, hot),
        "inferred_register_roles": {
            "p0": "selected Q4NX weight buffer at microkernel call",
            "p1": "selected activation chunk buffer at microkernel call",
            "p2": "fixed local scratch/accumulator base 0x73c80",
            "p3": "phase-body group-sum scratch stream",
            "x11": "32-lane activation vector whose lanes feed vextbcst.16",
            "r7": "current 16-bit activation group sum loaded from p3",
        },
    }


def render_report(manifest: dict[str, object]) -> str:
    key_static = manifest["key_static_counts"]
    key_dynamic = manifest["key_dynamic_counts"]
    lines = [
        "# Q4NX Hot Body Schedule",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Source ELF: `{manifest['source_elf']}`",
        f"- Hot loop: `{manifest['ranges']['hot_loop'][0]}..{manifest['ranges']['hot_loop'][1]}`",
        f"- Loop count: `{manifest['loop_count']}`",
        f"- Hot instruction lines: `{manifest['hot_instruction_lines']}`",
        f"- Hot op slots: `{manifest['hot_op_slots']}`",
        "",
        "## Key Counts",
        "",
        "| op | static | dynamic |",
        "| --- | ---: | ---: |",
    ]
    for op, static_count in key_static.items():
        lines.append(f"| `{op}` | `{static_count}` | `{key_dynamic[op]}` |")
    lines.extend(
        [
            "",
            "## Activation Lane Groups",
            "",
            "| group | range | lanes | complete | vmac.f | vextbcst.16 | vups.4x | vconv.bf16.fp32 | vst |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in manifest["lane_groups"]:
        counts = group["counts"]
        lines.append(
            f"| `{group['index']}` | `{group['start']}..{group['end']}` | `{group['lane_count']}` | "
            f"`{group['lanes_complete']}` | `{counts['vmac.f']}` | `{counts['vextbcst.16']}` | "
            f"`{counts['vups.4x']}` | `{counts['vconv.bf16.fp32']}` | `{counts['vst']}` |"
        )
    phase = manifest["phase_group_sum"]
    lines.extend(
        [
            "",
            "## Phase Group-Sum Producer",
            "",
            f"- Activation vector loads: `{phase['activation_vector_loads']}`",
            f"- Scalar extracts: `{phase['scalar_extracts']}`",
            f"- Scratch stores: `{phase['scratch_stores']}`",
            "",
            "## Register Roles",
            "",
        ]
    )
    for register, meaning in manifest["inferred_register_roles"].items():
        lines.append(f"- `{register}`: {meaning}")
    lines.extend(
        [
            "",
            "## First Group Flow",
            "",
            "This is a best-effort def/use table from the first activation-lane group. "
            "Memory side effects and packed vector aliases still need manual review.",
            "",
            "| address | op | defs | uses | fragment |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in manifest["first_group_flow"][:80]:
        defs = ", ".join(row["defs"])
        uses = ", ".join(row["uses"])
        fragment = str(row["fragment"]).replace("|", "\\|")
        lines.append(f"| `{row['address']}` | `{row['op']}` | `{defs}` | `{uses}` | `{fragment}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The MyLM hot body is not just a `vextbcst.16` probe. It has a fixed "
            "eight-group activation-lane schedule, no hot-loop `vst` spill, and a "
            "separate phase-body group-sum producer feeding `p3`. Matching this shape "
            "requires preserving the register lifetime plan across group boundaries, "
            "not only replacing individual broadcasts in the current IRON body.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        manifest = build_manifest()
    except Exception:
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Hot Body Schedule\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
