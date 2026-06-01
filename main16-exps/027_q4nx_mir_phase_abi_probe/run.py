#!/usr/bin/env python3
"""Identify the ABI layer required to run the generated Q4NX MIR hot body."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
LLVM_OBJDUMP = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump"
MYLM_ELF = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_phase_abi_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_phase_abi_probe.md"


@dataclass(frozen=True)
class RangeSummary:
    name: str
    start: str
    end: str
    instruction_lines: int
    op_counts: dict[str, int]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


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


def disassemble(start: int, end: int) -> str:
    return run_cmd(
        (
            str(LLVM_OBJDUMP),
            "-d",
            f"--start-address=0x{start:x}",
            f"--stop-address=0x{end:x}",
            str(MYLM_ELF),
        )
    )


def instruction_payloads(text: str) -> tuple[str, ...]:
    payloads: list[str] = []
    for line in text.splitlines():
        if not re.match(r"^\s*[0-9a-fA-F]+:", line):
            continue
        payload = line.split("\t", 1)[1] if "\t" in line else ""
        payloads.extend(fragment.strip() for fragment in payload.split(";") if fragment.strip())
    return tuple(payloads)


def op_name(fragment: str) -> str:
    match = re.match(r"^\s*([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)", fragment)
    return match.group(1) if match is not None else ""


def summarize(name: str, start: int, end: int) -> RangeSummary:
    text = disassemble(start, end)
    payloads = instruction_payloads(text)
    counts = Counter(op_name(fragment) for fragment in payloads if op_name(fragment))
    return RangeSummary(
        name=name,
        start=hex(start),
        end=hex(end),
        instruction_lines=len(re.findall(r"^\s*[0-9a-fA-F]+:", text, re.MULTILINE)),
        op_counts=dict(sorted(counts.items())),
    )


def contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def checks(prologue: str, qkv_setup: str, call_site: str, record_emit: str) -> tuple[Check, ...]:
    return (
        Check("prologue_sets_lc_2", contains(prologue, r"\bmova\s+lc,\s*#0x2\b"), "0x01f0 prologue owns hot-loop LC setup"),
        Check("prologue_sets_loop_bounds", contains(prologue, r"\bmovxm\s+ls,\s*#0x260\b") and contains(prologue, r"\bmovxm\s+le,\s*#0x1850\b"), "0x01f0 prologue sets LS/LE to the hot loop range"),
        Check("prologue_rebases_weight_pointer", contains(prologue, r"\bpaddb\s+\[p0\],\s*m0\b"), "0x01f0 advances p0 by 0x400 before hot loop"),
        Check("prologue_sets_constants", all(contains(prologue, pattern) for pattern in (r"\bmova\s+r1,\s*#0x200\b", r"\bmova\s+r3,\s*#0x2\b", r"\bmova\s+r4,\s*#0x33c\b", r"\bmova\s+r5,\s*#0x3c\b", r"\bmov\s+s0,\s*#0x0\b")), "hot loop constants are not provided by the external caller"),
        Check("qkv_setup_writes_group_sums", len(re.findall(r"\bst\.s16\b", qkv_setup)) == 8, "Q/K/V phase body writes 8 group-sum halfwords before calling 0x01f0"),
        Check("call_site_calls_microkernel", contains(call_site, r"\bjl\s+#0x1f0\b"), "Q/K/V phase body calls the shared Q4NX microkernel"),
        Check("call_site_sets_p0_weight", contains(call_site, r"\bmov\s+p0,\s*r18\b"), "delay slot maps p0 to selected Q4NX weight buffer"),
        Check("call_site_sets_p1_activation", contains(call_site, r"\bmovs\s+p1,\s*r16\b"), "delay slot maps p1 to selected activation chunk buffer"),
        Check("call_site_sets_p2_scratch", contains(call_site, r"\bmovxm\s+p2,\s*#0x73c80\b"), "delay slot maps p2 to scratch/control base"),
        Check("call_site_sets_p3_group_sums", contains(call_site, r"\bmovs\s+p3,\s*r15\b"), "delay slot maps p3 to group-sum scratch stream"),
        Check("record_emit_uses_q4_result", contains(record_emit, r"\bvst\.conv\.bf16\.fp32\b"), "post-call phase body emits compact record payload from q4 result state"),
    )


def render_report(
    status: str,
    range_summaries: tuple[RangeSummary, ...],
    check_rows: tuple[Check, ...],
) -> str:
    lines = [
        "# Q4NX MIR Phase ABI Probe",
        "",
        f"Status: `{status}`",
        "",
        "This experiment explains why the full hot-loop MIR object from experiment 026 is not yet directly runnable.",
        "The generated MIR covers `0x260..0x1850`; MyLM enters it through a `0x01f0` prologue and the Q/K/V phase body.",
        "",
        "## Checks",
        "",
        "| check | pass | detail |",
        "| --- | --- | --- |",
    ]
    for check in check_rows:
        lines.append(f"| `{check.name}` | `{check.passed}` | {check.detail} |")
    lines.extend(["", "## Range Summaries", ""])
    for summary in range_summaries:
        lines.append(f"### `{summary.name}`")
        lines.append("")
        lines.append(f"- range: `{summary.start}..{summary.end}`")
        lines.append(f"- instruction_lines: `{summary.instruction_lines}`")
        for op, count in summary.op_counts.items():
            lines.append(f"- `{op}`: `{count}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Experiment 026 is a valid object-shape proof, but it starts too late for numeric execution.",
            "- A runnable generated kernel must either include an equivalent `0x01f0` prologue or generate a function whose caller sets the same LC/LS/LE/constants/control registers.",
            "- It must also include the phase-body group-sum producer and the post-call compact-record emitter, or reuse the MyLM phase body while replacing only the shared microkernel.",
            "- The next experiment should therefore package `prologue + generated hot loop + record emit` as one tiny direct-QKV program, not call the hot-loop body alone.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    if not LLVM_OBJDUMP.exists():
        raise FileNotFoundError(LLVM_OBJDUMP)
    if not MYLM_ELF.exists():
        raise FileNotFoundError(MYLM_ELF)
    prologue = disassemble(0x1F0, 0x260)
    qkv_setup = disassemble(0x1A1E, 0x1D70)
    call_site = disassemble(0x1D70, 0x1D8E)
    record_emit = disassemble(0x1D8C, 0x1E12)
    check_rows = checks(prologue, qkv_setup, call_site, record_emit)
    status = "passed" if all(check.passed for check in check_rows) else "failed"
    range_summaries = (
        summarize("microkernel_prologue", 0x1F0, 0x260),
        summarize("qkv_group_sum_setup", 0x1A1E, 0x1D70),
        summarize("qkv_call_delay_slots", 0x1D70, 0x1D8E),
        summarize("qkv_record_emit", 0x1D8C, 0x1E12),
    )
    manifest = {
        "experiment": "027_q4nx_mir_phase_abi_probe",
        "status": status,
        "source": str(MYLM_ELF.relative_to(REPO_ROOT)),
        "checks": [asdict(check) for check in check_rows],
        "ranges": [asdict(summary) for summary in range_summaries],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(status, range_summaries, check_rows), encoding="utf-8")


def main() -> int:
    try:
        run()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
