#!/usr/bin/env python3
"""Compile the complete MyLM Q4NX hot loop as direct AIE2P MIR."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP024_RUN = REPO_ROOT / "main16-exps/024_q4nx_mir_opcode_coverage_map/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_full_hotloop_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_full_hotloop_probe.md"
EXPLICIT_NOOPS = {"nop", "nopa", "nopb"}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Candidate:
    name: str
    goal: str
    mir: str
    expected_counts: dict[str, int]
    source_event_count: int
    translated_event_count: int
    source_span_bytes: int


@dataclass(frozen=True)
class CandidateResult:
    name: str
    goal: str
    status: str
    llc: CommandResult
    asm: CommandResult
    objdump: CommandResult
    counts: dict[str, int]
    expected_counts: dict[str, int]
    source_event_count: int
    translated_event_count: int
    expected_translated_event_count: int
    source_span_bytes: int
    files: dict[str, str]
    reasons: list[str]


def load_exp024() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp024_for_full_hotloop", EXP024_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp024 helper: {EXP024_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP024 = load_exp024()


def full_hotloop_events() -> list:
    return EXP024.EXP006.build_events()


def full_hotloop_lines() -> list[str]:
    lines: list[str] = []
    for event in full_hotloop_events():
        line = EXP024.mir_for_event(event)
        if line is not None:
            lines.append(line)
    return lines


def source_span_bytes() -> int:
    events = full_hotloop_events()
    addresses = [event.address for event in events]
    return max(addresses) - min(addresses)


def expected_translated_event_count() -> int:
    return sum(1 for event in full_hotloop_events() if event.op not in EXPLICIT_NOOPS)


def candidates() -> tuple[Candidate, ...]:
    lines = full_hotloop_lines()
    return (
        Candidate(
            name="mylm_full_hotloop_straightline",
            goal="Compile the complete MyLM Q4NX hot-loop event order as one straight-line MIR block.",
            mir=EXP024.mir_function("mylm_full_hotloop_straightline", lines, loop=False),
            expected_counts=EXP024.expected_counts_for(lines),
            source_event_count=len(full_hotloop_events()),
            translated_event_count=len(lines),
            source_span_bytes=source_span_bytes(),
        ),
        Candidate(
            name="mylm_full_hotloop_zol",
            goal="Compile the complete MyLM Q4NX hot-loop event order inside a hardware-loop MIR block.",
            mir=EXP024.mir_function("mylm_full_hotloop_zol", lines, loop=True),
            expected_counts=EXP024.expected_counts_for(lines),
            source_event_count=len(full_hotloop_events()),
            translated_event_count=len(lines),
            source_span_bytes=source_span_bytes(),
        ),
    )


def schedule_counts(text: str) -> dict[str, int]:
    bundle_matches = [int(value) for value in re.findall(r"BundleCount:\s+'(\d+)'", text)]
    loop_bundle = re.search(r"BasicBlock:\s+loop\s*\n\s+-\s+BundleCount:\s+'(\d+)'", text)
    loop_byte = re.search(
        r"BasicBlock:\s+loop\s*\n\s+-\s+BundleCount:\s+'\d+'\s*\n\s+-\s+ByteCount:\s+'(\d+)'",
        text,
    )
    return {
        "schedule_found": len(re.findall(r"Schedule found", text)),
        "schedule_missed": len(re.findall(r"No schedule found|Longest circuit does not fit II", text)),
        "bundle_count_entries": len(bundle_matches),
        "loop_bundle_count": int(loop_bundle.group(1)) if loop_bundle is not None else -1,
        "loop_byte_count": int(loop_byte.group(1)) if loop_byte is not None else -1,
    }


def run_candidate(candidate: Candidate) -> CandidateResult:
    mir_path = BUILD_DIR / f"{candidate.name}.mir"
    object_path = BUILD_DIR / f"{candidate.name}.o"
    asm_path = BUILD_DIR / f"{candidate.name}.s"
    objdump_path = BUILD_DIR / f"{candidate.name}.objdump"
    mir_path.write_text(candidate.mir, encoding="utf-8")
    common_args = (
        "-verify-machineinstrs",
        "--mtriple=aie2p",
        "-O2",
        "--start-before=postmisched",
        "-pass-remarks-output=-",
        "-pass-remarks-filter=pipeliner|aie-asm-printer",
    )
    llc = EXP024.run_command((str(EXP024.LLC), *common_args, "--filetype=obj", str(mir_path), "-o", str(object_path)))
    asm = CommandResult(1, "", "object build did not complete")
    objdump = CommandResult(1, "", "object build did not complete")
    if llc.returncode == 0:
        asm = EXP024.run_command((str(EXP024.LLC), *common_args, str(mir_path), "-o", str(asm_path)))
        objdump = EXP024.run_command(
            (
                str(EXP024.OBJDUMP),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                str(object_path),
            )
        )
        objdump_path.write_text(objdump.stdout + objdump.stderr, encoding="utf-8")
    counts = EXP024.count_objdump(objdump.stdout + objdump.stderr) | schedule_counts(llc.stdout + llc.stderr)
    reasons: list[str] = []
    if llc.returncode != 0:
        reasons.append(f"llc failed with return code {llc.returncode}")
    if llc.returncode == 0 and objdump.returncode != 0:
        reasons.append(f"llvm-objdump failed with return code {objdump.returncode}")
    if candidate.translated_event_count != expected_translated_event_count():
        reasons.append(
            "translated event count mismatch: "
            f"expected {expected_translated_event_count()}, got {candidate.translated_event_count}"
        )
    for key, expected in candidate.expected_counts.items():
        got = counts.get(key, 0)
        if got != expected:
            reasons.append(f"expected {key}={expected}, got {got}")
    if counts["vextbcst_32"] != 0:
        reasons.append("unexpected vextbcst.32")
    if counts["vst"] != 0:
        reasons.append("unexpected vector store")
    status = "pass" if not reasons else "fail"
    return CandidateResult(
        name=candidate.name,
        goal=candidate.goal,
        status=status,
        llc=CommandResult(llc.returncode, llc.stdout, llc.stderr),
        asm=CommandResult(asm.returncode, asm.stdout, asm.stderr),
        objdump=CommandResult(objdump.returncode, objdump.stdout, objdump.stderr),
        counts=counts,
        expected_counts=candidate.expected_counts,
        source_event_count=candidate.source_event_count,
        translated_event_count=candidate.translated_event_count,
        expected_translated_event_count=expected_translated_event_count(),
        source_span_bytes=candidate.source_span_bytes,
        files={
            "mir": str(mir_path),
            "object": str(object_path),
            "asm": str(asm_path),
            "objdump": str(objdump_path),
        },
        reasons=reasons,
    )


def op_counts() -> dict[str, int]:
    return dict(Counter(event.op for event in full_hotloop_events()))


def render_report(results: list[CandidateResult]) -> str:
    lines = [
        "# Q4NX MIR Full Hotloop Probe",
        "",
        "Status: `passed`" if all(result.status == "pass" for result in results) else "Status: `failed`",
        "",
        "This experiment compiles the complete MyLM Q4NX hot-loop event stream as direct AIE2P MIR.",
        "It checks whether the llvm-aie route can preserve the full object-level instruction shape before any NPU numeric packaging.",
        "",
        "## Source Op Counts",
        "",
    ]
    for op, count in sorted(op_counts().items()):
        lines.append(f"- `{op}`: `{count}`")
    lines.extend(["", "## Results", ""])
    for result in results:
        lines.append(f"### `{result.name}`")
        lines.append("")
        lines.append(f"Status: `{result.status}`")
        lines.append("")
        lines.append(result.goal)
        lines.append("")
        lines.append(f"- `source_event_count`: `{result.source_event_count}`")
        lines.append(f"- `translated_event_count`: `{result.translated_event_count}`")
        lines.append(f"- `expected_translated_event_count`: `{result.expected_translated_event_count}`")
        lines.append(f"- `source_span_bytes`: `{result.source_span_bytes}`")
        lines.append(f"- `llc_returncode`: `{result.llc.returncode}`")
        lines.append(f"- `asm_returncode`: `{result.asm.returncode}`")
        lines.append(f"- `objdump_returncode`: `{result.objdump.returncode}`")
        for key, value in result.counts.items():
            lines.append(f"- `{key}`: `{value}`")
        if result.reasons:
            lines.append("")
            lines.append("Reasons:")
            for reason in result.reasons:
                lines.append(f"- {reason}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The direct MIR route now covers the complete hot-loop instruction vocabulary, including scalar pointer setup and drain instructions.",
            "- Passing this gate means the compiler can assemble the MyLM-shaped full hot body without falling back to `vextbcst.32` or spilling with `vst`.",
            "- This still is not a replacement kernel: the next gate is packaging the generated object into a direct-QKV numeric harness and comparing payloads against MyLM raw `0x1870`.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    if not EXP024.LLC.exists():
        raise FileNotFoundError(EXP024.LLC)
    if not EXP024.OBJDUMP.exists():
        raise FileNotFoundError(EXP024.OBJDUMP)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_candidate(candidate) for candidate in candidates()]
    manifest = {
        "experiment": "026_q4nx_mir_full_hotloop_probe",
        "status": "passed" if all(result.status == "pass" for result in results) else "failed",
        "source": str(EXP024.EXP006.EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "source_op_counts": op_counts(),
        "results": [asdict(result) for result in results],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(results), encoding="utf-8")


def main() -> int:
    try:
        run()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
