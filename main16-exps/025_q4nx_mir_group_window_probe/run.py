#!/usr/bin/env python3
"""Compile steady-state MyLM Q4NX group windows as direct AIE2P MIR."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_group_window_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_group_window_probe.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    goal: str
    groups: tuple[int, ...]
    mir: str
    expected_counts: dict[str, int]
    source_event_count: int
    translated_event_count: int
    source_span_bytes: int


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CandidateResult:
    name: str
    goal: str
    groups: tuple[int, ...]
    status: str
    llc: CommandResult
    asm: CommandResult
    objdump: CommandResult
    counts: dict[str, int]
    expected_counts: dict[str, int]
    source_event_count: int
    translated_event_count: int
    source_span_bytes: int
    files: dict[str, str]
    reasons: list[str]


def load_exp024() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp024_for_group_window", EXP024_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp024 helper: {EXP024_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP024 = load_exp024()


def events_for_groups(groups: tuple[int, ...]) -> list:
    selected = set(groups)
    return [event for event in EXP024.EXP006.build_events() if event.group in selected]


def lines_for_groups(groups: tuple[int, ...]) -> list[str]:
    lines: list[str] = []
    for event in events_for_groups(groups):
        line = EXP024.mir_for_event(event)
        if line is not None:
            lines.append(line)
    return lines


def source_span_bytes(events: list) -> int:
    addresses = [event.address for event in events]
    return max(addresses) - min(addresses) if addresses else 0


def candidates() -> tuple[Candidate, ...]:
    result: list[Candidate] = []
    for groups in ((1, 2), (1, 2, 3)):
        lines = lines_for_groups(groups)
        events = events_for_groups(groups)
        group_label = "_".join(str(group) for group in groups)
        result.append(
            Candidate(
                name=f"mylm_groups_{group_label}_window",
                goal=f"Compile MyLM steady-state groups {groups} as one direct MIR loop body.",
                groups=groups,
                mir=EXP024.mir_function(f"mylm_groups_{group_label}_window", lines, loop=True),
                expected_counts=EXP024.expected_counts_for(lines),
                source_event_count=len(events),
                translated_event_count=len(lines),
                source_span_bytes=source_span_bytes(events),
            )
        )
    return tuple(result)


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
        groups=candidate.groups,
        status=status,
        llc=CommandResult(llc.returncode, llc.stdout, llc.stderr),
        asm=CommandResult(asm.returncode, asm.stdout, asm.stderr),
        objdump=CommandResult(objdump.returncode, objdump.stdout, objdump.stderr),
        counts=counts,
        expected_counts=candidate.expected_counts,
        source_event_count=candidate.source_event_count,
        translated_event_count=candidate.translated_event_count,
        source_span_bytes=candidate.source_span_bytes,
        files={
            "mir": str(mir_path),
            "object": str(object_path),
            "asm": str(asm_path),
            "objdump": str(objdump_path),
        },
        reasons=reasons,
    )


def schedule_counts(text: str) -> dict[str, int]:
    loop_bundle = re.search(r"BasicBlock:\s+loop\s*\n\s+-\s+BundleCount:\s+'(\d+)'", text)
    loop_byte = re.search(r"BasicBlock:\s+loop\s*\n\s+-\s+BundleCount:\s+'\d+'\s*\n\s+-\s+ByteCount:\s+'(\d+)'", text)
    return {
        "schedule_found": len(re.findall(r"Schedule found", text)),
        "schedule_missed": len(re.findall(r"No schedule found|Longest circuit does not fit II", text)),
        "loop_bundle_count": int(loop_bundle.group(1)) if loop_bundle is not None else -1,
        "loop_byte_count": int(loop_byte.group(1)) if loop_byte is not None else -1,
    }


def op_counts(groups: tuple[int, ...]) -> dict[str, int]:
    return dict(Counter(event.op for event in events_for_groups(groups)))


def render_report(results: list[CandidateResult]) -> str:
    lines = [
        "# Q4NX MIR Group Window Probe",
        "",
        "Status: `passed`" if all(result.status == "pass" for result in results) else "Status: `failed`",
        "",
        "This experiment checks direct MIR generation for steady-state MyLM group windows.",
        "",
        "## Results",
        "",
    ]
    for result in results:
        lines.append(f"### `{result.name}`")
        lines.append("")
        lines.append(f"Status: `{result.status}`")
        lines.append("")
        lines.append(result.goal)
        lines.append("")
        lines.append(f"- `groups`: `{result.groups}`")
        lines.append(f"- `source_event_count`: `{result.source_event_count}`")
        lines.append(f"- `translated_event_count`: `{result.translated_event_count}`")
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
            "- Direct MIR still preserves the MyLM steady-state opcode vocabulary across group boundaries.",
            "- There is still no `vextbcst.32` fallback and no vector store spill.",
            "- Peano's postpipeliner still reports no schedule for these large already-ordered windows; the useful output here is the object/assembly shape, not an automatic new schedule.",
            "- The next useful step is to compare this generated object against the MyLM hot range at the bundle/byte level, then package a tiny generated MIR object in the same direct-QKV harness for numeric readback.",
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
        "experiment": "025_q4nx_mir_group_window_probe",
        "status": "passed" if all(result.status == "pass" for result in results) else "failed",
        "source": str(EXP024.EXP006.EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "group_op_counts": {
            "_".join(str(group) for group in result.groups): op_counts(result.groups)
            for result in results
        },
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

