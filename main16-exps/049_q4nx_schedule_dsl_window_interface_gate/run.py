#!/usr/bin/env python3
"""Check local schedule-window interfaces above generated MIR."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP048_RUN = REPO_ROOT / "main16-exps/048_q4nx_schedule_dsl_resource_manifest/run.py"
MANIFEST = EXPERIMENT_DIR / "q4nx_schedule_dsl_window_interface_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_schedule_dsl_window_interface_gate.md"

REGISTER_RE = re.compile(r"\$[A-Za-z0-9_]+")


@dataclass(frozen=True)
class OperationIo:
    defs: tuple[str, ...]
    uses: tuple[str, ...]


@dataclass(frozen=True)
class WindowInterface:
    start: str
    end: str
    live_ins: tuple[str, ...]
    boundary_defs: tuple[str, ...]


@dataclass(frozen=True)
class InterfaceDiff:
    field: str
    original: tuple[str, ...]
    candidate: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    replacements: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]
    expected_interface_ok: bool
    expected_resource_ok: bool


@dataclass(frozen=True)
class CandidateResult:
    name: str
    expected_interface_ok: bool
    interface_ok: bool
    expected_resource_ok: bool
    resource_ok: bool
    window: WindowInterface
    diffs: tuple[InterfaceDiff, ...]
    resource_violations: tuple


@dataclass(frozen=True)
class WindowInterfaceManifest:
    status: str
    candidates: tuple[CandidateResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP048 = load_module("mylm_exp048_for_window_interface", EXP048_RUN)


def operation_io(line: str) -> OperationIo:
    stripped = line.strip()
    if " = " not in stripped:
        return OperationIo((), ())
    left, right = stripped.split(" = ", 1)
    defs = [register for register in REGISTER_RE.findall(left)]
    uses: list[str] = []
    tokens = right.split(",")
    for token in tokens:
        registers = REGISTER_RE.findall(token)
        if not registers:
            continue
        if "implicit-def" in token:
            defs.extend(registers)
        else:
            uses.extend(registers)
    return OperationIo(tuple(dict.fromkeys(defs)), tuple(dict.fromkeys(uses)))


def bundle_window(program, replacements: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]):
    changed = tuple(sorted(replacements))
    start = changed[0]
    end = changed[-1]
    return tuple(bundle for bundle in program.bundles if start <= bundle.address <= end)


def window_interface(bundles) -> WindowInterface:
    defined: set[str] = set()
    live_ins: set[str] = set()
    boundary_defs: set[str] = set()
    for bundle in bundles:
        for operation in bundle.operations:
            io = operation_io(operation.text)
            for register in io.uses:
                if register not in defined:
                    live_ins.add(register)
            for register in io.defs:
                defined.add(register)
                boundary_defs.add(register)
    return WindowInterface(
        hex(bundles[0].address),
        hex(bundles[-1].address),
        tuple(sorted(live_ins)),
        tuple(sorted(boundary_defs)),
    )


def interface_diffs(original: WindowInterface, candidate: WindowInterface) -> tuple[InterfaceDiff, ...]:
    diffs: list[InterfaceDiff] = []
    if original.live_ins != candidate.live_ins:
        diffs.append(InterfaceDiff("live_ins", original.live_ins, candidate.live_ins))
    if original.boundary_defs != candidate.boundary_defs:
        diffs.append(InterfaceDiff("boundary_defs", original.boundary_defs, candidate.boundary_defs))
    return tuple(diffs)


CANDIDATES = (
    Candidate("highhalf_swap_045", EXP048.CANDIDATES[0].replacements, True, True),
    Candidate("vups_advance_046", EXP048.CANDIDATES[1].replacements, True, False),
    Candidate(
        "drop_bmhh_copy_negative_control",
        {0x700: ((EXP048.BMHH_COPY,), ("      NOP",))},
        False,
        True,
    ),
)


def run_candidate(candidate: Candidate) -> CandidateResult:
    original = EXP048.original_program()
    resources = EXP048.resource_manifest(original)
    changed = original.with_replacements(candidate.replacements)
    original_interface = window_interface(bundle_window(original, candidate.replacements))
    candidate_interface = window_interface(bundle_window(changed, candidate.replacements))
    diffs = interface_diffs(original_interface, candidate_interface)
    resource_violations = resources.validate(changed, original)
    return CandidateResult(
        candidate.name,
        candidate.expected_interface_ok,
        not diffs,
        candidate.expected_resource_ok,
        not resource_violations,
        candidate_interface,
        diffs,
        resource_violations,
    )


def build_manifest() -> WindowInterfaceManifest:
    results = tuple(run_candidate(candidate) for candidate in CANDIDATES)
    status = "passed" if all(
        result.interface_ok == result.expected_interface_ok and result.resource_ok == result.expected_resource_ok
        for result in results
    ) else "failed"
    return WindowInterfaceManifest(status, results)


def render_tuple(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def render_diffs(diffs: tuple[InterfaceDiff, ...]) -> str:
    if not diffs:
        return ""
    rows = []
    for diff in diffs:
        rows.append(f"{diff.field}: {render_tuple(diff.original)} -> {render_tuple(diff.candidate)}")
    return "<br>".join(rows)


def render_report(manifest: WindowInterfaceManifest) -> str:
    lines = [
        "# Q4NX Schedule DSL Window Interface Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        "| candidate | interface ok | resource ok | window | diffs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in manifest.candidates:
        lines.append(
            f"| `{result.name}` | `{result.interface_ok}` | `{result.resource_ok}` | "
            f"`{result.window.start}..{result.window.end}` | `{render_diffs(result.diffs)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The high-half swap keeps the same local window interface and passes the resource manifest.",
            "- The `vups.4x` advance keeps the same textual live-in/def interface, but is still rejected by the resource manifest.",
            "- The negative control drops the high-half copy; the resource manifest allows a single `NOP`, but the window interface rejects the changed live-ins and boundary defs.",
            "- This confirms that the DSL needs both checks: resource legality and window interface preservation catch different classes of bad schedules.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
