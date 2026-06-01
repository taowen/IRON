#!/usr/bin/env python3
"""Run all direct-QKV cases for alternate acc2 source mutations at 0x700."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import cast


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP015_RUN = REPO_ROOT / "main16-exps/015_q4nx_tiny_codegen_numeric_gate/run.py"
EXP029_RUN = REPO_ROOT / "main16-exps/029_q4nx_mir_prebundled_exact_replay/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_acc2_alternate_source_numeric_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_acc2_alternate_source_numeric_gate.md"

SITE_ADDRESS = 0x700

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Replacement:
    name: str
    asm: str
    bytes_hex: str


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    expected_record: tuple[str, ...]
    observed_record: tuple[str, ...]
    runtime_status: str
    matches_expected: bool
    runtime_output: str


@dataclass(frozen=True)
class ReplacementResult:
    name: str
    asm: str
    replacement_bytes: str
    original_bytes: str
    patched_raw: str
    cases: tuple[RuntimeCase, ...]

    @property
    def matches_expected(self) -> bool:
        return all(case.matches_expected for case in self.cases)


@dataclass(frozen=True)
class GateManifest:
    status: str
    site_address: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    results: tuple[ReplacementResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_acc2_alternate_source_gate", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_acc2_alternate_source_gate", EXP029_RUN)


REPLACEMENTS = (
    Replacement("from_bmll2", "vmov bmhh1, bmll2", "f812c819"),
    Replacement("from_bmlh2", "vmov bmhh1, bmlh2", "f812c919"),
    Replacement("from_bmhl2", "vmov bmhh1, bmhl2", "f812ca19"),
    Replacement("original_bmhh2", "vmov bmhh1, bmhh2", "f812cb19"),
)


def configure_helpers(patched_raw: Path | None = None) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    if patched_raw is not None:
        EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def selected_replacements(max_replacements: int) -> tuple[Replacement, ...]:
    if max_replacements > 0:
        return REPLACEMENTS[:max_replacements]
    return REPLACEMENTS


def selected_cases(max_cases: int) -> tuple:
    cases = EXP015.cases()
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def make_patched_raw(replacement: Replacement) -> tuple[Path, str]:
    build = BUILD_DIR / "patched_raw" / replacement.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    raw = bytearray(EXP029.RAW_PROGRAM.read_bytes())
    original = bytes(raw[SITE_ADDRESS : SITE_ADDRESS + 4])
    raw[SITE_ADDRESS : SITE_ADDRESS + 4] = bytes.fromhex(replacement.bytes_hex)
    patched = build / f"mylm_c2r2_program.site_{SITE_ADDRESS:x}.{replacement.name}.bin"
    patched.write_bytes(raw)
    return patched, original.hex()


def build_patched_direct_qkv(patched_raw: Path) -> None:
    configure_helpers(patched_raw)
    EXP015.EXP008.build_qkv_direct_xclbin()


def run_case(case, timeout: int) -> RuntimeCase:
    expected = tuple(hex(word & 0xFFFFFFFF) for word in EXP015.expected_record_words(case))
    runtime = EXP015.run_npu_case(
        EXP015.EXP008.EXP005.EXP132.EXP130.XCLBIN,
        EXP015.EXP008.EXP005.EXP132.EXP130.INSTS,
        case,
        EXP015.EXP008.RECORDS,
        timeout,
    )
    observed = tuple(runtime["record"])
    return RuntimeCase(
        name=case.name,
        expected_record=expected,
        observed_record=observed,
        runtime_status=runtime["status"],
        matches_expected=runtime["status"] == "record_observed" and observed == expected,
        runtime_output=runtime["runtime_output"],
    )


def run_replacement(replacement: Replacement, cases: tuple, timeout: int) -> ReplacementResult:
    patched, original = make_patched_raw(replacement)
    build_patched_direct_qkv(patched)
    runtime_cases = tuple(run_case(case, timeout) for case in cases)
    return ReplacementResult(
        name=replacement.name,
        asm=replacement.asm,
        replacement_bytes=replacement.bytes_hex,
        original_bytes=original,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        cases=runtime_cases,
    )


def build_manifest(args: argparse.Namespace) -> GateManifest:
    configure_helpers()
    before = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return GateManifest("skipped_topology_not_ok", hex(SITE_ADDRESS), before, None, ())
    cases = selected_cases(args.max_cases)
    results = tuple(
        run_replacement(replacement, cases, args.runtime_timeout)
        for replacement in selected_replacements(args.max_replacements)
    )
    after = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    status = "passed" if after["status"] == "ok" and all(result.matches_expected for result in results) else "failed"
    return GateManifest(status, hex(SITE_ADDRESS), before, after, results)


def render_report(manifest: GateManifest) -> str:
    lines = [
        "# Q4NX Acc2 Alternate Source Numeric Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- site: `{manifest.site_address}`",
        "",
    ]
    if not manifest.results:
        lines.append("No replacement was run.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Results",
            "",
            "| replacement | asm | cases | all match | failed cases | observed first words |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for result in manifest.results:
        failed = tuple(case.name for case in result.cases if not case.matches_expected)
        failed_text = ", ".join(failed) if failed else "none"
        observed = ", ".join(result.cases[-1].observed_record[:6]) if result.cases else ""
        lines.append(
            f"| `{result.name}` | `{result.asm}` | `{len(result.cases)}` | "
            f"`{result.matches_expected}` | `{failed_text}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A passing alternate source is a real non-byte-copy source-bundle mutation under the current direct-QKV synthetic gate.",
            "- This does not make the mutation a performance optimization; it only proves we can change a source bundle without breaking current numeric coverage.",
            "- If all `acc2` quadrants pass, the synthetic cases do not distinguish the quadrant value at this template point.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    parser.add_argument("--max-replacements", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
        MANIFEST.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
