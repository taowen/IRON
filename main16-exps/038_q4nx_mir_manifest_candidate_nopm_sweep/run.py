#!/usr/bin/env python3
"""Sweep manifest-selected source vmov candidates with an MV-slot nopm patch."""

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
EXP033_RUN = REPO_ROOT / "main16-exps/033_q4nx_mir_bundle_mutation_manifest/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_manifest_candidate_nopm_sweep.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_manifest_candidate_nopm_sweep.md"

NOPM_BYTES = bytes.fromhex("f84a0318")
NOPM_ASM = "nopm"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class MutationSite:
    address: int
    original_bytes: str
    original_ops: tuple[str, ...]
    original_reason: str


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    expected_record: tuple[str, ...]
    observed_record: tuple[str, ...]
    runtime_status: str
    matches_expected: bool
    runtime_output: str


@dataclass(frozen=True)
class SiteResult:
    address: str
    original_bytes: str
    original_ops: tuple[str, ...]
    original_reason: str
    replacement_bytes: str
    replacement_asm: str
    patched_raw: str
    cases: tuple[RuntimeCase, ...]

    @property
    def matches_expected(self) -> bool:
        return all(case.matches_expected for case in self.cases)


@dataclass(frozen=True)
class SweepManifest:
    status: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    results: tuple[SiteResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_manifest_candidate_nopm_sweep", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_manifest_candidate_nopm_sweep", EXP029_RUN)
EXP033 = load_module("mylm_exp033_for_manifest_candidate_nopm_sweep", EXP033_RUN)


def configure_helpers(patched_raw: Path | None = None) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    if patched_raw is not None:
        EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def selected_sites(max_sites: int) -> tuple[MutationSite, ...]:
    manifest = EXP033.build_manifest()
    rows = manifest.weak_source_candidates
    if not rows:
        rows = tuple(row for row in manifest.known_failed_source_mutations if "experiment 038" in row.reason)
    sites: list[MutationSite] = []
    for row in rows:
        if row.length != 4 or row.ops != ("vmov",):
            continue
        sites.append(
            MutationSite(
                address=int(row.address, 16),
                original_bytes=row.bytes_hex,
                original_ops=row.ops,
                original_reason=row.reason,
            )
        )
    if max_sites > 0:
        return tuple(sites[:max_sites])
    return tuple(sites)


def selected_cases(max_cases: int) -> tuple:
    cases = EXP015.cases()
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def make_patched_raw(site: MutationSite) -> Path:
    build = BUILD_DIR / "patched_raw" / f"site_{site.address:x}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    raw = bytearray(EXP029.RAW_PROGRAM.read_bytes())
    original = bytes.fromhex(site.original_bytes)
    found = bytes(raw[site.address : site.address + len(original)])
    if found != original:
        raise RuntimeError(
            f"unexpected bytes at {hex(site.address)}: got {found.hex()}, expected {original.hex()}"
        )
    raw[site.address : site.address + len(NOPM_BYTES)] = NOPM_BYTES
    patched = build / f"mylm_c2r2_program.site_{site.address:x}.nopm.bin"
    patched.write_bytes(raw)
    return patched


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


def run_site(site: MutationSite, cases: tuple, timeout: int) -> SiteResult:
    patched = make_patched_raw(site)
    build_patched_direct_qkv(patched)
    runtime_cases = tuple(run_case(case, timeout) for case in cases)
    return SiteResult(
        address=hex(site.address),
        original_bytes=site.original_bytes,
        original_ops=site.original_ops,
        original_reason=site.original_reason,
        replacement_bytes=NOPM_BYTES.hex(),
        replacement_asm=NOPM_ASM,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        cases=runtime_cases,
    )


def build_manifest(args: argparse.Namespace) -> SweepManifest:
    configure_helpers()
    before = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return SweepManifest("skipped_topology_not_ok", before, None, ())
    cases = selected_cases(args.max_cases)
    results = tuple(run_site(site, cases, args.runtime_timeout) for site in selected_sites(args.max_sites))
    after = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    status = "passed" if after["status"] == "ok" and all(result.matches_expected for result in results) else "failed"
    return SweepManifest(status, before, after, results)


def render_report(manifest: SweepManifest) -> str:
    lines = [
        "# Q4NX MIR Manifest Candidate NOPM Sweep",
        "",
        f"Status: `{manifest.status}`",
        "",
    ]
    if not manifest.results:
        lines.append("No runnable candidate was selected.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Results",
            "",
            "| address | cases | all match | failed cases | observed first words |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for result in manifest.results:
        failed = tuple(case.name for case in result.cases if not case.matches_expected)
        observed = ", ".join(result.cases[0].observed_record[:4]) if result.cases else ""
        failed_text = ", ".join(failed) if failed else "none"
        lines.append(
            f"| `{result.address}` | `{len(result.cases)}` | `{result.matches_expected}` | "
            f"`{failed_text}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Each row is a real source-bundle byte mutation, not a byte-exact replay.",
            "- A passing row means the updated manifest found a source bundle that can be removed for the tested synthetic cases.",
            "- A failing row means the static dead-def model is still missing timing, alias, or data-dependent semantics for that site.",
            "- In the full sweep, every candidate failed only `zero0_allq4`, with the expected zero/offset payload one bf16 step higher than the observed payload.",
            "- This is still a source-bundle mutation gate, not a complete self-generated Q4NX hot block.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    parser.add_argument("--max-sites", type=int, default=0)
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
