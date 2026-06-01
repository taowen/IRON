#!/usr/bin/env python3
"""Sweep explicit MyLM hot-loop padding gaps with the same no-op byte mutation."""

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


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP015_RUN = REPO_ROOT / "main16-exps/015_q4nx_tiny_codegen_numeric_gate/run.py"
EXP029_RUN = REPO_ROOT / "main16-exps/029_q4nx_mir_prebundled_exact_replay/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_padding_gap_sweep.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_padding_gap_sweep.md"

ORIGINAL_BYTES = bytes.fromhex("00000000")
REPLACEMENT_BYTES = bytes.fromhex("f8a0df1f")
REPLACEMENT_ASM = "mov r31, r31"


@dataclass(frozen=True)
class MutationSite:
    name: str
    address: int
    reason: str


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
    name: str
    address: str
    reason: str
    original_bytes: str
    replacement_bytes: str
    replacement_asm: str
    patched_raw: str
    runtime: RuntimeCase


SITES = (
    MutationSite("gap_before_0x2c2", 0x2BA, "first explicit padding gap before source address 0x2c2"),
    MutationSite("gap_before_0x1828", 0x1820, "late explicit padding gap before source address 0x1828"),
    MutationSite("gap_before_0x1844", 0x183C, "late explicit padding gap before source address 0x1844"),
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_padding_gap_sweep", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_padding_gap_sweep", EXP029_RUN)


def original_raw() -> bytes:
    return EXP029.RAW_PROGRAM.read_bytes()


def make_patched_raw(site: MutationSite) -> Path:
    build = BUILD_DIR / "patched_raw" / site.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    raw = bytearray(original_raw())
    found = bytes(raw[site.address : site.address + len(ORIGINAL_BYTES)])
    if found != ORIGINAL_BYTES:
        raise RuntimeError(
            f"{site.name} has unexpected bytes at {hex(site.address)}: "
            f"got {found.hex()}, expected {ORIGINAL_BYTES.hex()}"
        )
    raw[site.address : site.address + len(REPLACEMENT_BYTES)] = REPLACEMENT_BYTES
    patched = build / f"mylm_c2r2_program.{site.name}.bin"
    patched.write_bytes(raw)
    return patched


def configure_helpers(patched_raw: Path) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


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


def run_site(site: MutationSite, timeout: int) -> SiteResult:
    patched = make_patched_raw(site)
    build_patched_direct_qkv(patched)
    runtime = run_case(EXP015.cases()[0], timeout)
    return SiteResult(
        name=site.name,
        address=hex(site.address),
        reason=site.reason,
        original_bytes=ORIGINAL_BYTES.hex(),
        replacement_bytes=REPLACEMENT_BYTES.hex(),
        replacement_asm=REPLACEMENT_ASM,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        runtime=runtime,
    )


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    results = [run_site(site, args.runtime_timeout) for site in SITES]
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    status = "passed" if after["status"] == "ok" and all(result.runtime.matches_expected for result in results) else "failed"
    return {
        "status": status,
        "topology_before": before,
        "topology_after": after,
        "results": [asdict(result) for result in results],
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Q4NX MIR Padding Gap Sweep",
        "",
        f"Status: `{manifest['status']}`",
        "",
    ]
    if "results" not in manifest:
        lines.append("The runtime topology was not healthy enough to run this gate.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Mutation",
            "",
            f"- original bytes: `{ORIGINAL_BYTES.hex()}`",
            f"- replacement bytes: `{REPLACEMENT_BYTES.hex()}`",
            f"- replacement asm: `{REPLACEMENT_ASM}`",
            "",
            "## Results",
            "",
            "| site | address | matches expected | observed first words |",
            "| --- | --- | --- | --- |",
        ]
    )
    for result in manifest["results"]:
        runtime = result["runtime"]
        observed = ", ".join(runtime["observed_record"][:4])
        lines.append(
            f"| `{result['name']}` | `{result['address']}` | "
            f"`{runtime['matches_expected']}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The same self-move byte mutation is applied to each explicit hot-loop padding gap independently.",
            "- Any payload mismatch means that gap is a timing contract, not spare code space.",
            "- A passing site can be used as a byte-diff packaging canary, but not as evidence that real Q4NX bundles are safe to reorder.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
