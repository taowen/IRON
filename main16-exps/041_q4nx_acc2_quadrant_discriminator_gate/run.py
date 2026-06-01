#!/usr/bin/env python3
"""Compare acc2 source mutations against MyLM reference on asymmetric zero cases."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_acc2_quadrant_discriminator_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_acc2_quadrant_discriminator_gate.md"

SITE_ADDRESS = 0x700
ZERO_LOW_ONLY = 0x00003C80
ZERO_HIGH_ONLY = 0x3C800000

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Replacement:
    name: str
    asm: str
    bytes_hex: str


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    record: tuple[str, ...]
    runtime_status: str
    runtime_output: str


@dataclass(frozen=True)
class MutationCase:
    name: str
    reference_record: tuple[str, ...]
    observed_record: tuple[str, ...]
    runtime_status: str
    matches_reference: bool
    runtime_output: str


@dataclass(frozen=True)
class ReplacementResult:
    name: str
    asm: str
    replacement_bytes: str
    original_bytes: str
    patched_raw: str
    cases: tuple[MutationCase, ...]

    @property
    def matches_reference(self) -> bool:
        return all(case.matches_reference for case in self.cases)


@dataclass(frozen=True)
class GateManifest:
    status: str
    site_address: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    reference_cases: tuple[ReferenceCase, ...]
    results: tuple[ReplacementResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_acc2_quadrant_discriminator", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_acc2_quadrant_discriminator", EXP029_RUN)
TinyCase = EXP015.TinyCase


REPLACEMENTS = (
    Replacement("from_bmll2", "vmov bmhh1, bmll2", "f812c819"),
    Replacement("from_bmlh2", "vmov bmhh1, bmlh2", "f812c919"),
    Replacement("from_bmhl2", "vmov bmhh1, bmhl2", "f812ca19"),
)


def configure_helpers(patched_raw: Path | None = None) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    if patched_raw is not None:
        EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def discriminator_cases(max_cases: int) -> tuple:
    cases: list = []
    for zero_index in range(16):
        cases.append(
            TinyCase(
                f"zero{zero_index:02d}_low_half",
                EXP015.all_scale(),
                (zero_index,),
                EXP015.all_q4(),
                0x11111111,
                ZERO_LOW_ONLY,
            )
        )
        cases.append(
            TinyCase(
                f"zero{zero_index:02d}_high_half",
                EXP015.all_scale(),
                (zero_index,),
                EXP015.all_q4(),
                0x11111111,
                ZERO_HIGH_ONLY,
            )
        )
    if max_cases > 0:
        return tuple(cases[:max_cases])
    return tuple(cases)


def selected_replacements(max_replacements: int) -> tuple[Replacement, ...]:
    if max_replacements > 0:
        return REPLACEMENTS[:max_replacements]
    return REPLACEMENTS


def build_direct_qkv(patched_raw: Path | None) -> None:
    configure_helpers(patched_raw)
    EXP015.EXP008.build_qkv_direct_xclbin()


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


def run_raw_case(case, timeout: int) -> tuple[str, tuple[str, ...], str]:
    runtime = EXP015.run_npu_case(
        EXP015.EXP008.EXP005.EXP132.EXP130.XCLBIN,
        EXP015.EXP008.EXP005.EXP132.EXP130.INSTS,
        case,
        EXP015.EXP008.RECORDS,
        timeout,
    )
    return runtime["status"], tuple(runtime["record"]), runtime["runtime_output"]


def run_reference_cases(cases: tuple, timeout: int) -> tuple[ReferenceCase, ...]:
    build_direct_qkv(None)
    rows: list[ReferenceCase] = []
    for case in cases:
        status, record, output = run_raw_case(case, timeout)
        rows.append(ReferenceCase(case.name, record, status, output))
    return tuple(rows)


def run_replacement(
    replacement: Replacement,
    cases: tuple,
    references: dict[str, tuple[str, ...]],
    timeout: int,
) -> ReplacementResult:
    patched, original = make_patched_raw(replacement)
    build_direct_qkv(patched)
    rows: list[MutationCase] = []
    for case in cases:
        status, record, output = run_raw_case(case, timeout)
        reference = references[case.name]
        rows.append(
            MutationCase(
                name=case.name,
                reference_record=reference,
                observed_record=record,
                runtime_status=status,
                matches_reference=status == "record_observed" and record == reference,
                runtime_output=output,
            )
        )
    return ReplacementResult(
        name=replacement.name,
        asm=replacement.asm,
        replacement_bytes=replacement.bytes_hex,
        original_bytes=original,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        cases=tuple(rows),
    )


def build_manifest(args: argparse.Namespace) -> GateManifest:
    configure_helpers()
    before = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return GateManifest("skipped_topology_not_ok", hex(SITE_ADDRESS), before, None, (), ())
    cases = discriminator_cases(args.max_cases)
    reference_cases = run_reference_cases(cases, args.runtime_timeout)
    references = {case.name: case.record for case in reference_cases}
    results = tuple(
        run_replacement(replacement, cases, references, args.runtime_timeout)
        for replacement in selected_replacements(args.max_replacements)
    )
    after = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    reference_ok = all(case.runtime_status == "record_observed" for case in reference_cases)
    status = (
        "passed"
        if after["status"] == "ok" and reference_ok and all(result.matches_reference for result in results)
        else "failed"
    )
    return GateManifest(status, hex(SITE_ADDRESS), before, after, reference_cases, results)


def render_report(manifest: GateManifest) -> str:
    lines = [
        "# Q4NX Acc2 Quadrant Discriminator Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- site: `{manifest.site_address}`",
        f"- reference cases: `{len(manifest.reference_cases)}`",
        "",
    ]
    if not manifest.results:
        lines.append("No replacement was run.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Results",
            "",
            "| replacement | cases | all match reference | failed cases | first failed observed |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for result in manifest.results:
        failed = tuple(case for case in result.cases if not case.matches_reference)
        failed_text = ", ".join(case.name for case in failed[:8]) if failed else "none"
        observed = ", ".join(failed[0].observed_record[:8]) if failed else ""
        lines.append(
            f"| `{result.name}` | `{len(result.cases)}` | `{result.matches_reference}` | "
            f"`{failed_text}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The reference is the unmodified MyLM raw program, not the scalar formula.",
            "- A pass means the asymmetric zero cases still cannot distinguish the tested `acc2` source quadrants at this site.",
            "- A fail identifies the first case that can distinguish the quadrants and should become part of the default source-mutation gate.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-replacements", type=int, default=0)
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
