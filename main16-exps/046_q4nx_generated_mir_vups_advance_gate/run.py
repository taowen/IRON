#!/usr/bin/env python3
"""Advance one vups.4x bundle and run MyLM-reference NPU gates."""

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
EXP042_RUN = REPO_ROOT / "main16-exps/042_q4nx_generated_mir_hotloop_variant_gate/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_generated_mir_vups_advance_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_generated_mir_vups_advance_gate.md"

WINDOW_START = 0x700
WINDOW_END = 0x704
VUPS4X = "      $dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0"
BMHH_COPY = "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2"
VADD_DM2 = "      $dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0"
BUNDLE_REPLACEMENTS = {
    0x700: ((BMHH_COPY,), (VUPS4X,)),
    0x704: ((VUPS4X, VADD_DM2), (BMHH_COPY, VADD_DM2)),
}

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class BundleReplacement:
    address: str
    original_mir: tuple[str, ...]
    variant_mir: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedVupsAdvanceVariant:
    window_start: str
    window_end: str
    replacements: tuple[BundleReplacement, ...]
    unpadded_mir: str
    padded_mir: str
    object: str
    objdump: str
    patched_raw: str
    text_bytes: int
    diff_count: int
    first_diffs: tuple


@dataclass(frozen=True)
class VupsAdvanceGateManifest:
    status: str
    case_set: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    generated: GeneratedVupsAdvanceVariant | None
    cases: tuple


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP042 = load_module("mylm_exp042_for_vups_advance", EXP042_RUN)


def configure_exp042() -> None:
    EXP042.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP042.BUILD_DIR = BUILD_DIR


def variant_source_bundles() -> tuple:
    rows = []
    seen: set[int] = set()
    for bundle in EXP042.EXP029.source_bundles():
        if bundle.address in BUNDLE_REPLACEMENTS:
            original, variant = BUNDLE_REPLACEMENTS[bundle.address]
            if bundle.lines != original:
                raise RuntimeError(f"unexpected bundle at {hex(bundle.address)}: {bundle.lines}")
            rows.append(EXP042.EXP029.SourceBundle(bundle.address, variant))
            seen.add(bundle.address)
        else:
            rows.append(bundle)
    missing = tuple(sorted(set(BUNDLE_REPLACEMENTS) - seen))
    if missing:
        raise RuntimeError(f"missing replacement bundles: {', '.join(hex(address) for address in missing)}")
    return tuple(rows)


def generate_variant() -> GeneratedVupsAdvanceVariant:
    build = BUILD_DIR / "generated_vups_advance"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP042.EXP029.BUILD_DIR = build
    original = EXP042.EXP029.original_hot_bytes()
    bundles = variant_source_bundles()
    unpadded_text = EXP042.EXP029.unpadded_mir(bundles)
    unpadded = EXP042.EXP029.compile_mir("q4nx_hotloop_vups_advance_unpadded", unpadded_text, original)
    if unpadded.llc.returncode != 0:
        raise RuntimeError(unpadded.llc.stderr)
    lengths = EXP042.EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, _ = EXP042.EXP029.padded_mir(bundles, lengths)
    padded = EXP042.EXP029.compile_mir("q4nx_hotloop_vups_advance_padded", padded_text, original)
    if padded.llc.returncode != 0:
        raise RuntimeError(padded.llc.stderr)
    generated_text = EXP042.EXP029.read_elf_section(build / "q4nx_hotloop_vups_advance_padded.o", ".text")
    if len(generated_text) != len(original):
        raise RuntimeError(f"generated hot-loop length mismatch: {len(generated_text)} != {len(original)}")
    diffs = EXP042.byte_diffs(original, generated_text)
    raw = bytearray(EXP042.EXP029.RAW_PROGRAM.read_bytes())
    raw[EXP042.EXP029.HOT_START : EXP042.EXP029.HOT_END] = generated_text
    patched = build / "mylm_c2r2_program.generated_mir_vups_advance.bin"
    patched.write_bytes(raw)
    replacements = tuple(
        BundleReplacement(hex(address), original_mir, variant_mir)
        for address, (original_mir, variant_mir) in sorted(BUNDLE_REPLACEMENTS.items())
    )
    return GeneratedVupsAdvanceVariant(
        window_start=hex(WINDOW_START),
        window_end=hex(WINDOW_END),
        replacements=replacements,
        unpadded_mir=unpadded.mir,
        padded_mir=padded.mir,
        object=padded.object,
        objdump=padded.objdump,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        text_bytes=len(generated_text),
        diff_count=len(diffs),
        first_diffs=diffs[:16],
    )


def build_manifest(args: argparse.Namespace) -> VupsAdvanceGateManifest:
    configure_exp042()
    EXP042.configure_helpers()
    before = cast(dict[str, JsonValue], EXP042.EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return VupsAdvanceGateManifest("skipped_topology_not_ok", args.case_set, before, None, None, ())
    cases = EXP042.selected_cases(args.case_set, args.max_cases)
    generated = generate_variant()
    references = EXP042.reference_records(cases, args.runtime_timeout)
    variant_cases = EXP042.run_variant_cases(cases, references, REPO_ROOT / generated.patched_raw, args.runtime_timeout)
    after = cast(dict[str, JsonValue], EXP042.EXP015.EXP008.topology_status(args.xrt_timeout))
    status = (
        "passed"
        if after["status"] == "ok" and generated.diff_count > 0 and all(case.matches_reference for case in variant_cases)
        else "failed"
    )
    return VupsAdvanceGateManifest(status, args.case_set, before, after, generated, variant_cases)


def render_bundle(lines: tuple[str, ...]) -> str:
    return "<br>".join(line.strip() for line in lines)


def render_report(manifest: VupsAdvanceGateManifest) -> str:
    lines = [
        "# Q4NX Generated MIR VUPS Advance Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- case set: `{manifest.case_set}`",
        "",
    ]
    if manifest.generated is None:
        lines.append("No generated variant was built.")
        return "\n".join(lines)
    generated = manifest.generated
    lines.extend(
        [
            "## Generated Variant",
            "",
            f"- patched raw: `{generated.patched_raw}`",
            f"- object: `{generated.object}`",
            f"- window: `{generated.window_start}..{generated.window_end}`",
            f"- replacements: `{len(generated.replacements)}`",
            f"- text bytes: `{generated.text_bytes}`",
            f"- byte diffs vs MyLM hot loop: `{generated.diff_count}`",
            "",
            "| address | original MIR | variant MIR |",
            "| --- | --- | --- |",
        ]
    )
    for replacement in generated.replacements:
        lines.append(
            f"| `{replacement.address}` | `{render_bundle(replacement.original_mir)}` | `{render_bundle(replacement.variant_mir)}` |"
        )
    lines.extend(["", "## Byte Diffs", "", "| address | original | generated |", "| --- | --- | --- |"])
    for diff in generated.first_diffs:
        lines.append(f"| `{diff.address}` | `{diff.original}` | `{diff.generated}` |")
    lines.extend(
        [
            "",
            "## Numeric Gate",
            "",
            "| case | status | matches reference | observed first words |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case in manifest.cases:
        observed = ", ".join(case.observed_record[:6])
        lines.append(f"| `{case.name}` | `{case.runtime_status}` | `{case.matches_reference}` | `{observed}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a schedule-changing mutation: `vups.4x` is issued one bundle earlier and the high-half move is delayed.",
            "- A passed gate means this local `vups -> vadd` spacing change preserves selected MyLM-reference numeric coverage.",
            "- A failed gate means the original immediate `vups.4x` placement is part of the local numeric contract.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-set", choices=("direct", "asymmetric-zero"), default="direct")
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
