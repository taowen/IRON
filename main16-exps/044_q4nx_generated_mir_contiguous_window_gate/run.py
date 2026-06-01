#!/usr/bin/env python3
"""Generate a contiguous-window MIR variant and run MyLM-reference NPU gates."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_generated_mir_contiguous_window_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_generated_mir_contiguous_window_gate.md"

WINDOW_START = 0x6EE
WINDOW_END = 0x704
REPLACEMENTS = {
    0x6EE: (
        "      $bmlh1 = VMOV_alu_mv_mv_x $bmlh2",
        "      $bmlh1 = VMOV_alu_mv_mv_x $bmll2",
    ),
    0x6F6: (
        "      $bmhl1 = VMOV_alu_mv_mv_x $bmhl2",
        "      $bmhl1 = VMOV_alu_mv_mv_x $bmll2",
    ),
    0x700: (
        "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2",
        "      $bmhh1 = VMOV_alu_mv_mv_x $bmll2",
    ),
}

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class WindowReplacement:
    address: str
    original_mir: str
    variant_mir: str


@dataclass(frozen=True)
class GeneratedWindowVariant:
    window_start: str
    window_end: str
    replacements: tuple[WindowReplacement, ...]
    unpadded_mir: str
    padded_mir: str
    object: str
    objdump: str
    patched_raw: str
    text_bytes: int
    diff_count: int
    first_diffs: tuple


@dataclass(frozen=True)
class WindowGateManifest:
    status: str
    case_set: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    generated: GeneratedWindowVariant | None
    cases: tuple


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP042 = load_module("mylm_exp042_for_contiguous_window", EXP042_RUN)


def configure_exp042() -> None:
    EXP042.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP042.BUILD_DIR = BUILD_DIR


def replace_line(lines: tuple[str, ...], original: str, variant: str, address: int) -> tuple[str, ...]:
    count = sum(1 for line in lines if line == original)
    if count != 1:
        raise RuntimeError(f"expected one line at {hex(address)}: {original}, found {count}")
    return tuple(variant if line == original else line for line in lines)


def window_variant_source_bundles() -> tuple:
    rows = []
    seen: set[int] = set()
    for bundle in EXP042.EXP029.source_bundles():
        if bundle.address in REPLACEMENTS:
            original, variant = REPLACEMENTS[bundle.address]
            rows.append(EXP042.EXP029.SourceBundle(bundle.address, replace_line(bundle.lines, original, variant, bundle.address)))
            seen.add(bundle.address)
        else:
            rows.append(bundle)
    missing = tuple(sorted(set(REPLACEMENTS) - seen))
    if missing:
        raise RuntimeError(f"missing replacement bundles: {', '.join(hex(address) for address in missing)}")
    return tuple(rows)


def generate_variant() -> GeneratedWindowVariant:
    build = BUILD_DIR / "generated_window_variant"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP042.EXP029.BUILD_DIR = build
    original = EXP042.EXP029.original_hot_bytes()
    bundles = window_variant_source_bundles()
    unpadded_text = EXP042.EXP029.unpadded_mir(bundles)
    unpadded = EXP042.EXP029.compile_mir("q4nx_hotloop_window_variant_unpadded", unpadded_text, original)
    if unpadded.llc.returncode != 0:
        raise RuntimeError(unpadded.llc.stderr)
    lengths = EXP042.EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, _ = EXP042.EXP029.padded_mir(bundles, lengths)
    padded = EXP042.EXP029.compile_mir("q4nx_hotloop_window_variant_padded", padded_text, original)
    if padded.llc.returncode != 0:
        raise RuntimeError(padded.llc.stderr)
    generated_text = EXP042.EXP029.read_elf_section(build / "q4nx_hotloop_window_variant_padded.o", ".text")
    if len(generated_text) != len(original):
        raise RuntimeError(f"generated hot-loop length mismatch: {len(generated_text)} != {len(original)}")
    diffs = EXP042.byte_diffs(original, generated_text)
    raw = bytearray(EXP042.EXP029.RAW_PROGRAM.read_bytes())
    raw[EXP042.EXP029.HOT_START : EXP042.EXP029.HOT_END] = generated_text
    patched = build / "mylm_c2r2_program.generated_mir_window_variant.bin"
    patched.write_bytes(raw)
    replacements = tuple(
        WindowReplacement(hex(address), original_mir, variant_mir)
        for address, (original_mir, variant_mir) in sorted(REPLACEMENTS.items())
    )
    return GeneratedWindowVariant(
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


def build_manifest(args: argparse.Namespace) -> WindowGateManifest:
    configure_exp042()
    EXP042.configure_helpers()
    before = cast(dict[str, JsonValue], EXP042.EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return WindowGateManifest("skipped_topology_not_ok", args.case_set, before, None, None, ())
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
    return WindowGateManifest(status, args.case_set, before, after, generated, variant_cases)


def render_report(manifest: WindowGateManifest) -> str:
    lines = [
        "# Q4NX Generated MIR Contiguous Window Gate",
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
        lines.append(f"| `{replacement.address}` | `{replacement.original_mir.strip()}` | `{replacement.variant_mir.strip()}` |")
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
            "- This is a contiguous producer/consumer window mutation, while preserving the original bundle schedule shape.",
            "- Passing the gate shows this local window can be generated and changed without breaking current MyLM-reference coverage.",
            "- The variant still preserves the original schedule shape; it is not a performance optimization yet.",
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
