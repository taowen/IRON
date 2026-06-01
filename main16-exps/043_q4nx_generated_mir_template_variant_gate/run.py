#!/usr/bin/env python3
"""Generate a repeated-template MIR hot-loop variant and run NPU numeric gates."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_generated_mir_template_variant_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_generated_mir_template_variant_gate.md"

ORIGINAL_MIR = "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2"
VARIANT_MIR = "      $bmhh1 = VMOV_alu_mv_mv_x $bmll2"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class GeneratedTemplateVariant:
    mutation_addresses: tuple[str, ...]
    unpadded_mir: str
    padded_mir: str
    object: str
    objdump: str
    patched_raw: str
    text_bytes: int
    diff_count: int
    first_diffs: tuple


@dataclass(frozen=True)
class TemplateGateManifest:
    status: str
    case_set: str
    original_mir: str
    variant_mir: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    generated: GeneratedTemplateVariant | None
    cases: tuple


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP042 = load_module("mylm_exp042_for_generated_template_variant", EXP042_RUN)


def configure_exp042() -> None:
    EXP042.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP042.BUILD_DIR = BUILD_DIR


def template_variant_source_bundles() -> tuple:
    rows = []
    addresses: list[int] = []
    for bundle in EXP042.EXP029.source_bundles():
        if bundle.lines == (ORIGINAL_MIR,):
            rows.append(EXP042.EXP029.SourceBundle(bundle.address, (VARIANT_MIR,)))
            addresses.append(bundle.address)
        else:
            rows.append(bundle)
    if not addresses:
        raise RuntimeError("no template bundles were replaced")
    return tuple(rows), tuple(addresses)


def generate_variant() -> GeneratedTemplateVariant:
    build = BUILD_DIR / "generated_template_variant"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP042.EXP029.BUILD_DIR = build
    original = EXP042.EXP029.original_hot_bytes()
    bundles, addresses = template_variant_source_bundles()
    unpadded_text = EXP042.EXP029.unpadded_mir(bundles)
    unpadded = EXP042.EXP029.compile_mir("q4nx_hotloop_template_variant_unpadded", unpadded_text, original)
    if unpadded.llc.returncode != 0:
        raise RuntimeError(unpadded.llc.stderr)
    lengths = EXP042.EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, _ = EXP042.EXP029.padded_mir(bundles, lengths)
    padded = EXP042.EXP029.compile_mir("q4nx_hotloop_template_variant_padded", padded_text, original)
    if padded.llc.returncode != 0:
        raise RuntimeError(padded.llc.stderr)
    generated_text = EXP042.EXP029.read_elf_section(build / "q4nx_hotloop_template_variant_padded.o", ".text")
    if len(generated_text) != len(original):
        raise RuntimeError(f"generated hot-loop length mismatch: {len(generated_text)} != {len(original)}")
    diffs = EXP042.byte_diffs(original, generated_text)
    raw = bytearray(EXP042.EXP029.RAW_PROGRAM.read_bytes())
    raw[EXP042.EXP029.HOT_START : EXP042.EXP029.HOT_END] = generated_text
    patched = build / "mylm_c2r2_program.generated_mir_template_variant.bin"
    patched.write_bytes(raw)
    return GeneratedTemplateVariant(
        mutation_addresses=tuple(hex(address) for address in addresses),
        unpadded_mir=unpadded.mir,
        padded_mir=padded.mir,
        object=padded.object,
        objdump=padded.objdump,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        text_bytes=len(generated_text),
        diff_count=len(diffs),
        first_diffs=diffs[:24],
    )


def build_manifest(args: argparse.Namespace) -> TemplateGateManifest:
    configure_exp042()
    EXP042.configure_helpers()
    before = cast(dict[str, JsonValue], EXP042.EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return TemplateGateManifest("skipped_topology_not_ok", args.case_set, ORIGINAL_MIR, VARIANT_MIR, before, None, None, ())
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
    return TemplateGateManifest(status, args.case_set, ORIGINAL_MIR, VARIANT_MIR, before, after, generated, variant_cases)


def render_report(manifest: TemplateGateManifest) -> str:
    lines = [
        "# Q4NX Generated MIR Template Variant Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- case set: `{manifest.case_set}`",
        f"- original MIR: `{manifest.original_mir}`",
        f"- variant MIR: `{manifest.variant_mir}`",
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
            f"- text bytes: `{generated.text_bytes}`",
            f"- mutation sites: `{len(generated.mutation_addresses)}`",
            f"- mutation addresses: `{', '.join(generated.mutation_addresses)}`",
            f"- byte diffs vs MyLM hot loop: `{generated.diff_count}`",
            "",
            "| address | original | generated |",
            "| --- | --- | --- |",
        ]
    )
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
            "- This regenerates the full hot loop and changes every matching source template bundle.",
            "- Passing the gate shows the MIR route can carry a template-level generated variant.",
            "- The generated variant is still functionally equivalent under the selected case set; it is not a speedup yet.",
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
