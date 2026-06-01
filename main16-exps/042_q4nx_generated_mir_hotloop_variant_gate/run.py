#!/usr/bin/env python3
"""Generate a non-byte-copy MIR hot-loop variant and run direct-QKV numeric gate."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_generated_mir_hotloop_variant_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_generated_mir_hotloop_variant_gate.md"

MUTATION_ADDRESS = 0x700
ORIGINAL_MIR = "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2"
VARIANT_MIR = "      $bmhh1 = VMOV_alu_mv_mv_x $bmll2"
ZERO_LOW_ONLY = 0x00003C80
ZERO_HIGH_ONLY = 0x3C800000

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ByteDiff:
    address: str
    original: str
    generated: str


@dataclass(frozen=True)
class GeneratedVariant:
    unpadded_mir: str
    padded_mir: str
    object: str
    objdump: str
    patched_raw: str
    text_bytes: int
    diff_count: int
    first_diffs: tuple[ByteDiff, ...]


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    reference_record: tuple[str, ...]
    observed_record: tuple[str, ...]
    runtime_status: str
    matches_reference: bool
    runtime_output: str


@dataclass(frozen=True)
class GateManifest:
    status: str
    case_set: str
    mutation_address: str
    original_mir: str
    variant_mir: str
    topology_before: dict[str, JsonValue]
    topology_after: dict[str, JsonValue] | None
    generated: GeneratedVariant | None
    cases: tuple[RuntimeCase, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_generated_mir_hotloop_variant", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_generated_mir_hotloop_variant", EXP029_RUN)
TinyCase = EXP015.TinyCase


def configure_helpers(patched_raw: Path | None = None) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    if patched_raw is not None:
        EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def direct_cases() -> tuple:
    return EXP015.cases()


def asymmetric_zero_cases() -> tuple:
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
    return tuple(cases)


def selected_cases(case_set: str, max_cases: int) -> tuple:
    cases = asymmetric_zero_cases() if case_set == "asymmetric-zero" else direct_cases()
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def variant_source_bundles() -> tuple:
    rows = []
    replaced = False
    for bundle in EXP029.source_bundles():
        if bundle.address == MUTATION_ADDRESS:
            if bundle.lines != (ORIGINAL_MIR,):
                raise RuntimeError(f"unexpected source bundle at {hex(MUTATION_ADDRESS)}: {bundle.lines}")
            rows.append(EXP029.SourceBundle(bundle.address, (VARIANT_MIR,)))
            replaced = True
        else:
            rows.append(bundle)
    if not replaced:
        raise RuntimeError(f"missing source bundle at {hex(MUTATION_ADDRESS)}")
    return tuple(rows)


def byte_diffs(original: bytes, generated: bytes) -> tuple[ByteDiff, ...]:
    diffs: list[ByteDiff] = []
    for offset, (left, right) in enumerate(zip(original, generated)):
        if left != right:
            diffs.append(ByteDiff(hex(EXP029.HOT_START + offset), f"{left:02x}", f"{right:02x}"))
    if len(original) != len(generated):
        diffs.append(ByteDiff("length", str(len(original)), str(len(generated))))
    return tuple(diffs)


def generate_variant() -> GeneratedVariant:
    build = BUILD_DIR / "generated_variant"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP029.BUILD_DIR = build
    original = EXP029.original_hot_bytes()
    bundles = variant_source_bundles()
    unpadded_text = EXP029.unpadded_mir(bundles)
    unpadded = EXP029.compile_mir("q4nx_hotloop_variant_unpadded", unpadded_text, original)
    if unpadded.llc.returncode != 0:
        raise RuntimeError(unpadded.llc.stderr)
    lengths = EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, _ = EXP029.padded_mir(bundles, lengths)
    padded = EXP029.compile_mir("q4nx_hotloop_variant_padded", padded_text, original)
    if padded.llc.returncode != 0:
        raise RuntimeError(padded.llc.stderr)
    generated_text = EXP029.read_elf_section(build / "q4nx_hotloop_variant_padded.o", ".text")
    if len(generated_text) != len(original):
        raise RuntimeError(f"generated hot-loop length mismatch: {len(generated_text)} != {len(original)}")
    diffs = byte_diffs(original, generated_text)
    raw = bytearray(EXP029.RAW_PROGRAM.read_bytes())
    raw[EXP029.HOT_START : EXP029.HOT_END] = generated_text
    patched = build / "mylm_c2r2_program.generated_mir_variant.bin"
    patched.write_bytes(raw)
    return GeneratedVariant(
        unpadded_mir=unpadded.mir,
        padded_mir=padded.mir,
        object=padded.object,
        objdump=padded.objdump,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        text_bytes=len(generated_text),
        diff_count=len(diffs),
        first_diffs=diffs[:16],
    )


def run_raw_case(case, timeout: int) -> tuple[str, tuple[str, ...], str]:
    runtime = EXP015.run_npu_case(
        EXP015.EXP008.EXP005.EXP132.EXP130.XCLBIN,
        EXP015.EXP008.EXP005.EXP132.EXP130.INSTS,
        case,
        EXP015.EXP008.RECORDS,
        timeout,
    )
    return runtime["status"], tuple(runtime["record"]), runtime["runtime_output"]


def reference_records(cases: tuple, timeout: int) -> dict[str, tuple[str, ...]]:
    configure_helpers()
    EXP015.EXP008.build_qkv_direct_xclbin()
    out: dict[str, tuple[str, ...]] = {}
    for case in cases:
        status, record, _ = run_raw_case(case, timeout)
        if status != "record_observed":
            raise RuntimeError(f"reference case failed: {case.name} status={status}")
        out[case.name] = record
    return out


def run_variant_cases(cases: tuple, references: dict[str, tuple[str, ...]], patched_raw: Path, timeout: int) -> tuple[RuntimeCase, ...]:
    configure_helpers(patched_raw)
    EXP015.EXP008.build_qkv_direct_xclbin()
    rows: list[RuntimeCase] = []
    for case in cases:
        status, record, output = run_raw_case(case, timeout)
        reference = references[case.name]
        rows.append(
            RuntimeCase(
                name=case.name,
                reference_record=reference,
                observed_record=record,
                runtime_status=status,
                matches_reference=status == "record_observed" and record == reference,
                runtime_output=output,
            )
        )
    return tuple(rows)


def build_manifest(args: argparse.Namespace) -> GateManifest:
    configure_helpers()
    before = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return GateManifest(
            "skipped_topology_not_ok",
            args.case_set,
            hex(MUTATION_ADDRESS),
            ORIGINAL_MIR,
            VARIANT_MIR,
            before,
            None,
            None,
            (),
        )
    cases = selected_cases(args.case_set, args.max_cases)
    generated = generate_variant()
    references = reference_records(cases, args.runtime_timeout)
    variant_cases = run_variant_cases(cases, references, REPO_ROOT / generated.patched_raw, args.runtime_timeout)
    after = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    status = (
        "passed"
        if after["status"] == "ok" and generated.diff_count > 0 and all(case.matches_reference for case in variant_cases)
        else "failed"
    )
    return GateManifest(
        status,
        args.case_set,
        hex(MUTATION_ADDRESS),
        ORIGINAL_MIR,
        VARIANT_MIR,
        before,
        after,
        generated,
        variant_cases,
    )


def render_report(manifest: GateManifest) -> str:
    lines = [
        "# Q4NX Generated MIR Hotloop Variant Gate",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- case set: `{manifest.case_set}`",
        f"- mutation address: `{manifest.mutation_address}`",
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
            "- This replaces the whole hot-loop range with generated pre-bundled MIR bytes, not a direct 4-byte patch.",
            "- Passing the gate means the MIR generator route can produce a controlled non-byte-copy hot-loop variant.",
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
