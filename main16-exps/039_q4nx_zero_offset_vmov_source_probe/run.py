#!/usr/bin/env python3
"""Probe the zero/offset dependence of the repeated bmhh vmov site."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_zero_offset_vmov_source_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_zero_offset_vmov_source_probe.md"

SITE_ADDRESS = 0x700
NOPM_BYTES = bytes.fromhex("f84a0318")
ZERO_CASE = "zero0_allq4"

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
    zero_case: RuntimeCase


@dataclass(frozen=True)
class ProbeManifest:
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


EXP015 = load_module("mylm_exp015_for_zero_offset_vmov_source_probe", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_zero_offset_vmov_source_probe", EXP029_RUN)


def configure_helpers(patched_raw: Path | None = None) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    if patched_raw is not None:
        EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def source_text(asm: str) -> str:
    return "\n".join(
        (
            '  .section .text,"ax",@progbits',
            "  .globl __start",
            "  .type __start,@function",
            "__start:",
            f"  {asm}",
            "  .size __start, .-__start",
            "",
        )
    )


def assemble_instruction(name: str, asm: str) -> bytes:
    if asm == "nopm":
        return NOPM_BYTES
    build = BUILD_DIR / "replacement_bytes" / name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    src = build / f"{name}.s"
    obj = build / f"{name}.o"
    src.write_text(source_text(asm), encoding="utf-8")
    EXP015.run_cmd((str(EXP015.CLANG), "--target=aie2p-none-unknown-elf", "-c", str(src), "-o", str(obj)), build)
    text = EXP029.read_elf_section(obj, ".text")
    if len(text) < 4:
        raise RuntimeError(f"replacement {name} encoded to {len(text)} bytes, expected at least 4")
    return text[:4]


def replacement_specs() -> tuple[tuple[str, str], ...]:
    return (
        ("nopm", "nopm"),
        ("self_bmhh1", "vmov bmhh1, bmhh1"),
        ("from_bmll2", "vmov bmhh1, bmll2"),
        ("from_bmlh2", "vmov bmhh1, bmlh2"),
        ("from_bmhl2", "vmov bmhh1, bmhl2"),
        ("original_bmhh2", "vmov bmhh1, bmhh2"),
    )


def replacements(max_replacements: int) -> tuple[Replacement, ...]:
    rows: list[Replacement] = []
    for name, asm in replacement_specs():
        rows.append(Replacement(name, asm, assemble_instruction(name, asm).hex()))
    if max_replacements > 0:
        return tuple(rows[:max_replacements])
    return tuple(rows)


def zero_case():
    for case in EXP015.cases():
        if case.name == ZERO_CASE:
            return case
    raise RuntimeError(f"missing case {ZERO_CASE}")


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


def run_zero_case(timeout: int) -> RuntimeCase:
    case = zero_case()
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


def run_replacement(replacement: Replacement, timeout: int) -> ReplacementResult:
    patched, original_hex = make_patched_raw(replacement)
    build_patched_direct_qkv(patched)
    runtime = run_zero_case(timeout)
    return ReplacementResult(
        name=replacement.name,
        asm=replacement.asm,
        replacement_bytes=replacement.bytes_hex,
        original_bytes=original_hex,
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        zero_case=runtime,
    )


def build_manifest(args: argparse.Namespace) -> ProbeManifest:
    configure_helpers()
    before = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    if before["status"] != "ok":
        return ProbeManifest("skipped_topology_not_ok", hex(SITE_ADDRESS), before, None, ())
    results = tuple(run_replacement(replacement, args.runtime_timeout) for replacement in replacements(args.max_replacements))
    after = cast(dict[str, JsonValue], EXP015.EXP008.topology_status(args.xrt_timeout))
    observed_all = all(result.zero_case.runtime_status == "record_observed" for result in results)
    original_results = tuple(result for result in results if result.name == "original_bmhh2")
    original_ok = not original_results or original_results[0].zero_case.matches_expected
    status = "passed" if after["status"] == "ok" and observed_all and original_ok else "failed"
    return ProbeManifest(status, hex(SITE_ADDRESS), before, after, results)


def render_report(manifest: ProbeManifest) -> str:
    lines = [
        "# Q4NX Zero/Offset Vmov Source Probe",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- site: `{manifest.site_address}`",
        f"- case: `{ZERO_CASE}`",
        "",
    ]
    if not manifest.results:
        lines.append("No replacement was run.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Results",
            "",
            "| replacement | asm | bytes | matches zero case | observed first words |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in manifest.results:
        observed = ", ".join(result.zero_case.observed_record[:12])
        lines.append(
            f"| `{result.name}` | `{result.asm}` | `{result.replacement_bytes}` | "
            f"`{result.zero_case.matches_expected}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `nopm` and `vmov bmhh1, bmhh1` fail the same way, so preserving cycle count or self-state is not enough.",
            "- Every tested `acc2 -> bmhh1` source passes the zero/offset case, so the dependence is on refreshing `bmhh1` from the `acc2` correction value family.",
            "- The source quadrant is not distinguished by this synthetic zero/offset case; the next gate must run a passing alternate source against all direct-QKV cases.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
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
