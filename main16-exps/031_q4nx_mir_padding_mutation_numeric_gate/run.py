#!/usr/bin/env python3
"""Run a first controlled hot-loop byte-diff through the direct-QKV NPU gate."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_padding_mutation_numeric_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_padding_mutation_numeric_gate.md"

HOT_START = 0x260
HOT_END = 0x1850
MUTATION_ADDRESS = 0x2BA
ORIGINAL_BYTES = bytes.fromhex("00000000")
REPLACEMENT_BYTES = bytes.fromhex("f8a0df1f")
REPLACEMENT_ASM = "mov r31, r31"


@dataclass(frozen=True)
class ByteDiff:
    address: str
    got: str
    expected: str


@dataclass(frozen=True)
class PatchInfo:
    patched_raw: str
    mutation_address: str
    original_bytes: str
    replacement_bytes: str
    replacement_asm: str
    diff_count: int
    diffs: tuple[ByteDiff, ...]


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    expected_record: tuple[str, ...]
    observed_record: tuple[str, ...]
    runtime_status: str
    matches_expected: bool
    runtime_output: str


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_module("mylm_exp015_for_padding_mutation_numeric", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_padding_mutation_numeric", EXP029_RUN)


def expected_diffs() -> tuple[ByteDiff, ...]:
    diffs: list[ByteDiff] = []
    for index, (got, expected) in enumerate(zip(REPLACEMENT_BYTES, ORIGINAL_BYTES)):
        if got != expected:
            diffs.append(
                ByteDiff(
                    address=hex(MUTATION_ADDRESS + index),
                    got=f"0x{got:02x}",
                    expected=f"0x{expected:02x}",
                )
            )
    return tuple(diffs)


def original_raw() -> bytes:
    return EXP029.RAW_PROGRAM.read_bytes()


def make_patched_raw() -> PatchInfo:
    build = BUILD_DIR / "patched_raw"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    raw = bytearray(original_raw())
    found = bytes(raw[MUTATION_ADDRESS : MUTATION_ADDRESS + len(ORIGINAL_BYTES)])
    if found != ORIGINAL_BYTES:
        raise RuntimeError(
            "unexpected mutation site bytes at "
            f"{hex(MUTATION_ADDRESS)}: got {found.hex()}, expected {ORIGINAL_BYTES.hex()}"
        )
    raw[MUTATION_ADDRESS : MUTATION_ADDRESS + len(REPLACEMENT_BYTES)] = REPLACEMENT_BYTES
    patched = build / "mylm_c2r2_program.padding_mov_self.bin"
    patched.write_bytes(raw)
    return PatchInfo(
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        mutation_address=hex(MUTATION_ADDRESS),
        original_bytes=ORIGINAL_BYTES.hex(),
        replacement_bytes=REPLACEMENT_BYTES.hex(),
        replacement_asm=REPLACEMENT_ASM,
        diff_count=len(expected_diffs()),
        diffs=expected_diffs(),
    )


def selected_cases(max_cases: int) -> tuple:
    cases = EXP015.cases()
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def configure_helpers(patched_raw: Path) -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR
    EXP015.configure_exp008()
    EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW = patched_raw


def build_patched_direct_qkv(patched_raw: Path) -> dict[str, str]:
    configure_helpers(patched_raw)
    build = EXP015.EXP008.build_qkv_direct_xclbin()
    return {
        "xclbin": str(EXP015.EXP008.EXP005.EXP132.EXP130.XCLBIN.relative_to(REPO_ROOT)),
        "insts": str(EXP015.EXP008.EXP005.EXP132.EXP130.INSTS.relative_to(REPO_ROOT)),
        "elf": str(EXP015.EXP008.EXP005.EXP132.EXP130.ELF.relative_to(REPO_ROOT)),
        "source_build": str(Path(build["build_dir"])),
    }


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


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    patch = make_patched_raw()
    artifacts = build_patched_direct_qkv(REPO_ROOT / patch.patched_raw)
    cases = [run_case(case, args.runtime_timeout) for case in selected_cases(args.max_cases)]
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    status = "passed" if after["status"] == "ok" and all(case.matches_expected for case in cases) else "failed"
    return {
        "status": status,
        "topology_before": before,
        "topology_after": after,
        "patch": asdict(patch),
        "artifacts": artifacts,
        "cases": [asdict(case) for case in cases],
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Q4NX MIR Padding Mutation Numeric Gate",
        "",
        f"Status: `{manifest['status']}`",
        "",
    ]
    if "patch" not in manifest:
        lines.append("The runtime topology was not healthy enough to run this gate.")
        return "\n".join(lines)
    patch = manifest["patch"]
    lines.extend(
        [
            "## Patch",
            "",
            f"- patched raw: `{patch['patched_raw']}`",
            f"- mutation address: `{patch['mutation_address']}`",
            f"- original bytes: `{patch['original_bytes']}`",
            f"- replacement bytes: `{patch['replacement_bytes']}`",
            f"- replacement asm: `{patch['replacement_asm']}`",
            f"- diff count: `{patch['diff_count']}`",
            "",
            "## Byte Diff",
            "",
            "| address | original | replacement |",
            "| --- | --- | --- |",
        ]
    )
    for diff in patch["diffs"]:
        lines.append(f"| `{diff['address']}` | `{diff['expected']}` | `{diff['got']}` |")
    if "cases" in manifest:
        lines.extend(
            [
                "",
                "## Cases",
                "",
                "| case | status | matches expected | observed first words |",
                "| --- | --- | --- | --- |",
            ]
        )
        for case in manifest["cases"]:
            observed = ", ".join(case["observed_record"][:4])
            lines.append(
                f"| `{case['name']}` | `{case['runtime_status']}` | "
                f"`{case['matches_expected']}` | `{observed}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first intentional hot-loop byte-diff after the byte-exact no-op gate.",
            "- The mutation replaces two scalar NOP slots in the first explicit padding gap with a self-move.",
            "- A pass means this specific padding gap has enough slack for a one-cycle shorter no-op mutation.",
            "- A fail means padding bytes are part of the timing contract and future mutations must preserve cycle count exactly.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=60)
    parser.add_argument("--xrt-timeout", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=1)
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
