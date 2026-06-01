#!/usr/bin/env python3
"""Run the byte-exact pre-bundled MIR hot loop as a no-op NPU numeric gate."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_byte_exact_noop_numeric_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_byte_exact_noop_numeric_gate.md"

HOT_START = 0x260
HOT_END = 0x1850
PADDED_OBJECT = "mylm_hotloop_prebundled_padded.o"


@dataclass(frozen=True)
class PatchInfo:
    source_object: str
    patched_raw: str
    hot_start: str
    hot_end: str
    patch_bytes: int
    original_raw_matches: bool
    patched_raw_matches: bool
    first_diff: str | None


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


EXP015 = load_module("mylm_exp015_for_mir_noop_numeric", EXP015_RUN)
EXP029 = load_module("mylm_exp029_for_mir_noop_numeric", EXP029_RUN)


def original_raw() -> bytes:
    return EXP029.RAW_PROGRAM.read_bytes()


def first_difference(got: bytes, expected: bytes) -> str | None:
    limit = min(len(got), len(expected))
    for index in range(limit):
        if got[index] != expected[index]:
            return f"0x{index:x}: got 0x{got[index]:02x}, expected 0x{expected[index]:02x}"
    if len(got) != len(expected):
        return f"length mismatch: got {len(got)}, expected {len(expected)}"
    return None


def ensure_byte_exact_object() -> Path:
    object_path = EXP029.BUILD_DIR / PADDED_OBJECT
    if not object_path.exists():
        EXP029.run()
    if not object_path.exists():
        raise FileNotFoundError(object_path)
    hot = EXP029.read_elf_section(object_path, ".text")
    original_hot = EXP029.original_hot_bytes()
    if hot != original_hot:
        diff = first_difference(hot, original_hot)
        raise RuntimeError(f"experiment 029 padded object is not byte exact: {diff}")
    return object_path


def make_patched_raw() -> PatchInfo:
    build = BUILD_DIR / "patched_raw"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    object_path = ensure_byte_exact_object()
    hot = EXP029.read_elf_section(object_path, ".text")
    if len(hot) != HOT_END - HOT_START:
        raise RuntimeError(f"unexpected hot-loop byte count: {len(hot)}")
    original = bytearray(original_raw())
    patched = bytearray(original)
    patched[HOT_START:HOT_END] = hot
    patched_path = build / "mylm_c2r2_program.byte_exact_mir_noop.bin"
    patched_path.write_bytes(patched)
    diff = first_difference(bytes(patched), bytes(original))
    return PatchInfo(
        source_object=str(object_path.relative_to(REPO_ROOT)),
        patched_raw=str(patched_path.relative_to(REPO_ROOT)),
        hot_start=hex(HOT_START),
        hot_end=hex(HOT_END),
        patch_bytes=len(hot),
        original_raw_matches=hot == bytes(original[HOT_START:HOT_END]),
        patched_raw_matches=bytes(patched) == bytes(original),
        first_diff=diff,
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
    if not patch.patched_raw_matches:
        return {"status": "failed_raw_mismatch", "topology_before": before, "patch": asdict(patch)}
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
        "# Q4NX MIR Byte-Exact No-Op Numeric Gate",
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
            f"- source object: `{patch['source_object']}`",
            f"- patched raw: `{patch['patched_raw']}`",
            f"- hot range: `{patch['hot_start']}..{patch['hot_end']}`",
            f"- patch bytes: `{patch['patch_bytes']}`",
            f"- hot bytes match original: `{patch['original_raw_matches']}`",
            f"- full raw matches original: `{patch['patched_raw_matches']}`",
            f"- first diff: `{patch['first_diff']}`",
            "",
        ]
    )
    if "cases" in manifest:
        lines.extend(
            [
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
            "- This is a no-op replacement: the MIR object from experiment 029 is byte-exact against MyLM's hot loop.",
            "- A pass means the MIR object extraction, raw patching, packaging, and direct-QKV runtime path preserve MyLM numerics.",
            "- This is the baseline for future one-bundle-at-a-time MIR mutations.",
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
