#!/usr/bin/env python3
"""Patch the generated MIR hot loop into MyLM and run a direct-QKV numeric gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP015_RUN = REPO_ROOT / "main16-exps/015_q4nx_tiny_codegen_numeric_gate/run.py"
EXP026_RUN = REPO_ROOT / "main16-exps/026_q4nx_mir_full_hotloop_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_hotloop_patch_numeric_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_hotloop_patch_numeric_gate.md"

HOT_START = 0x260
HOT_END = 0x1850
MIR_LOOP_START = 0x10
MIR_LOOP_END = 0x15F0
NOP16 = b"\x00\x00" * 8


@dataclass(frozen=True)
class PatchInfo:
    source_object: str
    patched_raw: str
    hot_start: str
    hot_end: str
    mir_loop_start: str
    mir_loop_end: str
    patch_bytes: int
    nop_fill_bytes: int


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


EXP015 = load_module("mylm_exp015_for_mir_patch_numeric", EXP015_RUN)
EXP026 = load_module("mylm_exp026_for_mir_patch_numeric", EXP026_RUN)


def read_elf_section(path: Path, section_name: str) -> bytes:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1:
        raise ValueError(f"not an ELF32 file: {path}")
    section_header_offset = struct.unpack_from("<I", data, 32)[0]
    section_header_size = struct.unpack_from("<H", data, 46)[0]
    section_count = struct.unpack_from("<H", data, 48)[0]
    string_index = struct.unpack_from("<H", data, 50)[0]
    string_header = section_header_offset + string_index * section_header_size
    string_offset = struct.unpack_from("<I", data, string_header + 16)[0]
    string_size = struct.unpack_from("<I", data, string_header + 20)[0]
    strings = data[string_offset : string_offset + string_size]
    for index in range(section_count):
        header = section_header_offset + index * section_header_size
        name_offset = struct.unpack_from("<I", data, header)[0]
        end = strings.find(b"\x00", name_offset)
        name = strings[name_offset:end].decode("ascii")
        if name == section_name:
            offset = struct.unpack_from("<I", data, header + 16)[0]
            size = struct.unpack_from("<I", data, header + 20)[0]
            return data[offset : offset + size]
    raise ValueError(f"missing section {section_name}: {path}")


def ensure_exp026_object() -> Path:
    object_path = EXP026.BUILD_DIR / "mylm_full_hotloop_zol.o"
    if object_path.exists():
        return object_path
    EXP026.run()
    if not object_path.exists():
        raise FileNotFoundError(object_path)
    return object_path


def make_patched_raw() -> PatchInfo:
    build = BUILD_DIR / "patched_raw"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    object_path = ensure_exp026_object()
    mir_text = read_elf_section(object_path, ".text")
    loop = mir_text[MIR_LOOP_START:MIR_LOOP_END]
    if len(loop) != MIR_LOOP_END - MIR_LOOP_START:
        raise RuntimeError(f"unexpected MIR loop bytes: {len(loop)}")
    hot_len = HOT_END - HOT_START
    if len(loop) > hot_len:
        raise RuntimeError(f"MIR loop does not fit hot range: {len(loop)} > {hot_len}")
    raw = bytearray(EXP015.EXP008.EXP005.EXP132.DEFAULT_RAW.read_bytes())
    raw[HOT_START : HOT_START + len(loop)] = loop
    fill_start = HOT_START + len(loop)
    fill_len = HOT_END - fill_start
    if fill_len < 0 or fill_len % len(NOP16) != 0:
        raise RuntimeError(f"invalid nop fill length: {fill_len}")
    raw[fill_start:HOT_END] = NOP16 * (fill_len // len(NOP16))
    patched = build / "mylm_c2r2_program.mir_hotloop_patch.bin"
    patched.write_bytes(raw)
    return PatchInfo(
        source_object=str(object_path.relative_to(REPO_ROOT)),
        patched_raw=str(patched.relative_to(REPO_ROOT)),
        hot_start=hex(HOT_START),
        hot_end=hex(HOT_END),
        mir_loop_start=hex(MIR_LOOP_START),
        mir_loop_end=hex(MIR_LOOP_END),
        patch_bytes=len(loop),
        nop_fill_bytes=fill_len,
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


def build_manifest(args: argparse.Namespace) -> dict:
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


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX MIR Hotloop Patch Numeric Gate",
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
            f"- MIR loop range: `{patch['mir_loop_start']}..{patch['mir_loop_end']}`",
            f"- patch bytes: `{patch['patch_bytes']}`",
            f"- nop fill bytes: `{patch['nop_fill_bytes']}`",
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
            f"| `{case['name']}` | `{case['runtime_status']}` | `{case['matches_expected']}` | `{observed}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first direct numeric gate for the MIR-generated MyLM-style hot body.",
            "- It preserves MyLM's prologue, phase body, group-sum producer, record emitter, DMA, and lock behavior.",
            "- A pass means the generated MIR hot loop is numerically compatible for the selected synthetic cases.",
            "- A fail should be debugged as a hot-loop semantic/register-schedule mismatch, not as a dataflow or phase-body issue.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
