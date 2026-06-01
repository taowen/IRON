#!/usr/bin/env python3
"""Run VUPS.4x lane-pattern value probes on real NPU."""

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
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "vups4x_lane_pattern_probe.json"
REPORT = EXPERIMENT_DIR / "vups4x_lane_pattern_probe.md"

SCRATCH_ADDR = 0x73D00
OUTPUT_CELLS = ("bmll0", "bmlh0", "bmhl0", "bmhh0")
CELL_SENTINELS = {
    "bmll0": 0x3C80,
    "bmlh0": 0x4000,
    "bmhl0": 0x4040,
    "bmhh0": 0x4080,
}
LANE_PATTERN_WORDS = (
    0x3C804000,
    0x40404080,
    0x40A040C0,
    0x40E04100,
    0x41104120,
    0x41304140,
    0x41504160,
    0x41704180,
    0x419041A0,
    0x41B041C0,
    0x41D041E0,
    0x41F04200,
    0x42104220,
    0x42304240,
    0x42504260,
    0x42704280,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ProbeCase:
    name: str
    target: str
    sign: int
    shift: int
    output_cell: str


@dataclass(frozen=True)
class ProbeResult:
    name: str
    target: str
    sign: int
    shift: int
    output_cell: str
    runtime_status: str
    record: tuple[str, ...]
    unique_record_words: tuple[str, ...]
    runtime_output: str


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_vups4x_lane_pattern", EXP015_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 015 helper: {EXP015_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_exp015()
TinyCase = EXP015.TinyCase


def configure_helpers() -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR


def host_case() -> TinyCase:
    return TinyCase(
        "vups4x_lane_pattern_input",
        EXP015.all_scale(),
        (0,),
        (0, 512),
        0x11111111,
        EXP015.BF16_SCALE_PAIR,
    )


def focused_cases() -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    for target in ("x2d", "cml", "cmh"):
        for output_cell in OUTPUT_CELLS:
            cases.append(ProbeCase(f"{target}_sign0_s0_{output_cell}", target, 0, 0, output_cell))
    cases.extend(
        (
            ProbeCase("x2d_sign1_s0_bmll0", "x2d", 1, 0, "bmll0"),
            ProbeCase("x2d_sign1_s0_bmhh0", "x2d", 1, 0, "bmhh0"),
            ProbeCase("cml_sign1_s0_bmll0", "cml", 1, 0, "bmll0"),
            ProbeCase("cml_sign1_s0_bmhl0", "cml", 1, 0, "bmhl0"),
            ProbeCase("cmh_sign1_s0_bmll0", "cmh", 1, 0, "bmll0"),
            ProbeCase("cmh_sign1_s0_bmhl0", "cmh", 1, 0, "bmhl0"),
            ProbeCase("x2d_sign0_s1_bmll0", "x2d", 0, 1, "bmll0"),
            ProbeCase("x2d_sign0_s1_bmhh0", "x2d", 0, 1, "bmhh0"),
            ProbeCase("cml_sign0_s1_bmll0", "cml", 0, 1, "bmll0"),
            ProbeCase("cml_sign0_s1_bmhl0", "cml", 0, 1, "bmhl0"),
            ProbeCase("cmh_sign0_s1_bmll0", "cmh", 0, 1, "bmll0"),
            ProbeCase("cmh_sign0_s1_bmhl0", "cmh", 0, 1, "bmhl0"),
        )
    )
    return tuple(cases)


def all_cases() -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    for target in ("x2d", "cml", "cmh"):
        for sign in (0, 1):
            for shift in (0, 1):
                for output_cell in OUTPUT_CELLS:
                    cases.append(ProbeCase(f"{target}_sign{sign}_s{shift}_{output_cell}", target, sign, shift, output_cell))
    return tuple(cases)


def probe_cases(case_set: str, max_cases: int) -> tuple[ProbeCase, ...]:
    cases = all_cases() if case_set == "all" else focused_cases()
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def nops(lines: list[str], count: int) -> None:
    lines.extend(["  nop"] * count)


def consume_one_chunk(lines: list[str]) -> None:
    lines.extend(
        [
            "  movx r14, #-1",
            "  acq #49, r14",
            "  nop",
            "  nop",
            "  nop",
            "  acq #51, r14",
            "  nop",
            "  nop",
            "  nop",
        ]
    )


def release_one_chunk(lines: list[str]) -> None:
    lines.extend(
        [
            "  mova r15, #1",
            "  rel #48, r15",
            "  nop",
            "  nop",
            "  nop",
            "  rel #50, r15",
            "  nop",
            "  nop",
            "  nop",
        ]
    )


def set_s0(lines: list[str], shift: int) -> None:
    lines.extend(
        [
            f"  mova r1, #{shift}",
            "  mov s0, r1",
        ]
    )
    nops(lines, 6)


def init_dm0_sentinels(lines: list[str]) -> None:
    set_s0(lines, 0)
    for cell, value in CELL_SENTINELS.items():
        lines.extend(
            [
                f"  movxm r2, #0x{value:x}",
                "  vbcst.16 x0, r2",
            ]
        )
        nops(lines, 10)
        lines.append(f"  vups.2x {cell}, wl0, s0, upssign0")
        nops(lines, 6)


def write_lane_pattern(lines: list[str]) -> None:
    lines.append(f"  movxm p5, #0x{SCRATCH_ADDR:x}")
    for word in LANE_PATTERN_WORDS:
        lines.extend(
            [
                f"  movxm r3, #0x{word:x}",
                "  st r3, [p5], #4",
            ]
        )
    nops(lines, 12)
    lines.extend(
        [
            f"  movxm p5, #0x{SCRATCH_ADDR:x}",
            "  vlda x1, [p5], #0x40",
        ]
    )
    nops(lines, 16)


def run_target_vups(lines: list[str], case: ProbeCase) -> None:
    set_s0(lines, case.shift)
    sign = f"upssign{case.sign}"
    if case.target == "x2d":
        lines.append(f"  vups.4x dm0, x1, s0, {sign}")
    elif case.target == "cml":
        lines.append(f"  vups.4x cml0, wl1, s0, {sign}")
    elif case.target == "cmh":
        lines.append(f"  vups.4x cmh0, wl1, s0, {sign}")
    else:
        raise ValueError(f"unknown target: {case.target}")
    nops(lines, 24)


def probe_asm(case: ProbeCase) -> str:
    lines = [
        '  .section .text,"ax",@progbits',
        "  .globl __start",
        "  .type __start,@function",
        "  .p2align 4",
        "__start:",
        "  movxm sp, #0x70000",
    ]
    consume_one_chunk(lines)
    lines.extend(
        [
            "  movx r14, #-1",
            "  acq #52, r14",
            "  nop",
            "  nop",
            "  nop",
            "  movxm p0, #0x73c1c",
            "  mov crupsmode, #0",
        ]
    )
    nops(lines, 8)
    init_dm0_sentinels(lines)
    write_lane_pattern(lines)
    run_target_vups(lines, case)
    lines.append(f"  vst {case.output_cell}, [p0, #0]")
    nops(lines, 16)
    release_one_chunk(lines)
    for _ in range(EXP015.RECORD_CHUNKS - 1):
        consume_one_chunk(lines)
        release_one_chunk(lines)
    lines.extend(
        [
            "  mova r15, #1",
            "  rel #53, r15",
            "  nop",
            "  nop",
            "  nop",
            "  done",
            ".Lspin:",
            "  j #.Lspin",
            "  nop",
            "  nop",
            "  nop",
            "  nop",
            "  nop",
            "  .size __start, .-__start",
            "",
        ]
    )
    return "\n".join(lines)


def build_case_xclbin(case: ProbeCase) -> dict[str, str]:
    build = BUILD_DIR / case.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / f"{case.name}.s"
    obj = build / f"{case.name}.o"
    ld = build / f"{case.name}.ld"
    elf = build / f"{case.name}.elf"
    disasm = build / f"{case.name}.disasm.s"
    asm.write_text(probe_asm(case), encoding="utf-8")
    ld.write_text(EXP015.tiny_linker_script(), encoding="utf-8")
    EXP015.run_cmd((str(EXP015.CLANG), "--target=aie2p-none-unknown-elf", "-c", str(asm), "-o", str(obj)), build)
    EXP015.run_cmd((str(EXP015.LD_LLD), "-T", str(ld), str(obj), "-o", str(elf)), build)
    disasm.write_text(EXP015.run_cmd((str(EXP015.LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf)), build).stdout, encoding="utf-8")
    EXP015.configure_exp130(build, elf.name)
    EXP015.EXP130.write_mlir(1)
    EXP015.EXP130.package_with_aiecc()
    return {
        "build_dir": str(build.relative_to(REPO_ROOT)),
        "asm": str(asm.relative_to(REPO_ROOT)),
        "elf": str(elf.relative_to(REPO_ROOT)),
        "xclbin": str(EXP015.EXP130.XCLBIN.relative_to(REPO_ROOT)),
        "insts": str(EXP015.EXP130.INSTS.relative_to(REPO_ROOT)),
        "disasm": str(disasm.relative_to(REPO_ROOT)),
    }


def unique_record(record: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for word in record:
        if word not in seen:
            seen.append(word)
    return tuple(seen)


def run_probe(case: ProbeCase, timeout: int) -> tuple[ProbeResult, dict[str, str]]:
    artifacts = build_case_xclbin(case)
    runtime = EXP015.run_npu_case(EXP015.EXP130.XCLBIN, EXP015.EXP130.INSTS, host_case(), 1, timeout)
    record = tuple(runtime["record"])
    return (
        ProbeResult(
            name=case.name,
            target=case.target,
            sign=case.sign,
            shift=case.shift,
            output_cell=case.output_cell,
            runtime_status=runtime["status"],
            record=record,
            unique_record_words=unique_record(record),
            runtime_output=runtime["runtime_output"],
        ),
        artifacts,
    )


def build_manifest(args: argparse.Namespace) -> dict[str, JsonValue]:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    results: list[dict[str, JsonValue]] = []
    artifacts: dict[str, dict[str, str]] = {}
    for case in probe_cases(args.case_set, args.max_cases):
        result, case_artifacts = run_probe(case, args.runtime_timeout)
        results.append(cast(dict[str, JsonValue], asdict(result)))
        artifacts[case.name] = case_artifacts
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    status = "passed" if after["status"] == "ok" and all(row["runtime_status"] == "record_observed" for row in results) else "failed"
    return {
        "status": status,
        "case_set": args.case_set,
        "scratch_addr": hex(SCRATCH_ADDR),
        "cell_sentinels": {key: hex(value) for key, value in CELL_SENTINELS.items()},
        "lane_pattern_words": [hex(word) for word in LANE_PATTERN_WORDS],
        "topology_before": before,
        "topology_after": after,
        "results": results,
        "artifacts": artifacts,
    }


def render_report(manifest: dict[str, JsonValue]) -> str:
    lines = [
        "# VUPS.4x Lane Pattern Probe",
        "",
        f"Status: `{manifest['status']}`",
        "",
        f"- case set: `{manifest.get('case_set')}`",
        f"- scratch addr: `{manifest.get('scratch_addr')}`",
        "",
        "## Results",
        "",
        "| case | status | unique record words | first 8 record words |",
        "| --- | --- | --- | --- |",
    ]
    for item in cast(list[dict[str, JsonValue]], manifest.get("results", [])):
        unique = ", ".join(cast(list[str], item["unique_record_words"])[:8])
        first = ", ".join(cast(list[str], item["record"])[:8])
        lines.append(f"| `{item['name']}` | `{item['runtime_status']}` | `{unique}` | `{first}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This probe uses a non-uniform source vector loaded from local scratch memory.",
            "- `x2d` updates all four `dm0` quadrants with lane-pattern data.",
            "- `cml` updates only the low half (`bmll0`, `bmlh0`) and preserves the high-half sentinels.",
            "- `cmh` updates only the high half (`bmhl0`, `bmhh0`) and preserves the low-half sentinels.",
            "- `upssign1` sign-extends affected lane bytes before the destination writeback.",
            "- `s0=1` shifts the observed lane-pattern values left by one step in the sampled cases.",
            "- These value-level facts are the missing input for a stricter bundle manifest; compiler metadata alone was not enough.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-set", choices=("focused", "all"), default="focused")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        REPORT.write_text(
            "# VUPS.4x Lane Pattern Probe\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n",
            encoding="utf-8",
        )
        return 1
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
