#!/usr/bin/env python3
"""Run isolated NPU readback probes for VUPS.4x accumulator-cell effects."""

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
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "vups4x_cell_readback_probe.json"
REPORT = EXPERIMENT_DIR / "vups4x_cell_readback_probe.md"

SENTINEL_BF16 = 0x3C80
PROBE_BF16 = 0x4000
OUTPUT_CELLS = ("bmll0", "bmlh0", "bmhl0", "bmhh0")


@dataclass(frozen=True)
class ProbeCase:
    name: str
    variant: str
    output_cell: str


@dataclass(frozen=True)
class ProbeResult:
    name: str
    variant: str
    output_cell: str
    runtime_status: str
    record: tuple[str, ...]
    unique_record_words: tuple[str, ...]
    runtime_output: str


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_vups4x_cell_readback", EXP015_RUN)
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
        "vups4x_cell_readback_input",
        EXP015.all_scale(),
        (0,),
        (0, 512),
        0x11111111,
        EXP015.BF16_SCALE_PAIR,
    )


def probe_cases(max_cases: int) -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    for variant in ("x2d_upssign0", "w2c_upssign0"):
        for output_cell in OUTPUT_CELLS:
            cases.append(ProbeCase(f"{variant}_{output_cell}", variant, output_cell))
    if max_cases > 0:
        return tuple(cases[:max_cases])
    return tuple(cases)


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


def nops(lines: list[str], count: int) -> None:
    lines.extend(["  nop"] * count)


def init_dm0_sentinel(lines: list[str]) -> None:
    lines.extend(
        [
            f"  movxm r2, #0x{SENTINEL_BF16:x}",
            "  vbcst.16 x0, r2",
        ]
    )
    nops(lines, 12)
    for cell in OUTPUT_CELLS:
        lines.append(f"  vups.2x {cell}, wl0, s0, upssign0")
        nops(lines, 6)


def run_target_vups(lines: list[str], variant: str) -> None:
    lines.extend(
        [
            f"  movxm r3, #0x{PROBE_BF16:x}",
            "  vbcst.16 x1, r3",
        ]
    )
    nops(lines, 12)
    if variant == "x2d_upssign0":
        lines.append("  vups.4x dm0, x1, s0, upssign0")
    elif variant == "w2c_upssign0":
        lines.append("  vups.4x cml0, wl1, s0, upssign0")
    else:
        raise ValueError(f"unknown variant: {variant}")
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
            "  mova r1, #0",
            "  mov s0, r1",
        ]
    )
    nops(lines, 8)
    init_dm0_sentinel(lines)
    run_target_vups(lines, case.variant)
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
            variant=case.variant,
            output_cell=case.output_cell,
            runtime_status=runtime["status"],
            record=record,
            unique_record_words=unique_record(record),
            runtime_output=runtime["runtime_output"],
        ),
        artifacts,
    )


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    results: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, str]] = {}
    for case in probe_cases(args.max_cases):
        result, case_artifacts = run_probe(case, args.runtime_timeout)
        results.append(asdict(result))
        artifacts[case.name] = case_artifacts
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    status = "passed" if after["status"] == "ok" and all(row["runtime_status"] == "record_observed" for row in results) else "failed"
    return {
        "status": status,
        "sentinel_bf16": hex(SENTINEL_BF16),
        "probe_bf16": hex(PROBE_BF16),
        "topology_before": before,
        "topology_after": after,
        "results": results,
        "artifacts": artifacts,
        "interpretation": {
            "same_word_across_record": (
                "For this first probe, a uniform record means the selected quadrant was internally consistent."
            ),
            "different_words": (
                "Mixed record words indicate lane-level behavior and need a follow-up lane-pattern probe."
            ),
        },
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# VUPS.4x Cell Readback Probe",
        "",
        f"Status: `{manifest['status']}`",
        "",
        f"- sentinel bf16: `{manifest.get('sentinel_bf16')}`",
        f"- probe bf16: `{manifest.get('probe_bf16')}`",
        "",
        "## Results",
        "",
        "| case | status | unique record words | first record words |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest.get("results", []):
        unique = ", ".join(row["unique_record_words"])
        first = ", ".join(row["record"][:4])
        lines.append(f"| `{row['name']}` | `{row['runtime_status']}` | `{unique}` | `{first}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a value-semantics probe, not a performance probe.",
            "- The compact record header is intentionally not meaningful here; the selected vector cell is stored from word 0.",
            "- `x2d` should reveal whether a full `dm` destination overwrites all four accumulator quadrants.",
            "- `w2c` should reveal whether a `cml/cmh` destination preserves the other half of `dm0` initialized by the sentinel.",
            "- If output words are neither sentinel-like nor probe-like, the next experiment must use lane-pattern inputs rather than uniform broadcasts.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=0)
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
            "# VUPS.4x Cell Readback Probe\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n",
            encoding="utf-8",
        )
        return 1
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
