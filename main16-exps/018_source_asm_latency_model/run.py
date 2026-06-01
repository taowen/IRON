#!/usr/bin/env python3
"""Sweep tiny source-asm producer/consumer gaps on real NPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP015_RUN = REPO_ROOT / "main16-exps/015_q4nx_tiny_codegen_numeric_gate/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "source_asm_latency_model.json"
REPORT = EXPERIMENT_DIR / "source_asm_latency_model.md"

PASS_MARKER = 0x5A5AA55A
FAIL_MARKER = 0xA55A5A5A


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_latency", EXP015_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 015 helper: {EXP015_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_exp015()
TinyCase = EXP015.TinyCase


@dataclass(frozen=True)
class Probe:
    kind: str
    gap: int
    expected: int


def configure_helpers() -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR


def input_case() -> TinyCase:
    return TinyCase(
        "latency_probe_input",
        EXP015.all_scale(),
        (0,),
        (0, 512),
        0x11111111,
        EXP015.BF16_SCALE_PAIR,
    )


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


def branch_delay(lines: list[str]) -> None:
    nops(lines, 8)


def emit_payload_tail(lines: list[str]) -> None:
    for _ in range(EXP015.RECORD_DWORDS - 2):
        lines.extend(
            [
                "  movxm r0, #0",
                "  st r0, [p0], #4",
            ]
        )


def emit_load_scale0(lines: list[str], dst: str, gap: int) -> None:
    lines.extend(
        [
            "  movxm p1, #0x72800",
            "  movxm m0, #0",
            "  padda [p1], m0",
            f"  lda {dst}, [p1, #0]",
        ]
    )
    nops(lines, gap)


def emit_probe(lines: list[str], probe: Probe) -> None:
    if probe.kind == "lda_to_st":
        lines.append("  movxm r0, #0")
        emit_load_scale0(lines, "r0", probe.gap)
        lines.append("  st r0, [p0], #4")
        return
    if probe.kind == "lda_to_eq_jnz":
        emit_load_scale0(lines, "r3", probe.gap)
        lines.extend(
            [
                "  mova r6, #0",
                "  eq r7, r3, r6",
            ]
        )
        nops(lines, 16)
        lines.append("  jnz r7, #.Lbad")
        branch_delay(lines)
        lines.append(f"  movxm r0, #0x{PASS_MARKER:x}")
        lines.append("  j #.Ldone")
        branch_delay(lines)
        lines.append(".Lbad:")
        lines.append(f"  movxm r0, #0x{FAIL_MARKER:x}")
        lines.append(".Ldone:")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        return
    if probe.kind == "lda_to_eq_jz":
        emit_load_scale0(lines, "r3", probe.gap)
        lines.extend(
            [
                "  mova r6, #0",
                "  eq r7, r3, r6",
            ]
        )
        nops(lines, 16)
        lines.append("  jz r7, #.Ltaken")
        branch_delay(lines)
        lines.append(f"  movxm r0, #0x{FAIL_MARKER:x}")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        lines.append("  j #.Ldone")
        branch_delay(lines)
        lines.append(".Ltaken:")
        lines.append(f"  movxm r0, #0x{PASS_MARKER:x}")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        lines.append(".Ldone:")
        return
    if probe.kind == "eq_to_jnz":
        lines.extend(
            [
                "  mova r3, #1",
                "  mova r6, #0",
                "  mova r7, #1",
            ]
        )
        nops(lines, 16)
        lines.append("  eq r7, r3, r6")
        nops(lines, probe.gap)
        lines.append("  jnz r7, #.Lbad")
        branch_delay(lines)
        lines.append(f"  movxm r0, #0x{PASS_MARKER:x}")
        lines.append("  j #.Ldone")
        branch_delay(lines)
        lines.append(".Lbad:")
        lines.append(f"  movxm r0, #0x{FAIL_MARKER:x}")
        lines.append(".Ldone:")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        return
    if probe.kind == "eq_to_jz":
        lines.extend(
            [
                "  mova r3, #1",
                "  mova r6, #0",
            ]
        )
        nops(lines, 16)
        lines.append("  eq r7, r3, r6")
        nops(lines, probe.gap)
        lines.append("  jz r7, #.Ltaken")
        branch_delay(lines)
        lines.append(f"  movxm r0, #0x{FAIL_MARKER:x}")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        lines.append("  j #.Ldone")
        branch_delay(lines)
        lines.append(".Ltaken:")
        lines.append(f"  movxm r0, #0x{PASS_MARKER:x}")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        lines.append(".Ldone:")
        return
    if probe.kind == "and_to_eq":
        lines.extend(
            [
                "  mova r5, #0",
                "  movxm r3, #0x11111111",
                "  movxm r4, #0xf",
                "  mova r6, #0",
            ]
        )
        nops(lines, 16)
        lines.append("  and r5, r3, r4")
        nops(lines, probe.gap)
        lines.append("  eq r7, r5, r6")
        nops(lines, 16)
        lines.append("  jnz r7, #.Lbad")
        branch_delay(lines)
        lines.append(f"  movxm r0, #0x{PASS_MARKER:x}")
        lines.append("  j #.Ldone")
        branch_delay(lines)
        lines.append(".Lbad:")
        lines.append(f"  movxm r0, #0x{FAIL_MARKER:x}")
        lines.append(".Ldone:")
        nops(lines, 8)
        lines.append("  st r0, [p0], #4")
        return
    if probe.kind == "and_to_eq_value":
        lines.extend(
            [
                "  mova r5, #0",
                "  movxm r3, #0x11111111",
                "  movxm r4, #0xf",
                "  mova r6, #0",
            ]
        )
        nops(lines, 16)
        lines.append("  and r5, r3, r4")
        nops(lines, probe.gap)
        lines.append("  eq r7, r5, r6")
        nops(lines, 16)
        lines.append("  st r7, [p0], #4")
        return
    if probe.kind == "lshl_to_or":
        lines.extend(
            [
                "  movxm r8, #0",
                "  movxm r9, #0x3c80",
                "  mova r10, #16",
            ]
        )
        nops(lines, 16)
        lines.append("  lshl r9, r9, r10")
        nops(lines, probe.gap)
        lines.append("  or r8, r8, r9")
        nops(lines, 16)
        lines.append("  st r8, [p0], #4")
        return
    raise ValueError(f"unknown probe kind: {probe.kind}")


def probe_asm(probe: Probe) -> str:
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
            "  movxm r0, #0x1",
            "  st r0, [p0], #4",
        ]
    )
    emit_probe(lines, probe)
    emit_payload_tail(lines)
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


def safe_name(probe: Probe) -> str:
    return f"{probe.kind}_gap{probe.gap}"


def build_probe_xclbin(probe: Probe) -> dict[str, str]:
    build = BUILD_DIR / safe_name(probe)
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / "source_asm_latency_probe.s"
    obj = build / "source_asm_latency_probe.o"
    ld = build / "source_asm_latency_probe.ld"
    elf = build / "source_asm_latency_probe.elf"
    disasm = build / "source_asm_latency_probe.disasm.s"
    asm.write_text(probe_asm(probe))
    ld.write_text(EXP015.tiny_linker_script())
    EXP015.run_cmd((str(EXP015.CLANG), "--target=aie2p-none-unknown-elf", "-c", str(asm), "-o", str(obj)), build)
    EXP015.run_cmd((str(EXP015.LD_LLD), "-T", str(ld), str(obj), "-o", str(elf)), build)
    disasm.write_text(EXP015.run_cmd((str(EXP015.LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf)), build).stdout)
    EXP015.configure_exp130(build, elf.name)
    EXP015.EXP130.write_mlir(1)
    EXP015.EXP130.package_with_aiecc()
    return {
        "asm": str(asm.relative_to(REPO_ROOT)),
        "elf": str(elf.relative_to(REPO_ROOT)),
        "xclbin": str(EXP015.EXP130.XCLBIN.relative_to(REPO_ROOT)),
        "insts": str(EXP015.EXP130.INSTS.relative_to(REPO_ROOT)),
        "disasm": str(disasm.relative_to(REPO_ROOT)),
    }


def run_probe(probe: Probe, runtime_timeout: int) -> dict[str, object]:
    artifacts = build_probe_xclbin(probe)
    runtime = EXP015.run_npu_case(EXP015.EXP130.XCLBIN, EXP015.EXP130.INSTS, input_case(), 1, runtime_timeout)
    record = tuple(int(word, 16) for word in runtime["record"])
    observed = record[1] if len(record) > 1 else None
    passed = runtime["status"] == "record_observed" and observed == probe.expected
    return {
        "kind": probe.kind,
        "gap": probe.gap,
        "expected": hex(probe.expected & 0xFFFFFFFF),
        "observed": None if observed is None else hex(observed & 0xFFFFFFFF),
        "passed": passed,
        "runtime_status": runtime["status"],
        "runtime_output": runtime.get("runtime_output", ""),
        "artifacts": artifacts,
    }


def probes(max_load_gap: int, max_alu_gap: int) -> tuple[Probe, ...]:
    items: list[Probe] = []
    for gap in range(max_load_gap + 1):
        items.append(Probe("lda_to_st", gap, EXP015.BF16_SCALE_PAIR))
        items.append(Probe("lda_to_eq_jz", gap, PASS_MARKER))
    for gap in range(max_alu_gap + 1):
        items.append(Probe("eq_to_jz", gap, PASS_MARKER))
        items.append(Probe("and_to_eq_value", gap, 0))
        items.append(Probe("lshl_to_or", gap, 0x3C800000))
    return tuple(items)


def summarize_min_gaps(results: list[dict[str, object]]) -> dict[str, int | None]:
    summary: dict[str, int | None] = {}
    for result in results:
        kind = str(result["kind"])
        if kind not in summary:
            summary[kind] = None
        if result["passed"] and summary[kind] is None:
            summary[kind] = int(result["gap"])
    return summary


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    results = [run_probe(probe, args.runtime_timeout) for probe in probes(args.max_load_gap, args.max_alu_gap)]
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    min_gaps = summarize_min_gaps(results)
    complete = after["status"] == "ok" and all(value is not None for value in min_gaps.values())
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "min_safe_gaps": min_gaps,
        "results": results,
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Source ASM Latency Model",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Minimum Safe Gaps",
        "",
        "| dependency | min nops |",
        "| --- | --- |",
    ]
    min_gaps = manifest.get("min_safe_gaps", {})
    if isinstance(min_gaps, dict):
        for kind, gap in min_gaps.items():
            lines.append(f"| `{kind}` | `{gap}` |")
    lines.extend(
        [
            "",
            "## Sweep",
            "",
            "| dependency | gap | observed | expected | pass | runtime |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in manifest.get("results", []):
        if not isinstance(result, dict):
            continue
        lines.append(
            f"| `{result['kind']}` | `{result['gap']}` | `{result['observed']}` | "
            f"`{result['expected']}` | `{result['passed']}` | `{result['runtime_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are conservative real-NPU gaps for source assembly emitted without "
            "Peano scheduling. A future MyLM-style generator should fill the gaps "
            "with independent work, not literal nops. The values are still useful "
            "as a hazard-checking lower bound for generated tiny kernels.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-load-gap", type=int, default=16)
    parser.add_argument("--max-alu-gap", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Source ASM Latency Model\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
