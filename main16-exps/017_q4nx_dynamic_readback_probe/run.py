#!/usr/bin/env python3
"""Read back candidate DMA buffer addresses from a tiny source-asm core."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_dynamic_readback_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_dynamic_readback_probe.md"


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_readback", EXP015_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 015 helper: {EXP015_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_exp015()
TinyCase = EXP015.TinyCase


@dataclass(frozen=True)
class ReadWord:
    name: str
    base: int
    offset: int
    expected: int


READ_WORDS = (
    ReadWord("full_weight_scale0", 0x72800, 0x0, EXP015.BF16_SCALE_PAIR),
    ReadWord("full_weight_zero0", 0x72800, 0x200, EXP015.BF16_SCALE_PAIR),
    ReadWord("full_weight_q4word0", 0x72800, 0x400, 0x11111111),
    ReadWord("full_weight_q4word512", 0x72800, 0xC00, 0x11111111),
    ReadWord("local_weight_scale0", 0x2800, 0x0, EXP015.BF16_SCALE_PAIR),
    ReadWord("local_weight_zero0", 0x2800, 0x200, EXP015.BF16_SCALE_PAIR),
    ReadWord("local_weight_q4word0", 0x2800, 0x400, 0x11111111),
    ReadWord("local_weight_q4word512", 0x2800, 0xC00, 0x11111111),
    ReadWord("full_activation_word0", 0x78000, 0x0, EXP015.BF16_ONE_PAIR),
    ReadWord("local_activation_word0", 0x8000, 0x0, EXP015.BF16_ONE_PAIR),
    ReadWord("full_record_header", 0x73C1C, 0x0, 0x1),
    ReadWord("local_record_header", 0x3C1C, 0x0, 0x1),
    ReadWord("full_weight_pong_scale0", 0x74000, 0x0, 0x0),
    ReadWord("full_activation_pong_word0", 0x7C000, 0x0, 0x0),
    ReadWord("full_weight_scale1", 0x72800, 0x4, EXP015.BF16_SCALE_PAIR),
    ReadWord("full_weight_q4word1", 0x72800, 0x404, 0x0),
)


def configure_helpers() -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR


def readback_case() -> TinyCase:
    return TinyCase(
        "readback_active_chunk0",
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


def emit_read_word(lines: list[str], word: ReadWord) -> None:
    lines.extend(
        [
            f"  // {word.name}",
            f"  movxm p1, #0x{word.base:x}",
            f"  movxm m0, #0x{word.offset:x}",
            "  padda [p1], m0",
            "  lda r0, [p1, #0]",
        ]
    )
    lines.extend(["  nop"] * 16)
    lines.append("  st r0, [p0], #4")


def readback_asm() -> str:
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
    for word in READ_WORDS:
        emit_read_word(lines, word)
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


def build_readback_xclbin() -> dict:
    build = BUILD_DIR / "readback"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / "q4nx_dynamic_readback.s"
    obj = build / "q4nx_dynamic_readback.o"
    ld = build / "q4nx_dynamic_readback.ld"
    elf = build / "q4nx_dynamic_readback.elf"
    disasm = build / "q4nx_dynamic_readback.disasm.s"
    asm.write_text(readback_asm())
    ld.write_text(EXP015.tiny_linker_script())
    EXP015.run_cmd((str(EXP015.CLANG), "--target=aie2p-none-unknown-elf", "-c", str(asm), "-o", str(obj)), build)
    EXP015.run_cmd((str(EXP015.LD_LLD), "-T", str(ld), str(obj), "-o", str(elf)), build)
    disasm.write_text(EXP015.run_cmd((str(EXP015.LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf)), build).stdout)
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


def classify_record(record: tuple[int, ...]) -> list[dict[str, object]]:
    observations = []
    for index, word in enumerate(READ_WORDS, start=1):
        observed = record[index] if index < len(record) else None
        observations.append(
            {
                "name": word.name,
                "base": hex(word.base),
                "offset": hex(word.offset),
                "observed": None if observed is None else hex(observed & 0xFFFFFFFF),
                "expected": hex(word.expected & 0xFFFFFFFF),
                "matches_expected": observed == word.expected,
            }
        )
    return observations


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    artifacts = build_readback_xclbin()
    runtime = EXP015.run_npu_case(EXP015.EXP130.XCLBIN, EXP015.EXP130.INSTS, readback_case(), 1, args.runtime_timeout)
    record = tuple(int(word, 16) for word in runtime["record"])
    observations = classify_record(record)
    full_weight_ok = all(item["matches_expected"] for item in observations[:4])
    full_activation_ok = observations[8]["matches_expected"]
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    status = "passed" if runtime["status"] == "record_observed" and full_weight_ok and full_activation_ok and after["status"] == "ok" else "failed"
    return {
        "status": status,
        "topology_before": before,
        "topology_after": after,
        "artifacts": artifacts,
        "runtime": runtime,
        "observations": observations,
        "interpretation": {
            "full_weight_address_visible": full_weight_ok,
            "full_activation_address_visible": full_activation_ok,
        },
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Dynamic Readback Probe",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Observations",
        "",
        "| name | base | offset | observed | expected | match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest.get("observations", []):
        lines.append(
            f"| `{item['name']}` | `{item['base']}` | `{item['offset']}` | "
            f"`{item['observed']}` | `{item['expected']}` | `{item['matches_expected']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass means source assembly can see the DMA-written full-address "
            "weight and activation buffers immediately after the normal full-lock "
            "acquire. Experiment 016 can then treat any remaining mismatch as "
            "generated arithmetic/control-flow, not buffer visibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Dynamic Readback Probe\n\nExperiment failed.\n\n```text\n"
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
