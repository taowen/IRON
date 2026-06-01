#!/usr/bin/env python3
"""Probe scalar eq/jz/jnz semantics in source assembly."""

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
MANIFEST = EXPERIMENT_DIR / "source_asm_branch_semantics_probe.json"
REPORT = EXPERIMENT_DIR / "source_asm_branch_semantics_probe.md"

TAKEN = 0x11110001
FALLTHROUGH = 0x22220002


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_branch", EXP015_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 015 helper: {EXP015_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP015 = load_exp015()
TinyCase = EXP015.TinyCase


@dataclass(frozen=True)
class PayloadWord:
    name: str
    observed: int
    meaning: str


def configure_helpers() -> None:
    EXP015.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP015.BUILD_DIR = BUILD_DIR


def input_case() -> TinyCase:
    return TinyCase("branch_semantics_input", (), (), (), 0, 0)


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


def store_r(lines: list[str], register: str) -> None:
    nops(lines, 8)
    lines.append(f"  st {register}, [p0], #4")


def store_marker(lines: list[str], marker: int) -> None:
    lines.append(f"  movxm r0, #0x{marker:x}")
    store_r(lines, "r0")


def emit_eq_value(lines: list[str], name: str, left: int, right: int) -> None:
    lines.append(f"  // {name}")
    lines.extend(
        [
            f"  mova r3, #{left}",
            f"  mova r6, #{right}",
        ]
    )
    nops(lines, 16)
    lines.append("  eq r7, r3, r6")
    store_r(lines, "r7")


def emit_branch_value(lines: list[str], name: str, mnemonic: str, value: int) -> None:
    target = f".L{name}_taken"
    done = f".L{name}_done"
    lines.append(f"  // {name}")
    lines.append(f"  mova r7, #{value}")
    nops(lines, 16)
    lines.append(f"  {mnemonic} r7, #{target}")
    nops(lines, 8)
    store_marker(lines, FALLTHROUGH)
    lines.append(f"  j #{done}")
    nops(lines, 8)
    lines.append(f"{target}:")
    store_marker(lines, TAKEN)
    lines.append(f"{done}:")


def emit_eq_branch(lines: list[str], name: str, mnemonic: str, left: int, right: int) -> None:
    target = f".L{name}_taken"
    done = f".L{name}_done"
    lines.append(f"  // {name}")
    lines.extend(
        [
            f"  mova r3, #{left}",
            f"  mova r6, #{right}",
        ]
    )
    nops(lines, 16)
    lines.append("  eq r7, r3, r6")
    nops(lines, 16)
    lines.append(f"  {mnemonic} r7, #{target}")
    nops(lines, 8)
    store_marker(lines, FALLTHROUGH)
    lines.append(f"  j #{done}")
    nops(lines, 8)
    lines.append(f"{target}:")
    store_marker(lines, TAKEN)
    lines.append(f"{done}:")


def emit_payload_tail(lines: list[str], used_payload_words: int) -> None:
    for _ in range(16 - used_payload_words):
        store_marker(lines, 0)


def branch_asm() -> str:
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
    emit_eq_value(lines, "eq_equal", 0, 0)
    emit_eq_value(lines, "eq_notequal", 1, 0)
    emit_branch_value(lines, "jz_zero", "jz", 0)
    emit_branch_value(lines, "jz_one", "jz", 1)
    emit_branch_value(lines, "jnz_zero", "jnz", 0)
    emit_branch_value(lines, "jnz_one", "jnz", 1)
    emit_eq_branch(lines, "eq_equal_jz", "jz", 0, 0)
    emit_eq_branch(lines, "eq_notequal_jz", "jz", 1, 0)
    emit_eq_branch(lines, "eq_equal_jnz", "jnz", 0, 0)
    emit_eq_branch(lines, "eq_notequal_jnz", "jnz", 1, 0)
    emit_payload_tail(lines, 10)
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


def build_xclbin() -> dict[str, str]:
    build = BUILD_DIR / "branch_semantics"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / "source_asm_branch_semantics.s"
    obj = build / "source_asm_branch_semantics.o"
    ld = build / "source_asm_branch_semantics.ld"
    elf = build / "source_asm_branch_semantics.elf"
    disasm = build / "source_asm_branch_semantics.disasm.s"
    asm.write_text(branch_asm())
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


def explain_payload(record: tuple[int, ...]) -> tuple[PayloadWord, ...]:
    names = (
        "eq_equal",
        "eq_notequal",
        "jz_zero",
        "jz_one",
        "jnz_zero",
        "jnz_one",
        "eq_equal_jz",
        "eq_notequal_jz",
        "eq_equal_jnz",
        "eq_notequal_jnz",
    )
    words: list[PayloadWord] = []
    for index, name in enumerate(names, start=1):
        observed = record[index] if index < len(record) else 0
        if observed == TAKEN:
            meaning = "branch_taken"
        elif observed == FALLTHROUGH:
            meaning = "fallthrough"
        else:
            meaning = f"value_{observed}"
        words.append(PayloadWord(name, observed, meaning))
    return tuple(words)


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    artifacts = build_xclbin()
    runtime = EXP015.run_npu_case(EXP015.EXP130.XCLBIN, EXP015.EXP130.INSTS, input_case(), 1, args.runtime_timeout)
    record = tuple(int(word, 16) for word in runtime["record"])
    payload = explain_payload(record)
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    return {
        "status": "passed" if runtime["status"] == "record_observed" and after["status"] == "ok" else "failed",
        "topology_before": before,
        "topology_after": after,
        "artifacts": artifacts,
        "runtime": runtime,
        "payload": [
            {
                "name": word.name,
                "observed": hex(word.observed & 0xFFFFFFFF),
                "meaning": word.meaning,
            }
            for word in payload
        ],
        "markers": {
            "branch_taken": hex(TAKEN),
            "fallthrough": hex(FALLTHROUGH),
        },
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Source ASM Branch Semantics Probe",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Payload",
        "",
        "| test | observed | meaning |",
        "| --- | --- | --- |",
    ]
    for item in manifest.get("payload", []):
        if isinstance(item, dict):
            lines.append(f"| `{item['name']}` | `{item['observed']}` | `{item['meaning']}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`eq` and conditional branch semantics must be taken from this table when "
            "generating scalar control flow. The experiment exists because the "
            "mnemonic spelling alone was not a safe guide for `eq` + `jz/jnz`.",
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
            "# Source ASM Branch Semantics Probe\n\nExperiment failed.\n\n```text\n"
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
