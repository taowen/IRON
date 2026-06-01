#!/usr/bin/env python3
"""Generate dynamic tiny Q4NX arithmetic and compare it with MyLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP015_RUN = REPO_ROOT / "main16-exps/015_q4nx_tiny_codegen_numeric_gate/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_dynamic_tiny_arithmetic_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_dynamic_tiny_arithmetic_gate.md"


def load_exp015() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp015_for_dynamic_tiny", EXP015_RUN)
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


def dynamic_cases() -> tuple[TinyCase, ...]:
    return (
        TinyCase("dyn_q4word0_allnibbles", EXP015.all_scale(), (), (0,), 0x11111111, 0),
        TinyCase("dyn_q4word512_allnibbles", EXP015.all_scale(), (), (512,), 0x11111111, 0),
        TinyCase("dyn_q4word0_nibble3", EXP015.all_scale(), (), (0,), 1 << 12, 0),
        TinyCase("dyn_q4word0_plus2_accum", EXP015.all_scale(), (), (0, 2), 0x11111111, 0),
        TinyCase("dyn_zero0_q4word0", EXP015.all_scale(), (0,), (0,), 0x11111111, EXP015.BF16_SCALE_PAIR),
    )


def q4_offset(q4_index: int) -> int:
    return (EXP015.SCALE_DWORDS + EXP015.ZERO_DWORDS + q4_index) * 4


def scale_offset(word_index: int) -> int:
    return word_index * 4


def zero_offset(word_index: int) -> int:
    return (EXP015.SCALE_DWORDS + word_index) * 4


def q4_nibbles_for_word(case: TinyCase, word_index: int) -> tuple[tuple[int, int], ...]:
    items: list[tuple[int, int]] = []
    for q4_index in case.q4_indices:
        half_base = 0 if q4_index < 512 else 8
        quartet = half_base + (0 if q4_index % 2 == 0 else 4)
        if quartet <= word_index < quartet + 4:
            low_nibble = (word_index - quartet) * 2
            items.append((q4_index, low_nibble))
            items.append((q4_index, low_nibble + 1))
    return tuple(items)


def emit_load_word(lines: list[str], pointer: str, offset: int, dst_reg: str) -> None:
    lines.extend(
        [
            f"  movxm {pointer}, #0x72800",
            f"  movxm m0, #0x{offset:x}",
            f"  padda [{pointer}], m0",
            f"  lda {dst_reg}, [{pointer}, #0]",
        ]
    )
    lines.extend(["  nop"] * 16)


def emit_branch_nops(lines: list[str]) -> None:
    lines.extend(["  nop"] * 8)


def emit_jump(lines: list[str], label: str) -> None:
    lines.append(f"  j #{label}")
    emit_branch_nops(lines)


def emit_branch_if_zero(lines: list[str], condition_reg: str, label: str) -> None:
    lines.append(f"  jz {condition_reg}, #{label}")
    emit_branch_nops(lines)


def emit_add_if_q4_nibble(lines: list[str], q4_index: int, nibble_index: int, count_reg: str, label: str) -> None:
    mask = 0xF << (4 * nibble_index)
    emit_load_word(lines, "p1", q4_offset(q4_index), "r3")
    lines.extend(
        [
            f"  movxm r4, #0x{mask:x}",
            "  and r5, r3, r4",
            "  nop",
            "  nop",
            "  nop",
            "  mova r6, #0",
            "  eq r7, r5, r6",
            "  nop",
            "  nop",
            "  nop",
        ]
    )
    add_label = f"{label}_add"
    emit_branch_if_zero(lines, "r7", add_label)
    emit_jump(lines, label)
    lines.append(f"{add_label}:")
    lines.append(f"  add {count_reg}, {count_reg}, #1")
    lines.append(f"{label}:")


def emit_add_if_zero_slot(lines: list[str], word_index: int, count_reg: str, label: str) -> None:
    emit_load_word(lines, "p1", zero_offset(word_index), "r3")
    lines.extend(
        [
            "  mova r6, #0",
            "  eq r7, r3, r6",
            "  nop",
            "  nop",
            "  nop",
        ]
    )
    add_label = f"{label}_add"
    emit_branch_if_zero(lines, "r7", add_label)
    emit_jump(lines, label)
    lines.append(f"{add_label}:")
    lines.append(f"  add {count_reg}, {count_reg}, #32")
    lines.append(f"{label}:")


def emit_skip_if_scale_zero(lines: list[str], word_index: int, label: str) -> None:
    body_label = f"{label}_body"
    emit_load_word(lines, "p1", scale_offset(word_index), "r3")
    lines.extend(
        [
            "  mova r6, #0",
            "  eq r7, r3, r6",
            "  nop",
            "  nop",
            "  nop",
        ]
    )
    emit_branch_if_zero(lines, "r7", body_label)
    emit_jump(lines, label)
    lines.append(f"{body_label}:")


def count_to_bf16(count: int) -> int:
    return EXP015.float_to_bf16(float(count) / 64.0)


def emit_count_to_bf16(lines: list[str], count_reg: str, out_reg: str, max_count: int, prefix: str) -> None:
    lines.append(f"  movxm {out_reg}, #0")
    for count in range(max_count + 1):
        lines.extend(
            [
                f"  mova r6, #{count}",
                f"  eq r7, {count_reg}, r6",
                "  nop",
                "  nop",
                "  nop",
            ]
        )
        next_label = f".L{prefix}_next_{count}"
        emit_branch_if_zero(lines, "r7", next_label)
        lines.append(f"  movxm {out_reg}, #0x{count_to_bf16(count):x}")
        emit_jump(lines, f".L{prefix}_done")
        lines.append(f"{next_label}:")
    lines.append(f".L{prefix}_done:")


def emit_compute_payload_word(lines: list[str], case: TinyCase, word_index: int, max_count: int) -> None:
    skip_q4 = f".Lword_{word_index}_scale_zero"
    lines.extend(
        [
            f"  // payload word {word_index}",
            "  mova r1, #0",
            "  mova r2, #0",
        ]
    )
    emit_skip_if_scale_zero(lines, word_index, skip_q4)
    for item_index, (q4_index, nibble_index) in enumerate(q4_nibbles_for_word(case, word_index)):
        target_reg = "r1" if nibble_index % 2 == 0 else "r2"
        emit_add_if_q4_nibble(
            lines,
            q4_index,
            nibble_index,
            target_reg,
            f".Lword_{word_index}_q4_{item_index}_zero",
        )
    lines.append(f"{skip_q4}:")
    emit_add_if_zero_slot(lines, word_index, "r1", f".Lword_{word_index}_zero_low_absent")
    emit_add_if_zero_slot(lines, word_index, "r2", f".Lword_{word_index}_zero_high_absent")
    emit_count_to_bf16(lines, "r1", "r8", max_count, f"word_{word_index}_low")
    emit_count_to_bf16(lines, "r2", "r9", max_count, f"word_{word_index}_high")
    lines.extend(
        [
            "  mova r10, #16",
            "  lshl r9, r9, r10",
            "  nop",
            "  nop",
            "  nop",
            "  or r8, r8, r9",
            "  nop",
            "  nop",
            "  nop",
            "  st r8, [p0], #4",
        ]
    )


def max_count_for_case(case: TinyCase) -> int:
    max_count = 0
    for word_index in range(16):
        low = 32 if word_index in {index % 16 for index in case.zero_indices} else 0
        high = low
        for _q4_index, nibble_index in q4_nibbles_for_word(case, word_index):
            if nibble_index % 2 == 0:
                low += 1
            else:
                high += 1
        max_count = max(max_count, low, high)
    return max_count


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


def dynamic_asm(case: TinyCase) -> str:
    max_count = max_count_for_case(case)
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
    for word_index in range(16):
        emit_compute_payload_word(lines, case, word_index, max_count)
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


def build_dynamic_xclbin(case: TinyCase) -> dict:
    build = BUILD_DIR / f"dynamic_{case.name}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / "dynamic_tiny_q4nx.s"
    obj = build / "dynamic_tiny_q4nx.o"
    ld = build / "dynamic_tiny_q4nx.ld"
    elf = build / "dynamic_tiny_q4nx.elf"
    disasm = build / "dynamic_tiny_q4nx.disasm.s"
    asm.write_text(dynamic_asm(case))
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
        "disasm_counts": {
            "acq": disasm.read_text().count("acq"),
            "rel": disasm.read_text().count("rel"),
            "lda": disasm.read_text().count("lda"),
            "and": disasm.read_text().count("and"),
            "st": disasm.read_text().count("\tst\t"),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    EXP015.configure_exp008()
    before = EXP015.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    EXP015.EXP008.build_qkv_direct_xclbin()
    selected_cases = dynamic_cases()
    if args.max_cases > 0:
        selected_cases = selected_cases[: args.max_cases]
    results = []
    for case in selected_cases:
        expected = EXP015.expected_record_words(case)
        mylm = EXP015.run_npu_case(
            EXP015.EXP008.EXP005.EXP132.EXP130.XCLBIN,
            EXP015.EXP008.EXP005.EXP132.EXP130.INSTS,
            case,
            EXP015.EXP008.RECORDS,
            args.runtime_timeout,
        )
        mylm_record = tuple(int(word, 16) for word in mylm["record"])
        dynamic_artifacts = build_dynamic_xclbin(case)
        dynamic = EXP015.run_npu_case(EXP015.EXP130.XCLBIN, EXP015.EXP130.INSTS, case, 1, args.runtime_timeout)
        dynamic_record = tuple(int(word, 16) for word in dynamic["record"])
        results.append(
            {
                "name": case.name,
                "expected_record": [hex(word & 0xFFFFFFFF) for word in expected],
                "mylm": mylm,
                "dynamic": dynamic,
                "dynamic_artifacts": dynamic_artifacts,
                "formula_matches_mylm": mylm["status"] == "record_observed" and mylm_record == expected,
                "dynamic_matches_expected": dynamic["status"] == "record_observed" and dynamic_record == expected,
                "dynamic_matches_mylm": dynamic["status"] == "record_observed"
                and mylm["status"] == "record_observed"
                and dynamic_record == mylm_record,
            }
        )
    after = EXP015.EXP008.topology_status(args.xrt_timeout)
    complete = (
        after["status"] == "ok"
        and all(result["formula_matches_mylm"] for result in results)
        and all(result["dynamic_matches_expected"] for result in results)
        and all(result["dynamic_matches_mylm"] for result in results)
    )
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "results": results,
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Dynamic Tiny Arithmetic Gate",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Results",
        "",
        "| case | formula == MyLM | dynamic == expected | dynamic == MyLM | dynamic status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in manifest.get("results", []):
        lines.append(
            f"| `{result['name']}` | `{result['formula_matches_mylm']}` | "
            f"`{result['dynamic_matches_expected']}` | `{result['dynamic_matches_mylm']}` | "
            f"`{result['dynamic']['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This generated whole-core source-assembly body reads q4/scale/zero fields "
            "from the active weight chunk, computes payload-word low/high bf16 "
            "halves through a tiny count-to-bf16 path, consumes the same 16 stream "
            "chunks, and emits one compact record. It is intentionally narrow and "
            "not yet the MyLM software-pipelined hot loop.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Dynamic Tiny Arithmetic Gate\n\nExperiment failed.\n\n```text\n"
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
