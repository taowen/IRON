#!/usr/bin/env python3
"""Generate a tiny source-assembly record body and compare it with MyLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP008_RUN = REPO_ROOT / "main16-exps/008_q4nx_payload_formula_probe/run.py"
EXP130_RUN = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/run.py"
LLVM_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
CLANG = LLVM_AIE_BIN / "clang"
LD_LLD = LLVM_AIE_BIN / "ld.lld"
LLVM_OBJDUMP = LLVM_AIE_BIN / "llvm-objdump"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_tiny_codegen_numeric_gate.json"
REPORT = EXPERIMENT_DIR / "q4nx_tiny_codegen_numeric_gate.md"

RECORD_DWORDS = 17
RECORD_CHUNKS = 16
ACTIVATION_DWORDS_PER_CHUNK = 128
WEIGHT_DWORDS_PER_CHUNK = 1280
SCALE_DWORDS = 128
ZERO_DWORDS = 128
Q4_DATA_DWORDS = 1024
Q4_CHUNK_DWORDS = SCALE_DWORDS + ZERO_DWORDS + Q4_DATA_DWORDS
BF16_ONE_PAIR = 0x3F803F80
BF16_SCALE_PAIR = 0x3C803C80


@dataclass(frozen=True)
class TinyCase:
    name: str
    scale_indices: tuple[int, ...]
    zero_indices: tuple[int, ...]
    q4_indices: tuple[int, ...]
    q4_word: int
    zero_pair: int
    active_chunk: int = 0


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP008 = load_module("mylm_exp008_for_tiny_codegen", EXP008_RUN)
EXP130 = load_module("mylm_exp130_for_tiny_codegen", EXP130_RUN)


def all_scale() -> tuple[int, ...]:
    return tuple(range(SCALE_DWORDS))


def all_q4() -> tuple[int, ...]:
    return tuple(range(Q4_DATA_DWORDS))


def cases() -> tuple[TinyCase, ...]:
    return (
        TinyCase("q4word0_allnibbles", all_scale(), (), (0,), 0x11111111, 0),
        TinyCase("q4word512_allnibbles", all_scale(), (), (512,), 0x11111111, 0),
        TinyCase("q4word0_nibble3", all_scale(), (), (0,), 1 << 12, 0),
        TinyCase("zero0_allq4", all_scale(), (0,), all_q4(), 0x11111111, BF16_SCALE_PAIR),
    )


def bf16_pair(value: int) -> int:
    return ((value & 0xFFFF) << 16) | (value & 0xFFFF)


def bf16_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (value & 0xFFFF) << 16))[0]


def float_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def expected_payload_words(case: TinyCase) -> tuple[int, ...]:
    halves = [[0.0, 0.0] for _ in range(16)]
    active_scales = {index % 16 for index in case.scale_indices}
    scale_value = bf16_to_float(0x3C80)
    for q4_index in case.q4_indices:
        half = 0 if q4_index < 512 else 8
        quartet = half + (0 if q4_index % 2 == 0 else 4)
        for nibble_index in range(8):
            word_index = quartet + nibble_index // 2
            half_index = nibble_index % 2
            if word_index not in active_scales:
                continue
            nibble = (case.q4_word >> (4 * nibble_index)) & 0xF
            halves[word_index][half_index] += float(nibble) * scale_value
    zero_value = bf16_to_float((case.zero_pair >> 16) & 0xFFFF)
    for zero_index in case.zero_indices:
        word_index = zero_index % 16
        halves[word_index][0] += zero_value * 32.0
        halves[word_index][1] += zero_value * 32.0
    words = []
    for low, high in halves:
        words.append((float_to_bf16(high) << 16) | float_to_bf16(low))
    return tuple(words)


def expected_record_words(case: TinyCase) -> tuple[int, ...]:
    return (0x1, *expected_payload_words(case))


def configure_exp008() -> None:
    EXP008.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP008.BUILD_DIR = BUILD_DIR / "mylm_direct"


def configure_exp130(build: Path, elf_name: str) -> None:
    EXP130.RAW_COPY = build / "unused_raw.bin"
    EXP130.STATIC_73C80_COPY = build / "unused_73c80.bin"
    EXP130.STATIC_73D00_COPY = build / "unused_73d00.bin"
    EXP130.ELF = build / elf_name
    EXP130.MLIR = build / "design.mlir"
    EXP130.TXN = build / "design.txn.mlir"
    EXP130.XCLBIN = build / "design.xclbin"
    EXP130.INSTS = build / "design.bin"
    EXP130.PRJ_DIR = build / "prj"


def run_cmd(cmd: tuple[str, ...], cwd: Path = EXPERIMENT_DIR) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def make_host_code(xclbin: Path, insts: Path, case: TinyCase, records: int) -> str:
    out_dwords = records * RECORD_DWORDS
    act_dwords = records * RECORD_CHUNKS * ACTIVATION_DWORDS_PER_CHUNK
    wt_dwords = records * RECORD_CHUNKS * WEIGHT_DWORDS_PER_CHUNK
    payload = json.dumps(
        {
            "active_chunk": case.active_chunk,
            "scale_indices": list(case.scale_indices),
            "zero_indices": list(case.zero_indices),
            "q4_indices": list(case.q4_indices),
            "q4_word": case.q4_word,
            "zero_pair": case.zero_pair,
        },
        sort_keys=True,
    )
    env_path = f"{EXP130.XRT_BIN}:{os.environ.get('PATH', '')}"
    return f"""
import json
import os
import time
import traceback
os.environ['PATH'] = {env_path!r}
import numpy as np
import torch
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

RECORD_CHUNKS = {RECORD_CHUNKS}
ACTIVATION_DWORDS_PER_CHUNK = {ACTIVATION_DWORDS_PER_CHUNK}
Q4_CHUNK_DWORDS = {Q4_CHUNK_DWORDS}
SCALE_DWORDS = {SCALE_DWORDS}
ZERO_DWORDS = {ZERO_DWORDS}
BF16_ONE_PAIR = {BF16_ONE_PAIR}
BF16_SCALE_PAIR = {BF16_SCALE_PAIR}
CASE = json.loads({payload!r})

def make_activation():
    activation = np.zeros(({act_dwords},), dtype=np.int32)
    start = int(CASE['active_chunk']) * ACTIVATION_DWORDS_PER_CHUNK
    activation[start:start + ACTIVATION_DWORDS_PER_CHUNK] = np.int32(BF16_ONE_PAIR)
    return activation

def make_weights():
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    chunks = weights.reshape((-1, Q4_CHUNK_DWORDS))
    chunk = chunks[int(CASE['active_chunk'])]
    for index in CASE['scale_indices']:
        chunk[int(index)] = np.int32(BF16_SCALE_PAIR)
    zero_base = SCALE_DWORDS
    for index in CASE['zero_indices']:
        chunk[zero_base + int(index)] = np.int32(int(CASE['zero_pair']))
    q4_base = SCALE_DWORDS + ZERO_DWORDS
    for index in CASE['q4_indices']:
        chunk[q4_base + int(index)] = np.int32(int(CASE['q4_word']))
    return weights

try:
    aie_utils.DefaultNPURuntime.cleanup()
except Exception:
    pass

try:
    kernel = NPUKernel(
        xclbin_path={str(xclbin)!r},
        kernel_name='MLIR_AIE',
        insts_path={str(insts)!r},
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    out_buf = XRTTensor(({out_dwords},), dtype=np.int32)
    activation_buf = XRTTensor.from_torch(torch.from_numpy(make_activation()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(make_weights()).to(torch.int32))
    start_time = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf, weights_buf, activation_buf])
    elapsed = time.time() - start_time
    got = out_buf.to_torch().numpy().astype(np.int32)
    print('run_ok')
    print('elapsed=' + str(elapsed))
    print('npu_time=' + str(getattr(result, 'npu_time', None)))
    print('record=' + ','.join(hex(int(x) & 0xffffffff) for x in got.tolist()))
except Exception:
    print('exception_begin')
    traceback.print_exc()
    raise
finally:
    try:
        aie_utils.DefaultNPURuntime.cleanup()
    except Exception:
        traceback.print_exc()
"""


def parse_record(stdout: str) -> tuple[int, ...]:
    match = re.search(r"^record=([^\n]+)", stdout, re.MULTILINE)
    if match is None:
        return ()
    return tuple(int(word, 16) for word in match.group(1).split(","))


def run_npu_case(xclbin: Path, insts: Path, case: TinyCase, records: int, timeout: int) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{EXP130.XRT_BIN}:{env.get('PATH', '')}"
    code = make_host_code(xclbin, insts, case, records)
    try:
        result = EXP130.run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "record": [],
            "runtime_output": EXP130._short_output(
                EXP130.as_text(exc.stdout) + EXP130.as_text(exc.stderr),
                max_chars=1200,
            ),
        }
    record = parse_record(result.stdout)
    status = "record_observed" if result.returncode == 0 and len(record) >= RECORD_DWORDS else "runtime_failed"
    return {
        "status": status,
        "record": [hex(word & 0xFFFFFFFF) for word in record[:RECORD_DWORDS]],
        "returncode": result.returncode,
        "runtime_output": EXP130._short_output(result.stdout + result.stderr, max_chars=1200),
    }


def tiny_asm(record_words: tuple[int, ...]) -> str:
    lines = [
        '  .section .text,"ax",@progbits',
        "  .globl __start",
        "  .type __start,@function",
        "  .p2align 4",
        "__start:",
        "  movxm sp, #0x70000",
    ]
    for _ in range(RECORD_CHUNKS):
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
    lines.extend(
        [
        "  movx r14, #-1",
        "  acq #52, r14",
        "  nop",
        "  nop",
        "  nop",
        "  movxm p0, #0x73c1c",
        ]
    )
    for index, word in enumerate(record_words):
        lines.extend(
            [
                f"  movxm r0, #0x{word & 0xFFFFFFFF:08x}",
                "  st r0, [p0], #4",
            ]
        )
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


def tiny_linker_script() -> str:
    return "\n".join(
        [
            "MEMORY { program (RX) : ORIGIN = 0, LENGTH = 0x20000 }",
            "ENTRY(__start)",
            "SECTIONS { . = 0x0; .text : { *(.text*) } > program }",
            "",
        ]
    )


def build_tiny_xclbin(case: TinyCase, record_words: tuple[int, ...]) -> dict:
    build = BUILD_DIR / f"tiny_{case.name}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    asm = build / "tiny_codegen.s"
    obj = build / "tiny_codegen.o"
    ld = build / "tiny_codegen.ld"
    elf = build / "tiny_codegen.elf"
    disasm = build / "tiny_codegen.disasm.s"
    asm.write_text(tiny_asm(record_words))
    ld.write_text(tiny_linker_script())
    run_cmd((str(CLANG), "--target=aie2p-none-unknown-elf", "-c", str(asm), "-o", str(obj)), build)
    run_cmd((str(LD_LLD), "-T", str(ld), str(obj), "-o", str(elf)), build)
    disasm.write_text(run_cmd((str(LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf)), build).stdout)
    configure_exp130(build, elf.name)
    EXP130.write_mlir(1)
    EXP130.package_with_aiecc()
    return {
        "build_dir": str(build.relative_to(REPO_ROOT)),
        "asm": str(asm.relative_to(REPO_ROOT)),
        "elf": str(elf.relative_to(REPO_ROOT)),
        "xclbin": str(EXP130.XCLBIN.relative_to(REPO_ROOT)),
        "insts": str(EXP130.INSTS.relative_to(REPO_ROOT)),
        "disasm": str(disasm.relative_to(REPO_ROOT)),
        "disasm_counts": {
            "acq": disasm.read_text().count("acq"),
            "rel": disasm.read_text().count("rel"),
            "st": disasm.read_text().count("\tst\t"),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict:
    configure_exp008()
    before = EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    EXP008.build_qkv_direct_xclbin()
    selected_cases = cases()
    if args.max_cases > 0:
        selected_cases = selected_cases[: args.max_cases]
    results = []
    for case in selected_cases:
        expected = expected_record_words(case)
        mylm = run_npu_case(
            EXP008.EXP005.EXP132.EXP130.XCLBIN,
            EXP008.EXP005.EXP132.EXP130.INSTS,
            case,
            EXP008.RECORDS,
            args.runtime_timeout,
        )
        mylm_record = tuple(int(word, 16) for word in mylm["record"])
        formula_matches_mylm = mylm["status"] == "record_observed" and mylm_record == expected
        tiny_artifacts = build_tiny_xclbin(case, expected)
        tiny = run_npu_case(EXP130.XCLBIN, EXP130.INSTS, case, 1, args.runtime_timeout)
        tiny_record = tuple(int(word, 16) for word in tiny["record"])
        tiny_matches_expected = tiny["status"] == "record_observed" and tiny_record == expected
        tiny_matches_mylm = tiny["status"] == "record_observed" and mylm["status"] == "record_observed" and tiny_record == mylm_record
        results.append(
            {
                "name": case.name,
                "expected_record": [hex(word & 0xFFFFFFFF) for word in expected],
                "mylm": mylm,
                "tiny": tiny,
                "tiny_artifacts": tiny_artifacts,
                "formula_matches_mylm": formula_matches_mylm,
                "tiny_matches_expected": tiny_matches_expected,
                "tiny_matches_mylm": tiny_matches_mylm,
            }
        )
    after = EXP008.topology_status(args.xrt_timeout)
    complete = (
        after["status"] == "ok"
        and all(result["formula_matches_mylm"] for result in results)
        and all(result["tiny_matches_expected"] for result in results)
        and all(result["tiny_matches_mylm"] for result in results)
    )
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "results": results,
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Tiny Codegen Numeric Gate",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Results",
        "",
        "| case | formula == MyLM | tiny == expected | tiny == MyLM | tiny status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in manifest.get("results", []):
        lines.append(
            f"| `{result['name']}` | `{result['formula_matches_mylm']}` | "
            f"`{result['tiny_matches_expected']}` | `{result['tiny_matches_mylm']}` | "
            f"`{result['tiny']['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an emit-only generated whole-core source-assembly gate. It proves "
            "that the observed Q4NX layout/parity formula can be turned into a "
            "standalone raw AIE2P program that consumes the same stream count and "
            "emits a compact record exactly matching the MyLM raw body. The next "
            "step is to replace the generated constants with dynamic loads and "
            "arithmetic for the same tiny cases.",
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
            "# Q4NX Tiny Codegen Numeric Gate\n\nExperiment failed.\n\n```text\n"
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
