#!/usr/bin/env python3
"""Run exp58 on real NPU and verify MyLM patch-pair row1 split."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import aie.utils as aie_utils
import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from generate import (
    CHUNK_BF16,
    MAIN_COLUMN,
    PATCH_PAIR_BF16,
    ROWS_PER_COLUMN,
    TOTAL_INPUT_I32,
    TOTAL_OUTPUT_DWORDS,
    WORDS_PER_RECORD,
    generate_mlir,
)
from reference import expected_output, make_input_i32

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "patch_pair_split.cc"
    obj = EXPERIMENT_DIR / "patch_pair_split.o"
    cmd = [
        str(clang),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    print("  Compiling patch_pair_split.cc...")
    run_command(cmd)


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    cmd = [
        str(aiecc),
        "-v",
        "-j1",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--peano",
        str(peano_dir),
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("  Compiling MLIR...")
    run_command(cmd)


def check_mlir_structure(mlir_text: str) -> list[str]:
    expected_counts = {
        "aie.flow(": 6,
        "aie.packet_flow": ROWS_PER_COLUMN,
        "aie.core(": ROWS_PER_COLUMN,
        "aie.mem(": ROWS_PER_COLUMN,
        "aie.memtile_dma": 1,
        "aiex.npu.writebd": 3,
        "aiex.npu.address_patch": 3,
        "aiex.npu.push_queue": 2,
        "aiex.npu.sync": 2,
        "use_next_bd = 1": 1,
        "func.call @record_patch_chunk": ROWS_PER_COLUMN,
    }
    errors: list[str] = []
    for marker, expected in expected_counts.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")

    match = re.search(r"aie\.runtime_sequence\(([^)]+)\)", mlir_text)
    if match is None:
        errors.append("No runtime_sequence found")
    else:
        runtime_args = [arg.strip() for arg in match.group(1).split(",")]
        expected_args = [
            f"%input: memref<{TOTAL_INPUT_I32}xi32>",
            f"%output: memref<{TOTAL_OUTPUT_DWORDS}xi32>",
        ]
        if runtime_args != expected_args:
            errors.append(f"Unexpected runtime args: {runtime_args}")
    return errors


def build_kernel() -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  STRUCTURE FAIL: {error}" for error in errors))

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def _records_by_row(records: np.ndarray, label: str) -> np.ndarray:
    shaped = records.reshape(ROWS_PER_COLUMN, WORDS_PER_RECORD)
    ordered = np.empty_like(shaped)
    seen: set[int] = set()
    for record in shaped:
        row = int(record[0])
        if row < 0 or row >= ROWS_PER_COLUMN:
            raise RuntimeError(f"{label}: bad row header {row}")
        if row in seen:
            raise RuntimeError(f"{label}: duplicate row header {row}")
        seen.add(row)
        ordered[row] = record
    if len(seen) != ROWS_PER_COLUMN:
        raise RuntimeError(f"{label}: missing rows {set(range(ROWS_PER_COLUMN)) - seen}")
    return ordered.reshape(-1)


def run_on_npu() -> bool:
    print("=" * 78)
    print("Experiment 58: MyLM Patch-Pair Row1 Split")
    print("=" * 78)
    print(f"  main column: c{MAIN_COLUMN}")
    print(f"  chunk: {CHUNK_BF16} bf16")
    print(f"  patch pair: {PATCH_PAIR_BF16} bf16")
    print("  runtime: linked patch0 -> patch1 descriptors, row1 splits to four rows")
    print()

    compile_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    input_data = make_input_i32()
    expected = expected_output()
    input_buf = XRTTensor.from_torch(torch.from_numpy(input_data.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    got_arrival = output_buf.to_torch().numpy().astype(np.int32)
    got = _records_by_row(got_arrival, "got")
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected:\n{expected.reshape(ROWS_PER_COLUMN, WORDS_PER_RECORD)}")
    print(f"  got arrival:\n{got_arrival.reshape(ROWS_PER_COLUMN, WORDS_PER_RECORD)}")
    print(f"  got ordered:\n{got.reshape(ROWS_PER_COLUMN, WORDS_PER_RECORD)}")

    mismatch = np.where(got != expected)[0]
    if mismatch.size == 0:
        print("  PASS: two 64-row MyLM patch pairs were split by row1 into four compute rows.")
        return True

    print(f"  FAIL: mismatches={mismatch.size}/{TOTAL_OUTPUT_DWORDS}")
    for idx in mismatch[:16]:
        print(f"    out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
    return False


def main() -> bool:
    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    return run_on_npu()


if __name__ == "__main__":
    try:
        success = main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()
    raise SystemExit(0 if success else 1)
