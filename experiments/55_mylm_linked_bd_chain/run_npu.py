#!/usr/bin/env python3
"""Run exp55 on real NPU and verify MyLM-style linked MM2S BDs."""

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

from generate import CHUNK_DWORDS, NUM_CHUNKS, TOTAL_INPUT_DWORDS, TOTAL_OUTPUT_DWORDS, generate_mlir
from reference import WORDS_PER_RECORD, expected_output, make_input

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "linked_bd_chain.cc"
    obj = EXPERIMENT_DIR / "linked_bd_chain.o"
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
    print("  Compiling linked_bd_chain.cc...")
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
        "aie.flow(": 2,
        "aie.core(": 1,
        "aie.mem(": 1,
        "aiex.npu.writebd": NUM_CHUNKS + 1,
        "aiex.npu.address_patch": NUM_CHUNKS + 1,
        "aiex.npu.push_queue": 2,
        "aiex.npu.sync": 2,
        "func.call @record_descriptor_chunk": 2,
        "use_next_bd = 1": NUM_CHUNKS - 1,
        "use_next_bd = 0": 2,
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
            f"%input: memref<{TOTAL_INPUT_DWORDS}xi32>",
            f"%output: memref<{TOTAL_OUTPUT_DWORDS}xi32>",
        ]
        if runtime_args != expected_args:
            errors.append(f"Unexpected runtime args: {runtime_args}")

    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    if runtime_body.count("direction = 1") != 1:
        errors.append("Expected one final MM2S sync")
    if "dma_configure_task_for" in mlir_text or "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found high-level DMA helpers")
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


def run_on_npu() -> bool:
    print("=" * 78)
    print("Experiment 55: MyLM-Style Linked BD Chain")
    print("=" * 78)
    print(f"  chunks: {NUM_CHUNKS}")
    print(f"  chunk size: {CHUNK_DWORDS} i32")
    print("  runtime: patch BD0..BD7, push BD0 once, sync once")
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

    input_data = make_input()
    expected = expected_output()
    input_buf = XRTTensor.from_torch(torch.from_numpy(input_data.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected:\n{expected.reshape(NUM_CHUNKS, WORDS_PER_RECORD)}")
    print(f"  got:\n{got.reshape(NUM_CHUNKS, WORDS_PER_RECORD)}")

    mismatch = np.where(got != expected)[0]
    if mismatch.size == 0:
        print("  PASS: linked raw MM2S BD chain reached the static ping-pong ring in order.")
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
