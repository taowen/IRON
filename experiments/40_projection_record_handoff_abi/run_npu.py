#!/usr/bin/env python3
"""Run exp40 on real NPU and verify the projection-record handoff ABI."""

import os
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
    ALL_RECORD_DWORDS,
    RAW_COLUMN_DWORDS,
    TOTAL_OUTPUT_DWORDS,
    generate_mlir,
)
from reference import SEGMENTS, expected_output, make_all_records

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "record_handoff.cc"
    obj = EXPERIMENT_DIR / "record_handoff.o"
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
    print("  Compiling record_handoff.cc...")
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
    checks = {
        "aie.packet_flow": 4,
        "aie.packet_source<%src": 4,
        "aie.packet_dest<%mt0, DMA : 0>": 4,
        "aie.flow(%mt0, DMA : 1, %agg, DMA : 0)": 1,
        "aie.flow(%shim1, DMA : 0, %agg, DMA : 1)": 1,
        "aie.flow(%agg, DMA : 0, %shim1, DMA : 0)": 1,
        "aiex.npu.writebd": 3,
        "aiex.npu.address_patch": 3,
        "aiex.npu.push_queue": 3,
        "aiex.npu.sync": 1,
    }
    errors: list[str] = []
    for marker, expected in checks.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")
    required_markers = (
        f"memref<{RAW_COLUMN_DWORDS}xi32>",
        f"memref<{ALL_RECORD_DWORDS}xi32>",
        f"memref<{TOTAL_OUTPUT_DWORDS}xi32>",
        "func.func private @emit_projection_record",
        "func.func private @aggregate_record_ladder",
        "func.call @emit_projection_record",
        "func.call @aggregate_record_ladder",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")
    if "dma_configure_task_for" in mlir_text:
        errors.append("Expected raw writebd runtime, found dma_configure_task_for")
    if "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found shim_dma_allocation")
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
        message = "\n".join(f"  STRUCTURE FAIL: {error}" for error in errors)
        raise RuntimeError(message)

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def report_segment(name: str, got: np.ndarray, expected: np.ndarray, offset: int, length: int) -> bool:
    got_segment = got[offset:offset + length]
    exp_segment = expected[offset:offset + length]
    mismatches = np.where(got_segment != exp_segment)[0]
    if mismatches.size == 0:
        print(f"  PASS {name:<10} len={length}")
        return True

    print(f"  FAIL {name:<10} mismatches={mismatches.size}/{length}")
    for local_idx in mismatches[:8]:
        global_idx = offset + int(local_idx)
        print(
            f"    out[{global_idx}]: expected={int(exp_segment[local_idx])} "
            f"got={int(got_segment[local_idx])}"
        )
    return False


def run_on_npu() -> bool:
    print("=" * 72)
    print("Experiment 40: Projection Record Handoff ABI")
    print("=" * 72)
    print(f"  source column records: 4 x 17 dwords -> {RAW_COLUMN_DWORDS} raw dwords")
    print(f"  host ladder input: {ALL_RECORD_DWORDS} dwords")
    print(f"  output ladder: {TOTAL_OUTPUT_DWORDS} dwords")
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

    all_records = make_all_records()
    expected = expected_output()
    all_records_buf = XRTTensor.from_torch(torch.from_numpy(all_records.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [all_records_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")

    passed = True
    for name, offset, length in SEGMENTS:
        passed &= report_segment(name, got, expected, offset, length)
    return passed


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
