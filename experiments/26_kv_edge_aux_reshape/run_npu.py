#!/usr/bin/env python3
"""Run exp26 on real NPU and verify selector/sideband calibration."""

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
    CURRENT_DWORDS,
    HISTORY_SOURCE_DWORDS,
    SIDEBAND_DWORDS,
    TOTAL_OUT_DWORDS,
    VARIANTS,
    generate_mlir,
)
from reference import (
    expected_output,
    make_current_payload,
    make_history_payload,
    make_sideband_payload,
)

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "debug_kernels.cc"
    obj = EXPERIMENT_DIR / "debug_kernels.o"
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
    print("  Compiling debug_kernels.cc...")
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


def check_mlir_structure(mlir_text: str, variant_name: str) -> list[str]:
    variant = VARIANTS[variant_name]
    checks = {
        f"aie.packet_flow({variant.packet_id})": 1,
        "aie.packet_source<%current_src, DMA : 0>": 1,
        f"aie.packet_dest<%{variant.shape_tile}, DMA : 0>": 1,
        f"aie.flow(%{variant.mem_tile}, DMA : 0, %{variant.shape_tile}, DMA : 1)": 1,
        "aie.flow(%side_sink, DMA : 1, %shape_b, DMA : 0)": 1,
        "aie.runtime_sequence": 1,
        "aiex.npu.writebd": 6,
        "aiex.npu.address_patch": 6,
        "aiex.npu.push_queue": 6,
        "aiex.npu.sync": 3,
    }
    errors: list[str] = []
    for marker, expected in checks.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{variant_name}: {marker}: expected {expected}, got {actual}")
    return errors


def run_variant(variant_name: str) -> bool:
    variant = VARIANTS[variant_name]
    print("-" * 72)
    print(f"Variant {variant.name}: packet{variant.packet_id} -> {variant.shape_tile}")
    print("-" * 72)

    build_dir = EXPERIMENT_DIR / "build" / variant.name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir(variant.name)
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text, variant.name)
    if errors:
        for error in errors:
            print(f"  STRUCTURE FAIL: {error}")
        return False

    compile_mlir(mlir_path, xclbin_path, insts_path)

    print("  Loading NPU kernel...")
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    current = make_current_payload()
    history = make_history_payload()
    sideband = make_sideband_payload()
    expected = expected_output()

    current_buf = XRTTensor.from_torch(torch.from_numpy(current.copy()).to(torch.int32))
    history_buf = XRTTensor.from_torch(torch.from_numpy(history.copy()).to(torch.int32))
    sideband_buf = XRTTensor.from_torch(torch.from_numpy(sideband.copy()).to(torch.int32))
    out_buf = XRTTensor((TOTAL_OUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(
        handle, [current_buf, history_buf, sideband_buf, out_buf]
    )
    npu_time_us = result.npu_time / 1e3
    got = out_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  got: {got.tolist()}")
    print(f"  exp: {expected.tolist()}")

    mismatches = np.where(got != expected)[0]
    if mismatches.size == 0:
        print(f"  PASS: {variant.name}")
        return True

    print(f"  FAIL: {variant.name}: {mismatches.size} mismatches")
    for idx in mismatches[:12]:
        print(f"    out[{idx}]: expected={expected[idx]} got={got[idx]}")
    return False


def run_on_npu() -> bool:
    print("=" * 72)
    print("Experiment 26: Edge Selector And Sideband Calibration")
    print("=" * 72)
    print(f"  current: {CURRENT_DWORDS} dwords via packet14/15 variants")
    print(f"  history: {HISTORY_SOURCE_DWORDS} dwords through row1 ring, first 2048 consumed")
    print(f"  sideband: {SIDEBAND_DWORDS} dwords c2r2 -> c6r2 -> c6r3")
    print()

    compile_kernel()
    all_passed = True
    for variant_name in ("left14", "right15"):
        all_passed &= run_variant(variant_name)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: exp26 selector, history ring, and shape-B handoff verified.")
    else:
        print("FAIL: at least one exp26 variant failed.")
    print("=" * 72)
    return all_passed


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
