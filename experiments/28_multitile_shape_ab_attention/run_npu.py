#!/usr/bin/env python3
"""Run exp28 on real NPU and verify multi-tile shape-A/B online attention."""

from __future__ import annotations

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
    ATTENTION_OUT_DWORDS,
    CURRENT_DWORDS,
    HISTORY_SOURCE_DWORDS,
    SIDEBAND_DEBUG_DWORDS,
    TOTAL_OUT_DWORDS,
    generate_mlir,
    last_valid_for_context,
    num_tiles_for_context,
)
from reference import expected_output, make_current_payload, make_history_payload

EXPERIMENT_DIR = Path(__file__).parent
CONTEXT_LENGTHS = (17, 31, 32, 64, 128, 129)


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "attention_kernels.cc"
    obj = EXPERIMENT_DIR / "attention_kernels.o"
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
    print("  Compiling attention_kernels.cc...")
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
        "aie.packet_flow(14)": 1,
        "aie.packet_source<%current_src, DMA : 0>": 1,
        "aie.packet_dest<%shape_a, DMA : 0>": 1,
        "aie.flow(%mem0, DMA : 0, %shape_a, DMA : 1)": 1,
        "aie.flow(%shape_a, DMA : 0, %side_sink, DMA : 0)": 1,
        "aie.flow(%side_sink, DMA : 1, %shape_b, DMA : 0)": 1,
        "aie.flow(%mem7, DMA : 0, %shape_b, DMA : 1)": 1,
        "aie.runtime_sequence": 1,
        "aiex.npu.writebd": 5,
        "aiex.npu.address_patch": 5,
        "aiex.npu.push_queue": 5,
        "aiex.npu.sync": 2,
        "aie.memtile_dma": 2,
        "scf.for": 3,
        "func.call @shape_a_tile_sideband": 1,
        "func.call @shape_b_accumulate_tile": 1,
    }
    errors: list[str] = []
    for marker, expected in checks.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")
    if "dma_configure_task_for" in mlir_text:
        errors.append("Expected raw writebd runtime, found dma_configure_task_for")
    if "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found shim_dma_allocation")
    return errors


def build_case(context_len: int) -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / f"L{context_len}"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"
    mlir_text = generate_mlir(context_len)
    mlir_path.write_text(mlir_text)

    errors = check_mlir_structure(mlir_text)
    if errors:
        message = "\n".join(f"  STRUCTURE FAIL: {error}" for error in errors)
        raise RuntimeError(message)

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def run_case(context_len: int, xclbin_path: Path, insts_path: Path) -> bool:
    num_tiles = num_tiles_for_context(context_len)
    last_valid = last_valid_for_context(context_len)
    print(f"\n--- L={context_len} tiles={num_tiles} last_valid={last_valid} ---")
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    current = make_current_payload(context_len)
    k_history = make_history_payload(context_len, "k")
    v_history = make_history_payload(context_len, "v")
    expected = expected_output(context_len)

    current_buf = XRTTensor.from_torch(torch.from_numpy(current.copy()).to(torch.float32))
    k_history_buf = XRTTensor.from_torch(torch.from_numpy(k_history.copy()).to(torch.float32))
    v_history_buf = XRTTensor.from_torch(torch.from_numpy(v_history.copy()).to(torch.float32))
    out_buf = XRTTensor((TOTAL_OUT_DWORDS,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(
        handle, [current_buf, k_history_buf, v_history_buf, out_buf]
    )
    npu_time_us = result.npu_time / 1e3
    got = out_buf.to_torch().numpy().astype(np.float32)
    close = np.isclose(got, expected, rtol=1e-4, atol=1e-5)
    max_err = float(np.max(np.abs(got - expected)))

    if bool(np.all(close)):
        print(f"  PASS  time={npu_time_us:.1f}us max_err={max_err:.6g}")
        print(f"    sideband_debug={got[:SIDEBAND_DEBUG_DWORDS].tolist()}")
        print(
            "    attn_head0[0:8]="
            f"{got[SIDEBAND_DEBUG_DWORDS:SIDEBAND_DEBUG_DWORDS + 8].tolist()}"
        )
        return True

    mismatches = np.where(~close)[0]
    print(f"  FAIL  time={npu_time_us:.1f}us mismatches={mismatches.size} max_err={max_err:.6g}")
    for idx in mismatches[:16]:
        print(f"    out[{idx}]: expected={expected[idx]:.8f} got={got[idx]:.8f}")
    print(f"    sideband_debug got={got[:SIDEBAND_DEBUG_DWORDS].tolist()}")
    print(f"    sideband_debug exp={expected[:SIDEBAND_DEBUG_DWORDS].tolist()}")
    return False


def main() -> bool:
    print("=" * 72)
    print("Experiment 28: Multi-Tile Shape-A/Shape-B Online Attention")
    print("=" * 72)
    print(f"  current packet: {CURRENT_DWORDS} f32 via packet14, sent once")
    print(f"  K/V ring tile: {HISTORY_SOURCE_DWORDS} f32 source -> first 2048 consumed")
    print(f"  output: {SIDEBAND_DEBUG_DWORDS} f32 sideband debug + {ATTENTION_OUT_DWORDS} f32 attention")
    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")

    compile_kernel()
    all_passed = True
    for context_len in CONTEXT_LENGTHS:
        xclbin_path, insts_path = build_case(context_len)
        all_passed &= run_case(context_len, xclbin_path, insts_path)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: exp28 multi-tile shape-A/B online attention verified on NPU.")
    else:
        print("FAIL: at least one exp28 context length failed.")
    print("=" * 72)
    return all_passed


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
