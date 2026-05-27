#!/usr/bin/env python3
"""Run exp33 on NPU and compare against the CPU reference."""

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
    DIM_GROUPS,
    GROUP_DWORDS,
    OUTPUT_DWORDS,
    PLANE_TILE_DWORDS,
    QUERY_DWORDS,
    RESHAPED_TILE_DWORDS,
    TOKENS_PER_TILE,
    generate_mlir,
    num_tiles_for_context,
)
from reference import make_case

EXPERIMENT_DIR = Path(__file__).parent
CONTEXT_LENGTHS = (17, 31, 32, 79)


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def check_mlir_structure(mlir_text: str, num_tiles: int) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "aiex.npu.writebd": 2 * num_tiles + 2,
        "aiex.npu.address_patch": 2 * num_tiles + 2,
        "aiex.npu.push_queue": 2 * num_tiles + 2,
        "aiex.npu.sync": 1,
        "aie.flow": 5,
        "aie.memtile_dma": 1,
        "aie.core(": 1,
        "aie.mem(": 1,
    }
    for needle, expected in expected_counts.items():
        actual = mlir_text.count(needle)
        if actual != expected:
            errors.append(f"Expected {expected} {needle}, got {actual}")

    if "dma_configure_task_for" in mlir_text:
        errors.append("Expected raw writebd runtime, found dma_configure_task_for")
    if "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found shim_dma_allocation")

    match = re.search(r"aie\.runtime_sequence\(([^)]+)\)", mlir_text)
    if match is None:
        errors.append("No runtime_sequence found")
    else:
        args = match.group(1).split(",")
        if len(args) != 3:
            errors.append(f"Expected 3 runtime args, got {len(args)}")

    reshape_dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = 128>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    if mlir_text.count(reshape_dims) != 4:
        errors.append("Expected K/V ping-pong row1 reshape dimensions")
    if f"memref<{RESHAPED_TILE_DWORDS}xf32>" not in mlir_text:
        errors.append("Missing reshaped tile buffer type")
    return errors


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "reshape_attention.cc"
    obj = EXPERIMENT_DIR / "reshape_attention.o"
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
    print("  Compiling reshape_attention.cc...")
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


def build_case(context_len: int) -> tuple[Path, Path]:
    num_tiles = num_tiles_for_context(context_len)
    build_dir = EXPERIMENT_DIR / "build" / f"L{context_len}"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir(context_len)
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text, num_tiles)
    if errors:
        message = "\n".join(f"  STRUCTURE FAIL: {error}" for error in errors)
        raise RuntimeError(message)

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def run_case(context_len: int) -> bool:
    num_tiles = num_tiles_for_context(context_len)
    print(f"\n--- L={context_len}, tiles={num_tiles} ---")
    xclbin_path, insts_path = build_case(context_len)
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    query, kv_cache, expected = make_case(context_len)
    assert query.shape[0] == QUERY_DWORDS
    assert kv_cache.shape[0] == 2 * num_tiles * PLANE_TILE_DWORDS

    query_buf = XRTTensor.from_torch(torch.from_numpy(query.copy()).to(torch.float32))
    kv_buf = XRTTensor.from_torch(torch.from_numpy(kv_cache.copy()).to(torch.float32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(handle, [query_buf, kv_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    npu_output = output_buf.to_torch().numpy().astype(np.float32)

    abs_err = np.abs(npu_output - expected)
    bad = np.where(abs_err > 2e-5)[0]
    passed = bad.size == 0
    if passed:
        print(
            f"  PASS  time={npu_time_us:.1f}us  max_abs={float(abs_err.max()):.7f}  "
            f"out[0:4]={npu_output[:4]}"
        )
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  max_abs={float(abs_err.max()):.7f}")
        print(f"    mismatches: {bad.size}/{OUTPUT_DWORDS}")
        for idx in bad[:8]:
            print(
                f"      out[{idx}]: expected={expected[idx]:.7f} "
                f"got={npu_output[idx]:.7f} err={abs_err[idx]:.7f}"
            )
    return passed


def main() -> bool:
    print("=" * 72)
    print("Experiment 33: Row1 KV Reshape Attention")
    print("=" * 72)
    print(f"Query dwords: {QUERY_DWORDS}")
    print(f"Plane tile dwords: {PLANE_TILE_DWORDS}, reshaped tile dwords: {RESHAPED_TILE_DWORDS}")
    print()

    print("Compiling kernel...")
    compile_kernel()

    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    all_passed = True
    for context_len in CONTEXT_LENGTHS:
        all_passed &= run_case(context_len)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: row1 dim-group-major KV reshape feeds attention.")
    else:
        print("FAIL: at least one length failed.")
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
