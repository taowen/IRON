#!/usr/bin/env python3
"""Run experiment 21 on NPU and compare against CPU reference."""

import os
import re
import sys
import traceback
from math import ceil
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
    HEAD_DIM,
    NUM_KV_HEADS_PER_GROUP,
    OUTPUT_DWORDS_PER_WORKER,
    PLANE_TILE_DWORDS,
    TOKEN_DWORDS,
    TOKENS_PER_TILE,
    generate_mlir,
)
from reference import apply_current_to_cache, layer_reference_after_current_write

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "aiex.npu.writebd": 14,
        "aiex.npu.address_patch": 14,
        "aiex.npu.push_queue": 14,
        "aiex.npu.sync": 4,
        "aie.flow": 10,
        "aie.memtile_dma": 2,
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

    markers = [
        "current_k",
        "current_v",
        "derive_query",
        "running_max",
        "running_sum",
        "online_softmax_attention",
        "layer_epilogue",
    ]
    for marker in markers:
        if marker not in mlir_text:
            errors.append(f"Missing marker: {marker}")

    return errors


def compile_kernel() -> bool:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "kv_attention.cc"
    obj = EXPERIMENT_DIR / "kv_attention.o"
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
    print("  Compiling kv_attention.cc...")
    return os.system(" ".join(cmd)) == 0


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> bool:
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
    return os.system(" ".join(cmd)) == 0


def _check_current_cache_write(L: int, before: np.ndarray, after_npu: np.ndarray, current: np.ndarray) -> list[str]:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    token_offset = (L - 1) * TOKEN_DWORDS
    expected_after = apply_current_to_cache(L, before, current)
    errors: list[str] = []
    for plane_idx, plane_name in enumerate(["k03", "v03", "k47", "v47"]):
        start = plane_idx * plane_dwords + token_offset
        end = start + TOKEN_DWORDS
        if not np.allclose(after_npu[start:end], expected_after[start:end], rtol=0.0, atol=0.0):
            errors.append(f"{plane_name} current token cache write mismatch")
    return errors


def run_case(L: int) -> bool:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    print(f"\n--- L={L}, tiles={num_tiles}, last_valid={last_valid} ---")

    build_dir = EXPERIMENT_DIR / "build" / f"L{L}"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir(L)
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text)
    if errors:
        for error in errors:
            print(f"  STRUCTURE FAIL: {error}")
        return False

    if not compile_mlir(mlir_path, xclbin_path, insts_path):
        print("  FAIL: MLIR compilation error")
        return False

    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    current, cache_before, expected = layer_reference_after_current_write(L)
    current_buf = XRTTensor.from_torch(torch.from_numpy(current.copy()).to(torch.float32))
    cache_buf = XRTTensor.from_torch(torch.from_numpy(cache_before.copy()).to(torch.float32))
    output_buf = XRTTensor((2 * OUTPUT_DWORDS_PER_WORKER,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(handle, [current_buf, cache_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    npu_output = output_buf.to_torch().numpy().astype(np.float32)
    cache_after = cache_buf.to_torch().numpy().astype(np.float32)

    cache_errors = _check_current_cache_write(L, cache_before, cache_after, current)
    abs_err = np.abs(npu_output - expected)
    bad = np.where(abs_err > 2e-4)[0]
    passed = len(cache_errors) == 0 and bad.size == 0
    if passed:
        print(
            f"  PASS  time={npu_time_us:.1f}us  max_abs={float(abs_err.max()):.6f}  "
            f"w0[0:4]={npu_output[:4]}  w1[0:4]={npu_output[128:132]}"
        )
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  max_abs={float(abs_err.max()):.6f}")
        for error in cache_errors:
            print(f"    {error}")
        if bad.size:
            print(f"    output mismatches: {bad.size}/256")
            for idx in bad[:8]:
                worker = "w0" if idx < 128 else "w1"
                local = idx if idx < 128 else idx - 128
                print(
                    f"      {worker}[{local}]: expected={expected[idx]:.7f} "
                    f"got={npu_output[idx]:.7f} err={abs_err[idx]:.7f}"
                )
    return passed


def main() -> bool:
    print("=" * 72)
    print("Experiment 21: Single-Layer Decode Contract")
    print("=" * 72)
    print(f"Plane tile: 16 tokens x {NUM_KV_HEADS_PER_GROUP} heads x {HEAD_DIM} dims")
    print("Local phases: derive query -> online attention -> layer epilogue")
    print()

    print("Compiling kernel...")
    if not compile_kernel():
        return False

    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    all_passed = True
    for L in [1, 15, 16, 17, 31, 32, 79]:
        all_passed &= run_case(L)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: single-layer decode contract verified.")
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
