#!/usr/bin/env python3
"""Run exp37 on NPU and compare against the CPU reference."""

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
    HEAD_DIM,
    HEAD_PHASE_I32,
    HIDDEN_DIM,
    HIDDEN_I32,
    KV_COL,
    NUM_COLS,
    OUTPUT_DWORDS,
    PLANE_TILE_DWORDS,
    QUERY_DWORDS,
    RESHAPED_TILE_DWORDS,
    TOKENS_PER_TILE,
    TOTAL_WEIGHT_I32,
    current_offsets,
    generate_mlir,
    num_tiles_for_context,
)
from reference import (
    attention_reference_after_current_write,
    make_case_data,
)

EXPERIMENT_DIR = Path(__file__).parent
CONTEXT_LENGTHS = (17, 31, 32, 79)
OUTPUT_ABS_TOLERANCE = 4e-2
CACHE_ABS_TOLERANCE = 5e-3


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def check_mlir_structure(mlir_text: str, num_tiles: int) -> list[str]:
    errors: list[str] = []
    expected_runtime_ops = 13 + 2 * num_tiles
    expected_counts = {
        "aiex.npu.writebd": expected_runtime_ops,
        "aiex.npu.address_patch": expected_runtime_ops,
        "aiex.npu.push_queue": expected_runtime_ops,
        "aiex.npu.sync": NUM_COLS + 1,
        "aie.flow": NUM_COLS * 3 + 2 + NUM_COLS,
        "aie.memtile_dma": NUM_COLS + 1,
        "aie.core(": NUM_COLS,
        "aie.mem(": NUM_COLS,
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
        if len(args) != 4:
            errors.append(f"Expected 4 runtime args, got {len(args)}")

    reshape_dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    if mlir_text.count(reshape_dims) != NUM_COLS * 4:
        errors.append("Expected K/V ping-pong row1 reshape dimensions on each fanout channel")
    if f"memref<{RESHAPED_TILE_DWORDS}xf32>" not in mlir_text:
        errors.append("Missing reshaped tile buffer type")
    if mlir_text.count(f"buffer_length = {HEAD_PHASE_I32} : i32") != NUM_COLS + 2:
        errors.append("Expected four Q-head weight BDs plus K/V weight BDs")

    cache_patch_count = len(re.findall(r"arg_idx = 2 : i32", mlir_text))
    expected_cache_patches = 2 * num_tiles + 2
    if cache_patch_count != expected_cache_patches:
        errors.append(f"Expected current-write plus one-read K/V cache patches ({expected_cache_patches}), got {cache_patch_count}")
    return errors


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "qkv_reshape_attention.cc"
    obj = EXPERIMENT_DIR / "qkv_reshape_attention.o"
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
    print("  Compiling qkv_reshape_attention.cc...")
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


def _bf16_to_i32_buffer(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.view(np.uint8).tobytes(), dtype=np.int32)


def _bytes_to_i32_buffer(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(), dtype=np.int32)


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


def _check_current_cache_write(
    context_len: int,
    cache_after_npu: np.ndarray,
    current_k: np.ndarray,
    current_v: np.ndarray,
) -> list[str]:
    k_offset, v_offset = current_offsets(context_len)
    k_err = np.abs(cache_after_npu[k_offset:k_offset + HEAD_DIM] - current_k)
    v_err = np.abs(cache_after_npu[v_offset:v_offset + HEAD_DIM] - current_v)
    errors: list[str] = []
    if np.any(k_err > CACHE_ABS_TOLERANCE):
        errors.append(f"K current token cache write mismatch, max_abs={float(k_err.max()):.6f}")
    if np.any(v_err > CACHE_ABS_TOLERANCE):
        errors.append(f"V current token cache write mismatch, max_abs={float(v_err.max()):.6f}")
    return errors


def run_case(context_len: int, hidden: np.ndarray, packed_weight: np.ndarray) -> bool:
    num_tiles = num_tiles_for_context(context_len)
    print(f"\n--- L={context_len}, tiles={num_tiles} ---")
    xclbin_path, insts_path = build_case(context_len)
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    query, current_k, current_v, cache_before, cache_after_ref, expected = attention_reference_after_current_write(
        context_len,
        hidden,
        packed_weight,
    )
    hidden_i32 = _bf16_to_i32_buffer(hidden)
    weight_i32 = _bytes_to_i32_buffer(packed_weight)
    assert hidden_i32.shape[0] == HIDDEN_I32
    assert weight_i32.shape[0] == TOTAL_WEIGHT_I32
    assert cache_before.shape[0] == 2 * num_tiles * PLANE_TILE_DWORDS

    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden_i32.copy()).to(torch.int32))
    weight_buf = XRTTensor.from_torch(torch.from_numpy(weight_i32.copy()).to(torch.int32))
    kv_buf = XRTTensor.from_torch(torch.from_numpy(cache_before.copy()).to(torch.float32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(handle, [hidden_buf, weight_buf, kv_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    npu_output = output_buf.to_torch().numpy().astype(np.float32)
    npu_cache = kv_buf.to_torch().numpy().astype(np.float32)

    output_err = np.abs(npu_output - expected)
    output_bad = np.where(~np.isfinite(npu_output) | (output_err > OUTPUT_ABS_TOLERANCE))[0]
    cache_errors = _check_current_cache_write(context_len, npu_cache, current_k, current_v)
    unchanged_mask = np.ones(cache_before.shape[0], dtype=bool)
    k_offset, v_offset = current_offsets(context_len)
    unchanged_mask[k_offset:k_offset + HEAD_DIM] = False
    unchanged_mask[v_offset:v_offset + HEAD_DIM] = False
    unchanged_bad = np.where(npu_cache[unchanged_mask] != cache_after_ref[unchanged_mask])[0]
    passed = output_bad.size == 0 and len(cache_errors) == 0 and unchanged_bad.size == 0

    if passed:
        print(
            f"  PASS  time={npu_time_us:.1f}us  max_abs={float(output_err.max()):.6f}  "
            f"q[0:4]={query[:4]}  out[0:4]={npu_output[:4]}"
        )
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  max_abs={float(output_err.max()):.6f}")
        for error in cache_errors:
            print(f"    {error}")
        if unchanged_bad.size:
            print(f"    unchanged cache mismatches: {unchanged_bad.size}")
        if output_bad.size:
            print(f"    output mismatches: {output_bad.size}/{OUTPUT_DWORDS}")
            for idx in output_bad[:8]:
                print(
                    f"      out[{idx}]: expected={expected[idx]:.7f} "
                    f"got={npu_output[idx]:.7f} err={output_err[idx]:.7f}"
                )
    return passed


def main() -> bool:
    print("=" * 72)
    print("Experiment 37: Projected One-Read Fanout Attention")
    print("=" * 72)
    print(f"Hidden dim: {HIDDEN_DIM} bf16, per-head phase: {HEAD_PHASE_I32} i32")
    print(f"Columns: {NUM_COLS}, KV reader column: {KV_COL}, output dwords: {OUTPUT_DWORDS}")
    print(f"Plane tile dwords: {PLANE_TILE_DWORDS}, reshaped tile dwords: {RESHAPED_TILE_DWORDS}")
    print()

    hidden, packed_weight, query, current_k, current_v = make_case_data()
    print(f"Reference Q[0:4]: {query[:4]}")
    print(f"Reference current K[0:4]: {current_k[:4]}")
    print(f"Reference current V[0:4]: {current_v[:4]}")

    print("Compiling kernel...")
    compile_kernel()

    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    all_passed = True
    for context_len in CONTEXT_LENGTHS:
        all_passed &= run_case(context_len, hidden, packed_weight)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: NPU Q/K/V projection feeds current-write one-read fanout attention.")
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
