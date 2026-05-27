#!/usr/bin/env python3
"""Run exp39 on NPU and compare against the CPU reference."""

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
    HIDDEN_DIM,
    HIDDEN_I32,
    NUM_KV_GROUPS,
    OUTPUT_DWORDS,
    OUTPUT_GROUP_DWORDS,
    PLANE_TILE_DWORDS,
    Q_HEADS_PER_GROUP,
    RESHAPED_TILE_DWORDS,
    TOKENS_PER_TILE,
    current_offsets,
    generate_mlir,
    kv_group_stride_dwords,
    num_tiles_for_context,
)
from reference import make_case

EXPERIMENT_DIR = Path(__file__).parent
CONTEXT_LENGTHS = (17, 31, 32, 79)
OUTPUT_ABS_TOLERANCE = 1e-3
CACHE_ABS_TOLERANCE = 1e-6


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _bf16_to_i32_buffer(values: np.ndarray) -> np.ndarray:
    raw_u16 = values.view(np.uint16)
    return raw_u16.reshape(-1, 2).view(np.int32).reshape(-1)


def check_mlir_structure(mlir_text: str, num_tiles: int) -> list[str]:
    errors: list[str] = []
    per_group_runtime_ops = 2 * num_tiles + 4
    expected_counts = {
        "aiex.npu.writebd": NUM_KV_GROUPS * per_group_runtime_ops,
        "aiex.npu.address_patch": NUM_KV_GROUPS * per_group_runtime_ops,
        "aiex.npu.push_queue": NUM_KV_GROUPS * per_group_runtime_ops,
        "aiex.npu.sync": NUM_KV_GROUPS * 2,
        "aie.packet_flow": NUM_KV_GROUPS * Q_HEADS_PER_GROUP,
        "aie.memtile_dma": NUM_KV_GROUPS,
        "aie.core(": NUM_KV_GROUPS * Q_HEADS_PER_GROUP,
        "aie.mem(": NUM_KV_GROUPS * Q_HEADS_PER_GROUP,
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
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    if mlir_text.count(reshape_dims) != NUM_KV_GROUPS * Q_HEADS_PER_GROUP * 4:
        errors.append("Expected K/V ping-pong row1 reshape dimensions on every fanout channel")
    if f"memref<{HIDDEN_DIM}xbf16>" not in mlir_text:
        errors.append("Missing hidden buffer type")
    if f"memref<{RESHAPED_TILE_DWORDS}xf32>" not in mlir_text:
        errors.append("Missing reshaped tile buffer type")

    hidden_patches = len(re.findall(r"arg_idx = 0 : i32", mlir_text))
    if hidden_patches != NUM_KV_GROUPS:
        errors.append(f"Expected one hidden patch per KV group, got {hidden_patches}")

    cache_patches = len(re.findall(r"arg_idx = 1 : i32", mlir_text))
    expected_cache_patches = NUM_KV_GROUPS * (2 * num_tiles + 2)
    if cache_patches != expected_cache_patches:
        errors.append(f"Expected current-write plus K/V cache patches ({expected_cache_patches}), got {cache_patches}")
    return errors


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "projected_attention.cc"
    obj = EXPERIMENT_DIR / "projected_attention.o"
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
    print("  Compiling projected_attention.cc...")
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


def _current_cache_errors(
    context_len: int,
    npu_cache: np.ndarray,
    current_k: np.ndarray,
    current_v: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    for group in range(NUM_KV_GROUPS):
        k_offset, v_offset = current_offsets(context_len, group)
        k_err = np.abs(npu_cache[k_offset:k_offset + HEAD_DIM] - current_k[group])
        v_err = np.abs(npu_cache[v_offset:v_offset + HEAD_DIM] - current_v[group])
        if np.any(k_err > CACHE_ABS_TOLERANCE):
            errors.append(f"group {group} current K max_abs={float(k_err.max()):.7f}")
        if np.any(v_err > CACHE_ABS_TOLERANCE):
            errors.append(f"group {group} current V max_abs={float(v_err.max()):.7f}")
    return errors


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

    hidden, query, current_k, current_v, cache_before, cache_after, expected = make_case(context_len)
    hidden_i32 = _bf16_to_i32_buffer(hidden)
    assert hidden_i32.shape[0] == HIDDEN_I32
    assert cache_before.shape[0] == NUM_KV_GROUPS * kv_group_stride_dwords(context_len)

    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden_i32.copy()).to(torch.int32))
    kv_buf = XRTTensor.from_torch(torch.from_numpy(cache_before.copy()).to(torch.float32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(handle, [hidden_buf, kv_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    npu_output = output_buf.to_torch().numpy().astype(np.float32)
    npu_cache = kv_buf.to_torch().numpy().astype(np.float32)

    output_err = np.abs(npu_output - expected)
    output_bad = np.where(~np.isfinite(npu_output) | (output_err > OUTPUT_ABS_TOLERANCE))[0]
    cache_errors = _current_cache_errors(context_len, npu_cache, current_k, current_v)
    unchanged_mask = np.ones(cache_before.shape[0], dtype=bool)
    for group in range(NUM_KV_GROUPS):
        k_offset, v_offset = current_offsets(context_len, group)
        unchanged_mask[k_offset:k_offset + HEAD_DIM] = False
        unchanged_mask[v_offset:v_offset + HEAD_DIM] = False
    unchanged_bad = np.where(npu_cache[unchanged_mask] != cache_after[unchanged_mask])[0]
    passed = output_bad.size == 0 and len(cache_errors) == 0 and unchanged_bad.size == 0

    if passed:
        print(
            f"  PASS  time={npu_time_us:.1f}us  max_abs={float(output_err.max()):.7f}  "
            f"q[0:4]={query[:4]}  out[0:4]={npu_output[:4]}"
        )
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  max_abs={float(output_err.max()):.7f}")
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
    print("Experiment 39: Projected Current-Write Full Attention")
    print("=" * 72)
    print(f"Hidden dim: {HIDDEN_DIM} bf16")
    print(f"KV groups: {NUM_KV_GROUPS}, Q heads per group: {Q_HEADS_PER_GROUP}, output dwords: {OUTPUT_DWORDS}")
    print(f"Plane tile dwords: {PLANE_TILE_DWORDS}, per-group output dwords: {OUTPUT_GROUP_DWORDS}")
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
        print("SUCCESS: NPU projected Q/K/V feeds current-write full attention.")
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
