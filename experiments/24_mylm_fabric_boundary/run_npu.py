#!/usr/bin/env python3
"""Run exp24 on NPU and compare against the CPU reference."""

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
    HIDDEN_DIM,
    HIDDEN_I32,
    OUTPUT_DWORDS,
    PLANE_TILE_DWORDS,
    TOKEN_DWORDS,
    TOKENS_PER_TILE,
    TOTAL_WEIGHT_I32,
    generate_mlir,
)
from reference import (
    apply_current_to_cache,
    layer_reference_after_current_write,
    make_case_data,
)

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str, num_tiles: int) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "aiex.npu.writebd": 12,
        "aiex.npu.address_patch": 12,
        "aiex.npu.push_queue": 12,
        "aiex.npu.sync": 2,
        "aie.flow": 11,
        "aie.memtile_dma": 2,
        "aie.core(": 2,
        "aie.mem(": 2,
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

    markers = [
        "zero_token",
        "q4nx_project_chunk_accum",
        "proj_current_k",
        "edge_current_k",
        "mem1_query",
        "mem1_current",
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
    src = EXPERIMENT_DIR / "qkv_attention.cc"
    obj = EXPERIMENT_DIR / "qkv_attention.o"
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
    print("  Compiling qkv_attention.cc...")
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


def _bf16_to_i32_buffer(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.view(np.uint8).tobytes(), dtype=np.int32)


def _bytes_to_i32_buffer(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(), dtype=np.int32)


def _check_current_cache_write(
    L: int,
    before: np.ndarray,
    after_npu: np.ndarray,
    current_k: np.ndarray,
    current_v: np.ndarray,
) -> list[str]:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    token_offset = (L - 1) * TOKEN_DWORDS
    expected_after = apply_current_to_cache(L, before, current_k, current_v)
    errors: list[str] = []
    k_start = token_offset
    v_start = plane_dwords + token_offset
    if not np.allclose(
        after_npu[k_start:k_start + TOKEN_DWORDS],
        expected_after[k_start:k_start + TOKEN_DWORDS],
        rtol=1e-3,
        atol=5e-3,
    ):
        errors.append("K current token cache write mismatch")
    if not np.allclose(
        after_npu[v_start:v_start + TOKEN_DWORDS],
        expected_after[v_start:v_start + TOKEN_DWORDS],
        rtol=1e-3,
        atol=5e-3,
    ):
        errors.append("V current token cache write mismatch")
    return errors


def run_case(L: int, hidden: np.ndarray, packed_weight: np.ndarray) -> bool:
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
    errors = check_mlir_structure(mlir_text, num_tiles)
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

    query, current_k, current_v, cache_before, expected = layer_reference_after_current_write(L, hidden, packed_weight)
    hidden_i32 = _bf16_to_i32_buffer(hidden)
    weight_i32 = _bytes_to_i32_buffer(packed_weight)
    assert hidden_i32.shape[0] == HIDDEN_I32
    assert weight_i32.shape[0] == TOTAL_WEIGHT_I32

    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden_i32.copy()).to(torch.int32))
    weight_buf = XRTTensor.from_torch(torch.from_numpy(weight_i32.copy()).to(torch.int32))
    cache_buf = XRTTensor.from_torch(torch.from_numpy(cache_before.copy()).to(torch.float32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.float32)

    result = aie_utils.DefaultNPURuntime.run(handle, [hidden_buf, weight_buf, cache_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    npu_output = output_buf.to_torch().numpy().astype(np.float32)
    cache_after = cache_buf.to_torch().numpy().astype(np.float32)

    cache_errors = _check_current_cache_write(L, cache_before, cache_after, current_k, current_v)
    finite = np.isfinite(npu_output) & np.isfinite(expected)
    abs_err = np.abs(npu_output - expected)
    bad = np.where((~finite) | (abs_err > 5e-3))[0]
    passed = len(cache_errors) == 0 and bad.size == 0
    finite_abs = abs_err[finite]
    max_abs = float(finite_abs.max()) if finite_abs.size else float("nan")
    if passed:
        print(
            f"  PASS  time={npu_time_us:.1f}us  max_abs={max_abs:.6f}  "
            f"query[0:4]={query[:4]}  out[0:4]={npu_output[:4]}"
        )
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  max_abs={max_abs:.6f}")
        for error in cache_errors:
            print(f"    {error}")
        if bad.size:
            print(f"    output mismatches: {bad.size}/{OUTPUT_DWORDS}")
            for idx in bad[:8]:
                print(
                    f"      out[{idx}]: expected={expected[idx]:.7f} "
                    f"got={npu_output[idx]:.7f} err={abs_err[idx]:.7f}"
                )
    return passed


def main() -> bool:
    print("=" * 72)
    print("Experiment 24: MyLM Fabric Boundary")
    print("=" * 72)
    print(f"Hidden dim: {HIDDEN_DIM} bf16, output token dwords: {TOKEN_DWORDS}")
    print(f"Q/K/V weight blocks: {TOTAL_WEIGHT_I32} i32, attention tile: {TOKENS_PER_TILE} tokens")
    print("Topology: projection fabric on col0, edge/KV fabric on col1")
    print()

    hidden, packed_weight, query, current_k, current_v = make_case_data()
    print(f"Reference query[0:4]: {query[:4]}")
    print(f"Reference current K[0:4]: {current_k[:4]}")
    print(f"Reference current V[0:4]: {current_v[:4]}")

    print("Compiling kernel...")
    if not compile_kernel():
        return False

    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    all_passed = True
    for L in [1, 15, 16, 17, 31, 32, 79]:
        all_passed &= run_case(L, hidden, packed_weight)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: projection and edge fabrics exchange Q/K/V and scan updated cache.")
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
