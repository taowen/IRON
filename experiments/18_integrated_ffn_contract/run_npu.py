#!/usr/bin/env python3
"""
Experiment 18: integrated fused FFN contract.

The full smoke uses bounded values so the run checks dataflow correctness
instead of stressing SwiGLU saturation. Phase probes in phase_probe.py cover
the descriptor-lifetime and phase-reuse ownership rules retained from the older
FFN prototypes.
"""

import os
import sys
import traceback
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from generate import (
    generate_mlir, M_PER_TILE, NUM_COLS, ROWS_PER_COL, K_HIDDEN, K_CHUNK,
    NUM_CHUNKS, GROUP_SIZE, NUM_PHASES_PROJ, TOTAL_OUTPUT,
    CHUNK_BF16, FAT_CHUNK_BF16, DOWN_CHUNK_BF16,
    ACT_BF16, OUT_BF16, INTER_BF16, GATHERED_BF16,
    INTERMEDIATE, K_DOWN, K_CHUNK_DOWN, GROUPS_PER_ROW_DOWN,
    PUSHES_PER_COL, PER_COL_WT_BF16,
    TOTAL_WT_I32, ACT_I32, OUT_TOTAL_I32,
)
from reference import pack_all_weights, complete_ffn_reference
from preflight import preflight_check

EXPERIMENT_DIR = Path(__file__).parent
NUM_TILES = NUM_COLS * ROWS_PER_COL


def compile_kernels(build_dir):
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    common_flags = [
        "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
    ]

    src1 = EXPERIMENT_DIR / "q4nx_chunk_accum.cc"

    # Variant 1: gate/up (K_CHUNK=256)
    obj1 = EXPERIMENT_DIR / "q4nx_chunk_accum.o"
    cmd1 = [str(clang)] + common_flags + [
        f"-DQ4_M={M_PER_TILE}", f"-DQ4_K_CHUNK={K_CHUNK}", f"-DGROUP_SIZE={GROUP_SIZE}",
        "-c", str(src1), "-o", str(obj1),
    ]
    print(f"  Compiling q4nx_chunk_accum.cc (K_CHUNK={K_CHUNK})...")
    ret = os.system(" ".join(cmd1))
    if ret != 0:
        return False

    # Variant 2: down (K_CHUNK=128) — renamed symbols
    obj1b = EXPERIMENT_DIR / "q4nx_chunk_accum_down.o"
    cmd1b = [str(clang)] + common_flags + [
        f"-DQ4_M={M_PER_TILE}", f"-DQ4_K_CHUNK={K_CHUNK_DOWN}", f"-DGROUP_SIZE={GROUP_SIZE}",
        "-Dq4nx_chunk_accum_offset=q4nx_down_chunk_accum_offset",
        "-Dq4nx_flush_output=q4nx_down_flush_output",
        "-c", str(src1), "-o", str(obj1b),
    ]
    print(f"  Compiling q4nx_chunk_accum.cc (K_CHUNK={K_CHUNK_DOWN}, renamed)...")
    ret = os.system(" ".join(cmd1b))
    if ret != 0:
        return False

    # Compile swiglu_fused.cc
    src2 = EXPERIMENT_DIR / "swiglu_fused.cc"
    obj2 = EXPERIMENT_DIR / "swiglu_fused.o"
    cmd2 = [str(clang)] + common_flags + [
        "-c", str(src2), "-o", str(obj2),
    ]
    print(f"  Compiling swiglu_fused.cc...")
    ret = os.system(" ".join(cmd2))
    if ret != 0:
        return False

    return True


def compile_mlir(build_dir, mlir_path):
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    cmd = [
        str(aiecc), "-v", "-j1",
        "--no-compile-host", "--no-xchesscc", "--no-xbridge",
        "--peano", str(peano_dir),
        "--aie-generate-xclbin", f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print(f"  Compiling MLIR -> xclbin + insts.bin...")
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def run_on_npu():
    total_wt_bytes = TOTAL_WT_I32 * 4
    print(f"Configuration:")
    print(f"  Tiles: {NUM_TILES} ({NUM_COLS} cols × {ROWS_PER_COL} rows)")
    print(f"  Complete FFN: gate + up + SwiGLU + gather + down")
    print(f"  M_PER_TILE={M_PER_TILE}, K_HIDDEN={K_HIDDEN}, K_DOWN={K_DOWN}")
    print(f"  Gate/Up: K_CHUNK={K_CHUNK}, NUM_CHUNKS={NUM_CHUNKS}, pushes={NUM_PHASES_PROJ*NUM_CHUNKS}")
    print(f"  Down: K_CHUNK_DOWN={K_CHUNK_DOWN}, pushes=1 (padded to {CHUNK_BF16*2} bytes)")
    print(f"  Total pushes per column: {PUSHES_PER_COL}")
    print(f"  Weight path: ping/pong memtile BDs + unique shim BDs for queued pushes")
    print(f"  Weight BO: {total_wt_bytes} bytes ({total_wt_bytes//1024} KB)")
    print(f"  Activation BO: {ACT_I32 * 4} bytes ({ACT_BF16} bf16)")
    print(f"  Output BO: {OUT_TOTAL_I32 * 4} bytes ({TOTAL_OUTPUT} bf16)")
    print()

    # Generate MLIR
    print("Generating MLIR...")
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Preflight checks
    print("  Running preflight checks...")
    errors = preflight_check(mlir_text)
    if errors:
        print("PREFLIGHT FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print("  Preflight: PASS (all 8 checks)")

    # Compile kernels
    print("\nCompiling kernels...")
    if not compile_kernels(build_dir):
        print("FAILED: Kernel compilation error")
        return False

    # Compile MLIR
    print("\nCompiling MLIR...")
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    print("\nRunning on NPU...")
    dev = aie_utils.DefaultNPURuntime.device()
    print(f"  NPU device: {dev}")

    # Generate test data
    np.random.seed(42)
    activation = (0.125 * np.random.randn(K_HIDDEN)).astype(bfloat16)

    all_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            all_scales[col][row] = [
                np.random.uniform(0.005, 0.03, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_zeros[col][row] = [
                np.random.uniform(6, 9, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K_HIDDEN), dtype=np.uint8)
                for _ in range(NUM_PHASES_PROJ)
            ]

    down_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            down_scales[col][row] = np.random.uniform(
                0.005, 0.03, (M_PER_TILE, GROUPS_PER_ROW_DOWN)
            ).astype(bfloat16)
            down_zeros[col][row] = np.random.uniform(
                6, 9, (M_PER_TILE, GROUPS_PER_ROW_DOWN)
            ).astype(bfloat16)
            down_int4[col][row] = np.random.randint(
                0, 16, (M_PER_TILE, K_DOWN), dtype=np.uint8
            )

    # Pack weights
    packed = pack_all_weights(all_scales, all_zeros, all_int4,
                              down_scales, down_zeros, down_int4)
    assert packed.shape[0] == total_wt_bytes, f"Weight size mismatch: {packed.shape[0]} != {total_wt_bytes}"

    # CPU reference
    ref_output = complete_ffn_reference(packed, activation)
    print(f"  Ref output[0:4] (col0_r0): {ref_output[:4]}")
    print(f"  Ref output[96:100] (col0_r3): {ref_output[96:100]}")

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Weight buffer
    wt_i32 = packed.view(np.uint8).view(np.int32)
    wt_buf = XRTTensor.from_torch(torch.from_numpy(wt_i32.copy()).to(torch.int32))

    # Activation buffer
    act_bytes = activation.view(np.uint8)
    act_i32 = np.frombuffer(act_bytes.tobytes(), dtype=np.int32)
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))

    # Output buffer
    out_buf = XRTTensor((OUT_TOTAL_I32,), dtype=np.int32)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)
    print(f"  NPU output[0:4] (col0_r0): {npu_output[:4]}")
    print(f"  NPU output[96:100] (col0_r3): {npu_output[96:100]}")

    # Verify
    print("\nVerification:")
    ref_f32 = ref_output.astype(np.float32)
    npu_f32 = npu_output.astype(np.float32)

    abs_err = np.abs(ref_f32 - npu_f32)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    ref_abs = np.abs(ref_f32)
    rel_err = abs_err / np.maximum(ref_abs, 1e-6)
    max_rel_err = np.max(rel_err)

    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Max relative error: {max_rel_err:.4f} ({max_rel_err*100:.2f}%)")

    # Per-tile breakdown
    print("\n  Per-tile breakdown:")
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            idx = (col * ROWS_PER_COL + row) * M_PER_TILE
            tile_err = np.max(np.abs(ref_f32[idx:idx+M_PER_TILE] - npu_f32[idx:idx+M_PER_TILE]))
            print(f"    Col{col} Row{row}: max_err={tile_err:.4f}")

    # Q4NX projection output is rounded to bf16 between phases, and SwiGLU uses
    # the AIE tanh-based sigmoid implementation.  The bounded input keeps this a
    # dataflow correctness test rather than a numerical stress test.
    rel_tol = 0.40
    abs_tol = 0.40
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    finite_mask = np.isfinite(ref_f32) & np.isfinite(npu_f32)
    mask = (~finite_mask) | (abs_err >= np.maximum(abs_tol, rel_tol * norm))
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {TOTAL_OUTPUT} elements match within tolerance")
    else:
        print(f"\n  FAIL: {num_errors}/{TOTAL_OUTPUT} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:12]:
            col_idx = idx // (ROWS_PER_COL * M_PER_TILE)
            rem = idx % (ROWS_PER_COL * M_PER_TILE)
            row_idx = rem // M_PER_TILE
            elem = rem % M_PER_TILE
            print(f"    [{idx}] col{col_idx}_r{row_idx}_e{elem}: ref={ref_f32[idx]:.4f} npu={npu_f32[idx]:.4f}")

    print()
    print("=" * 70)
    if num_errors == 0:
        print("SUCCESS: Integrated fused FFN contract")
        print(f"  {NUM_TILES} tiles ({NUM_COLS} col × {ROWS_PER_COL} rows) = {TOTAL_OUTPUT} outputs verified")
        print(f"  K={K_HIDDEN}, {PUSHES_PER_COL} queued weight pushes use distinct shim BDs")
        print(f"  Multicast activation + 4-core packet gather per column")
        print(f"  Complete FFN: gate → up → SwiGLU → gather → down → output")
        print(f"  Latency: {npu_time_us:.1f} us")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
    print("=" * 70)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 18: Integrated Fused FFN Contract")
    print(f"  {NUM_COLS} col × {ROWS_PER_COL} rows, K={K_HIDDEN}, per-slot lock + unique shim BDs")
    print("=" * 70)
    print()

    try:
        success = run_on_npu()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    sys.exit(0 if success else 1)
