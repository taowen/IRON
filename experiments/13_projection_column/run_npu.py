#!/usr/bin/env python3
"""
Experiment 13: Low-Level Projection Column — Memtile Broadcast + Writebd on NPU.

Proves:
1. Memtile activation broadcast (2-consumer lock, DDR→memtile ONCE, fan-out to 2 cores)
2. Memtile weight distribution (fat chunk → offset split-read to different cores)
3. Multi-column (2 independent column pipes on separate shim/memtile/core)
4. writebd/address_patch/push_queue (raw NPU instructions for ALL shim DMA)
5. Static BD rings (explicit aie.memtile_dma and aie.mem BD chains)
6. Activation hold (core holds activation in local memory across 2 projections)
7. Phase reuse (8 weight BD submissions per column queued upfront)
"""

import sys
import os
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path

from generate import (
    generate_mlir, M_PER_TILE, NUM_COLS, ROWS_PER_COL, K, K_CHUNK,
    NUM_CHUNKS, GROUP_SIZE, NUM_PROJECTIONS, TOTAL_OUTPUT,
    CHUNK_BF16, FAT_CHUNK_BF16, ACT_BF16, OUT_BF16,
    TOTAL_WT_I32, ACT_I32, OUT_TOTAL_I32,
)
from reference import pack_all_weights_fat, projection_column_reference

EXPERIMENT_DIR = Path(__file__).parent
NUM_TILES = NUM_COLS * ROWS_PER_COL


def check_mlir_structure(mlir_text: str) -> list:
    """Static checker: verify writebd runtime invariants in generated MLIR."""
    errors = []

    # No dma_configure_task_for (fully replaced)
    if "dma_configure_task_for" in mlir_text:
        errors.append("Found dma_configure_task_for — should use writebd")

    # No shim_dma_allocation
    if "shim_dma_allocation" in mlir_text:
        errors.append("Found shim_dma_allocation — should use writebd")

    # Expected writebd count: 2 cols × (1 act + 8 wt + 4 out) = 26
    import re
    n_writebd = len(re.findall(r"aiex\.npu\.writebd", mlir_text))
    expected_writebd = NUM_COLS * (1 + NUM_PROJECTIONS * NUM_CHUNKS + NUM_PROJECTIONS * ROWS_PER_COL)
    if n_writebd != expected_writebd:
        errors.append(f"Expected {expected_writebd} writebd, got {n_writebd}")

    # Expected address_patch = same as writebd
    n_patch = len(re.findall(r"aiex\.npu\.address_patch", mlir_text))
    if n_patch != expected_writebd:
        errors.append(f"Expected {expected_writebd} address_patch, got {n_patch}")

    # Expected push_queue = same as writebd
    n_push = len(re.findall(r"aiex\.npu\.push_queue", mlir_text))
    if n_push != expected_writebd:
        errors.append(f"Expected {expected_writebd} push_queue, got {n_push}")

    # 2 syncs (one per S2MM channel on last column)
    n_sync = len(re.findall(r"aiex\.npu\.sync", mlir_text))
    if n_sync != 2:
        errors.append(f"Expected 2 sync, got {n_sync}")

    # 3 runtime_sequence args
    rt_match = re.search(r'aie\.runtime_sequence\(([^)]+)\)', mlir_text)
    if rt_match:
        args = rt_match.group(1).split(',')
        if len(args) != 3:
            errors.append(f"Expected 3 runtime_sequence args, got {len(args)}")
    else:
        errors.append("No runtime_sequence found")

    # 16 flows (8 per column)
    n_flows = len(re.findall(r"aie\.flow", mlir_text))
    if n_flows != 16:
        errors.append(f"Expected 16 flows, got {n_flows}")

    # 2 memtile_dma blocks
    n_mt_dma = len(re.findall(r"aie\.memtile_dma", mlir_text))
    if n_mt_dma != 2:
        errors.append(f"Expected 2 memtile_dma, got {n_mt_dma}")

    # 4 core programs
    n_core = len(re.findall(r"aie\.core\(", mlir_text))
    if n_core != 4:
        errors.append(f"Expected 4 aie.core, got {n_core}")

    # 4 aie.mem blocks
    n_mem = len(re.findall(r"aie\.mem\(", mlir_text))
    if n_mem != 4:
        errors.append(f"Expected 4 aie.mem, got {n_mem}")

    return errors


def compile_kernel(build_dir):
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    src = EXPERIMENT_DIR / "q4nx_chunk_accum.cc"
    obj = EXPERIMENT_DIR / "q4nx_chunk_accum.o"

    cmd = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        f"-DQ4_M={M_PER_TILE}", f"-DQ4_K_CHUNK={K_CHUNK}", f"-DGROUP_SIZE={GROUP_SIZE}",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c", str(src), "-o", str(obj),
    ]
    print(f"  Compiling q4nx_chunk_accum.cc...")
    ret = os.system(" ".join(cmd))
    return ret == 0


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
    print(f"  Tiles: {NUM_TILES} ({NUM_COLS} cols x {ROWS_PER_COL} rows)")
    print(f"  Projections: {NUM_PROJECTIONS}")
    print(f"  M_PER_TILE={M_PER_TILE}, K={K}, K_CHUNK={K_CHUNK}, NUM_CHUNKS={NUM_CHUNKS}")
    print(f"  Q4NX chunk: {CHUNK_BF16 * 2} bytes ({CHUNK_BF16} bf16)")
    print(f"  Fat chunk: {FAT_CHUNK_BF16 * 2} bytes ({FAT_CHUNK_BF16} bf16)")
    print(f"  Weight BO: {total_wt_bytes} bytes ({TOTAL_WT_I32} i32)")
    print(f"  Activation BO: {ACT_I32 * 4} bytes ({ACT_I32} i32)")
    print(f"  Output BO: {OUT_TOTAL_I32 * 4} bytes ({OUT_TOTAL_I32} i32)")
    print()

    # Generate MLIR
    print("Generating MLIR...")
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Static structure check
    errors = check_mlir_structure(mlir_text)
    if errors:
        print("MLIR structure check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    print("  MLIR structure check: PASS")

    # Compile kernel
    print("\nCompiling kernel...")
    if not compile_kernel(build_dir):
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
    activation = np.random.randn(K).astype(bfloat16)

    all_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            all_scales[col][row] = [
                np.random.uniform(0.01, 0.5, (M_PER_TILE, K // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PROJECTIONS)
            ]
            all_zeros[col][row] = [
                np.random.uniform(4, 12, (M_PER_TILE, K // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PROJECTIONS)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K), dtype=np.uint8)
                for _ in range(NUM_PROJECTIONS)
            ]

    # Pack weights (fat-chunk layout grouped by column)
    packed = pack_all_weights_fat(all_scales, all_zeros, all_int4)
    assert packed.shape[0] == total_wt_bytes, f"Weight size mismatch: {packed.shape[0]} != {total_wt_bytes}"

    # CPU reference
    ref_output = projection_column_reference(packed, activation)
    print(f"  Ref output[0:4] (col0_r0_p0): {ref_output[:4]}")
    print(f"  Ref output[32:36] (col0_r0_p1): {ref_output[32:36]}")

    # Load xclbin
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Weight buffer: pack as i32 view
    wt_i32 = packed.view(np.uint8).view(np.int32)
    wt_buf = XRTTensor.from_torch(torch.from_numpy(wt_i32.copy()).to(torch.int32))

    # Activation buffer: bf16 → i32 view
    act_bytes = activation.view(np.uint8)
    act_i32 = np.frombuffer(act_bytes.tobytes(), dtype=np.int32)
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))

    # Output buffer
    out_buf = XRTTensor((OUT_TOTAL_I32,), dtype=np.int32)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output: i32 → bf16
    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)
    print(f"  NPU output[0:4] (col0_r0_p0): {npu_output[:4]}")
    print(f"  NPU output[32:36] (col0_r0_p1): {npu_output[32:36]}")

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

    # Per-tile/projection breakdown
    print("\n  Per-tile breakdown:")
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            for proj in range(NUM_PROJECTIONS):
                idx = (col * ROWS_PER_COL * NUM_PROJECTIONS + row * NUM_PROJECTIONS + proj) * M_PER_TILE
                tile_err = np.max(np.abs(ref_f32[idx:idx+M_PER_TILE] - npu_f32[idx:idx+M_PER_TILE]))
                print(f"    Col{col} Row{row} Proj{proj}: max_err={tile_err:.4f}")

    # Q4 tolerance check (abs_tol slightly above bf16 ULP to account for memtile hop)
    rel_tol = 0.15
    abs_tol = 0.125
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    mask = abs_err >= np.maximum(abs_tol, rel_tol * norm)
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {TOTAL_OUTPUT} elements match within tolerance (rel={rel_tol}, abs={abs_tol})")
    else:
        print(f"\n  FAIL: {num_errors}/{TOTAL_OUTPUT} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:12]:
            col = idx // (ROWS_PER_COL * NUM_PROJECTIONS * M_PER_TILE)
            rem = idx % (ROWS_PER_COL * NUM_PROJECTIONS * M_PER_TILE)
            row = rem // (NUM_PROJECTIONS * M_PER_TILE)
            proj = (rem % (NUM_PROJECTIONS * M_PER_TILE)) // M_PER_TILE
            elem = rem % M_PER_TILE
            print(f"    [{idx}] col{col}_r{row}_p{proj}_e{elem}: ref={ref_f32[idx]:.4f} npu={npu_f32[idx]:.4f} err={abs_err[idx]:.4f}")

    print()
    print("=" * 70)
    if num_errors == 0:
        print("SUCCESS: Low-Level Projection Column produces CORRECT output!")
        print(f"  {NUM_TILES} tiles ({NUM_COLS} cols x {ROWS_PER_COL} rows) x {NUM_PROJECTIONS} projections = {TOTAL_OUTPUT} outputs verified")
        print(f"  Memtile activation broadcast (2-consumer lock, single DDR read)")
        print(f"  Memtile weight distribution (fat chunk offset split-read)")
        print(f"  writebd/address_patch/push_queue runtime (no dma_configure_task_for)")
        print(f"  Static BD rings (explicit aie.memtile_dma/aie.mem)")
        print(f"  Activation hold across projections (no DDR re-fetch)")
        print(f"  Phase reuse: 8 weight BDs queued upfront per column")
        print(f"  Latency: {npu_time_us:.1f} us")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
    print("=" * 70)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 13: Low-Level Projection Column")
    print("  Memtile Broadcast + Writebd Runtime")
    print("=" * 70)
    print()

    try:
        success = run_on_npu()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    sys.exit(0 if success else 1)
