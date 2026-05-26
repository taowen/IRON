#!/usr/bin/env python3
"""
Experiment 15: Complete FFN with Down Projection + Bidirectional Memtile Flow.

Proves:
1. Bidirectional memtile flow — cores send intermediate UP, memtile gathers, broadcasts back DOWN
2. Sequential BD chain on S2MM ch0 — activation THEN gathered intermediate (same channel, different data)
3. Complete FFN: gate → up → SwiGLU → gather → down projection
4. Down projection uses gathered intermediate (K=64) as input
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
    generate_mlir, M_PER_TILE, NUM_COLS, ROWS_PER_COL, K_HIDDEN, K_CHUNK,
    NUM_CHUNKS, GROUP_SIZE, NUM_PHASES_PROJ, TOTAL_OUTPUT,
    CHUNK_BF16, FAT_CHUNK_BF16, DOWN_CHUNK_BF16, FAT_DOWN_CHUNK_BF16,
    ACT_BF16, OUT_BF16, INTER_BF16, GATHERED_BF16,
    INTERMEDIATE, K_DOWN, K_CHUNK_DOWN, GROUPS_PER_ROW_DOWN,
    PER_COL_WT_BF16, TOTAL_WT_I32, ACT_I32, OUT_TOTAL_I32,
)
from reference import pack_all_weights, complete_ffn_reference

EXPERIMENT_DIR = Path(__file__).parent
NUM_TILES = NUM_COLS * ROWS_PER_COL


def check_mlir_structure(mlir_text: str) -> list:
    """Static checker: verify structural invariants in generated MLIR."""
    errors = []
    import re

    if "dma_configure_task_for" in mlir_text:
        errors.append("Found dma_configure_task_for — should use writebd")

    if "shim_dma_allocation" in mlir_text:
        errors.append("Found shim_dma_allocation — should use writebd")

    # Expected writebd: 2 cols × (1 act + 4 gate/up_wt + 1 down_wt + 2 out) = 16
    n_writebd = len(re.findall(r"aiex\.npu\.writebd", mlir_text))
    expected_writebd = NUM_COLS * (1 + NUM_PHASES_PROJ * NUM_CHUNKS + 1 + ROWS_PER_COL)  # 16
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

    # 4 syncs (2 per column)
    n_sync = len(re.findall(r"aiex\.npu\.sync", mlir_text))
    if n_sync != 4:
        errors.append(f"Expected 4 sync, got {n_sync}")

    # 3 runtime_sequence args
    rt_match = re.search(r'aie\.runtime_sequence\(([^)]+)\)', mlir_text)
    if rt_match:
        args = rt_match.group(1).split(',')
        if len(args) != 3:
            errors.append(f"Expected 3 runtime_sequence args, got {len(args)}")
    else:
        errors.append("No runtime_sequence found")

    # 8 circuit-switched flows per column + 2 packet flows per column
    n_flows = len(re.findall(r"aie\.flow", mlir_text))
    n_pkt_flows = len(re.findall(r"aie\.packet_flow", mlir_text))
    expected_flows = NUM_COLS * 8
    expected_pkt_flows = NUM_COLS * 2
    if n_flows != expected_flows:
        errors.append(f"Expected {expected_flows} aie.flow, got {n_flows}")
    if n_pkt_flows != expected_pkt_flows:
        errors.append(f"Expected {expected_pkt_flows} aie.packet_flow, got {n_pkt_flows}")

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

    # Compile q4nx_chunk_accum.cc — NOTE: needs both K_CHUNK sizes
    # The kernel uses Q4_K_CHUNK=256 for gate/up, but also gets called with
    # the down_wt_buf which is only 640 bf16 (K=64). The kernel dispatches
    # based on num_rows and act_offset. Since we call it with the full buffer
    # ref and offset=0, the kernel always processes from offset. We compile
    # with Q4_K_CHUNK=256 since gate/up use it; for down projection we still
    # call the same kernel but the buffer is sized for K=64 chunk format.
    # Actually, the kernel needs a SEPARATE compilation for K_CHUNK=64.
    # We'll compile two variants.

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

    # Variant 2: down (K_CHUNK=64) — separate object file with renamed symbols
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
    print(f"  Tiles: {NUM_TILES} ({NUM_COLS} cols x {ROWS_PER_COL} rows)")
    print(f"  Complete FFN: gate + up + SwiGLU + down")
    print(f"  M_PER_TILE={M_PER_TILE}, K_HIDDEN={K_HIDDEN}, K_DOWN={K_DOWN}")
    print(f"  Gate/Up: K_CHUNK={K_CHUNK}, NUM_CHUNKS={NUM_CHUNKS}")
    print(f"  Down: K_CHUNK_DOWN={K_CHUNK_DOWN}, NUM_CHUNKS_DOWN=1")
    print(f"  Q4NX gate/up chunk: {CHUNK_BF16 * 2} bytes")
    print(f"  Q4NX down chunk: {DOWN_CHUNK_BF16 * 2} bytes")
    print(f"  Fat gate/up chunk: {FAT_CHUNK_BF16 * 2} bytes")
    print(f"  Fat down chunk: {FAT_DOWN_CHUNK_BF16 * 2} bytes")
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
    activation = np.random.randn(K_HIDDEN).astype(bfloat16)

    # Gate/Up weights
    all_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    all_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            all_scales[col][row] = [
                np.random.uniform(0.01, 0.5, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_zeros[col][row] = [
                np.random.uniform(4, 12, (M_PER_TILE, K_HIDDEN // GROUP_SIZE)).astype(bfloat16)
                for _ in range(NUM_PHASES_PROJ)
            ]
            all_int4[col][row] = [
                np.random.randint(0, 16, (M_PER_TILE, K_HIDDEN), dtype=np.uint8)
                for _ in range(NUM_PHASES_PROJ)
            ]

    # Down weights (K=64)
    down_scales = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_zeros = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]
    down_int4 = [[None] * ROWS_PER_COL for _ in range(NUM_COLS)]

    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            down_scales[col][row] = np.random.uniform(0.01, 0.5, (M_PER_TILE, GROUPS_PER_ROW_DOWN)).astype(bfloat16)
            down_zeros[col][row] = np.random.uniform(4, 12, (M_PER_TILE, GROUPS_PER_ROW_DOWN)).astype(bfloat16)
            down_int4[col][row] = np.random.randint(0, 16, (M_PER_TILE, K_DOWN), dtype=np.uint8)

    # Pack weights
    packed = pack_all_weights(all_scales, all_zeros, all_int4, down_scales, down_zeros, down_int4)
    assert packed.shape[0] == total_wt_bytes, f"Weight size mismatch: {packed.shape[0]} != {total_wt_bytes}"

    # CPU reference
    ref_output = complete_ffn_reference(packed, activation)
    print(f"  Ref output[0:4] (col0_r0): {ref_output[:4]}")
    print(f"  Ref output[32:36] (col0_r1): {ref_output[32:36]}")

    # Load xclbin
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

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
    print(f"  NPU output[32:36] (col0_r1): {npu_output[32:36]}")

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

    # Tolerance: wider than exp 14 — two Q4NX projections + SwiGLU amplifies rounding
    rel_tol = 0.35
    abs_tol = 0.35
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    mask = abs_err >= np.maximum(abs_tol, rel_tol * norm)
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {TOTAL_OUTPUT} elements match within tolerance (rel={rel_tol}, abs={abs_tol})")
    else:
        print(f"\n  FAIL: {num_errors}/{TOTAL_OUTPUT} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:12]:
            col = idx // (ROWS_PER_COL * M_PER_TILE)
            rem = idx % (ROWS_PER_COL * M_PER_TILE)
            row = rem // M_PER_TILE
            elem = rem % M_PER_TILE
            print(f"    [{idx}] col{col}_r{row}_e{elem}: ref={ref_f32[idx]:.4f} npu={npu_f32[idx]:.4f} err={abs_err[idx]:.4f}")

    print()
    print("=" * 70)
    if num_errors == 0:
        print("SUCCESS: Complete FFN with Bidirectional Memtile Flow!")
        print(f"  {NUM_TILES} tiles ({NUM_COLS} cols x {ROWS_PER_COL} rows) = {TOTAL_OUTPUT} outputs verified")
        print(f"  Bidirectional flow: cores→memtile (intermediate) + memtile→cores (gathered)")
        print(f"  Sequential BD chain: S2MM ch0 receives activation THEN gathered intermediate")
        print(f"  Complete FFN: gate → up → SwiGLU → gather → down → output")
        print(f"  No DDR round-trip for intermediate (stays in fabric)")
        print(f"  Latency: {npu_time_us:.1f} us")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
    print("=" * 70)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 15: Complete FFN")
    print("  Down Projection + Bidirectional Memtile Flow")
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
