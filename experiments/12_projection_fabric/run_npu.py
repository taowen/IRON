#!/usr/bin/env python3
"""
Experiment 12: Projection Fabric — 4-Tile Phase-Reuse Q4NX GEMV on NPU.

Proves:
1. 4-tile parallel fabric (2 cols x 2 rows), identical Q4NX programs
2. Activation hold/replay across 2 projections (no DDR re-fetch)
3. Phase reuse: projA → projB via weight DMA queue, zero host intervention
4. Weight distribution: per-tile TAPs into shared weight BO
5. Cross-chunk FP32 accumulation (K=1024, 4 chunks per projection per tile)
6. Numerical equivalence: all 256 output elements match CPU reference
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

from design import (
    projection_fabric_pipeline, M_PER_TILE, NUM_TILES, K, K_CHUNK,
    NUM_CHUNKS, GROUP_SIZE, NUM_PROJECTIONS, TOTAL_OUTPUT,
)
from reference import pack_all_weights, projection_fabric_reference

EXPERIMENT_DIR = Path(__file__).parent


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
        "--dynamic-objFifos",
        "--aie-generate-xclbin", f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print(f"  Compiling MLIR -> xclbin + insts.bin...")
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def run_on_npu():
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    per_tile_bytes = NUM_PROJECTIONS * NUM_CHUNKS * chunk_bytes
    total_wt_bytes = NUM_TILES * per_tile_bytes

    print(f"Configuration:")
    print(f"  Tiles: {NUM_TILES} (2 cols x 2 rows)")
    print(f"  Projections: {NUM_PROJECTIONS}")
    print(f"  M_PER_TILE={M_PER_TILE}, K={K}, K_CHUNK={K_CHUNK}, NUM_CHUNKS={NUM_CHUNKS}")
    print(f"  Q4NX chunk: {chunk_bytes} bytes ({chunk_bytes // 2} bf16)")
    print(f"  Weight per tile: {per_tile_bytes} bytes")
    print(f"  Weight BO total: {total_wt_bytes} bytes ({total_wt_bytes // 2} bf16)")
    print(f"  Activation BO: {K * 2} bytes")
    print(f"  Output BO: {TOTAL_OUTPUT * 2} bytes ({TOTAL_OUTPUT} bf16)")
    print()

    dev = aie_utils.DefaultNPURuntime.device()
    print(f"NPU device: {dev} (cols={dev.cols})")

    # Generate MLIR
    mlir_module = projection_fabric_pipeline(dev=dev)
    mlir_str = str(mlir_module)

    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_str)

    # Compile kernel
    print("Compiling kernel...")
    if not compile_kernel(build_dir):
        print("FAILED: Kernel compilation error")
        return False

    # Compile MLIR
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    print("\nRunning on NPU...")

    # Generate test data
    np.random.seed(42)
    activation = np.random.randn(K).astype(bfloat16)

    all_scales = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]
    all_zeros = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]
    all_int4 = [[None] * NUM_PROJECTIONS for _ in range(NUM_TILES)]

    for tile in range(NUM_TILES):
        for proj in range(NUM_PROJECTIONS):
            all_scales[tile][proj] = np.random.uniform(
                0.01, 0.5, (M_PER_TILE, K // GROUP_SIZE)
            ).astype(bfloat16)
            all_zeros[tile][proj] = np.random.uniform(
                4, 12, (M_PER_TILE, K // GROUP_SIZE)
            ).astype(bfloat16)
            all_int4[tile][proj] = np.random.randint(
                0, 16, (M_PER_TILE, K), dtype=np.uint8
            )

    # Pack all weights
    packed = pack_all_weights(all_scales, all_zeros, all_int4)
    assert packed.shape[0] == total_wt_bytes

    # CPU reference
    ref_output = projection_fabric_reference(packed, activation)
    print(f"  Activation[0:4]: {activation[:4]}")
    print(f"  Ref output[0:4] (tile0_proj0): {ref_output[:4]}")
    print(f"  Ref output[32:36] (tile0_proj1): {ref_output[32:36]}")

    # Run on NPU
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Weight buffer: reinterpret packed bytes as bf16
    packed_bf16 = packed.view(np.uint16).view(bfloat16)
    wt_t = torch.from_numpy(packed_bf16.view(np.uint16).copy()).view(torch.bfloat16)
    wt_buf = XRTTensor.from_torch(wt_t)

    # Activation buffer
    act_t = torch.from_numpy(activation.view(np.uint16)).view(torch.bfloat16)
    act_buf = XRTTensor.from_torch(act_t)

    # Output buffer
    output_buf = XRTTensor((TOTAL_OUTPUT,), dtype=bfloat16)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = output_buf.to_torch()
    npu_output = output_torch.view(dtype=torch.uint16).numpy().view(bfloat16)
    print(f"  NPU output[0:4] (tile0_proj0): {npu_output[:4]}")
    print(f"  NPU output[32:36] (tile0_proj1): {npu_output[32:36]}")

    # Verify all 256 elements
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
    for tile in range(NUM_TILES):
        for proj in range(NUM_PROJECTIONS):
            idx = tile * M_PER_TILE * NUM_PROJECTIONS + proj * M_PER_TILE
            tile_err = np.max(np.abs(ref_f32[idx:idx+M_PER_TILE] - npu_f32[idx:idx+M_PER_TILE]))
            print(f"    Tile {tile} Proj {proj}: max_err={tile_err:.4f}")

    # Q4 tolerance
    rel_tol = 0.15
    abs_tol = 0.1
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    mask = abs_err >= np.maximum(abs_tol, rel_tol * norm)
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {TOTAL_OUTPUT} elements match within tolerance (rel={rel_tol}, abs={abs_tol})")
    else:
        print(f"\n  FAIL: {num_errors}/{TOTAL_OUTPUT} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:12]:
            tile = idx // (M_PER_TILE * NUM_PROJECTIONS)
            rem = idx % (M_PER_TILE * NUM_PROJECTIONS)
            proj = rem // M_PER_TILE
            row = rem % M_PER_TILE
            print(f"    [{idx}] tile{tile}_proj{proj}_row{row}: ref={ref_f32[idx]:.4f} npu={npu_f32[idx]:.4f} err={abs_err[idx]:.4f}")

    print()
    print("=" * 70)
    if num_errors == 0:
        print("SUCCESS: Projection Fabric produces CORRECT output!")
        print(f"  {NUM_TILES} tiles x {NUM_PROJECTIONS} projections = {TOTAL_OUTPUT} outputs verified")
        print(f"  Activation held across projections (no DDR re-fetch)")
        print(f"  Phase reuse: projA -> projB via weight DMA queue")
        print(f"  Weight distribution: per-tile TAPs into shared {total_wt_bytes}-byte BO")
        print(f"  Latency: {npu_time_us:.1f} us for {NUM_TILES}x parallel Q4NX GEMV")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
    print("=" * 70)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 12: Projection Fabric - Multi-Tile Phase-Reuse Q4NX GEMV")
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
