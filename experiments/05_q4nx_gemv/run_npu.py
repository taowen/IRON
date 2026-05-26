#!/usr/bin/env python3
"""
Run Experiment 05: Q4NX Online Dequant+GEMV on NPU with correctness verification.

Packed int4 weights are unpacked, dequantized, and MAC'd with activation
entirely inside a single AIE compute tile.
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

from design import q4nx_gemv_pipeline
from reference import pack_q4nx_chunk, q4nx_matvec_reference


M = 32
K = 256
GROUP_SIZE = 32


def compile_kernel(build_dir):
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    src = Path(__file__).parent / "q4nx_matvec.cc"
    obj = build_dir / "q4nx_matvec.o"

    cmd = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        f"-DQ4_M={M}", f"-DQ4_K={K}", f"-DGROUP_SIZE={GROUP_SIZE}",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c", str(src), "-o", str(obj),
    ]
    print(f"  Compiling q4nx_matvec.cc...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        return False
    import shutil
    shutil.copy(obj, Path(__file__).parent / "q4nx_matvec.o")
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
        "--dynamic-objFifos",
        "--aie-generate-xclbin", f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print(f"  Compiling MLIR → xclbin + insts.bin...")
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def run_on_npu():
    groups_per_row = K // GROUP_SIZE
    chunk_bytes = M * groups_per_row * 2 + M * groups_per_row * 2 + M * K // 2

    print(f"Configuration: M={M}, K={K}, group_size={GROUP_SIZE}")
    print(f"Q4NX chunk: {chunk_bytes} bytes (scales={M*groups_per_row*2}B + zeros={M*groups_per_row*2}B + int4={M*K//2}B)")
    print()

    dev = aie_utils.DefaultNPURuntime.device()
    print(f"NPU device: {dev} (cols={dev.cols})")

    # Generate MLIR
    mlir_module = q4nx_gemv_pipeline(dev=dev, M=M, K=K, group_size=GROUP_SIZE)
    mlir_str = str(mlir_module)

    build_dir = Path(__file__).parent / "build"
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
    scales = np.random.uniform(0.01, 0.5, (M, groups_per_row)).astype(bfloat16)
    zeros = np.random.uniform(4, 12, (M, groups_per_row)).astype(bfloat16)
    int4_data = np.random.randint(0, 16, (M, K), dtype=np.uint8)

    # Pack Q4NX chunk
    packed = pack_q4nx_chunk(scales, zeros, int4_data, M, K, GROUP_SIZE)
    assert packed.shape[0] == chunk_bytes

    # CPU reference
    ref_output = q4nx_matvec_reference(packed, activation, M, K, GROUP_SIZE)
    print(f"  Activation[0:4]: {activation[:4]}")
    print(f"  Output[0:4] (CPU ref): {ref_output[:4]}")

    # Run on NPU
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Weights buffer: reinterpret packed bytes as bf16 for correct DMA/pointer behavior
    packed_bf16 = packed.view(np.uint16).view(bfloat16)
    wt_t = torch.from_numpy(packed_bf16.view(np.uint16).copy()).view(torch.bfloat16)
    wt_buf = XRTTensor.from_torch(wt_t)

    # Activation buffer: bf16
    act_t = torch.from_numpy(activation.view(np.uint16)).view(torch.bfloat16)
    act_buf = XRTTensor.from_torch(act_t)

    # Output buffer: bf16
    output_buf = XRTTensor((M,), dtype=bfloat16)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = output_buf.to_torch()
    npu_output = output_torch.view(dtype=torch.uint16).numpy().view(bfloat16)
    print(f"  Output[0:4] (NPU):     {npu_output[:4]}")

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

    # Q4 tolerance: quantization already introduces significant error
    rel_tol = 0.15
    abs_tol = 0.1
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    mask = abs_err >= np.maximum(abs_tol, rel_tol * norm)
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {M} elements match within tolerance (rel={rel_tol}, abs={abs_tol})")
    else:
        print(f"\n  FAIL: {num_errors}/{M} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:8]:
            print(f"    [{idx}] ref={ref_f32[idx]:.6f} npu={npu_f32[idx]:.6f} err={abs_err[idx]:.6f}")

    print()
    print("=" * 60)
    if num_errors == 0:
        print("SUCCESS: Q4NX online dequant+GEMV produces CORRECT output!")
        print(f"  Packed int4 weights consumed directly on-chip")
        print(f"  Latency: {npu_time_us:.1f} us for {M}×{K} Q4NX matmul")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
    print("=" * 60)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 05: Q4NX Online Dequant+GEMV (Single Tile)")
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
