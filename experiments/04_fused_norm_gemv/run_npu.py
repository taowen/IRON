#!/usr/bin/env python3
"""
Run Experiment 04: Fused RMSNorm → GEMV on NPU with correctness verification.

Success = output matches CPU reference within bf16 tolerance.
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

from design import fused_norm_gemv_pipeline
from reference import fused_norm_gemv


def compile_kernels(build_dir, K):
    peano_dir = Path(aie_utils.config.peano_install_dir())
    mlir_aie_dir = Path(aie_utils.config.root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    base_flags = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
    ]

    # Compile RMSNorm kernel
    rms_src = repo_root / "aie_kernels" / "aie2p" / "rms_norm.cc"
    rms_obj = build_dir / "rms_norm.o"
    cmd = base_flags + ["-c", str(rms_src), "-o", str(rms_obj)]
    print(f"  Compiling rms_norm.cc...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        return False

    # Compile GEMV kernel
    mv_src = repo_root / "aie_kernels" / "generic" / "mv.cc"
    mv_obj = build_dir / "mv.o"
    cmd = base_flags + [f"-DDIM_K={K}", "-DVEC_SIZE=64", "-c", str(mv_src), "-o", str(mv_obj)]
    print(f"  Compiling mv.cc (DIM_K={K})...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        return False

    return True


def compile_mlir(build_dir, mlir_path):
    mlir_aie_dir = Path(aie_utils.config.root_path())
    peano_dir = Path(aie_utils.config.peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    compile_cmd = [
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
    ret = os.system(" ".join(compile_cmd))
    return ret == 0, xclbin_path, insts_path


def run_on_npu():
    K = 128
    M = 64
    m_input = 4

    print(f"Configuration: K={K}, M={M}, m_input={m_input}")
    print(f"Pipeline: hidden[{K}] → RMSNorm → normed[{K}] (on-chip) → GEMV → output[{M}]")
    print()

    # Use runtime device
    dev = aie_utils.DefaultNPURuntime.device()
    print(f"NPU device: {dev} (cols={dev.cols})")

    # Generate MLIR
    mlir_module = fused_norm_gemv_pipeline(dev=dev, K=K, M=M, m_input=m_input)
    mlir_str = str(mlir_module)

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_str)

    # Compile kernels
    print("Compiling kernels...")
    if not compile_kernels(build_dir, K):
        print("FAILED: Kernel compilation error")
        return False

    # Compile MLIR
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    print("\nRunning on NPU...")

    # Prepare test data
    np.random.seed(42)
    hidden = np.random.randn(K).astype(bfloat16)
    W = np.random.randn(M, K).astype(bfloat16)

    # CPU reference
    ref_output, ref_normed = fused_norm_gemv(hidden, W)
    print(f"  Hidden[0:4]: {hidden[:4]}")
    print(f"  Normed[0:4] (CPU ref): {ref_normed[:4]}")
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

    hidden_t = torch.from_numpy(hidden.view(np.uint16)).view(torch.bfloat16)
    W_flat = W.reshape(-1)
    weights_t = torch.from_numpy(W_flat.view(np.uint16)).view(torch.bfloat16)

    hidden_buf = XRTTensor.from_torch(hidden_t)
    weights_buf = XRTTensor.from_torch(weights_t)
    output_buf = XRTTensor((M,), dtype=bfloat16)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [hidden_buf, weights_buf, output_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = output_buf.to_torch()
    npu_output = output_torch.view(dtype=torch.uint16).numpy().view(bfloat16)
    print(f"  Output[0:4] (NPU):     {npu_output[:4]}")

    # Verify correctness
    print("\nVerification:")
    ref_f32 = ref_output.astype(np.float32)
    npu_f32 = npu_output.astype(np.float32)

    abs_err = np.abs(ref_f32 - npu_f32)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    # Relative error (avoid div by zero)
    ref_abs = np.abs(ref_f32)
    rel_err = abs_err / np.maximum(ref_abs, 1e-6)
    max_rel_err = np.max(rel_err)

    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Max relative error: {max_rel_err:.4f} ({max_rel_err*100:.2f}%)")

    # Fused 2-stage bf16 pipeline tolerance (cumulative rounding across RMSNorm + GEMV)
    rel_tol = 0.10
    abs_tol = 0.05
    norm = np.minimum(np.abs(ref_f32) + np.abs(npu_f32), np.finfo(np.float32).max)
    mask = abs_err >= np.maximum(abs_tol, rel_tol * norm)
    num_errors = np.sum(mask)

    if num_errors == 0:
        print(f"\n  PASS: All {M} elements match within tolerance (rel={rel_tol}, abs={abs_tol})")
    else:
        print(f"\n  FAIL: {num_errors}/{M} elements exceed tolerance")
        error_indices = np.where(mask)[0]
        for idx in error_indices[:5]:
            print(f"    [{idx}] ref={ref_f32[idx]:.6f} npu={npu_f32[idx]:.6f} err={abs_err[idx]:.6f}")

    print()
    print("=" * 60)
    if num_errors == 0:
        print("SUCCESS: Fused RMSNorm → GEMV produces CORRECT output on NPU!")
        print(f"  Intermediate 'normed' stayed on-chip (tile-to-tile FIFO)")
        print(f"  Latency: {npu_time_us:.1f} us")
    else:
        print("PARTIAL: Pipeline ran but output has errors.")
        print("Check kernel linkage and data flow.")
    print("=" * 60)

    return num_errors == 0


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 04: Fused RMSNorm → GEMV (Numerically Verified)")
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
