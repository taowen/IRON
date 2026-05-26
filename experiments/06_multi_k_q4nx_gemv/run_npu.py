#!/usr/bin/env python3
"""
Experiment 06: Single-Descriptor Multi-K Q4NX GEMV (32×4096, fp32 Accumulation)

Proves:
1. Single DMA descriptor covers all 16 K-chunks
2. fp32 on-tile accumulation across K iterations
3. Dequant correctness with controlled test data
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
from reference import pack_q4nx_chunk, q4nx_gemv_reference


M = 32
K = 4096
K_CHUNK = 256
K_CHUNKS = K // K_CHUNK
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
        f"-DQ4_M={M}", f"-DQ4_K={K_CHUNK}", f"-DGROUP_SIZE={GROUP_SIZE}",
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


def pack_all_chunks(scales, zeros, int4_data):
    """Pack all K-chunks into contiguous BO layout."""
    groups_per_chunk = K_CHUNK // GROUP_SIZE
    chunks = []
    for i in range(K_CHUNKS):
        s_i = scales[:, i * groups_per_chunk : (i + 1) * groups_per_chunk]
        z_i = zeros[:, i * groups_per_chunk : (i + 1) * groups_per_chunk]
        d_i = int4_data[:, i * K_CHUNK : (i + 1) * K_CHUNK]
        chunks.append(pack_q4nx_chunk(s_i, z_i, d_i, M, K_CHUNK, GROUP_SIZE))
    return chunks, np.concatenate(chunks)


def run_test(handle, name, scales, zeros, int4_data, activation, expected_f32):
    """Run one test case and verify."""
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    chunks_list, packed_all = pack_all_chunks(scales, zeros, int4_data)

    # Prepare buffers
    packed_bf16 = packed_all.view(np.uint16).view(bfloat16)
    wt_t = torch.from_numpy(packed_bf16.view(np.uint16).copy()).view(torch.bfloat16)
    wt_buf = XRTTensor.from_torch(wt_t)

    act_t = torch.from_numpy(activation.view(np.uint16).copy()).view(torch.bfloat16)
    act_buf = XRTTensor.from_torch(act_t)

    output_buf = XRTTensor((M,), dtype=bfloat16)

    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, output_buf])
    npu_time_us = result.npu_time / 1e3

    output_torch = output_buf.to_torch()
    npu_output = output_torch.view(dtype=torch.uint16).numpy().view(bfloat16)
    npu_f32 = npu_output.astype(np.float32)

    abs_err = np.abs(expected_f32 - npu_f32)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    exp_mag = np.max(np.abs(expected_f32))
    rel_thr = max(exp_mag * 0.02, 1.0)

    passed = max_abs_err < rel_thr

    print(f"  [{name}] time={npu_time_us:.1f}us  max_err={max_abs_err:.2f}  threshold={rel_thr:.2f}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"    Expected[0:4]: {expected_f32[:4]}")
        print(f"    Got[0:4]:      {npu_f32[:4]}")
        for idx in np.argsort(-abs_err)[:4]:
            print(f"    [{idx}] exp={expected_f32[idx]:.4f} got={npu_f32[idx]:.4f} err={abs_err[idx]:.4f}")

    return passed


def run_on_npu():
    groups_per_row = K // GROUP_SIZE
    chunk_bytes = M * (K_CHUNK // GROUP_SIZE) * 2 * 2 + M * K_CHUNK // 2
    total_wt_bytes = K_CHUNKS * chunk_bytes

    print(f"Configuration: M={M}, K={K}, K_chunk={K_CHUNK}, K_chunks={K_CHUNKS}")
    print(f"Weight BO: {total_wt_bytes} bytes ({K_CHUNKS} × {chunk_bytes}B chunks)")
    print(f"Activation: {K} bf16 = {K*2} bytes")
    print(f"Runtime ops: 1 fill(wt) + 1 fill(act) + 1 drain(out) = 3 total")
    print()

    dev = aie_utils.DefaultNPURuntime.device()
    print(f"NPU device: {dev} (cols={dev.cols})")

    # Generate MLIR
    mlir_module = q4nx_gemv_pipeline(dev=dev, M=M, K=K, K_chunk=K_CHUNK, group_size=GROUP_SIZE)
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(str(mlir_module))

    # Compile
    print("Compiling kernel...")
    if not compile_kernel(build_dir):
        print("FAILED: Kernel compilation error")
        return False

    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    # Load
    from aie.utils.npukernel import NPUKernel
    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    print("\nRunning tests on NPU...")
    all_passed = True

    # Test 1: scale=1, zero=0, int4=7, act=1.0
    # Expected: (7 - 0) * 1 * 1.0 * K = 7 * 4096 = 28672
    scales_1 = np.ones((M, groups_per_row), dtype=bfloat16)
    zeros_1 = np.zeros((M, groups_per_row), dtype=bfloat16)
    int4_1 = np.full((M, K), 7, dtype=np.uint8)
    act_1 = np.ones(K, dtype=bfloat16)
    exp_1 = np.full(M, 7.0 * K, dtype=np.float32)
    all_passed &= run_test(handle, "scale=1,zero=0,int4=7,act=1", scales_1, zeros_1, int4_1, act_1, exp_1)

    # Test 2: scale=2, zero=8, int4=10, act=1.0
    # Expected: (10 - 8) * 2 * 1.0 * K = 4 * 4096 = 16384
    # WITHOUT dequant: 10 * 1.0 * K = 40960  (distinguishes conclusively)
    scales_2 = np.full((M, groups_per_row), 2.0, dtype=bfloat16)
    zeros_2 = np.full((M, groups_per_row), 8.0, dtype=bfloat16)
    int4_2 = np.full((M, K), 10, dtype=np.uint8)
    act_2 = np.ones(K, dtype=bfloat16)
    exp_2 = np.full(M, (10.0 - 8.0) * 2.0 * K, dtype=np.float32)
    all_passed &= run_test(handle, "scale=2,zero=8,int4=10,act=1", scales_2, zeros_2, int4_2, act_2, exp_2)

    # Test 3: scale=0.5, zero=4, int4=12, act=2.0
    # Expected: (12 - 4) * 0.5 * 2.0 * K = 8 * 4096 = 32768
    # WITHOUT dequant: 12 * 2.0 * K = 98304
    scales_3 = np.full((M, groups_per_row), 0.5, dtype=bfloat16)
    zeros_3 = np.full((M, groups_per_row), 4.0, dtype=bfloat16)
    int4_3 = np.full((M, K), 12, dtype=np.uint8)
    act_3 = np.full(K, 2.0, dtype=bfloat16)
    exp_3 = np.full(M, (12.0 - 4.0) * 0.5 * 2.0 * K, dtype=np.float32)
    all_passed &= run_test(handle, "scale=0.5,zero=4,int4=12,act=2", scales_3, zeros_3, int4_3, act_3, exp_3)

    # Test 4: Random data
    np.random.seed(42)
    scales_4 = np.random.uniform(0.01, 0.5, (M, groups_per_row)).astype(bfloat16)
    zeros_4 = np.random.uniform(4, 12, (M, groups_per_row)).astype(bfloat16)
    int4_4 = np.random.randint(0, 16, (M, K), dtype=np.uint8)
    act_4 = np.random.randn(K).astype(bfloat16)
    groups_per_chunk = K_CHUNK // GROUP_SIZE
    chunks_4 = []
    for i in range(K_CHUNKS):
        s_i = scales_4[:, i * groups_per_chunk : (i + 1) * groups_per_chunk]
        z_i = zeros_4[:, i * groups_per_chunk : (i + 1) * groups_per_chunk]
        d_i = int4_4[:, i * K_CHUNK : (i + 1) * K_CHUNK]
        chunks_4.append(pack_q4nx_chunk(s_i, z_i, d_i, M, K_CHUNK, GROUP_SIZE))
    exp_4_f32, _ = q4nx_gemv_reference(chunks_4, act_4, M, K, K_CHUNK, GROUP_SIZE)
    all_passed &= run_test(handle, "random", scales_4, zeros_4, int4_4, act_4, exp_4_f32)

    print()
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All tests pass.")
        print(f"  Q4NX dequant+GEMV with fp32 accumulation over K={K}")
        print(f"  Single descriptor per stream, {K_CHUNKS} chunks core-looped")
    else:
        print("FAIL: Some tests failed.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 06: Single-Descriptor Multi-K Q4NX GEMV (32×4096)")
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
