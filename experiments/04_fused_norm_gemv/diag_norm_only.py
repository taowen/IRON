#!/usr/bin/env python3
"""Diagnostic: Run ONLY RMSNorm on NPU, output normed to DDR directly."""

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

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

from reference import rms_norm


def norm_only_design(dev, K):
    dtype = np.dtype[bfloat16]
    L1_ty = np.ndarray[(K,), dtype]
    L3_ty = np.ndarray[(K,), dtype]

    rms_norm_kernel = Kernel(
        "rms_norm_bf16_vector",
        "rms_norm.o",
        [L1_ty, L1_ty, np.int32],
    )

    in_fifo = ObjectFifo(L1_ty, name="in", depth=2)
    out_fifo = ObjectFifo(L1_ty, name="out", depth=2)

    def worker_body(in_f, out_f, kernel):
        for _ in range_(1):
            x = in_f.acquire(1)
            y = out_f.acquire(1)
            kernel(x, y, K)
            out_f.release(1)
            in_f.release(1)

    worker = Worker(worker_body, [in_fifo.cons(), out_fifo.prod(), rms_norm_kernel])

    in_tap = TensorAccessPattern(
        tensor_dims=L3_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    out_tap = TensorAccessPattern(
        tensor_dims=L3_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (in_buf, out_buf):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(in_fifo.prod(), in_buf, in_tap, task_group=tg)
        rt.drain(out_fifo.cons(), out_buf, out_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def run():
    K = 128

    dev = aie_utils.DefaultNPURuntime.device()
    print(f"Device: {dev} (cols={dev.cols})")

    mlir_module = norm_only_design(dev, K)
    mlir_str = str(mlir_module)

    build_dir = Path(__file__).parent / "build_diag"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_str)

    # Compile RMSNorm kernel
    peano_dir = Path(aie_utils.config.peano_install_dir())
    mlir_aie_dir = Path(aie_utils.config.root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"

    rms_src = repo_root / "aie_kernels" / "aie2p" / "rms_norm.cc"
    rms_obj = build_dir / "rms_norm.o"
    cmd = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}", f"-I{runtime_lib_include}",
        "-c", str(rms_src), "-o", str(rms_obj),
    ]
    print("Compiling rms_norm.cc...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        return False

    # Compile MLIR
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
    print("Compiling MLIR...")
    ret = os.system(" ".join(compile_cmd))
    if ret != 0:
        return False

    # Run on NPU
    print("\nRunning on NPU...")
    np.random.seed(42)
    hidden = np.random.randn(K).astype(bfloat16)
    ref_normed = rms_norm(hidden)

    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    hidden_t = torch.from_numpy(hidden.view(np.uint16)).view(torch.bfloat16)
    hidden_buf = XRTTensor.from_torch(hidden_t)
    output_buf = XRTTensor((K,), dtype=bfloat16)

    result = aie_utils.DefaultNPURuntime.run(handle, [hidden_buf, output_buf])
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    output_torch = output_buf.to_torch()
    npu_normed = output_torch.view(dtype=torch.uint16).numpy().view(bfloat16)

    print(f"  Input[0:8]:      {hidden[:8]}")
    print(f"  CPU normed[0:8]: {ref_normed[:8]}")
    print(f"  NPU normed[0:8]: {npu_normed[:8]}")

    ref_f32 = ref_normed.astype(np.float32)
    npu_f32 = npu_normed.astype(np.float32)
    abs_err = np.abs(ref_f32 - npu_f32)
    max_err = np.max(abs_err)
    print(f"  Max abs error: {max_err:.6f}")

    if max_err < 0.01:
        print("\n  PASS: RMSNorm alone works correctly on NPU")
        return True
    else:
        print(f"\n  FAIL: RMSNorm produces wrong output (max_err={max_err})")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Diagnostic: RMSNorm only (single tile, DDR→tile→DDR)")
    print("=" * 60)
    try:
        success = run()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()
    sys.exit(0 if success else 1)
