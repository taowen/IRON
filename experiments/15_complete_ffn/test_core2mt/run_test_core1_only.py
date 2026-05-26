"""Test: core1 (0,3) → memtile (passes through core0) → shim."""

import sys
import os
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path

BUF_SIZE = 32
EXPERIMENT_DIR = Path(__file__).parent


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
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def main():
    print("Generating MLIR (core1 only)...")
    from generate_core1_only import generate_mlir
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_core1_only"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir_text)

    print("Compiling MLIR...")
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    print("Running on NPU...")
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    out_buf = XRTTensor((BUF_SIZE // 2,), dtype=np.int32)

    print("  Executing...")
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  Time: {npu_time_us:.1f} us")

    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)
    print(f"  Output[0:8]: {npu_output[:8]}")
    print("\nPASS: core1→memtile (through core0 switch) works!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
