"""Minimal test: core→memtile data flow.

Sends 32 bf16 values through: shim → memtile → core → memtile → shim
If output matches input, core→memtile flow works.
"""

import sys
import os
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path

BUF_SIZE = 32
EXPERIMENT_DIR = Path(__file__).parent


def compile_kernel():
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    src = EXPERIMENT_DIR / "copy_kernel.cc"
    obj = EXPERIMENT_DIR / "copy_kernel.o"

    cmd = [
        str(clang),
        "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c", str(src), "-o", str(obj),
    ]
    print(f"  Compiling copy_kernel.cc...")
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


def main():
    # Compile kernel
    print("Compiling kernel...")
    if not compile_kernel():
        print("FAILED: Kernel compilation error")
        return False

    # Generate MLIR
    print("Generating MLIR...")
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from generate import generate_mlir
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir_text)

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
    input_data = np.random.randn(BUF_SIZE).astype(bfloat16)

    # Load and run
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Input buffer (32 bf16 = 16 i32)
    input_i32 = np.frombuffer(input_data.tobytes(), dtype=np.int32)
    input_buf = XRTTensor.from_torch(torch.from_numpy(input_i32.copy()).to(torch.int32))

    # Output buffer (32 bf16 = 16 i32)
    out_buf = XRTTensor((BUF_SIZE // 2,), dtype=np.int32)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)

    # Verify
    print(f"\n  Input[0:8]:  {input_data[:8]}")
    print(f"  Output[0:8]: {npu_output[:8]}")

    matches = np.array_equal(input_data, npu_output)
    if matches:
        print("\nPASS: core→memtile data flow works!")
        return True
    else:
        mismatches = np.where(input_data != npu_output)[0]
        print(f"\nFAIL: {len(mismatches)} mismatches out of {BUF_SIZE}")
        if np.all(npu_output == 0):
            print("  Output is all zeros - data never arrived")
        else:
            print(f"  First few outputs: {npu_output[:8]}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
