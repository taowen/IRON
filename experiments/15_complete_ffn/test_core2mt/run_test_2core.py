"""Test: 2-core gather pattern (core0 + core1 → memtile → shim).

Sends 32 bf16 activation to both cores, each copies and sends back to memtile.
Memtile gathers (core0 at offset 0, core1 at offset 32) and outputs 64 bf16.
Expected output: [input, input] (same 32 values repeated).
"""

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
GATHERED_SIZE = 64
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
    print(f"  Compiling MLIR -> xclbin + insts.bin...")
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def main():
    # Compile kernel (reuse from single-core test)
    obj = EXPERIMENT_DIR / "copy_kernel.o"
    if not obj.exists():
        print("ERROR: copy_kernel.o not found. Run run_test.py first.")
        return False

    # Generate MLIR
    print("Generating MLIR (2-core gather)...")
    from generate_2core import generate_mlir
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_2core"
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

    # Output buffer (64 bf16 = 32 i32)
    out_buf = XRTTensor((GATHERED_SIZE // 2,), dtype=np.int32)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)

    # Expected: [input, input] — core0 writes offset 0, core1 writes offset 32
    expected = np.concatenate([input_data, input_data])

    # Verify
    print(f"\n  Input[0:4]:       {input_data[:4]}")
    print(f"  Output[0:4]:      {npu_output[:4]} (core0 @ offset 0)")
    print(f"  Output[32:36]:    {npu_output[32:36]} (core1 @ offset 32)")
    print(f"  Expected[0:4]:    {expected[:4]}")
    print(f"  Expected[32:36]:  {expected[32:36]}")

    matches = np.array_equal(expected, npu_output)
    if matches:
        print("\nPASS: 2-core gather pattern works!")
        return True
    else:
        mismatches = np.where(expected != npu_output)[0]
        print(f"\nFAIL: {len(mismatches)} mismatches out of {GATHERED_SIZE}")
        if np.all(npu_output == 0):
            print("  Output is all zeros - data never arrived")
        elif np.all(npu_output[:32] == 0):
            print("  First half (core0) is zeros - core0→memtile failed")
        elif np.all(npu_output[32:] == 0):
            print("  Second half (core1) is zeros - core1→memtile failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
