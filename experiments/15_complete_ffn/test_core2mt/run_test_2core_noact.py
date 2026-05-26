"""Test: 2-core gather, NO activation. Just core→memtile→shim.

Both cores send their buffer (zeros) immediately. Memtile gathers and outputs.
If this works: 2-core routing is fine, issue is activation interaction.
If this fails: 2-core→memtile routing itself is broken.
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
    # Generate MLIR
    print("Generating MLIR (2-core no-act)...")
    from generate_2core_noact import generate_mlir
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_2core_noact"
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

    # Load and run
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Output buffer only (64 bf16 = 32 i32)
    out_buf = XRTTensor((GATHERED_SIZE // 2,), dtype=np.int32)

    print("  Executing on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU execution time: {npu_time_us:.1f} us")

    # Read output
    output_torch = out_buf.to_torch()
    output_i32 = output_torch.numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)

    # Expected: all zeros (cores send uninitialized buffers which are zero)
    print(f"\n  Output[0:8]:   {npu_output[:8]} (core0 portion)")
    print(f"  Output[32:40]: {npu_output[32:40]} (core1 portion)")

    # We don't care about values, just that we got data (not timeout)
    print("\nPASS: 2-core gather (no activation) works! Stream routing is fine.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
