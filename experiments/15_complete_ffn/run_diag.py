#!/usr/bin/env python3
"""Diagnostic run: verify packet-switched gather doesn't deadlock in full 4-tile design.

The core program is in DIAGNOSTIC mode:
- Sends intermediate (uninitialized data) to memtile via packet-switch
- Waits for gathered intermediate from memtile
- Sends output (uninitialized data) to shim

Success = no timeout/deadlock. Output values don't matter.
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
    generate_mlir, TOTAL_WT_I32, ACT_I32, OUT_TOTAL_I32,
    NUM_COLS, ROWS_PER_COL, M_PER_TILE,
)

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
    print("=== Diagnostic: packet-switched gather in full 4-tile design ===\n")

    print("Generating MLIR...")
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_diag2"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir_text)

    print("Compiling MLIR...")
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: MLIR compilation error")
        return False

    print("\nRunning on NPU...")
    from aie.utils.npukernel import NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Weights: zeros (not used in diagnostic)
    wt_buf = XRTTensor((TOTAL_WT_I32,), dtype=np.int32)
    # Activation: zeros
    act_buf = XRTTensor((ACT_I32,), dtype=np.int32)
    # Output buffer
    out_buf = XRTTensor((OUT_TOTAL_I32,), dtype=np.int32)

    print("  Executing (timeout means deadlock)...")
    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  Time: {npu_time_us:.1f} us")

    output_torch = out_buf.to_torch()
    npu_output = np.frombuffer(output_torch.numpy().tobytes(), dtype=bfloat16)
    print(f"  Output[0:8]: {npu_output[:8]}")
    print(f"  Output[32:40]: {npu_output[32:40]}")
    print(f"  Output[64:72]: {npu_output[64:72]}")
    print(f"  Output[96:104]: {npu_output[96:104]}")

    print("\nPASS: No deadlock! Packet-switched gather works in full 4-tile design.")
    print("  Next step: restore full core program with actual compute.")
    return True


if __name__ == "__main__":
    try:
        success = main()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    sys.exit(0 if success else 1)
