"""Test: 4-tile (2 cols × 2 rows) packet-switched gather."""

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

ACT_SIZE = 32
INTER_SIZE = 32
GATHERED_SIZE = 64
NUM_COLS = 2
ROWS_PER_COL = 2
OUT_SIZE_I32 = NUM_COLS * ROWS_PER_COL * INTER_SIZE // 2

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
    print("=== 4-Tile (2×2) Packet-Switched Gather Test ===\n")

    from generate_4tile_gather import generate_mlir
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_4tile_gather"
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

    # Activation: [1, 2, ..., 32]
    activation = np.arange(1, ACT_SIZE + 1, dtype=np.float32).astype(bfloat16)
    act_i32 = np.frombuffer(activation.tobytes(), dtype=np.int32)
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))

    out_buf = XRTTensor((OUT_SIZE_I32,), dtype=np.int32)

    print("  Executing...")
    result = aie_utils.DefaultNPURuntime.run(handle, [act_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  Time: {npu_time_us:.1f} us")

    output_torch = out_buf.to_torch()
    npu_output = np.frombuffer(output_torch.numpy().tobytes(), dtype=bfloat16)

    # Expected: each tile outputs gathered[0:32] = activation (since both cores copy activation)
    expected = np.tile(activation, NUM_COLS * ROWS_PER_COL)

    print(f"\n  Output breakdown (each 32 bf16 = gathered[0:32]):")
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            idx = (col * ROWS_PER_COL + row) * INTER_SIZE
            print(f"    Col{col} Row{row}: {npu_output[idx:idx+4]}...")

    if np.array_equal(npu_output, expected):
        print("\nPASS: 4-tile packet-switched gather verified!")
        return True
    else:
        mismatches = np.where(npu_output != expected)[0]
        print(f"\nFAIL: {len(mismatches)} mismatches at: {mismatches[:10]}")
        print(f"  Expected first 4: {expected[:4]}")
        print(f"  Got first 4: {npu_output[:4]}")
        return False


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
