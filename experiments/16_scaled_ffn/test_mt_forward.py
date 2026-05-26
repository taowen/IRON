#!/usr/bin/env python3
"""Ultra-minimal test: shim → memtile → shim (passthrough).

Tests just the memtile forwarding path:
- Shim MM2S ch0 sends 128 bf16 to memtile S2MM ch0
- Memtile MM2S ch0 forwards to shim S2MM ch0
"""

import sys, os, numpy as np
os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.npukernel import NPUKernel
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

BUF_SIZE = 128  # bf16
BUF_I32 = BUF_SIZE * 2 // 4  # 64 i32

EXPERIMENT_DIR = Path(__file__).parent.resolve()


def generate_mlir():
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)

    %mt_buf = aie.buffer(%mt) {{sym_name = "mt_buf"}} : memref<{BUF_SIZE}xbf16>
    %mt_empty = aie.lock(%mt, 0) {{init = 1 : i32, sym_name = "mt_empty"}}
    %mt_full  = aie.lock(%mt, 1) {{init = 0 : i32, sym_name = "mt_full"}}

    // Flows
    aie.flow(%shim, DMA : 0, %mt, DMA : 0)   // shim → memtile (input)
    aie.flow(%mt, DMA : 1, %shim, DMA : 0)   // memtile → shim (output)

    // Memtile DMA
    %mtdma = aie.memtile_dma(%mt) {{
      // S2MM ch0 (even, BD 0): receive from shim
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt_full, Release, 1)
      aie.next_bd ^recv

      // MM2S ch1 (odd, BD 25): forward to shim
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send, ^end)
    ^send:
      aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 25 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%in_bo: memref<{BUF_I32}xi32>, %out_bo: memref<{BUF_I32}xi32>) {{
      // Send input: shim MM2S ch0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {BUF_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive output: shim S2MM ch0
      aiex.npu.writebd {{bd_id = 3 : i32, buffer_length = {BUF_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = 118884 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 3 : i32, issue_token = true, repeat_count = 0 : i32}}

      // Sync
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


def main():
    mlir = generate_mlir()
    build_dir = EXPERIMENT_DIR / "build_mt_forward"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir)

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
    print("Compiling MLIR...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        print("FAILED: compilation")
        return False

    print("\nRunning on NPU...")
    dev = aie_utils.DefaultNPURuntime.device()

    input_data = np.arange(BUF_SIZE, dtype=np.float32).astype(bfloat16)
    in_i32 = np.frombuffer(input_data.tobytes(), dtype=np.int32)

    npu_kernel = NPUKernel(xclbin_path=str(xclbin_path), kernel_name="MLIR_AIE", insts_path=str(insts_path))
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    in_buf = XRTTensor.from_torch(torch.from_numpy(in_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((BUF_I32,), dtype=np.int32)

    print("  Executing...")
    try:
        result = aie_utils.DefaultNPURuntime.run(handle, [in_buf, out_buf])
        print(f"  Time: {result.npu_time/1e3:.1f} us")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    output_i32 = out_buf.to_torch().numpy()
    output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)
    print(f"  Input[0:4]: {input_data[:4]}")
    print(f"  Output[0:4]: {output[:4]}")

    max_err = np.max(np.abs(input_data.astype(np.float32) - output.astype(np.float32)))
    if max_err == 0:
        print("\nPASS: Memtile passthrough works!")
    else:
        print(f"\nFAIL: max_err = {max_err}")
    return max_err == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
