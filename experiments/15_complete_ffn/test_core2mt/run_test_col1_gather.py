"""Test: Only column 1 (tiles 1,0-3) with sequential act→gathered BD."""

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
EXPERIMENT_DIR = Path(__file__).parent


def generate_mlir():
    """Generate 1-column design using column 1 (not column 0)."""
    experiment_dir = EXPERIMENT_DIR.resolve()
    col = 1  # Use column 1 specifically

    return f"""module {{
  aie.device(npu2) {{
    %shim{col} = aie.tile({col}, 0)
    %mt{col} = aie.tile({col}, 1)
    %c{col}r0 = aie.tile({col}, 2)
    %c{col}r1 = aie.tile({col}, 3)

    %mt{col}_act_buf = aie.buffer(%mt{col}) {{sym_name = "mt{col}_act_buf"}} : memref<{ACT_SIZE}xbf16>
    %mt{col}_gathered_buf = aie.buffer(%mt{col}) {{sym_name = "mt{col}_gathered_buf"}} : memref<{GATHERED_SIZE}xbf16>

    %c{col}r0_act = aie.buffer(%c{col}r0) {{sym_name = "c{col}r0_act"}} : memref<{ACT_SIZE}xbf16>
    %c{col}r0_send = aie.buffer(%c{col}r0) {{sym_name = "c{col}r0_send"}} : memref<{INTER_SIZE}xbf16>
    %c{col}r0_gathered = aie.buffer(%c{col}r0) {{sym_name = "c{col}r0_gathered"}} : memref<{GATHERED_SIZE}xbf16>
    %c{col}r0_out = aie.buffer(%c{col}r0) {{sym_name = "c{col}r0_out"}} : memref<{INTER_SIZE}xbf16>
    %c{col}r1_act = aie.buffer(%c{col}r1) {{sym_name = "c{col}r1_act"}} : memref<{ACT_SIZE}xbf16>
    %c{col}r1_send = aie.buffer(%c{col}r1) {{sym_name = "c{col}r1_send"}} : memref<{INTER_SIZE}xbf16>
    %c{col}r1_gathered = aie.buffer(%c{col}r1) {{sym_name = "c{col}r1_gathered"}} : memref<{GATHERED_SIZE}xbf16>
    %c{col}r1_out = aie.buffer(%c{col}r1) {{sym_name = "c{col}r1_out"}} : memref<{INTER_SIZE}xbf16>

    %mt{col}_act_empty = aie.lock(%mt{col}, 0) {{init = 2 : i32, sym_name = "mt{col}_act_empty"}}
    %mt{col}_act_full = aie.lock(%mt{col}, 1) {{init = 0 : i32, sym_name = "mt{col}_act_full"}}
    %mt{col}_gathered_empty = aie.lock(%mt{col}, 2) {{init = 2 : i32, sym_name = "mt{col}_gathered_empty"}}
    %mt{col}_gathered_full = aie.lock(%mt{col}, 3) {{init = 0 : i32, sym_name = "mt{col}_gathered_full"}}

    %c{col}r0_act_empty = aie.lock(%c{col}r0, 0) {{init = 1 : i32, sym_name = "c{col}r0_act_empty"}}
    %c{col}r0_act_full = aie.lock(%c{col}r0, 1) {{init = 0 : i32, sym_name = "c{col}r0_act_full"}}
    %c{col}r0_send_prod = aie.lock(%c{col}r0, 2) {{init = 1 : i32, sym_name = "c{col}r0_send_prod"}}
    %c{col}r0_send_cons = aie.lock(%c{col}r0, 3) {{init = 0 : i32, sym_name = "c{col}r0_send_cons"}}
    %c{col}r0_gathered_empty = aie.lock(%c{col}r0, 4) {{init = 1 : i32, sym_name = "c{col}r0_gathered_empty"}}
    %c{col}r0_gathered_full = aie.lock(%c{col}r0, 5) {{init = 0 : i32, sym_name = "c{col}r0_gathered_full"}}
    %c{col}r0_out_prod = aie.lock(%c{col}r0, 6) {{init = 1 : i32, sym_name = "c{col}r0_out_prod"}}
    %c{col}r0_out_cons = aie.lock(%c{col}r0, 7) {{init = 0 : i32, sym_name = "c{col}r0_out_cons"}}

    %c{col}r1_act_empty = aie.lock(%c{col}r1, 0) {{init = 1 : i32, sym_name = "c{col}r1_act_empty"}}
    %c{col}r1_act_full = aie.lock(%c{col}r1, 1) {{init = 0 : i32, sym_name = "c{col}r1_act_full"}}
    %c{col}r1_send_prod = aie.lock(%c{col}r1, 2) {{init = 1 : i32, sym_name = "c{col}r1_send_prod"}}
    %c{col}r1_send_cons = aie.lock(%c{col}r1, 3) {{init = 0 : i32, sym_name = "c{col}r1_send_cons"}}
    %c{col}r1_gathered_empty = aie.lock(%c{col}r1, 4) {{init = 1 : i32, sym_name = "c{col}r1_gathered_empty"}}
    %c{col}r1_gathered_full = aie.lock(%c{col}r1, 5) {{init = 0 : i32, sym_name = "c{col}r1_gathered_full"}}
    %c{col}r1_out_prod = aie.lock(%c{col}r1, 6) {{init = 1 : i32, sym_name = "c{col}r1_out_prod"}}
    %c{col}r1_out_cons = aie.lock(%c{col}r1, 7) {{init = 0 : i32, sym_name = "c{col}r1_out_cons"}}

    aie.flow(%shim{col}, DMA : 0, %mt{col}, DMA : 0)
    aie.flow(%mt{col}, DMA : 0, %c{col}r0, DMA : 0)
    aie.flow(%mt{col}, DMA : 3, %c{col}r1, DMA : 0)
    aie.packet_flow(0) {{
      aie.packet_source<%c{col}r0, DMA : 1>
      aie.packet_dest<%mt{col}, DMA : 3>
    }}
    aie.packet_flow(1) {{
      aie.packet_source<%c{col}r1, DMA : 1>
      aie.packet_dest<%mt{col}, DMA : 3>
    }}
    aie.flow(%c{col}r0, DMA : 0, %shim{col}, DMA : 0)
    aie.flow(%c{col}r1, DMA : 0, %shim{col}, DMA : 1)

    func.func private @copy_bf16(memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/copy_kernel.o"}}

    %core_c{col}r0 = aie.core(%c{col}r0) {{
      %n = arith.constant {INTER_SIZE} : i32
      aie.use_lock(%c{col}r0_act_full, AcquireGreaterEqual, 1)
      func.call @copy_bf16(%c{col}r0_act, %c{col}r0_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      func.call @copy_bf16(%c{col}r0_act, %c{col}r0_out, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%c{col}r0_act_empty, Release, 1)
      aie.use_lock(%c{col}r0_send_cons, Release, 1)
      aie.use_lock(%c{col}r0_gathered_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{col}r0_gathered_empty, Release, 1)
      aie.use_lock(%c{col}r0_out_cons, Release, 1)
      aie.end
    }}
    %core_c{col}r1 = aie.core(%c{col}r1) {{
      %n = arith.constant {INTER_SIZE} : i32
      aie.use_lock(%c{col}r1_act_full, AcquireGreaterEqual, 1)
      func.call @copy_bf16(%c{col}r1_act, %c{col}r1_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      func.call @copy_bf16(%c{col}r1_act, %c{col}r1_out, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%c{col}r1_act_empty, Release, 1)
      aie.use_lock(%c{col}r1_send_cons, Release, 1)
      aie.use_lock(%c{col}r1_gathered_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{col}r1_gathered_empty, Release, 1)
      aie.use_lock(%c{col}r1_out_cons, Release, 1)
      aie.end
    }}

    %mem_c{col}r0 = aie.mem(%c{col}r0) {{
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%c{col}r0_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r0_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%c{col}r0_act_full, Release, 1)
      aie.next_bd ^gathered_bd
    ^gathered_bd:
      aie.use_lock(%c{col}r0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r0_gathered : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%c{col}r0_gathered_full, Release, 1)
      aie.next_bd ^act_bd
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^inter_start)
    ^out_bd:
      aie.use_lock(%c{col}r0_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r0_out : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%c{col}r0_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^inter_start:
      %2 = aie.dma_start(MM2S, 1, ^inter_bd, ^end)
    ^inter_bd:
      aie.use_lock(%c{col}r0_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r0_send : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}}
      aie.use_lock(%c{col}r0_send_prod, Release, 1)
      aie.next_bd ^inter_bd
    ^end:
      aie.end
    }}

    %mem_c{col}r1 = aie.mem(%c{col}r1) {{
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%c{col}r1_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r1_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%c{col}r1_act_full, Release, 1)
      aie.next_bd ^gathered_bd
    ^gathered_bd:
      aie.use_lock(%c{col}r1_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r1_gathered : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%c{col}r1_gathered_full, Release, 1)
      aie.next_bd ^act_bd
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^inter_start)
    ^out_bd:
      aie.use_lock(%c{col}r1_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r1_out : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%c{col}r1_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^inter_start:
      %2 = aie.dma_start(MM2S, 1, ^inter_bd, ^end)
    ^inter_bd:
      aie.use_lock(%c{col}r1_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{col}r1_send : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}}
      aie.use_lock(%c{col}r1_send_prod, Release, 1)
      aie.next_bd ^inter_bd
    ^end:
      aie.end
    }}

    %memtile_dma_mt{col} = aie.memtile_dma(%mt{col}) {{
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^inter_s2mm_start)
    ^act_s2mm:
      aie.use_lock(%mt{col}_act_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt{col}_act_full, Release, 2)
      aie.next_bd ^act_s2mm
    ^inter_s2mm_start:
      %1 = aie.dma_start(S2MM, 3, ^inter_bd0, ^act_mm2s_r0_start)
    ^inter_bd0:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_bd1
    ^inter_bd1:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_SIZE}xbf16>, {INTER_SIZE}, {INTER_SIZE}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_bd0
    ^act_mm2s_r0_start:
      %2 = aie.dma_start(MM2S, 0, ^act_mm2s_r0, ^act_mm2s_r1_start)
    ^act_mm2s_r0:
      aie.use_lock(%mt{col}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt{col}_act_empty, Release, 1)
      aie.next_bd ^gathered_mm2s_r0
    ^gathered_mm2s_r0:
      aie.use_lock(%mt{col}_gathered_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt{col}_gathered_empty, Release, 1)
      aie.next_bd ^act_mm2s_r0
    ^act_mm2s_r1_start:
      %3 = aie.dma_start(MM2S, 3, ^act_mm2s_r1, ^end)
    ^act_mm2s_r1:
      aie.use_lock(%mt{col}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt{col}_act_empty, Release, 1)
      aie.next_bd ^gathered_mm2s_r1
    ^gathered_mm2s_r1:
      aie.use_lock(%mt{col}_gathered_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt{col}_gathered_empty, Release, 1)
      aie.next_bd ^act_mm2s_r1
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%act_bo: memref<{ACT_SIZE // 2}xi32>, %out_bo: memref<{INTER_SIZE}xi32>) {{
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue({col}, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {INTER_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004 + 0x20} : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue({col}, 0, S2MM : 0) {{bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}}

      aiex.npu.writebd {{bd_id = 2 : i32, buffer_length = {INTER_SIZE // 2} : i32, buffer_offset = {INTER_SIZE * 2} : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004 + 2 * 0x20} : ui32, arg_idx = 1 : i32, arg_plus = {INTER_SIZE * 2} : i32}}
      aiex.npu.push_queue({col}, 0, S2MM : 1) {{bd_id = 2 : i32, issue_token = true, repeat_count = 0 : i32}}

      aiex.npu.sync {{channel = 0 : i32, column = {col} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
      aiex.npu.sync {{channel = 1 : i32, column = {col} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


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
    print("=== Column 1 Only: Sequential BD Gather ===\n")
    mlir_text = generate_mlir()

    build_dir = EXPERIMENT_DIR / "build_col1_gather"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir_text)

    print("Compiling...")
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("FAILED: compilation error")
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

    activation = np.arange(1, ACT_SIZE + 1, dtype=np.float32).astype(bfloat16)
    act_i32 = np.frombuffer(activation.tobytes(), dtype=np.int32)
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((INTER_SIZE,), dtype=np.int32)  # 2 rows × 32 bf16 = 32 i32

    print("  Executing...")
    result = aie_utils.DefaultNPURuntime.run(handle, [act_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    print(f"  Time: {npu_time_us:.1f} us")

    output_torch = out_buf.to_torch()
    npu_output = np.frombuffer(output_torch.numpy().tobytes(), dtype=bfloat16)
    expected = np.tile(activation, 2)

    print(f"  Output[0:4]: {npu_output[:4]}")
    print(f"  Output[32:36]: {npu_output[32:36]}")

    if np.array_equal(npu_output, expected):
        print("\nPASS: Column 1 sequential BD gather works!")
        return True
    else:
        mismatches = np.where(npu_output != expected)[0]
        print(f"\nFAIL: {len(mismatches)} mismatches")
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
