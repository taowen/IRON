"""Minimal test: core→memtile data flow.

Data path:
  shim MM2S ch0 → memtile S2MM ch0 → memtile MM2S ch0 → core S2MM ch0 (activation)
  core MM2S ch1 → memtile S2MM ch3 → memtile MM2S ch1 → shim S2MM ch0 (output)

Core program:
  1. Wait for activation (32 bf16)
  2. Copy activation to output buffer (kernel)
  3. Signal DMA to send output via MM2S ch1 → memtile

This tests whether core→memtile stream flow actually works.
"""

from pathlib import Path

BUF_SIZE = 32  # bf16 elements


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    // Tiles
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)

    // Memtile buffers
    %mt0_in_buf = aie.buffer(%mt0) {{sym_name = "mt0_in_buf"}} : memref<{BUF_SIZE}xbf16>
    %mt0_out_buf = aie.buffer(%mt0) {{sym_name = "mt0_out_buf"}} : memref<{BUF_SIZE}xbf16>

    // Core buffers
    %core0_act = aie.buffer(%core0) {{sym_name = "core0_act"}} : memref<{BUF_SIZE}xbf16>
    %core0_send = aie.buffer(%core0) {{sym_name = "core0_send"}} : memref<{BUF_SIZE}xbf16>

    // Memtile locks
    %mt0_in_empty = aie.lock(%mt0, 0) {{init = 1 : i32, sym_name = "mt0_in_empty"}}
    %mt0_in_full = aie.lock(%mt0, 1) {{init = 0 : i32, sym_name = "mt0_in_full"}}
    %mt0_out_empty = aie.lock(%mt0, 2) {{init = 1 : i32, sym_name = "mt0_out_empty"}}
    %mt0_out_full = aie.lock(%mt0, 3) {{init = 0 : i32, sym_name = "mt0_out_full"}}

    // Core locks
    %core0_act_empty = aie.lock(%core0, 0) {{init = 1 : i32, sym_name = "core0_act_empty"}}
    %core0_act_full = aie.lock(%core0, 1) {{init = 0 : i32, sym_name = "core0_act_full"}}
    %core0_send_prod = aie.lock(%core0, 2) {{init = 1 : i32, sym_name = "core0_send_prod"}}
    %core0_send_cons = aie.lock(%core0, 3) {{init = 0 : i32, sym_name = "core0_send_cons"}}

    // Flows
    aie.flow(%shim0, DMA : 0, %mt0, DMA : 0)       // input: shim → memtile
    aie.flow(%mt0, DMA : 0, %core0, DMA : 0)        // input: memtile → core
    aie.flow(%core0, DMA : 1, %mt0, DMA : 3)        // output: core → memtile (THE FLOW UNDER TEST)
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)        // output: memtile → shim

    // Kernel: simple copy
    func.func private @copy_bf16(memref<{BUF_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/copy_kernel.o"}}

    // Core program
    %core_prog = aie.core(%core0) {{
      %c0 = arith.constant 0 : index
      %n = arith.constant {BUF_SIZE} : i32

      // Wait for activation
      aie.use_lock(%core0_act_full, AcquireGreaterEqual, 1)

      // Copy act to send buffer
      func.call @copy_bf16(%core0_act, %core0_send, %n) : (memref<{BUF_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) -> ()

      // Release activation buffer
      aie.use_lock(%core0_act_empty, Release, 1)

      // Signal DMA to send
      aie.use_lock(%core0_send_cons, Release, 1)

      aie.end
    }}

    // Core DMA
    %mem_core0 = aie.mem(%core0) {{
      // S2MM ch0: receive activation
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%core0_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_act : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core0_act_full, Release, 1)
      aie.next_bd ^act_bd

      // MM2S ch1: send to memtile
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core0_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%core0_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

    // Memtile DMA
    %memtile_dma = aie.memtile_dma(%mt0) {{
      // S2MM ch0 (even, BD 0): receive input from shim
      %0 = aie.dma_start(S2MM, 0, ^in_s2mm, ^out_s2mm_start)
    ^in_s2mm:
      aie.use_lock(%mt0_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_in_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt0_in_full, Release, 1)
      aie.next_bd ^in_s2mm

      // S2MM ch3 (odd, BD 24): receive from core MM2S ch1
    ^out_s2mm_start:
      %1 = aie.dma_start(S2MM, 3, ^out_s2mm, ^in_mm2s_start)
    ^out_s2mm:
      aie.use_lock(%mt0_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_out_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_out_full, Release, 1)
      aie.next_bd ^out_s2mm

      // MM2S ch0 (even, BD 2): send input to core
    ^in_mm2s_start:
      %2 = aie.dma_start(MM2S, 0, ^in_mm2s, ^out_mm2s_start)
    ^in_mm2s:
      aie.use_lock(%mt0_in_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_in_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt0_in_empty, Release, 1)
      aie.next_bd ^in_mm2s

      // MM2S ch1 (odd, BD 25): send output to shim
    ^out_mm2s_start:
      %3 = aie.dma_start(MM2S, 1, ^out_mm2s, ^end)
    ^out_mm2s:
      aie.use_lock(%mt0_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_out_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 25 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt0_out_empty, Release, 1)
      aie.next_bd ^out_mm2s
    ^end:
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%input_bo: memref<{BUF_SIZE // 2}xi32>, %output_bo: memref<{BUF_SIZE // 2}xi32>) {{
      // Send input: shim MM2S ch0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {BUF_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive output: shim S2MM ch0
      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {BUF_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 1 * 0x20} : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}}

      // Sync on output
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
