"""Test: ONLY core1 (0,3) → memtile → shim. No core0 sends anything.

Tests whether core1's stream can pass through core0's switchbox to reach memtile.
"""

from pathlib import Path

BUF_SIZE = 32


def generate_mlir() -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    // Memtile buffer
    %mt0_out_buf = aie.buffer(%mt0) {{sym_name = "mt0_out_buf"}} : memref<{BUF_SIZE}xbf16>

    // Core1 buffer
    %core1_send = aie.buffer(%core1) {{sym_name = "core1_send"}} : memref<{BUF_SIZE}xbf16>

    // Memtile locks
    %mt0_out_empty = aie.lock(%mt0, 0) {{init = 1 : i32, sym_name = "mt0_out_empty"}}
    %mt0_out_full = aie.lock(%mt0, 1) {{init = 0 : i32, sym_name = "mt0_out_full"}}

    // Core1 locks
    %core1_send_prod = aie.lock(%core1, 0) {{init = 1 : i32, sym_name = "core1_send_prod"}}
    %core1_send_cons = aie.lock(%core1, 1) {{init = 0 : i32, sym_name = "core1_send_cons"}}

    // Flows
    aie.flow(%core1, DMA : 1, %mt0, DMA : 3)        // core1 → memtile (passes through core0 switch)
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)        // memtile → shim

    // Core1 program: just signal DMA to send
    %core1_prog = aie.core(%core1) {{
      aie.use_lock(%core1_send_cons, Release, 1)
      aie.end
    }}

    // Core1 DMA: MM2S ch1 only
    %mem_core1 = aie.mem(%core1) {{
      %0 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core1_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core1_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core1_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

    // Memtile DMA
    %memtile_dma = aie.memtile_dma(%mt0) {{
      // S2MM ch3 (odd, BD 24): receive from core1
      %0 = aie.dma_start(S2MM, 3, ^in_s2mm, ^out_mm2s_start)
    ^in_s2mm:
      aie.use_lock(%mt0_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_out_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_out_full, Release, 1)
      aie.next_bd ^in_s2mm

      // MM2S ch1 (odd, BD 25): send to shim
    ^out_mm2s_start:
      %1 = aie.dma_start(MM2S, 1, ^out_mm2s, ^end)
    ^out_mm2s:
      aie.use_lock(%mt0_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_out_buf : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 25 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt0_out_empty, Release, 1)
      aie.next_bd ^out_mm2s
    ^end:
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%output_bo: memref<{BUF_SIZE // 2}xi32>) {{
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {BUF_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}}
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
