"""Test: 2 cores → memtile, but with gathered_full pre-initialized to 2.

If memtile sends output immediately (gathered_full=2), this bypasses the
2-producer lock wait. If it STILL times out, the issue is something else
(perhaps stream routing or DMA configuration). If it passes, the issue is
specifically the AcquireGE 2 lock protocol.
"""

from pathlib import Path

BUF_SIZE = 32
GATHERED_SIZE = 64


def generate_mlir() -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    %mt0_gathered_buf = aie.buffer(%mt0) {{sym_name = "mt0_gathered_buf"}} : memref<{GATHERED_SIZE}xbf16>
    %core0_send = aie.buffer(%core0) {{sym_name = "core0_send"}} : memref<{BUF_SIZE}xbf16>
    %core1_send = aie.buffer(%core1) {{sym_name = "core1_send"}} : memref<{BUF_SIZE}xbf16>

    // PRE-INITIALIZE gathered_full=2 so memtile sends immediately
    %mt0_gathered_empty = aie.lock(%mt0, 0) {{init = 0 : i32, sym_name = "mt0_gathered_empty"}}
    %mt0_gathered_full = aie.lock(%mt0, 1) {{init = 2 : i32, sym_name = "mt0_gathered_full"}}

    %core0_send_prod = aie.lock(%core0, 0) {{init = 1 : i32, sym_name = "core0_send_prod"}}
    %core0_send_cons = aie.lock(%core0, 1) {{init = 0 : i32, sym_name = "core0_send_cons"}}
    %core1_send_prod = aie.lock(%core1, 0) {{init = 1 : i32, sym_name = "core1_send_prod"}}
    %core1_send_cons = aie.lock(%core1, 1) {{init = 0 : i32, sym_name = "core1_send_cons"}}

    // Flows (same as failing test)
    aie.flow(%core0, DMA : 1, %mt0, DMA : 3)
    aie.flow(%core1, DMA : 1, %mt0, DMA : 4)
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)

    // Core programs still signal sends (but memtile doesn't wait for them)
    %core0_prog = aie.core(%core0) {{
      aie.use_lock(%core0_send_cons, Release, 1)
      aie.end
    }}
    %core1_prog = aie.core(%core1) {{
      aie.use_lock(%core1_send_cons, Release, 1)
      aie.end
    }}

    %mem_core0 = aie.mem(%core0) {{
      %0 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core0_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core0_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

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

    %memtile_dma = aie.memtile_dma(%mt0) {{
      // S2MM ch3 (odd, BD 24): receive from core0 (won't fire - gathered_empty=0)
      %0 = aie.dma_start(S2MM, 3, ^inter_r0, ^inter_r1_start)
    ^inter_r0:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r0

      // S2MM ch4 (even, BD 1): receive from core1 (won't fire - gathered_empty=0)
    ^inter_r1_start:
      %1 = aie.dma_start(S2MM, 4, ^inter_r1, ^out_mm2s_start)
    ^inter_r1:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, {BUF_SIZE}, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r1

      // MM2S ch1 (odd, BD 25): send gathered to shim (fires immediately - gathered_full=2)
    ^out_mm2s_start:
      %2 = aie.dma_start(MM2S, 1, ^out_mm2s, ^end)
    ^out_mm2s:
      aie.use_lock(%mt0_gathered_full, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 25 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt0_gathered_empty, Release, 2)
      aie.next_bd ^out_mm2s
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%output_bo: memref<{GATHERED_SIZE // 2}xi32>) {{
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {GATHERED_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}}
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
