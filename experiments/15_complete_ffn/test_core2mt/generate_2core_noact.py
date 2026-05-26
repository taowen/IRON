"""Test: 2 cores → memtile gather → shim output (NO activation, just send).

Data path (no activation at all):
  core0 MM2S ch1 → memtile S2MM ch3 (32 bf16 at offset 0 in gathered)
  core1 MM2S ch1 → memtile S2MM ch4 (32 bf16 at offset 32 in gathered)
  memtile MM2S ch1 → shim S2MM ch0 (64 bf16 gathered output)

Core programs just signal their DMA to send whatever is in the buffer (zeros).
This tests whether 2-core→memtile routing works without any other data flow.
"""

from pathlib import Path

BUF_SIZE = 32  # per core
GATHERED_SIZE = 64  # total gathered


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    // Tiles
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    // Memtile buffers
    %mt0_gathered_buf = aie.buffer(%mt0) {{sym_name = "mt0_gathered_buf"}} : memref<{GATHERED_SIZE}xbf16>

    // Core0 buffers
    %core0_send = aie.buffer(%core0) {{sym_name = "core0_send"}} : memref<{BUF_SIZE}xbf16>

    // Core1 buffers
    %core1_send = aie.buffer(%core1) {{sym_name = "core1_send"}} : memref<{BUF_SIZE}xbf16>

    // Memtile locks
    %mt0_gathered_empty = aie.lock(%mt0, 0) {{init = 2 : i32, sym_name = "mt0_gathered_empty"}}
    %mt0_gathered_full = aie.lock(%mt0, 1) {{init = 0 : i32, sym_name = "mt0_gathered_full"}}

    // Core0 locks
    %core0_send_prod = aie.lock(%core0, 0) {{init = 1 : i32, sym_name = "core0_send_prod"}}
    %core0_send_cons = aie.lock(%core0, 1) {{init = 0 : i32, sym_name = "core0_send_cons"}}

    // Core1 locks
    %core1_send_prod = aie.lock(%core1, 0) {{init = 1 : i32, sym_name = "core1_send_prod"}}
    %core1_send_cons = aie.lock(%core1, 1) {{init = 0 : i32, sym_name = "core1_send_cons"}}

    // Flows (only 3 - minimal)
    aie.flow(%core0, DMA : 1, %mt0, DMA : 3)        // intermediate: core0 → memtile (offset 0)
    aie.flow(%core1, DMA : 1, %mt0, DMA : 4)        // intermediate: core1 → memtile (offset 32)
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)        // output: memtile → shim

    // Core0 program: just signal DMA to send
    %core0_prog = aie.core(%core0) {{
      aie.use_lock(%core0_send_cons, Release, 1)
      aie.end
    }}

    // Core1 program: just signal DMA to send
    %core1_prog = aie.core(%core1) {{
      aie.use_lock(%core1_send_cons, Release, 1)
      aie.end
    }}

    // Core0 DMA: MM2S ch1 only
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
      // S2MM ch3 (odd, BD 24): receive from core0 → offset 0
      %0 = aie.dma_start(S2MM, 3, ^inter_r0, ^inter_r1_start)
    ^inter_r0:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r0

      // S2MM ch4 (even, BD 1): receive from core1 → offset 32
    ^inter_r1_start:
      %1 = aie.dma_start(S2MM, 4, ^inter_r1, ^out_mm2s_start)
    ^inter_r1:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, {BUF_SIZE}, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r1

      // MM2S ch1 (odd, BD 25): send gathered to shim
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

    // Runtime sequence: no input needed, just receive gathered output
    aie.runtime_sequence(%output_bo: memref<{GATHERED_SIZE // 2}xi32>) {{
      // Receive output: shim S2MM ch0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {GATHERED_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}}

      // Sync on output
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
