"""Test: 2 cores → memtile gather → shim output.

Data path:
  shim MM2S ch0 → memtile → core0 S2MM ch0 (32 bf16 activation)
  shim MM2S ch0 → memtile → core1 S2MM ch0 (same 32 bf16 activation)
  core0 MM2S ch1 → memtile S2MM ch3 (32 bf16 at offset 0 in gathered)
  core1 MM2S ch1 → memtile S2MM ch4 (32 bf16 at offset 32 in gathered)
  memtile MM2S ch1 → shim S2MM ch0 (64 bf16 gathered output)

This tests the 2-producer gather pattern used in exp 15.
"""

from pathlib import Path

BUF_SIZE = 32  # per core
GATHERED_SIZE = 64  # total gathered
ACT_SIZE = 32  # activation per core


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
    %mt0_act_buf = aie.buffer(%mt0) {{sym_name = "mt0_act_buf"}} : memref<{ACT_SIZE}xbf16>
    %mt0_gathered_buf = aie.buffer(%mt0) {{sym_name = "mt0_gathered_buf"}} : memref<{GATHERED_SIZE}xbf16>

    // Core0 buffers
    %core0_act = aie.buffer(%core0) {{sym_name = "core0_act"}} : memref<{ACT_SIZE}xbf16>
    %core0_send = aie.buffer(%core0) {{sym_name = "core0_send"}} : memref<{BUF_SIZE}xbf16>

    // Core1 buffers
    %core1_act = aie.buffer(%core1) {{sym_name = "core1_act"}} : memref<{ACT_SIZE}xbf16>
    %core1_send = aie.buffer(%core1) {{sym_name = "core1_send"}} : memref<{BUF_SIZE}xbf16>

    // Memtile locks
    %mt0_act_empty = aie.lock(%mt0, 0) {{init = 2 : i32, sym_name = "mt0_act_empty"}}
    %mt0_act_full = aie.lock(%mt0, 1) {{init = 0 : i32, sym_name = "mt0_act_full"}}
    %mt0_gathered_empty = aie.lock(%mt0, 2) {{init = 2 : i32, sym_name = "mt0_gathered_empty"}}
    %mt0_gathered_full = aie.lock(%mt0, 3) {{init = 0 : i32, sym_name = "mt0_gathered_full"}}

    // Core0 locks
    %core0_act_empty = aie.lock(%core0, 0) {{init = 1 : i32, sym_name = "core0_act_empty"}}
    %core0_act_full = aie.lock(%core0, 1) {{init = 0 : i32, sym_name = "core0_act_full"}}
    %core0_send_prod = aie.lock(%core0, 2) {{init = 1 : i32, sym_name = "core0_send_prod"}}
    %core0_send_cons = aie.lock(%core0, 3) {{init = 0 : i32, sym_name = "core0_send_cons"}}

    // Core1 locks
    %core1_act_empty = aie.lock(%core1, 0) {{init = 1 : i32, sym_name = "core1_act_empty"}}
    %core1_act_full = aie.lock(%core1, 1) {{init = 0 : i32, sym_name = "core1_act_full"}}
    %core1_send_prod = aie.lock(%core1, 2) {{init = 1 : i32, sym_name = "core1_send_prod"}}
    %core1_send_cons = aie.lock(%core1, 3) {{init = 0 : i32, sym_name = "core1_send_cons"}}

    // Flows
    aie.flow(%shim0, DMA : 0, %mt0, DMA : 0)       // activation: shim → memtile
    aie.flow(%mt0, DMA : 0, %core0, DMA : 0)        // activation: memtile → core0
    aie.flow(%mt0, DMA : 3, %core1, DMA : 0)        // activation: memtile → core1
    aie.flow(%core0, DMA : 1, %mt0, DMA : 3)        // intermediate: core0 → memtile (offset 0)
    aie.flow(%core1, DMA : 1, %mt0, DMA : 4)        // intermediate: core1 → memtile (offset 32)
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)        // output: memtile → shim

    // Kernel: simple copy
    func.func private @copy_bf16(memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/copy_kernel.o"}}

    // Core0 program
    %core0_prog = aie.core(%core0) {{
      %n = arith.constant {BUF_SIZE} : i32

      // Wait for activation
      aie.use_lock(%core0_act_full, AcquireGreaterEqual, 1)

      // Copy act to send buffer
      func.call @copy_bf16(%core0_act, %core0_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) -> ()

      // Release activation
      aie.use_lock(%core0_act_empty, Release, 1)

      // Signal DMA to send intermediate
      aie.use_lock(%core0_send_cons, Release, 1)

      aie.end
    }}

    // Core1 program
    %core1_prog = aie.core(%core1) {{
      %n = arith.constant {BUF_SIZE} : i32

      // Wait for activation
      aie.use_lock(%core1_act_full, AcquireGreaterEqual, 1)

      // Copy act to send buffer
      func.call @copy_bf16(%core1_act, %core1_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) -> ()

      // Release activation
      aie.use_lock(%core1_act_empty, Release, 1)

      // Signal DMA to send intermediate
      aie.use_lock(%core1_send_cons, Release, 1)

      aie.end
    }}

    // Core0 DMA
    %mem_core0 = aie.mem(%core0) {{
      // S2MM ch0: receive activation
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%core0_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
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

    // Core1 DMA
    %mem_core1 = aie.mem(%core1) {{
      // S2MM ch0: receive activation
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%core1_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core1_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core1_act_full, Release, 1)
      aie.next_bd ^act_bd

      // MM2S ch1: send to memtile
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core1_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core1_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%core1_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

    // Memtile DMA
    %memtile_dma = aie.memtile_dma(%mt0) {{
      // S2MM ch0 (even, BD 0): receive activation from shim (2-consumer)
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^inter_r0_start)
    ^act_s2mm:
      aie.use_lock(%mt0_act_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt0_act_full, Release, 2)
      aie.next_bd ^act_s2mm

      // S2MM ch3 (odd, BD 24): receive intermediate from core0 → offset 0
    ^inter_r0_start:
      %1 = aie.dma_start(S2MM, 3, ^inter_r0, ^inter_r1_start)
    ^inter_r0:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r0

      // S2MM ch4 (even, BD 1): receive intermediate from core1 → offset 32
    ^inter_r1_start:
      %2 = aie.dma_start(S2MM, 4, ^inter_r1, ^act_mm2s_r0_start)
    ^inter_r1:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, {BUF_SIZE}, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_r1

      // MM2S ch0 (even, BD 2): send activation to core0
    ^act_mm2s_r0_start:
      %3 = aie.dma_start(MM2S, 0, ^act_mm2s_r0, ^act_mm2s_r1_start)
    ^act_mm2s_r0:
      aie.use_lock(%mt0_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt0_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r0

      // MM2S ch3 (odd, BD 25): send activation to core1
    ^act_mm2s_r1_start:
      %4 = aie.dma_start(MM2S, 3, ^act_mm2s_r1, ^out_mm2s_start)
    ^act_mm2s_r1:
      aie.use_lock(%mt0_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 25 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt0_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r1

      // MM2S ch1 (odd, BD 26): send gathered output to shim
    ^out_mm2s_start:
      %5 = aie.dma_start(MM2S, 1, ^out_mm2s, ^end)
    ^out_mm2s:
      aie.use_lock(%mt0_gathered_full, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 26 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt0_gathered_empty, Release, 2)
      aie.next_bd ^out_mm2s
    ^end:
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%input_bo: memref<{ACT_SIZE // 2}xi32>, %output_bo: memref<{GATHERED_SIZE // 2}xi32>) {{
      // Send input: shim MM2S ch0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive output: shim S2MM ch0
      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {GATHERED_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
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
