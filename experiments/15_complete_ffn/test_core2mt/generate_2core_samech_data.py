"""Test: 2 cores → same memtile S2MM channel, with actual data verification.

Same as samech test but with activation flowing to cores first.
Both cores copy activation to their send buffer, then send to memtile.
Expected output: [activation, activation] (64 bf16).
"""

from pathlib import Path

BUF_SIZE = 32
GATHERED_SIZE = 64
ACT_SIZE = 32


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    // Memtile buffers
    %mt0_act_buf = aie.buffer(%mt0) {{sym_name = "mt0_act_buf"}} : memref<{ACT_SIZE}xbf16>
    %mt0_gathered_buf = aie.buffer(%mt0) {{sym_name = "mt0_gathered_buf"}} : memref<{GATHERED_SIZE}xbf16>

    // Core buffers
    %core0_act = aie.buffer(%core0) {{sym_name = "core0_act"}} : memref<{ACT_SIZE}xbf16>
    %core0_send = aie.buffer(%core0) {{sym_name = "core0_send"}} : memref<{BUF_SIZE}xbf16>
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

    // Flows: activation via circuit-switch
    aie.flow(%shim0, DMA : 0, %mt0, DMA : 0)       // shim → memtile
    aie.flow(%mt0, DMA : 0, %core0, DMA : 0)        // memtile → core0
    aie.flow(%mt0, DMA : 3, %core1, DMA : 0)        // memtile → core1

    // Intermediate via packet-switch to SAME S2MM channel
    aie.packet_flow(0) {{
      aie.packet_source<%core0, DMA : 1>
      aie.packet_dest<%mt0, DMA : 3>
    }}
    aie.packet_flow(1) {{
      aie.packet_source<%core1, DMA : 1>
      aie.packet_dest<%mt0, DMA : 3>
    }}

    // Output via circuit-switch
    aie.flow(%mt0, DMA : 1, %shim0, DMA : 0)

    // Kernel
    func.func private @copy_bf16(memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/copy_kernel.o"}}

    // Core0 program
    %core0_prog = aie.core(%core0) {{
      %n = arith.constant {BUF_SIZE} : i32
      aie.use_lock(%core0_act_full, AcquireGreaterEqual, 1)
      func.call @copy_bf16(%core0_act, %core0_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%core0_act_empty, Release, 1)
      aie.use_lock(%core0_send_cons, Release, 1)
      aie.end
    }}

    // Core1 program
    %core1_prog = aie.core(%core1) {{
      %n = arith.constant {BUF_SIZE} : i32
      aie.use_lock(%core1_act_full, AcquireGreaterEqual, 1)
      func.call @copy_bf16(%core1_act, %core1_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{BUF_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%core1_act_empty, Release, 1)
      aie.use_lock(%core1_send_cons, Release, 1)
      aie.end
    }}

    // Core0 DMA
    %mem_core0 = aie.mem(%core0) {{
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%core0_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core0_act_full, Release, 1)
      aie.next_bd ^act_bd
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core0_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core0_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}}
      aie.use_lock(%core0_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

    // Core1 DMA
    %mem_core1 = aie.mem(%core1) {{
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%core1_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core1_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core1_act_full, Release, 1)
      aie.next_bd ^act_bd
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send_bd, ^end)
    ^send_bd:
      aie.use_lock(%core1_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%core1_send : memref<{BUF_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}}
      aie.use_lock(%core1_send_prod, Release, 1)
      aie.next_bd ^send_bd
    ^end:
      aie.end
    }}

    // Memtile DMA
    %memtile_dma = aie.memtile_dma(%mt0) {{
      // S2MM ch0 (even, BD 0): activation from shim (2-consumer)
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^inter_start)
    ^act_s2mm:
      aie.use_lock(%mt0_act_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt0_act_full, Release, 2)
      aie.next_bd ^act_s2mm

      // S2MM ch3 (odd, BD 24→25): both cores' intermediates sequentially
    ^inter_start:
      %1 = aie.dma_start(S2MM, 3, ^inter_bd0, ^act_mm2s_r0_start)
    ^inter_bd0:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {BUF_SIZE}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_bd1
    ^inter_bd1:
      aie.use_lock(%mt0_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, {BUF_SIZE}, {BUF_SIZE}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt0_gathered_full, Release, 1)
      aie.next_bd ^inter_bd0

      // MM2S ch0 (even, BD 2): activation to core0
    ^act_mm2s_r0_start:
      %2 = aie.dma_start(MM2S, 0, ^act_mm2s_r0, ^act_mm2s_r1_start)
    ^act_mm2s_r0:
      aie.use_lock(%mt0_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt0_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r0

      // MM2S ch3 (odd, BD 26): activation to core1
    ^act_mm2s_r1_start:
      %3 = aie.dma_start(MM2S, 3, ^act_mm2s_r1, ^out_mm2s_start)
    ^act_mm2s_r1:
      aie.use_lock(%mt0_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 26 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt0_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r1

      // MM2S ch1 (odd, BD 27): gathered output to shim
    ^out_mm2s_start:
      %4 = aie.dma_start(MM2S, 1, ^out_mm2s, ^end)
    ^out_mm2s:
      aie.use_lock(%mt0_gathered_full, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt0_gathered_buf : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 27 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt0_gathered_empty, Release, 2)
      aie.next_bd ^out_mm2s
    ^end:
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%input_bo: memref<{ACT_SIZE // 2}xi32>, %output_bo: memref<{GATHERED_SIZE // 2}xi32>) {{
      // Send activation
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive gathered output
      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {GATHERED_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {0x1D004 + 1 * 0x20} : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}}

      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
