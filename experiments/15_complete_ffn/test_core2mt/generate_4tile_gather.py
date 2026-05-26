"""Test: 4-tile (2 cols × 2 rows) packet-switched gather, NO weights.

This tests if the packet-switched gather pattern works with 2 columns.
Each column has 2 cores that send intermediates to memtile via packet-switch.
Memtile gathers and broadcasts back. Output goes to shim.

This strips out weights to isolate the bidirectional flow pattern.
"""

from pathlib import Path

ACT_SIZE = 32
INTER_SIZE = 32
GATHERED_SIZE = 64
NUM_COLS = 2
ROWS_PER_COL = 2


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    cols = []
    for col in range(NUM_COLS):
        cols.append(_gen_column(col, experiment_dir))

    rt = _gen_runtime_sequence()
    kernel_decl = f"    func.func private @copy_bf16(memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) attributes {{link_with = \"{experiment_dir}/copy_kernel.o\"}}"

    return f"""module {{
  aie.device(npu2) {{
{"".join(cols)}
{kernel_decl}

{rt}
  }}
}}
"""


def _gen_column(col, experiment_dir):
    lines = []

    # Tiles
    lines.append(f"    %shim{col} = aie.tile({col}, 0)")
    lines.append(f"    %mt{col} = aie.tile({col}, 1)")
    lines.append(f"    %c{col}r0 = aie.tile({col}, 2)")
    lines.append(f"    %c{col}r1 = aie.tile({col}, 3)")

    # Memtile buffers
    lines.append(f"    %mt{col}_act_buf = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_act_buf\"}} : memref<{ACT_SIZE}xbf16>")
    lines.append(f"    %mt{col}_gathered_buf = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_gathered_buf\"}} : memref<{GATHERED_SIZE}xbf16>")

    # Core buffers
    for row in range(ROWS_PER_COL):
        p = f"c{col}r{row}"
        lines.append(f"    %{p}_act = aie.buffer(%{p}) {{sym_name = \"{p}_act\"}} : memref<{ACT_SIZE}xbf16>")
        lines.append(f"    %{p}_send = aie.buffer(%{p}) {{sym_name = \"{p}_send\"}} : memref<{INTER_SIZE}xbf16>")
        lines.append(f"    %{p}_gathered = aie.buffer(%{p}) {{sym_name = \"{p}_gathered\"}} : memref<{GATHERED_SIZE}xbf16>")
        lines.append(f"    %{p}_out = aie.buffer(%{p}) {{sym_name = \"{p}_out\"}} : memref<{INTER_SIZE}xbf16>")

    # Memtile locks
    lines.append(f"    %mt{col}_act_empty = aie.lock(%mt{col}, 0) {{init = 2 : i32, sym_name = \"mt{col}_act_empty\"}}")
    lines.append(f"    %mt{col}_act_full = aie.lock(%mt{col}, 1) {{init = 0 : i32, sym_name = \"mt{col}_act_full\"}}")
    lines.append(f"    %mt{col}_gathered_empty = aie.lock(%mt{col}, 2) {{init = 2 : i32, sym_name = \"mt{col}_gathered_empty\"}}")
    lines.append(f"    %mt{col}_gathered_full = aie.lock(%mt{col}, 3) {{init = 0 : i32, sym_name = \"mt{col}_gathered_full\"}}")

    # Core locks
    for row in range(ROWS_PER_COL):
        p = f"c{col}r{row}"
        lines.append(f"    %{p}_act_empty = aie.lock(%{p}, 0) {{init = 1 : i32, sym_name = \"{p}_act_empty\"}}")
        lines.append(f"    %{p}_act_full = aie.lock(%{p}, 1) {{init = 0 : i32, sym_name = \"{p}_act_full\"}}")
        lines.append(f"    %{p}_send_prod = aie.lock(%{p}, 2) {{init = 1 : i32, sym_name = \"{p}_send_prod\"}}")
        lines.append(f"    %{p}_send_cons = aie.lock(%{p}, 3) {{init = 0 : i32, sym_name = \"{p}_send_cons\"}}")
        lines.append(f"    %{p}_gathered_empty = aie.lock(%{p}, 4) {{init = 1 : i32, sym_name = \"{p}_gathered_empty\"}}")
        lines.append(f"    %{p}_gathered_full = aie.lock(%{p}, 5) {{init = 0 : i32, sym_name = \"{p}_gathered_full\"}}")
        lines.append(f"    %{p}_out_prod = aie.lock(%{p}, 6) {{init = 1 : i32, sym_name = \"{p}_out_prod\"}}")
        lines.append(f"    %{p}_out_cons = aie.lock(%{p}, 7) {{init = 0 : i32, sym_name = \"{p}_out_cons\"}}")

    # Flows: activation (circuit-switched)
    lines.append(f"    aie.flow(%shim{col}, DMA : 0, %mt{col}, DMA : 0)")
    lines.append(f"    aie.flow(%mt{col}, DMA : 0, %c{col}r0, DMA : 0)")
    lines.append(f"    aie.flow(%mt{col}, DMA : 3, %c{col}r1, DMA : 0)")

    # Intermediate: packet-switched to SAME memtile S2MM ch3
    lines.append(f"    aie.packet_flow({col*2}) {{")
    lines.append(f"      aie.packet_source<%c{col}r0, DMA : 1>")
    lines.append(f"      aie.packet_dest<%mt{col}, DMA : 3>")
    lines.append(f"    }}")
    lines.append(f"    aie.packet_flow({col*2+1}) {{")
    lines.append(f"      aie.packet_source<%c{col}r1, DMA : 1>")
    lines.append(f"      aie.packet_dest<%mt{col}, DMA : 3>")
    lines.append(f"    }}")

    # Gathered: sent on SAME channel as activation (sequential BD)
    # No additional flows needed — MM2S ch0→core0 and ch3→core1 already handle it

    # Output: core → shim (circuit-switched)
    lines.append(f"    aie.flow(%c{col}r0, DMA : 0, %shim{col}, DMA : 0)")
    lines.append(f"    aie.flow(%c{col}r1, DMA : 0, %shim{col}, DMA : 1)")

    # Kernel
    # Kernel declared at module level

    # Core programs
    for row in range(ROWS_PER_COL):
        p = f"c{col}r{row}"
        lines.append(f"""    %core_{p} = aie.core(%{p}) {{
      %n = arith.constant {INTER_SIZE} : i32
      // Wait for activation
      aie.use_lock(%{p}_act_full, AcquireGreaterEqual, 1)
      // Copy activation to send buffer AND output buffer
      func.call @copy_bf16(%{p}_act, %{p}_send, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      func.call @copy_bf16(%{p}_act, %{p}_out, %n) : (memref<{ACT_SIZE}xbf16>, memref<{INTER_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%{p}_act_empty, Release, 1)
      // Trigger intermediate send
      aie.use_lock(%{p}_send_cons, Release, 1)
      // Wait for gathered (proves bidirectional flow works)
      aie.use_lock(%{p}_gathered_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{p}_gathered_empty, Release, 1)
      // Trigger output send
      aie.use_lock(%{p}_out_cons, Release, 1)
      aie.end
    }}""")

    # Core DMAs
    for row in range(ROWS_PER_COL):
        p = f"c{col}r{row}"
        lines.append(f"""    %mem_{p} = aie.mem(%{p}) {{
      // S2MM ch0: activation (BD0) → gathered (BD1) [sequential]
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^send_start)
    ^act_bd:
      aie.use_lock(%{p}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{p}_act_full, Release, 1)
      aie.next_bd ^gathered_bd
    ^gathered_bd:
      aie.use_lock(%{p}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_gathered : memref<{GATHERED_SIZE}xbf16>, 0, {GATHERED_SIZE}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{p}_gathered_full, Release, 1)
      aie.next_bd ^act_bd

      // MM2S ch0: output to shim
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^inter_start)
    ^out_bd:
      aie.use_lock(%{p}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_out : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{p}_out_prod, Release, 1)
      aie.next_bd ^out_bd

      // MM2S ch1: intermediate to memtile (packet-switched)
    ^inter_start:
      %2 = aie.dma_start(MM2S, 1, ^inter_bd, ^end)
    ^inter_bd:
      aie.use_lock(%{p}_send_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_send : memref<{INTER_SIZE}xbf16>, 0, {INTER_SIZE}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {col*2 + row}>}}
      aie.use_lock(%{p}_send_prod, Release, 1)
      aie.next_bd ^inter_bd
    ^end:
      aie.end
    }}""")

    # Memtile DMA
    lines.append(f"""    %memtile_dma_mt{col} = aie.memtile_dma(%mt{col}) {{
      // S2MM ch0 (even): activation from shim (2-consumer)
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^inter_s2mm_start)
    ^act_s2mm:
      aie.use_lock(%mt{col}_act_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt{col}_act_full, Release, 2)
      aie.next_bd ^act_s2mm

      // S2MM ch3 (odd): both intermediates via packet-switch (sequential BDs)
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

      // MM2S ch0 (even): act to core0, THEN gathered to core0 (sequential)
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

      // MM2S ch3 (odd): act to core1, THEN gathered to core1 (sequential)
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
    }}""")

    return "\n".join(lines) + "\n\n"


def _gen_runtime_sequence():
    OUT_SIZE_I32 = NUM_COLS * ROWS_PER_COL * INTER_SIZE // 2  # 64 i32
    lines = []
    lines.append(f"    aie.runtime_sequence(%act_bo: memref<{ACT_SIZE // 2}xi32>, %out_bo: memref<{OUT_SIZE_I32}xi32>) {{")

    for col in range(NUM_COLS):
        # Activation: shim MM2S ch0
        lines.append(f"      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_SIZE // 2} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}")
        lines.append(f"      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004 + 0 * 0x20} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}")
        lines.append(f"      aiex.npu.push_queue({col}, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}")

        # Output row0: shim S2MM ch0
        out_off_r0 = (col * ROWS_PER_COL + 0) * INTER_SIZE * 2
        lines.append(f"      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {INTER_SIZE // 2} : i32, buffer_offset = {out_off_r0} : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}")
        lines.append(f"      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004 + 1 * 0x20} : ui32, arg_idx = 1 : i32, arg_plus = {out_off_r0} : i32}}")
        lines.append(f"      aiex.npu.push_queue({col}, 0, S2MM : 0) {{bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}}")

        # Output row1: shim S2MM ch1
        out_off_r1 = (col * ROWS_PER_COL + 1) * INTER_SIZE * 2
        lines.append(f"      aiex.npu.writebd {{bd_id = 2 : i32, buffer_length = {INTER_SIZE // 2} : i32, buffer_offset = {out_off_r1} : i32, burst_length = 0 : i32, column = {col} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}")
        lines.append(f"      aiex.npu.address_patch {{addr = {col * 0x02000000 + 0x1D004 + 2 * 0x20} : ui32, arg_idx = 1 : i32, arg_plus = {out_off_r1} : i32}}")
        lines.append(f"      aiex.npu.push_queue({col}, 0, S2MM : 1) {{bd_id = 2 : i32, issue_token = true, repeat_count = 0 : i32}}")

    # Sync all outputs
    for col in range(NUM_COLS):
        lines.append(f"      aiex.npu.sync {{channel = 0 : i32, column = {col} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}")
        lines.append(f"      aiex.npu.sync {{channel = 1 : i32, column = {col} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}")
    lines.append("    }")
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_mlir())
