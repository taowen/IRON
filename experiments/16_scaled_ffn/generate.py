"""
Experiment 16: Scaled FFN — 4 cols × 4 rows, K=4096, BD double-buffer

Key patterns:
- 16 identical compute tiles (4 cols × 4 rows)
- Multicast activation: 1 memtile MM2S → 4 cores via stream switch
- BD double-buffer: only 2 shim BD IDs for 33 weight pushes (alternating bd1/bd2)
- 4-core packet-switched gather (pkt_id = col*4 + row)
- Sequential BD on core S2MM ch0: activation → gathered intermediate
- Uniform weight size: down weight padded to same 2560 bf16 as gate/up per-core slice
- 4-consumer weight distribution from memtile ping-pong
"""

from pathlib import Path

M_PER_TILE = 32
NUM_COLS = 1
ROWS_PER_COL = 4
K_HIDDEN = 512
K_CHUNK = 256
NUM_CHUNKS = K_HIDDEN // K_CHUNK  # 16
GROUP_SIZE = 32
NUM_PHASES_PROJ = 2  # gate + up

INTERMEDIATE = ROWS_PER_COL * M_PER_TILE  # 128
K_DOWN = INTERMEDIATE
K_CHUNK_DOWN = 128
NUM_CHUNKS_DOWN = 1
GROUPS_PER_ROW_DOWN = K_CHUNK_DOWN // GROUP_SIZE  # 4

CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COL  # 10240
DOWN_CHUNK_BF16 = 1280
FAT_DOWN_CHUNK_BF16 = DOWN_CHUNK_BF16 * ROWS_PER_COL  # 5120
DOWN_PADDED_BF16 = CHUNK_BF16  # pad down to uniform size
FAT_DOWN_PADDED_BF16 = DOWN_PADDED_BF16 * ROWS_PER_COL  # 10240

ACT_BF16 = K_HIDDEN  # 4096
OUT_BF16 = M_PER_TILE  # 32
INTER_BF16 = M_PER_TILE  # 32
GATHERED_BF16 = INTERMEDIATE  # 128
ACCUM_F32 = M_PER_TILE  # 32 floats = 128 bytes, explicit accumulator buffer

TOTAL_OUTPUT = NUM_COLS * ROWS_PER_COL * M_PER_TILE  # 512

PUSHES_PER_COL = NUM_PHASES_PROJ * NUM_CHUNKS + NUM_CHUNKS_DOWN  # 33
PER_COL_WT_BF16 = PUSHES_PER_COL * FAT_CHUNK_BF16  # 33 × 10240 = 337920
TOTAL_WT_I32 = (NUM_COLS * PER_COL_WT_BF16 * 2) // 4
ACT_I32 = ACT_BF16 * 2 // 4
OUT_TOTAL_I32 = TOTAL_OUTPUT * 2 // 4


def _shim_bd_address(column, bd_id):
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(column, bd_id, buffer_length, buffer_offset,
                 d0_size=0, d0_stride=0, d1_size=0, d1_stride=0,
                 d2_size=0, d2_stride=0,
                 iteration_size=0, iteration_stride=0):
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 0 : i32, column = {column} : i32, "
        f"d0_size = {d0_size} : i32, d0_stride = {d0_stride} : i32, "
        f"d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = {d1_size} : i32, d1_stride = {d1_stride} : i32, "
        f"d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = {d2_size} : i32, d2_stride = {d2_stride} : i32, "
        f"d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = {iteration_size} : i32, iteration_stride = {iteration_stride} : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(column, bd_id, arg_idx, arg_plus_bytes):
    addr = _shim_bd_address(column, bd_id)
    return f"      aiex.npu.address_patch {{addr = {addr} : ui32, arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"


def _npu_push_queue(column, direction, channel, bd_id, issue_token=False, repeat_count=0):
    token = "true" if issue_token else "false"
    return f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) {{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = {repeat_count} : i32}}"


def _npu_sync(column, channel=0):
    return f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"


def _core_program(col, row, prefix):
    """Core program: gate(NUM_CHUNKS) → up(NUM_CHUNKS) → swiglu → intermediate upload → down(1).

    DMA ping-pong state is continuous across phases. Gate uses transfers 0..N-1,
    UP uses transfers N..2N-1, down uses transfer 2N. The core must track the
    running transfer index to pick the correct ping/pong buffer.
    """
    # UP phase offset: after NUM_CHUNKS gate transfers, parity shifts by NUM_CHUNKS
    up_phase_offset = NUM_CHUNKS % 2
    # DOWN phase: after 2*NUM_CHUNKS transfers, always even → ping
    return f"""    %core_{prefix} = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %num_chunks = arith.constant {NUM_CHUNKS} : index
      %k_chunk_i32 = arith.constant {K_CHUNK} : i32
      %m_i32 = arith.constant {M_PER_TILE} : i32
      %up_offset_i32 = arith.constant {up_phase_offset} : i32

      // Acquire activation
      aie.use_lock(%{prefix}_act_full, AcquireGreaterEqual, 1)

      // Phase 1: GATE projection — {NUM_CHUNKS} chunks
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %k_chunk_i32 : i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        scf.if %is_pong {{
          aie.use_lock(%{prefix}_wt_pong_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_pong, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_pong_empty, Release, 1)
        }} else {{
          aie.use_lock(%{prefix}_wt_ping_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_ping_empty, Release, 1)
        }}
      }}
      func.call @q4nx_flush_output(%{prefix}_gate_buf, %m_i32, %{prefix}_accum_buf)
        : (memref<{OUT_BF16}xbf16>, i32, memref<{ACCUM_F32}xf32>) -> ()

      // Phase 2: UP projection — {NUM_CHUNKS} chunks (DMA parity offset by {up_phase_offset})
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %k_chunk_i32 : i32
        %shifted = arith.addi %chunk_i32, %up_offset_i32 : i32
        %rem = arith.remsi %shifted, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        scf.if %is_pong {{
          aie.use_lock(%{prefix}_wt_pong_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_pong, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_pong_empty, Release, 1)
        }} else {{
          aie.use_lock(%{prefix}_wt_ping_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_ping_empty, Release, 1)
        }}
      }}
      func.call @q4nx_flush_output(%{prefix}_up_buf, %m_i32, %{prefix}_accum_buf)
        : (memref<{OUT_BF16}xbf16>, i32, memref<{ACCUM_F32}xf32>) -> ()

      // Phase 3: SwiGLU → intermediate, then upload to memtile
      func.call @swiglu_fused(%{prefix}_gate_buf, %{prefix}_up_buf, %{prefix}_inter_buf, %m_i32)
        : (memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, memref<{INTER_BF16}xbf16>, i32) -> ()

      aie.use_lock(%{prefix}_act_empty, Release, 1)
      aie.use_lock(%{prefix}_inter_cons, Release, 1)

      // Phase 4: DOWN projection — transfer 2*NUM_CHUNKS is always even → wt_ping
      aie.use_lock(%{prefix}_gathered_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{prefix}_wt_ping_full, AcquireGreaterEqual, 1)

      func.call @q4nx_down_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_gathered_buf, %c0_i32, %m_i32, %{prefix}_accum_buf)
        : (memref<{CHUNK_BF16}xbf16>, memref<{GATHERED_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) -> ()
      aie.use_lock(%{prefix}_wt_ping_empty, Release, 1)

      aie.use_lock(%{prefix}_out_prod, AcquireGreaterEqual, 1)
      func.call @q4nx_down_flush_output(%{prefix}_out, %m_i32, %{prefix}_accum_buf)
        : (memref<{OUT_BF16}xbf16>, i32, memref<{ACCUM_F32}xf32>) -> ()
      aie.use_lock(%{prefix}_out_cons, Release, 1)

      aie.use_lock(%{prefix}_gathered_empty, Release, 1)

      aie.end
    }}"""


def _core_mem(prefix, col, row):
    """Core DMA: S2MM ch0 (act→gathered sequential), S2MM ch1 (wt ping-pong),
    MM2S ch0 (output), MM2S ch1 (intermediate with packet header)."""
    pkt_id = col * ROWS_PER_COL + row
    return f"""    %mem_{prefix} = aie.mem(%{prefix}) {{
      // S2MM ch0: activation → gathered (sequential)
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^wt_start)
    ^act_bd:
      aie.use_lock(%{prefix}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_act : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%{prefix}_act_full, Release, 1)
      aie.next_bd ^gathered_bd
    ^gathered_bd:
      aie.use_lock(%{prefix}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_gathered_buf : memref<{GATHERED_BF16}xbf16>, 0, {GATHERED_BF16}) {{bd_id = 6 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{prefix}_gathered_full, Release, 1)
      aie.next_bd ^act_bd

      // S2MM ch1: weight ping-pong (33 transfers of 2560 each)
    ^wt_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%{prefix}_wt_ping_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_wt_ping_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{prefix}_wt_pong_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_wt_pong_full, Release, 1)
      aie.next_bd ^wt_ping

      // MM2S ch0: final output (packet-switched to memtile)
    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^inter_start)
    ^out_bd:
      aie.use_lock(%{prefix}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUT_BF16}xbf16>, 0, {OUT_BF16}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {NUM_COLS * ROWS_PER_COL + col * ROWS_PER_COL + row}>}}
      aie.use_lock(%{prefix}_out_prod, Release, 1)
      aie.next_bd ^out_bd

      // MM2S ch1: intermediate upload (packet-switched)
    ^inter_start:
      %3 = aie.dma_start(MM2S, 1, ^inter_bd, ^end)
    ^inter_bd:
      aie.use_lock(%{prefix}_inter_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_inter_buf : memref<{INTER_BF16}xbf16>, 0, {INTER_BF16}) {{bd_id = 4 : i32, next_bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {pkt_id}>}}
      aie.use_lock(%{prefix}_inter_prod, Release, 1)
      aie.next_bd ^inter_bd
    ^end:
      aie.end
    }}"""


def _memtile_dma(col):
    """Memtile DMA for 4-core column.

    S2MM ch0 (even, BD 0): activation from shim
    S2MM ch1 (odd, BD 24-25): weights from shim, ping-pong, 33 cycles
    S2MM ch2 (even, BD 7-10): output collect from 4 cores via packet
    S2MM ch3 (odd, BD 26-29): intermediate gather from 4 cores via packet
    MM2S ch0 (even, BD 1→2): act then gathered, multicast to 4 cores
    MM2S ch1 (odd, BD 30-31): weight slice row0, ping-pong
    MM2S ch2 (even, BD 3-4): weight slice row1, ping-pong
    MM2S ch3 (odd, BD 32-33): weight slice row2, ping-pong
    MM2S ch4 (even, BD 5-6): weight slice row3, ping-pong
    MM2S ch5 (odd, BD 34): output forward to shim
    """
    # Weight slice offsets (each core gets 2560 bf16 from the 10240 fat chunk)
    offsets = [row * CHUNK_BF16 for row in range(ROWS_PER_COL)]

    return f"""    %memtile_dma_mt{col} = aie.memtile_dma(%mt{col}) {{
      // S2MM ch0 (even, BD 0): activation from shim
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^wt_s2mm_start)
    ^act_s2mm:
      aie.use_lock(%mt{col}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt{col}_act_full, Release, 1)
      aie.next_bd ^act_s2mm

      // S2MM ch1 (odd, BD 24-25): weights from shim, ping-pong (33 cycles)
    ^wt_s2mm_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_s2mm_ping, ^out_s2mm_start)
    ^wt_s2mm_ping:
      aie.use_lock(%mt{col}_wt_ping_empty, AcquireGreaterEqual, {ROWS_PER_COL})
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt{col}_wt_ping_full, Release, {ROWS_PER_COL})
      aie.next_bd ^wt_s2mm_pong
    ^wt_s2mm_pong:
      aie.use_lock(%mt{col}_wt_pong_empty, AcquireGreaterEqual, {ROWS_PER_COL})
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt{col}_wt_pong_full, Release, {ROWS_PER_COL})
      aie.next_bd ^wt_s2mm_ping

      // S2MM ch2 (even, BD 7-10): output collect from 4 cores via packet
    ^out_s2mm_start:
      %2 = aie.dma_start(S2MM, 2, ^out_s2mm_bd0, ^inter_s2mm_start)
    ^out_s2mm_bd0:
      aie.use_lock(%mt{col}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_out_buf : memref<{GATHERED_BF16}xbf16>, 0, {INTER_BF16}) {{bd_id = 7 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%mt{col}_out_full, Release, 1)
      aie.next_bd ^out_s2mm_bd1
    ^out_s2mm_bd1:
      aie.use_lock(%mt{col}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_out_buf : memref<{GATHERED_BF16}xbf16>, {INTER_BF16}, {INTER_BF16}) {{bd_id = 8 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%mt{col}_out_full, Release, 1)
      aie.next_bd ^out_s2mm_bd2
    ^out_s2mm_bd2:
      aie.use_lock(%mt{col}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_out_buf : memref<{GATHERED_BF16}xbf16>, {2*INTER_BF16}, {INTER_BF16}) {{bd_id = 9 : i32, next_bd_id = 10 : i32}}
      aie.use_lock(%mt{col}_out_full, Release, 1)
      aie.next_bd ^out_s2mm_bd3
    ^out_s2mm_bd3:
      aie.use_lock(%mt{col}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_out_buf : memref<{GATHERED_BF16}xbf16>, {3*INTER_BF16}, {INTER_BF16}) {{bd_id = 10 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%mt{col}_out_full, Release, 1)
      aie.next_bd ^out_s2mm_bd0

      // S2MM ch3 (odd, BD 26-29): intermediate gather from 4 cores, 4 sequential BDs
    ^inter_s2mm_start:
      %3 = aie.dma_start(S2MM, 3, ^inter_s2mm_bd0, ^act_mm2s_start)
    ^inter_s2mm_bd0:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_BF16}xbf16>, 0, {INTER_BF16}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_s2mm_bd1
    ^inter_s2mm_bd1:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_BF16}xbf16>, {INTER_BF16}, {INTER_BF16}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_s2mm_bd2
    ^inter_s2mm_bd2:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_BF16}xbf16>, {2*INTER_BF16}, {INTER_BF16}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_s2mm_bd3
    ^inter_s2mm_bd3:
      aie.use_lock(%mt{col}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_BF16}xbf16>, {3*INTER_BF16}, {INTER_BF16}) {{bd_id = 29 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt{col}_gathered_full, Release, 1)
      aie.next_bd ^inter_s2mm_bd0

      // MM2S ch0 (even, BD 1→2): act THEN gathered, multicast to all 4 cores
    ^act_mm2s_start:
      %4 = aie.dma_start(MM2S, 0, ^act_mm2s, ^wt_mm2s_r0_start)
    ^act_mm2s:
      aie.use_lock(%mt{col}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt{col}_act_empty, Release, 1)
      aie.next_bd ^gathered_mm2s
    ^gathered_mm2s:
      aie.use_lock(%mt{col}_gathered_full, AcquireGreaterEqual, {ROWS_PER_COL})
      aie.dma_bd(%mt{col}_gathered_buf : memref<{GATHERED_BF16}xbf16>, 0, {GATHERED_BF16}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt{col}_gathered_empty, Release, {ROWS_PER_COL})
      aie.next_bd ^act_mm2s

      // MM2S ch1 (odd, BD 30-31): weight slice row0, ping-pong
    ^wt_mm2s_r0_start:
      %5 = aie.dma_start(MM2S, 1, ^wt_mm2s_r0_ping, ^wt_mm2s_r1_start)
    ^wt_mm2s_r0_ping:
      aie.use_lock(%mt{col}_wt_ping_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[0]}, {CHUNK_BF16}) {{bd_id = 30 : i32, next_bd_id = 31 : i32}}
      aie.use_lock(%mt{col}_wt_ping_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r0_pong
    ^wt_mm2s_r0_pong:
      aie.use_lock(%mt{col}_wt_pong_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[0]}, {CHUNK_BF16}) {{bd_id = 31 : i32, next_bd_id = 30 : i32}}
      aie.use_lock(%mt{col}_wt_pong_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r0_ping

      // MM2S ch2 (even, BD 3-4): weight slice row1, ping-pong
    ^wt_mm2s_r1_start:
      %6 = aie.dma_start(MM2S, 2, ^wt_mm2s_r1_ping, ^wt_mm2s_r2_start)
    ^wt_mm2s_r1_ping:
      aie.use_lock(%mt{col}_wt_ping_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[1]}, {CHUNK_BF16}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mt{col}_wt_ping_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r1_pong
    ^wt_mm2s_r1_pong:
      aie.use_lock(%mt{col}_wt_pong_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[1]}, {CHUNK_BF16}) {{bd_id = 4 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mt{col}_wt_pong_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r1_ping

      // MM2S ch3 (odd, BD 32-33): weight slice row2, ping-pong
    ^wt_mm2s_r2_start:
      %7 = aie.dma_start(MM2S, 3, ^wt_mm2s_r2_ping, ^wt_mm2s_r3_start)
    ^wt_mm2s_r2_ping:
      aie.use_lock(%mt{col}_wt_ping_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[2]}, {CHUNK_BF16}) {{bd_id = 32 : i32, next_bd_id = 33 : i32}}
      aie.use_lock(%mt{col}_wt_ping_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r2_pong
    ^wt_mm2s_r2_pong:
      aie.use_lock(%mt{col}_wt_pong_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[2]}, {CHUNK_BF16}) {{bd_id = 33 : i32, next_bd_id = 32 : i32}}
      aie.use_lock(%mt{col}_wt_pong_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r2_ping

      // MM2S ch4 (even, BD 5-6): weight slice row3, ping-pong
    ^wt_mm2s_r3_start:
      %8 = aie.dma_start(MM2S, 4, ^wt_mm2s_r3_ping, ^out_mm2s_start)
    ^wt_mm2s_r3_ping:
      aie.use_lock(%mt{col}_wt_ping_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[3]}, {CHUNK_BF16}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%mt{col}_wt_ping_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r3_pong
    ^wt_mm2s_r3_pong:
      aie.use_lock(%mt{col}_wt_pong_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {offsets[3]}, {CHUNK_BF16}) {{bd_id = 6 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%mt{col}_wt_pong_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r3_ping

      // MM2S ch5 (odd, BD 34): output forward to shim
    ^out_mm2s_start:
      %9 = aie.dma_start(MM2S, 5, ^out_mm2s_bd, ^end)
    ^out_mm2s_bd:
      aie.use_lock(%mt{col}_out_full, AcquireGreaterEqual, {ROWS_PER_COL})
      aie.dma_bd(%mt{col}_out_buf : memref<{GATHERED_BF16}xbf16>, 0, {GATHERED_BF16}) {{bd_id = 34 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mt{col}_out_empty, Release, {ROWS_PER_COL})
      aie.next_bd ^out_mm2s_bd
    ^end:
      aie.end
    }}"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    # Runtime sequence — BD double-buffer strategy
    rt_lines = []
    rt_lines.append(f"    aie.runtime_sequence(%wt_bo: memref<{TOTAL_WT_I32}xi32>, %act_bo: memref<{ACT_I32}xi32>, %out_bo: memref<{OUT_TOTAL_I32}xi32>) {{")

    per_col_wt_bytes = PER_COL_WT_BF16 * 2
    fat_chunk_bytes = FAT_CHUNK_BF16 * 2  # 20480

    # Phase 1: Push ALL activations first (all columns start simultaneously)
    for col in range(NUM_COLS):
        rt_lines.append(_npu_writebd(col, 0, ACT_I32, 0))
        rt_lines.append(_npu_address_patch(col, 0, 1, 0))
        rt_lines.append(_npu_push_queue(col, "MM2S", 0, 0))

    # Phase 2: Arm ALL output receivers (before weights to avoid instruction stall)
    for col in range(NUM_COLS):
        out_offset = col * GATHERED_BF16 * 2
        rt_lines.append(_npu_writebd(col, 3, GATHERED_BF16 // 2, out_offset))
        rt_lines.append(_npu_address_patch(col, 3, 2, out_offset))
        rt_lines.append(_npu_push_queue(col, "S2MM", 0, 3, issue_token=True))

    # Phase 3: Push weights interleaved across columns (1 push per col round-robin).
    # Do not rewrite a shim BD descriptor after queueing it. The shim command
    # queue can consume descriptors asynchronously, so descriptor reuse before a
    # sync can make an older queued push read the newer offset.
    weight_bd_ids = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert PUSHES_PER_COL <= len(weight_bd_ids)
    for i in range(PUSHES_PER_COL):  # 33 rounds
        for col in range(NUM_COLS):
            base_offset = col * per_col_wt_bytes
            bd_id = weight_bd_ids[i]
            offset = base_offset + i * fat_chunk_bytes
            rt_lines.append(_npu_writebd(col, bd_id, FAT_CHUNK_BF16 // 2, offset))
            rt_lines.append(_npu_address_patch(col, bd_id, 0, offset))
            rt_lines.append(_npu_push_queue(col, "MM2S", 1, bd_id))

    # Sync on all columns (S2MM ch0 only)
    for col in range(NUM_COLS):
        rt_lines.append(_npu_sync(col, 0))
    rt_lines.append("    }")
    runtime_sequence = "\n".join(rt_lines)

    # Declarations
    tiles = ""
    buffers = ""
    locks = ""
    flows = ""

    for col in range(NUM_COLS):
        tiles += f"    %shim{col} = aie.tile({col}, 0)\n"
        tiles += f"    %mt{col} = aie.tile({col}, 1)\n"
        for row in range(ROWS_PER_COL):
            r = row + 2
            prefix = f"c{col}r{row}"
            tiles += f"    %{prefix} = aie.tile({col}, {r})\n"
    tiles += "\n"

    for col in range(NUM_COLS):
        # Memtile buffers
        buffers += f"    %mt{col}_act_buf = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_act_buf\"}} : memref<{ACT_BF16}xbf16>\n"
        buffers += f"    %mt{col}_wt_ping = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_wt_ping\"}} : memref<{FAT_CHUNK_BF16}xbf16>\n"
        buffers += f"    %mt{col}_wt_pong = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_wt_pong\"}} : memref<{FAT_CHUNK_BF16}xbf16>\n"
        buffers += f"    %mt{col}_gathered_buf = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_gathered_buf\"}} : memref<{GATHERED_BF16}xbf16>\n"
        buffers += f"    %mt{col}_out_buf = aie.buffer(%mt{col}) {{sym_name = \"mt{col}_out_buf\"}} : memref<{GATHERED_BF16}xbf16>\n"

        # Memtile locks
        locks += f"    %mt{col}_act_empty = aie.lock(%mt{col}, 0) {{init = 1 : i32, sym_name = \"mt{col}_act_empty\"}}\n"
        locks += f"    %mt{col}_act_full  = aie.lock(%mt{col}, 1) {{init = 0 : i32, sym_name = \"mt{col}_act_full\"}}\n"
        locks += f"    %mt{col}_wt_ping_empty = aie.lock(%mt{col}, 8) {{init = {ROWS_PER_COL} : i32, sym_name = \"mt{col}_wt_ping_empty\"}}\n"
        locks += f"    %mt{col}_wt_ping_full  = aie.lock(%mt{col}, 9) {{init = 0 : i32, sym_name = \"mt{col}_wt_ping_full\"}}\n"
        locks += f"    %mt{col}_wt_pong_empty = aie.lock(%mt{col}, 10) {{init = {ROWS_PER_COL} : i32, sym_name = \"mt{col}_wt_pong_empty\"}}\n"
        locks += f"    %mt{col}_wt_pong_full  = aie.lock(%mt{col}, 11) {{init = 0 : i32, sym_name = \"mt{col}_wt_pong_full\"}}\n"
        locks += f"    %mt{col}_gathered_empty = aie.lock(%mt{col}, 4) {{init = {ROWS_PER_COL} : i32, sym_name = \"mt{col}_gathered_empty\"}}\n"
        locks += f"    %mt{col}_gathered_full  = aie.lock(%mt{col}, 5) {{init = 0 : i32, sym_name = \"mt{col}_gathered_full\"}}\n"
        locks += f"    %mt{col}_out_empty = aie.lock(%mt{col}, 6) {{init = {ROWS_PER_COL} : i32, sym_name = \"mt{col}_out_empty\"}}\n"
        locks += f"    %mt{col}_out_full  = aie.lock(%mt{col}, 7) {{init = 0 : i32, sym_name = \"mt{col}_out_full\"}}\n"

        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            # Core buffers
            buffers += f"    %{prefix}_act = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_act\"}} : memref<{ACT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_wt_ping = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_wt_ping\"}} : memref<{CHUNK_BF16}xbf16>\n"
            buffers += f"    %{prefix}_wt_pong = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_wt_pong\"}} : memref<{CHUNK_BF16}xbf16>\n"
            buffers += f"    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_out\"}} : memref<{OUT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_gate_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_gate_buf\"}} : memref<{OUT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_up_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_up_buf\"}} : memref<{OUT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_inter_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_inter_buf\"}} : memref<{INTER_BF16}xbf16>\n"
            buffers += f"    %{prefix}_gathered_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_gathered_buf\"}} : memref<{GATHERED_BF16}xbf16>\n"
            buffers += f"    %{prefix}_accum_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_accum_buf\"}} : memref<{ACCUM_F32}xf32>\n"

            # Core locks
            locks += f"    %{prefix}_act_empty = aie.lock(%{prefix}, 0) {{init = 1 : i32, sym_name = \"{prefix}_act_empty\"}}\n"
            locks += f"    %{prefix}_act_full  = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = \"{prefix}_act_full\"}}\n"
            locks += f"    %{prefix}_wt_ping_empty = aie.lock(%{prefix}, 10) {{init = 1 : i32, sym_name = \"{prefix}_wt_ping_empty\"}}\n"
            locks += f"    %{prefix}_wt_ping_full  = aie.lock(%{prefix}, 11) {{init = 0 : i32, sym_name = \"{prefix}_wt_ping_full\"}}\n"
            locks += f"    %{prefix}_wt_pong_empty = aie.lock(%{prefix}, 12) {{init = 1 : i32, sym_name = \"{prefix}_wt_pong_empty\"}}\n"
            locks += f"    %{prefix}_wt_pong_full  = aie.lock(%{prefix}, 13) {{init = 0 : i32, sym_name = \"{prefix}_wt_pong_full\"}}\n"
            locks += f"    %{prefix}_out_prod  = aie.lock(%{prefix}, 4) {{init = 1 : i32, sym_name = \"{prefix}_out_prod\"}}\n"
            locks += f"    %{prefix}_out_cons  = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = \"{prefix}_out_cons\"}}\n"
            locks += f"    %{prefix}_inter_prod  = aie.lock(%{prefix}, 6) {{init = 0 : i32, sym_name = \"{prefix}_inter_prod\"}}\n"
            locks += f"    %{prefix}_inter_cons  = aie.lock(%{prefix}, 7) {{init = 0 : i32, sym_name = \"{prefix}_inter_cons\"}}\n"
            locks += f"    %{prefix}_gathered_empty = aie.lock(%{prefix}, 8) {{init = 1 : i32, sym_name = \"{prefix}_gathered_empty\"}}\n"
            locks += f"    %{prefix}_gathered_full  = aie.lock(%{prefix}, 9) {{init = 0 : i32, sym_name = \"{prefix}_gathered_full\"}}\n"

    # Flows
    for col in range(NUM_COLS):
        # Shim → Memtile
        flows += f"    aie.flow(%shim{col}, DMA : 0, %mt{col}, DMA : 0)\n"  # activation
        flows += f"    aie.flow(%shim{col}, DMA : 1, %mt{col}, DMA : 1)\n"  # weights

        # Memtile MM2S ch0 → ALL 4 cores S2MM ch0 (multicast: act + gathered)
        for row in range(ROWS_PER_COL):
            flows += f"    aie.flow(%mt{col}, DMA : 0, %c{col}r{row}, DMA : 0)\n"

        # Memtile weight distribution: MM2S ch(1+row) → core row S2MM ch1
        for row in range(ROWS_PER_COL):
            mt_ch = 1 + row  # MM2S ch1, ch2, ch3, ch4
            flows += f"    aie.flow(%mt{col}, DMA : {mt_ch}, %c{col}r{row}, DMA : 1)\n"

        # Core → Memtile: intermediate gather via packet-switch (all to memtile S2MM ch3)
        for row in range(ROWS_PER_COL):
            pkt_id = col * ROWS_PER_COL + row
            flows += f"    aie.packet_flow({pkt_id}) {{\n"
            flows += f"      aie.packet_source<%c{col}r{row}, DMA : 1>\n"
            flows += f"      aie.packet_dest<%mt{col}, DMA : 3>\n"
            flows += f"    }}\n"

        # Core → Memtile: output via packet-switch (all to memtile S2MM ch2)
        for row in range(ROWS_PER_COL):
            pkt_id = NUM_COLS * ROWS_PER_COL + col * ROWS_PER_COL + row  # 16..31
            flows += f"    aie.packet_flow({pkt_id}) {{\n"
            flows += f"      aie.packet_source<%c{col}r{row}, DMA : 0>\n"
            flows += f"      aie.packet_dest<%mt{col}, DMA : 2>\n"
            flows += f"    }}\n"

        # Memtile → Shim: aggregated output (circuit-switched, memtile MM2S ch5 → shim S2MM ch0)
        flows += f"    aie.flow(%mt{col}, DMA : 5, %shim{col}, DMA : 0)\n"

    # Core programs
    core_programs = ""
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            core_programs += _core_program(col, row, prefix) + "\n\n"

    # Memtile DMAs
    memtile_dmas = ""
    for col in range(NUM_COLS):
        memtile_dmas += _memtile_dma(col) + "\n\n"

    # Core DMAs
    core_mems = ""
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            core_mems += _core_mem(prefix, col, row) + "\n\n"

    return f"""module {{
  aie.device(npu2) {{
    // === Tile declarations ===
{tiles}
    // === Buffers ===
{buffers}
    // === Locks ===
{locks}
    // === Flows ===
{flows}
    // === Kernel declarations ===
    func.func private @q4nx_chunk_accum_offset(memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum.o"}}
    func.func private @q4nx_flush_output(memref<{OUT_BF16}xbf16>, i32, memref<{ACCUM_F32}xf32>) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum.o"}}
    func.func private @q4nx_down_chunk_accum_offset(memref<{CHUNK_BF16}xbf16>, memref<{GATHERED_BF16}xbf16>, i32, i32, memref<{ACCUM_F32}xf32>) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum_down.o"}}
    func.func private @q4nx_down_flush_output(memref<{OUT_BF16}xbf16>, i32, memref<{ACCUM_F32}xf32>) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum_down.o"}}
    func.func private @swiglu_fused(memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, memref<{INTER_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/swiglu_fused.o"}}

    // === Core programs ===
{core_programs}
    // === Memtile DMAs ===
{memtile_dmas}
    // === Core DMAs ===
{core_mems}
    // === Runtime sequence ===
{runtime_sequence}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
