"""
Experiment 14: FFN SwiGLU Fusion with Cross-Phase Residency + BD Iteration

Generates raw MLIR-AIE with:
- 2 columns × 2 rows = 4 compute tiles
- Cross-phase residency: gate/up outputs in core-local buffers (no DDR drain)
- SwiGLU kernel reads from local buffers (no DMA)
- BD iteration_stride compresses 8 weight transfers into 1 writebd+push_queue
- Same memtile broadcast + weight distribution as exp 13
"""

from pathlib import Path

M_PER_TILE = 32
NUM_COLS = 2
ROWS_PER_COL = 2
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
NUM_PHASES = 2  # gate + up

TOTAL_OUTPUT = NUM_COLS * ROWS_PER_COL * M_PER_TILE  # 128

CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COL
ACT_BF16 = K
OUT_BF16 = M_PER_TILE

TOTAL_WT_I32 = (NUM_COLS * NUM_PHASES * NUM_CHUNKS * FAT_CHUNK_BF16) // 2
ACT_I32 = K // 2
OUT_TOTAL_I32 = TOTAL_OUTPUT // 2


def _shim_bd_address(column: int, bd_id: int) -> int:
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
    """Generate core program: gate → up → swiglu (cross-phase residency)."""
    return f"""    %core_{prefix} = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_chunks = arith.constant {NUM_CHUNKS} : index
      %total_chunks = arith.constant {NUM_PHASES * NUM_CHUNKS} : index
      %k_chunk_i32 = arith.constant {K_CHUNK} : i32
      %m_i32 = arith.constant {M_PER_TILE} : i32
      %chunks_per_phase = arith.constant {NUM_CHUNKS} : index

      // Acquire activation — hold across BOTH gate and up phases
      aie.use_lock(%{prefix}_act_full, AcquireGreaterEqual, 1)

      // Phase 1: GATE projection — 4 chunks, flush to LOCAL gate_buf
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %k_chunk_i32 : i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%{prefix}_wt_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_pong, %{prefix}_act, %act_offset, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32) -> ()
        }} else {{
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_act, %act_offset, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32) -> ()
        }}

        aie.use_lock(%{prefix}_wt_empty, Release, 1)
      }}
      // Flush gate to LOCAL buffer (no DMA, no lock needed)
      func.call @q4nx_flush_output(%{prefix}_gate_buf, %m_i32)
        : (memref<{OUT_BF16}xbf16>, i32) -> ()

      // Phase 2: UP projection — 4 more chunks, flush to LOCAL up_buf
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %k_chunk_i32 : i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%{prefix}_wt_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_pong, %{prefix}_act, %act_offset, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32) -> ()
        }} else {{
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_act, %act_offset, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32) -> ()
        }}

        aie.use_lock(%{prefix}_wt_empty, Release, 1)
      }}
      // Flush up to LOCAL buffer (no DMA, no lock needed)
      func.call @q4nx_flush_output(%{prefix}_up_buf, %m_i32)
        : (memref<{OUT_BF16}xbf16>, i32) -> ()

      // Phase 3: SwiGLU — reads ONLY from local buffers, no DMA involved
      aie.use_lock(%{prefix}_out_prod, AcquireGreaterEqual, 1)
      func.call @swiglu_fused(%{prefix}_gate_buf, %{prefix}_up_buf, %{prefix}_out, %m_i32)
        : (memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, i32) -> ()
      aie.use_lock(%{prefix}_out_cons, Release, 1)

      // Release activation after all phases
      aie.use_lock(%{prefix}_act_empty, Release, 1)
      aie.end
    }}"""


def _memtile_dma(col, prefix):
    """Generate memtile DMA — identical to exp 13."""
    return f"""    %memtile_dma_{prefix} = aie.memtile_dma(%mt{col}) {{
      // S2MM ch0 (even → BD 0-23): activation from shim
      %0 = aie.dma_start(S2MM, 0, ^act_s2mm, ^act_mm2s_r0_start)
    ^act_s2mm:
      aie.use_lock(%mt{col}_act_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt{col}_act_full, Release, 2)
      aie.next_bd ^act_s2mm

      // MM2S ch0 (even → BD 0-23): activation to core row0
    ^act_mm2s_r0_start:
      %1 = aie.dma_start(MM2S, 0, ^act_mm2s_r0, ^act_mm2s_r1_start)
    ^act_mm2s_r0:
      aie.use_lock(%mt{col}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt{col}_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r0

      // MM2S ch3 (odd → BD 24-47): activation to core row1
    ^act_mm2s_r1_start:
      %2 = aie.dma_start(MM2S, 3, ^act_mm2s_r1, ^wt_s2mm_ch1_start)
    ^act_mm2s_r1:
      aie.use_lock(%mt{col}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_act_buf : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt{col}_act_empty, Release, 1)
      aie.next_bd ^act_mm2s_r1

      // S2MM ch1 (odd → BD 24-47): weight fat chunks (ping-pong, 2-consumer)
    ^wt_s2mm_ch1_start:
      %3 = aie.dma_start(S2MM, 1, ^wt_s2mm_ping, ^wt_mm2s_r0_start)
    ^wt_s2mm_ping:
      aie.use_lock(%mt{col}_wt_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt{col}_wt_full, Release, 2)
      aie.next_bd ^wt_s2mm_pong
    ^wt_s2mm_pong:
      aie.use_lock(%mt{col}_wt_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 26 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt{col}_wt_full, Release, 2)
      aie.next_bd ^wt_s2mm_ping

      // MM2S ch1 (odd → BD 24-47): weight first half to core row0
    ^wt_mm2s_r0_start:
      %4 = aie.dma_start(MM2S, 1, ^wt_mm2s_r0_ping, ^wt_mm2s_r1_start)
    ^wt_mm2s_r0_ping:
      aie.use_lock(%mt{col}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mt{col}_wt_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r0_pong
    ^wt_mm2s_r0_pong:
      aie.use_lock(%mt{col}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 28 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt{col}_wt_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r0_ping

      // MM2S ch2 (even → BD 0-23): weight second half to core row1
    ^wt_mm2s_r1_start:
      %5 = aie.dma_start(MM2S, 2, ^wt_mm2s_r1_ping, ^end)
    ^wt_mm2s_r1_ping:
      aie.use_lock(%mt{col}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {CHUNK_BF16}, {CHUNK_BF16}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%mt{col}_wt_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r1_pong
    ^wt_mm2s_r1_pong:
      aie.use_lock(%mt{col}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt{col}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {CHUNK_BF16}, {CHUNK_BF16}) {{bd_id = 5 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mt{col}_wt_empty, Release, 1)
      aie.next_bd ^wt_mm2s_r1_ping
    ^end:
      aie.end
    }}"""


def _core_mem(prefix):
    """Generate core tile DMA (S2MM ch0=act, S2MM ch1=wt, MM2S ch0=output)."""
    return f"""    %mem_{prefix} = aie.mem(%{prefix}) {{
      // S2MM ch0: activation (single buffer, hold)
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^wt_start)
    ^act_bd:
      aie.use_lock(%{prefix}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_act : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{prefix}_act_full, Release, 1)
      aie.next_bd ^act_bd

      // S2MM ch1: weight chunks (ping-pong)
    ^wt_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

      // MM2S ch0: output (single buffer)
    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{prefix}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUT_BF16}xbf16>, 0, {OUT_BF16}) {{bd_id = 3 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{prefix}_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    # Runtime sequence — explicit per-chunk writebd (proven pattern from exp 13)
    rt_lines = []
    rt_lines.append(f"    aie.runtime_sequence(%wt_bo: memref<{TOTAL_WT_I32}xi32>, %act_bo: memref<{ACT_I32}xi32>, %out_bo: memref<{OUT_TOTAL_I32}xi32>) {{")

    per_col_wt_bytes = NUM_PHASES * NUM_CHUNKS * FAT_CHUNK_BF16 * 2
    fat_chunk_bytes = FAT_CHUNK_BF16 * 2  # 10240

    for col in range(NUM_COLS):
        # Activation: shim MM2S ch0, bd_id=0
        rt_lines.append(_npu_writebd(col, 0, ACT_I32, 0))
        rt_lines.append(_npu_address_patch(col, 0, 1, 0))
        rt_lines.append(_npu_push_queue(col, "MM2S", 0, 0))

        # Weights: 8 fat chunks per column (unique BD IDs to avoid aliasing)
        base_offset = col * per_col_wt_bytes
        for i in range(NUM_PHASES * NUM_CHUNKS):
            bd_id = 1 + i  # BD IDs 1-8
            offset = base_offset + i * fat_chunk_bytes
            rt_lines.append(_npu_writebd(col, bd_id, FAT_CHUNK_BF16 // 2, offset))
            rt_lines.append(_npu_address_patch(col, bd_id, 0, offset))
            rt_lines.append(_npu_push_queue(col, "MM2S", 1, bd_id))

        # Output: 1 per core (S2MM ch0 for row0, S2MM ch1 for row1)
        is_last = (col == NUM_COLS - 1)

        # core row0 output
        out_off_r0 = (col * ROWS_PER_COL + 0) * OUT_BF16 * 2
        rt_lines.append(_npu_writebd(col, 9, OUT_BF16 // 2, out_off_r0))
        rt_lines.append(_npu_address_patch(col, 9, 2, out_off_r0))
        rt_lines.append(_npu_push_queue(col, "S2MM", 0, 9, issue_token=is_last))

        # core row1 output
        out_off_r1 = (col * ROWS_PER_COL + 1) * OUT_BF16 * 2
        rt_lines.append(_npu_writebd(col, 10, OUT_BF16 // 2, out_off_r1))
        rt_lines.append(_npu_address_patch(col, 10, 2, out_off_r1))
        rt_lines.append(_npu_push_queue(col, "S2MM", 1, 10, issue_token=is_last))

    # Sync on last column
    rt_lines.append(_npu_sync(NUM_COLS - 1, 0))
    rt_lines.append(_npu_sync(NUM_COLS - 1, 1))
    rt_lines.append("    }")
    runtime_sequence = "\n".join(rt_lines)

    # Tile/buffer/lock/flow declarations
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

        # Memtile locks
        locks += f"    %mt{col}_act_empty = aie.lock(%mt{col}, 0) {{init = 2 : i32, sym_name = \"mt{col}_act_empty\"}}\n"
        locks += f"    %mt{col}_act_full  = aie.lock(%mt{col}, 1) {{init = 0 : i32, sym_name = \"mt{col}_act_full\"}}\n"
        locks += f"    %mt{col}_wt_empty  = aie.lock(%mt{col}, 2) {{init = 4 : i32, sym_name = \"mt{col}_wt_empty\"}}\n"
        locks += f"    %mt{col}_wt_full   = aie.lock(%mt{col}, 3) {{init = 0 : i32, sym_name = \"mt{col}_wt_full\"}}\n"

        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            # Core buffers (DMA-connected)
            buffers += f"    %{prefix}_act = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_act\"}} : memref<{ACT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_wt_ping = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_wt_ping\"}} : memref<{CHUNK_BF16}xbf16>\n"
            buffers += f"    %{prefix}_wt_pong = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_wt_pong\"}} : memref<{CHUNK_BF16}xbf16>\n"
            buffers += f"    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_out\"}} : memref<{OUT_BF16}xbf16>\n"
            # Core-LOCAL buffers (NO DMA connection — cross-phase residency!)
            buffers += f"    %{prefix}_gate_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_gate_buf\"}} : memref<{OUT_BF16}xbf16>\n"
            buffers += f"    %{prefix}_up_buf = aie.buffer(%{prefix}) {{sym_name = \"{prefix}_up_buf\"}} : memref<{OUT_BF16}xbf16>\n"

            # Core locks
            locks += f"    %{prefix}_act_empty = aie.lock(%{prefix}, 0) {{init = 1 : i32, sym_name = \"{prefix}_act_empty\"}}\n"
            locks += f"    %{prefix}_act_full  = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = \"{prefix}_act_full\"}}\n"
            locks += f"    %{prefix}_wt_empty  = aie.lock(%{prefix}, 2) {{init = 2 : i32, sym_name = \"{prefix}_wt_empty\"}}\n"
            locks += f"    %{prefix}_wt_full   = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = \"{prefix}_wt_full\"}}\n"
            locks += f"    %{prefix}_out_prod  = aie.lock(%{prefix}, 4) {{init = 1 : i32, sym_name = \"{prefix}_out_prod\"}}\n"
            locks += f"    %{prefix}_out_cons  = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = \"{prefix}_out_cons\"}}\n"

    for col in range(NUM_COLS):
        flows += f"    aie.flow(%shim{col}, DMA : 0, %mt{col}, DMA : 0)\n"
        flows += f"    aie.flow(%shim{col}, DMA : 1, %mt{col}, DMA : 1)\n"
        flows += f"    aie.flow(%mt{col}, DMA : 0, %c{col}r0, DMA : 0)\n"
        flows += f"    aie.flow(%mt{col}, DMA : 3, %c{col}r1, DMA : 0)\n"
        flows += f"    aie.flow(%mt{col}, DMA : 1, %c{col}r0, DMA : 1)\n"
        flows += f"    aie.flow(%mt{col}, DMA : 2, %c{col}r1, DMA : 1)\n"
        flows += f"    aie.flow(%c{col}r0, DMA : 0, %shim{col}, DMA : 0)\n"
        flows += f"    aie.flow(%c{col}r1, DMA : 0, %shim{col}, DMA : 1)\n"

    # Core programs
    core_programs = ""
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            core_programs += _core_program(col, row, prefix) + "\n\n"

    # Memtile DMAs
    memtile_dmas = ""
    for col in range(NUM_COLS):
        prefix = f"mt{col}"
        memtile_dmas += _memtile_dma(col, prefix) + "\n\n"

    # Core MEMs
    core_mems = ""
    for col in range(NUM_COLS):
        for row in range(ROWS_PER_COL):
            prefix = f"c{col}r{row}"
            core_mems += _core_mem(prefix) + "\n\n"

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
    func.func private @q4nx_chunk_accum_offset(memref<{CHUNK_BF16}xbf16>, memref<{ACT_BF16}xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum.o"}}
    func.func private @q4nx_flush_output(memref<{OUT_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/q4nx_chunk_accum.o"}}
    func.func private @swiglu_fused(memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, memref<{OUT_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/swiglu_fused.o"}}

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
