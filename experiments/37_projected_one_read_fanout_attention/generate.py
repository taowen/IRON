"""Generate raw MLIR-AIE for exp37 projected one-read fanout attention."""

import sys
from math import ceil
from pathlib import Path

NUM_Q_HEADS = 4
NUM_COLS = NUM_Q_HEADS
KV_COL = 4
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

HIDDEN_DIM = 4096
HIDDEN_I32 = HIDDEN_DIM // 2
Q4_K_CHUNK = 256
Q4_CHUNKS = HIDDEN_DIM // Q4_K_CHUNK
GROUP_SIZE = 32
Q4_ROWS = 32
HEAD_ROW_BLOCKS = HEAD_DIM // Q4_ROWS

CHUNK_BF16 = 2560
ROWBLOCK_BF16 = Q4_CHUNKS * CHUNK_BF16
HEAD_PHASE_BF16 = HEAD_ROW_BLOCKS * ROWBLOCK_BF16
HEAD_PHASE_I32 = HEAD_PHASE_BF16 // 2
TOTAL_WEIGHT_PHASES = NUM_Q_HEADS + 2
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_PHASES * HEAD_PHASE_I32

QUERY_HEAD_DWORDS = HEAD_DIM
QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS
STATE_DWORDS = 4


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def last_valid_for_context(context_len: int) -> int:
    return context_len - (num_tiles_for_context(context_len) - 1) * TOKENS_PER_TILE


def current_offsets(context_len: int) -> tuple[int, int]:
    tile = (context_len - 1) // TOKENS_PER_TILE
    token = (context_len - 1) % TOKENS_PER_TILE
    k_plane_dwords = num_tiles_for_context(context_len) * PLANE_TILE_DWORDS
    k_offset = tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    v_offset = k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    return k_offset, v_offset


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 0 : i32, column = {column} : i32, "
        f"d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def _npu_push_queue(column: int, direction: str, channel: int, bd_id: int, issue_token: bool = False) -> str:
    token = "true" if issue_token else "false"
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"
    )


def _npu_sync(column: int, channel: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _runtime_sequence(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    total_kv_dwords = 2 * k_plane_dwords
    current_k_offset, current_v_offset = current_offsets(context_len)
    lines = [
        f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_I32}xi32>, "
        f"%qkv_weight: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%kv_cache: memref<{total_kv_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]

    lines += [
        _npu_writebd(KV_COL, 0, HIDDEN_I32, 0),
        _npu_address_patch(KV_COL, 0, 0, 0),
        _npu_push_queue(KV_COL, "MM2S", 0, 0),
    ]

    lines += [
        _npu_writebd(0, 13, HEAD_DIM, current_k_offset * 4),
        _npu_address_patch(0, 13, 2, current_k_offset * 4),
        _npu_push_queue(0, "S2MM", 0, 13),
        _npu_writebd(0, 14, HEAD_DIM, current_v_offset * 4),
        _npu_address_patch(0, 14, 2, current_v_offset * 4),
        _npu_push_queue(0, "S2MM", 0, 14, issue_token=True),
    ]

    for head in range(NUM_Q_HEADS):
        weight_offset = head * HEAD_PHASE_BF16 * 2
        lines += [
            _npu_writebd(head, 1, HEAD_PHASE_I32, weight_offset),
            _npu_address_patch(head, 1, 1, weight_offset),
            _npu_push_queue(head, "MM2S", 1, 1, issue_token=True),
        ]

    k_weight_offset = NUM_Q_HEADS * HEAD_PHASE_BF16 * 2
    v_weight_offset = (NUM_Q_HEADS + 1) * HEAD_PHASE_BF16 * 2
    lines += [
        _npu_writebd(0, 2, HEAD_PHASE_I32, k_weight_offset),
        _npu_address_patch(0, 2, 1, k_weight_offset),
        _npu_push_queue(0, "MM2S", 1, 2, issue_token=True),
        _npu_writebd(0, 3, HEAD_PHASE_I32, v_weight_offset),
        _npu_address_patch(0, 3, 1, v_weight_offset),
        _npu_push_queue(0, "MM2S", 1, 3, issue_token=True),
        _npu_sync(0, 0),
    ]

    for tile in range(num_tiles):
        k_offset = tile * PLANE_TILE_DWORDS * 4
        v_offset = (k_plane_dwords + tile * PLANE_TILE_DWORDS) * 4
        k_bd = 4 + tile
        v_bd = 8 + tile
        lines += [
            _npu_writebd(KV_COL, k_bd, PLANE_TILE_DWORDS, k_offset),
            _npu_address_patch(KV_COL, k_bd, 2, k_offset),
            _npu_push_queue(KV_COL, "MM2S", 0, k_bd),
            _npu_writebd(KV_COL, v_bd, PLANE_TILE_DWORDS, v_offset),
            _npu_address_patch(KV_COL, v_bd, 2, v_offset),
            _npu_push_queue(KV_COL, "MM2S", 1, v_bd),
        ]

    for column in range(NUM_COLS):
        output_offset = column * HEAD_DIM * 4
        lines += [
            _npu_writebd(column, 15, HEAD_DIM, output_offset),
            _npu_address_patch(column, 15, 3, output_offset),
            _npu_push_queue(column, "S2MM", 0, 15, issue_token=True),
        ]

    for column in range(NUM_COLS):
        lines.append(_npu_sync(column, 0))
    lines.append("    }")
    return "\n".join(lines)


def _projection_loop(prefix: str, name: str, target: str) -> str:
    return f"""
      scf.for %rb_{name}_{prefix} = %c0 to %head_rowblocks_idx step %c1 {{
        %rb_{name}_{prefix}_i32 = arith.index_cast %rb_{name}_{prefix} : index to i32
        %out_{name}_{prefix}_offset = arith.muli %rb_{name}_{prefix}_i32, %c32_i32 : i32

        scf.for %chunk_{name}_{prefix} = %c0 to %q4_chunks_idx step %c1 {{
          %chunk_{name}_{prefix}_i32 = arith.index_cast %chunk_{name}_{prefix} : index to i32
          %act_{name}_{prefix}_offset = arith.muli %chunk_{name}_{prefix}_i32, %k_chunk_i32 : i32
          %rem_{name}_{prefix} = arith.remsi %chunk_{name}_{prefix}_i32, %c2_i32 : i32
          %is_pong_{name}_{prefix} = arith.cmpi eq, %rem_{name}_{prefix}, %c1_i32 : i32

          aie.use_lock(%{prefix}_wt_full, AcquireGreaterEqual, 1)
          scf.if %is_pong_{name}_{prefix} {{
            func.call @q4nx_project_head_chunk(%{prefix}_wt_pong, %{prefix}_hidden, %{target}, %out_{name}_{prefix}_offset, %act_{name}_{prefix}_offset)
              : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, i32, i32) -> ()
          }} else {{
            func.call @q4nx_project_head_chunk(%{prefix}_wt_ping, %{prefix}_hidden, %{target}, %out_{name}_{prefix}_offset, %act_{name}_{prefix}_offset)
              : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, i32, i32) -> ()
          }}
          aie.use_lock(%{prefix}_wt_empty, Release, 1)
        }}
      }}"""


def _memtile_buffers_and_locks(column: int) -> str:
    prefix = f"mem{column}"
    return f"""
    %{prefix}_wt_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{prefix}_wt_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>

    %{prefix}_wt_empty = aie.lock(%{prefix}, 10) {{init = 2 : i32, sym_name = "{prefix}_wt_empty"}}
    %{prefix}_wt_full  = aie.lock(%{prefix}, 11) {{init = 0 : i32, sym_name = "{prefix}_wt_full"}}
"""


def _kv_memtile_buffers_and_locks() -> str:
    return f"""
    %kvmem_hidden = aie.buffer(%kvmem) {{sym_name = "kvmem_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %kvmem_k_ping = aie.buffer(%kvmem) {{sym_name = "kvmem_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %kvmem_k_pong = aie.buffer(%kvmem) {{sym_name = "kvmem_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %kvmem_v_ping = aie.buffer(%kvmem) {{sym_name = "kvmem_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %kvmem_v_pong = aie.buffer(%kvmem) {{sym_name = "kvmem_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>

    %kvmem_hidden_empty = aie.lock(%kvmem, 8) {{init = 4 : i32, sym_name = "kvmem_hidden_empty"}}
    %kvmem_hidden_full = aie.lock(%kvmem, 9) {{init = 0 : i32, sym_name = "kvmem_hidden_full"}}
    %kvmem_k_empty = aie.lock(%kvmem, 0) {{init = 8 : i32, sym_name = "kvmem_k_empty"}}
    %kvmem_k_full = aie.lock(%kvmem, 1) {{init = 0 : i32, sym_name = "kvmem_k_full"}}
    %kvmem_v_empty = aie.lock(%kvmem, 2) {{init = 8 : i32, sym_name = "kvmem_v_empty"}}
    %kvmem_v_full = aie.lock(%kvmem, 3) {{init = 0 : i32, sym_name = "kvmem_v_full"}}
"""


def _worker_buffers_and_locks(column: int) -> str:
    prefix = f"w{column}"
    return f"""
    %{prefix}_hidden = aie.buffer(%{prefix}) {{sym_name = "{prefix}_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %{prefix}_wt_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{prefix}_wt_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{prefix}_query = aie.buffer(%{prefix}) {{sym_name = "{prefix}_query"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_current_k = aie.buffer(%{prefix}) {{sym_name = "{prefix}_current_k"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_current_v = aie.buffer(%{prefix}) {{sym_name = "{prefix}_current_v"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_current_out = aie.buffer(%{prefix}) {{sym_name = "{prefix}_current_out"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_k_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_v_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = "{prefix}_out"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_running_max = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_max"}} : memref<{STATE_DWORDS}xf32>
    %{prefix}_running_sum = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_sum"}} : memref<{STATE_DWORDS}xf32>

    %{prefix}_hidden_empty = aie.lock(%{prefix}, 12) {{init = 1 : i32, sym_name = "{prefix}_hidden_empty"}}
    %{prefix}_hidden_full  = aie.lock(%{prefix}, 13) {{init = 0 : i32, sym_name = "{prefix}_hidden_full"}}
    %{prefix}_wt_empty = aie.lock(%{prefix}, 14) {{init = 2 : i32, sym_name = "{prefix}_wt_empty"}}
    %{prefix}_wt_full  = aie.lock(%{prefix}, 15) {{init = 0 : i32, sym_name = "{prefix}_wt_full"}}
    %{prefix}_k_empty = aie.lock(%{prefix}, 0) {{init = 1 : i32, sym_name = "{prefix}_k_empty"}}
    %{prefix}_k_full  = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = "{prefix}_k_full"}}
    %{prefix}_v_empty = aie.lock(%{prefix}, 2) {{init = 1 : i32, sym_name = "{prefix}_v_empty"}}
    %{prefix}_v_full  = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = "{prefix}_v_full"}}
    %{prefix}_out_prod = aie.lock(%{prefix}, 4) {{init = 1 : i32, sym_name = "{prefix}_out_prod"}}
    %{prefix}_out_cons = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = "{prefix}_out_cons"}}
    %{prefix}_current_out_prod = aie.lock(%{prefix}, 10) {{init = 1 : i32, sym_name = "{prefix}_current_out_prod"}}
    %{prefix}_current_out_cons = aie.lock(%{prefix}, 11) {{init = 0 : i32, sym_name = "{prefix}_current_out_cons"}}
"""


def _worker_core(column: int, num_tiles: int, last_valid: int) -> str:
    prefix = f"w{column}"
    q_loop = _projection_loop(prefix, "q", f"{prefix}_query")
    k_v_loops = ""
    current_write = ""
    if column == 0:
        k_loop = _projection_loop(prefix, "k", f"{prefix}_current_k")
        v_loop = _projection_loop(prefix, "v", f"{prefix}_current_v")
        k_v_loops = f"""
      func.call @zero_head(%{prefix}_current_k) : (memref<{HEAD_DIM}xf32>) -> ()
      func.call @zero_head(%{prefix}_current_v) : (memref<{HEAD_DIM}xf32>) -> ()
{k_loop}
{v_loop}"""
        current_write = f"""
      aie.use_lock(%{prefix}_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_head(%{prefix}_current_k, %{prefix}_current_out)
        : (memref<{HEAD_DIM}xf32>, memref<{HEAD_DIM}xf32>) -> ()
      aie.use_lock(%{prefix}_current_out_cons, Release, 1)

      aie.use_lock(%{prefix}_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_head(%{prefix}_current_v, %{prefix}_current_out)
        : (memref<{HEAD_DIM}xf32>, memref<{HEAD_DIM}xf32>) -> ()
      aie.use_lock(%{prefix}_current_out_cons, Release, 1)"""

    return f"""    %{prefix}_core = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c32_i32 = arith.constant {Q4_ROWS} : i32
      %k_chunk_i32 = arith.constant {Q4_K_CHUNK} : i32
      %q4_chunks_idx = arith.constant {Q4_CHUNKS} : index
      %head_rowblocks_idx = arith.constant {HEAD_ROW_BLOCKS} : index
      %num_tiles_idx = arith.constant {num_tiles} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      aie.use_lock(%{prefix}_hidden_full, AcquireGreaterEqual, 1)
      func.call @zero_head(%{prefix}_query) : (memref<{HEAD_DIM}xf32>) -> ()
{q_loop}
{k_v_loops}
      aie.use_lock(%{prefix}_hidden_empty, Release, 1)
{current_write}

      aie.use_lock(%{prefix}_out_prod, AcquireGreaterEqual, 1)
      func.call @init_attention_state_head(%{prefix}_out, %{prefix}_running_max, %{prefix}_running_sum)
        : (memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        aie.use_lock(%{prefix}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{prefix}_v_full, AcquireGreaterEqual, 1)
        func.call @online_attention_reshaped_head(%{prefix}_query, %{prefix}_k_tile, %{prefix}_v_tile, %{prefix}_out, %{prefix}_running_max, %{prefix}_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
          : (memref<{HEAD_DIM}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>, i32, i32, i32) -> ()
        aie.use_lock(%{prefix}_k_empty, Release, 1)
        aie.use_lock(%{prefix}_v_empty, Release, 1)
      }}

      func.call @finalize_attention_head(%{prefix}_out, %{prefix}_running_sum)
        : (memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>) -> ()
      aie.use_lock(%{prefix}_out_cons, Release, 1)
      aie.end
    }}"""


def _memtile_dma(column: int) -> str:
    prefix = f"mem{column}"
    return f"""    %{prefix}_dma = aie.memtile_dma(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 1, ^wt_in_ping, ^weight_out_start)
    ^wt_in_ping:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_in_pong
    ^wt_in_pong:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_in_ping

    ^weight_out_start:
      %1 = aie.dma_start(MM2S, 1, ^wt_out_ping, ^end)
    ^wt_out_ping:
      aie.use_lock(%{prefix}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%{prefix}_wt_empty, Release, 1)
      aie.next_bd ^wt_out_pong
    ^wt_out_pong:
      aie.use_lock(%{prefix}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%{prefix}_wt_empty, Release, 1)
      aie.next_bd ^wt_out_ping
    ^end:
      aie.end
    }}"""


def _kv_out_channel(channel: int, worker_column: int, bd_ids: tuple[int, int, int, int]) -> str:
    dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    k_ping_bd, v_ping_bd, k_pong_bd, v_pong_bd = bd_ids
    return f"""    ^out{worker_column}_start:
      %{worker_column + 2} = aie.dma_start(MM2S, {channel}, ^out{worker_column}_hidden, ^out{worker_column + 1}_start)
    ^out{worker_column}_hidden:
      aie.use_lock(%kvmem_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%kvmem_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = {k_ping_bd + 12} : i32, next_bd_id = {k_ping_bd} : i32}}
      aie.use_lock(%kvmem_hidden_empty, Release, 1)
      aie.next_bd ^out{worker_column}_k_ping
    ^out{worker_column}_k_ping:
      aie.use_lock(%kvmem_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%kvmem_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {k_ping_bd} : i32, next_bd_id = {v_ping_bd} : i32}}
      aie.use_lock(%kvmem_k_empty, Release, 1)
      aie.next_bd ^out{worker_column}_v_ping
    ^out{worker_column}_v_ping:
      aie.use_lock(%kvmem_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%kvmem_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {v_ping_bd} : i32, next_bd_id = {k_pong_bd} : i32}}
      aie.use_lock(%kvmem_v_empty, Release, 1)
      aie.next_bd ^out{worker_column}_k_pong
    ^out{worker_column}_k_pong:
      aie.use_lock(%kvmem_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%kvmem_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {k_pong_bd} : i32, next_bd_id = {v_pong_bd} : i32}}
      aie.use_lock(%kvmem_k_empty, Release, 1)
      aie.next_bd ^out{worker_column}_v_pong
    ^out{worker_column}_v_pong:
      aie.use_lock(%kvmem_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%kvmem_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {v_pong_bd} : i32, next_bd_id = {k_ping_bd} : i32}}
      aie.use_lock(%kvmem_v_empty, Release, 1)
      aie.next_bd ^out{worker_column}_k_ping"""


def _kv_memtile_dma() -> str:
    channel_bds = {
        0: (2, 3, 4, 5),
        1: (26, 27, 28, 29),
        2: (6, 7, 8, 9),
        3: (30, 31, 32, 33),
    }
    outputs = "\n".join(
        _kv_out_channel(channel, channel, channel_bds[channel])
        for channel in range(NUM_COLS)
    )
    return f"""    %kvmem_dma = aie.memtile_dma(%kvmem) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^v_in_start)
    ^hidden_in:
      aie.use_lock(%kvmem_hidden_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%kvmem_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 10 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%kvmem_hidden_full, Release, 4)
      aie.next_bd ^k_in_ping
    ^k_in_ping:
      aie.use_lock(%kvmem_k_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%kvmem_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%kvmem_k_full, Release, 4)
      aie.next_bd ^k_in_pong
    ^k_in_pong:
      aie.use_lock(%kvmem_k_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%kvmem_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%kvmem_k_full, Release, 4)
      aie.next_bd ^k_in_ping

    ^v_in_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in_ping, ^out0_start)
    ^v_in_ping:
      aie.use_lock(%kvmem_v_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%kvmem_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%kvmem_v_full, Release, 4)
      aie.next_bd ^v_in_pong
    ^v_in_pong:
      aie.use_lock(%kvmem_v_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%kvmem_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%kvmem_v_full, Release, 4)
      aie.next_bd ^v_in_ping

{outputs}
    ^out4_start:
      aie.end
    }}"""


def _worker_mem(column: int) -> str:
    prefix = f"w{column}"
    current_path = ""
    if column == 0:
        current_path = f"""
    ^current_k_out:
      aie.use_lock(%{prefix}_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_current_out : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%{prefix}_current_out_prod, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%{prefix}_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_current_out : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 6 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%{prefix}_current_out_prod, Release, 1)
      aie.next_bd ^out_bd"""
        out_start = "^current_k_out"
    else:
        out_start = "^out_bd"

    return f"""    %{prefix}_mem = aie.mem(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^weight_start)
    ^hidden_in:
      aie.use_lock(%{prefix}_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_hidden_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, 1)
      aie.next_bd ^v_in
    ^v_in:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, 1)
      aie.next_bd ^k_in

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{prefix}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 4 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{prefix}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, {out_start}, ^end)
{current_path}
    ^out_bd:
      aie.use_lock(%{prefix}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 7 : i32}}
      aie.use_lock(%{prefix}_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def generate_mlir(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    last_valid = last_valid_for_context(context_len)
    experiment_dir = Path(__file__).parent.resolve()

    tile_defs = "\n".join(
        f"    %shim{column} = aie.tile({column}, 0)\n"
        f"    %mem{column} = aie.tile({column}, 1)\n"
        f"    %w{column} = aie.tile({column}, 2)"
        for column in range(NUM_COLS)
    )
    tile_defs += (
        f"\n    %kvshim = aie.tile({KV_COL}, 0)\n"
        f"    %kvmem = aie.tile({KV_COL}, 1)"
    )
    buffers = "\n".join(
        _memtile_buffers_and_locks(column) + _worker_buffers_and_locks(column)
        for column in range(NUM_COLS)
    )
    flows = "\n".join(
        f"""    aie.flow(%shim{column}, DMA : 1, %mem{column}, DMA : 1)
    aie.flow(%mem{column}, DMA : 1, %w{column}, DMA : 1)
    aie.flow(%w{column}, DMA : 0, %shim{column}, DMA : 0)"""
        for column in range(NUM_COLS)
    )
    flows += (
        "\n    aie.flow(%kvshim, DMA : 0, %kvmem, DMA : 0)"
        "\n    aie.flow(%kvshim, DMA : 1, %kvmem, DMA : 1)"
    )
    flows += "\n" + "\n".join(
        f"    aie.flow(%kvmem, DMA : {column}, %w{column}, DMA : 0)"
        for column in range(NUM_COLS)
    )

    cores = "\n\n".join(_worker_core(column, num_tiles, last_valid) for column in range(NUM_COLS))
    memtiles = "\n\n".join(_memtile_dma(column) for column in range(NUM_COLS))
    worker_mems = "\n\n".join(_worker_mem(column) for column in range(NUM_COLS))
    return f"""module {{
  aie.device(npu2) {{
{tile_defs}

{buffers}
{_kv_memtile_buffers_and_locks()}

{flows}

    func.func private @zero_head(memref<{HEAD_DIM}xf32>) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}
    func.func private @q4nx_project_head_chunk(memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}
    func.func private @copy_head(memref<{HEAD_DIM}xf32>, memref<{HEAD_DIM}xf32>) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}
    func.func private @init_attention_state_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}
    func.func private @online_attention_reshaped_head(memref<{HEAD_DIM}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}
    func.func private @finalize_attention_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_reshape_attention.o"}}

{cores}

{memtiles}

{_kv_memtile_dma()}

{worker_mems}

{_runtime_sequence(context_len)}
  }}
}}
"""


if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
