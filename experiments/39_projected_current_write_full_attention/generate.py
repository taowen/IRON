"""Generate raw MLIR-AIE for exp39 projected current-write full attention."""

import sys
from math import ceil
from pathlib import Path

NUM_KV_GROUPS = 8
Q_HEADS_PER_GROUP = 4
NUM_Q_HEADS = NUM_KV_GROUPS * Q_HEADS_PER_GROUP
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

HIDDEN_DIM = 4096
HIDDEN_I32 = HIDDEN_DIM // 2
QUERY_GROUP_DWORDS = Q_HEADS_PER_GROUP * HEAD_DIM
QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_GROUP_DWORDS = QUERY_GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS
STATE_DWORDS = 4


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def last_valid_for_context(context_len: int) -> int:
    return context_len - (num_tiles_for_context(context_len) - 1) * TOKENS_PER_TILE


def kv_group_stride_dwords(context_len: int) -> int:
    return 2 * num_tiles_for_context(context_len) * PLANE_TILE_DWORDS


def current_offsets(context_len: int, group: int) -> tuple[int, int]:
    tile = (context_len - 1) // TOKENS_PER_TILE
    token = (context_len - 1) % TOKENS_PER_TILE
    group_base = group * kv_group_stride_dwords(context_len)
    k_plane_dwords = num_tiles_for_context(context_len) * PLANE_TILE_DWORDS
    k_offset = group_base + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    v_offset = group_base + k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
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


def _npu_sync(column: int, channel: int) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _runtime_sequence(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    group_stride = kv_group_stride_dwords(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    kv_total = NUM_KV_GROUPS * group_stride
    lines = [
        f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_I32}xi32>, "
        f"%kv_cache: memref<{kv_total}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]
    for group in range(NUM_KV_GROUPS):
        current_k_offset, current_v_offset = current_offsets(context_len, group)
        lines += [
            _npu_writebd(group, 0, HIDDEN_I32, 0),
            _npu_address_patch(group, 0, 0, 0),
            _npu_push_queue(group, "MM2S", 0, 0),
            _npu_writebd(group, 14, HEAD_DIM, current_k_offset * 4),
            _npu_address_patch(group, 14, 1, current_k_offset * 4),
            _npu_push_queue(group, "S2MM", 0, 14),
            _npu_writebd(group, 15, HEAD_DIM, current_v_offset * 4),
            _npu_address_patch(group, 15, 1, current_v_offset * 4),
            _npu_push_queue(group, "S2MM", 0, 15, issue_token=True),
            _npu_sync(group, 0),
        ]

        for tile in range(num_tiles):
            group_base = group * group_stride
            k_offset = (group_base + tile * PLANE_TILE_DWORDS) * 4
            v_offset = (group_base + k_plane_dwords + tile * PLANE_TILE_DWORDS) * 4
            k_bd = 1 + tile
            v_bd = 8 + tile
            lines += [
                _npu_writebd(group, k_bd, PLANE_TILE_DWORDS, k_offset),
                _npu_address_patch(group, k_bd, 1, k_offset),
                _npu_push_queue(group, "MM2S", 0, k_bd),
                _npu_writebd(group, v_bd, PLANE_TILE_DWORDS, v_offset),
                _npu_address_patch(group, v_bd, 1, v_offset),
                _npu_push_queue(group, "MM2S", 1, v_bd),
            ]

        output_offset = group * OUTPUT_GROUP_DWORDS * 4
        lines += [
            _npu_writebd(group, 13, OUTPUT_GROUP_DWORDS, output_offset),
            _npu_address_patch(group, 13, 2, output_offset),
            _npu_push_queue(group, "S2MM", 1, 13, issue_token=True),
            _npu_sync(group, 1),
        ]
    lines.append("    }")
    return "\n".join(lines)


def _worker_buffers_and_locks(column: int, row: int) -> str:
    prefix = f"w{column}_{row}"
    current = ""
    if row == 0:
        current = f"""
    %{prefix}_current_k = aie.buffer(%{prefix}) {{sym_name = "{prefix}_current_k"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_current_v = aie.buffer(%{prefix}) {{sym_name = "{prefix}_current_v"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_current_k_empty = aie.lock(%{prefix}, 10) {{init = 1 : i32, sym_name = "{prefix}_current_k_empty"}}
    %{prefix}_current_k_full = aie.lock(%{prefix}, 11) {{init = 0 : i32, sym_name = "{prefix}_current_k_full"}}
    %{prefix}_current_v_empty = aie.lock(%{prefix}, 12) {{init = 1 : i32, sym_name = "{prefix}_current_v_empty"}}
    %{prefix}_current_v_full = aie.lock(%{prefix}, 13) {{init = 0 : i32, sym_name = "{prefix}_current_v_full"}}
"""
    return f"""
    %{prefix}_hidden = aie.buffer(%{prefix}) {{sym_name = "{prefix}_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %{prefix}_query = aie.buffer(%{prefix}) {{sym_name = "{prefix}_query"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_k_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_v_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = "{prefix}_out"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_running_max = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_max"}} : memref<{STATE_DWORDS}xf32>
    %{prefix}_running_sum = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_sum"}} : memref<{STATE_DWORDS}xf32>
{current}
    %{prefix}_hidden_empty = aie.lock(%{prefix}, 8) {{init = 1 : i32, sym_name = "{prefix}_hidden_empty"}}
    %{prefix}_hidden_full = aie.lock(%{prefix}, 9) {{init = 0 : i32, sym_name = "{prefix}_hidden_full"}}
    %{prefix}_k_empty = aie.lock(%{prefix}, 0) {{init = 1 : i32, sym_name = "{prefix}_k_empty"}}
    %{prefix}_k_full = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = "{prefix}_k_full"}}
    %{prefix}_v_empty = aie.lock(%{prefix}, 2) {{init = 1 : i32, sym_name = "{prefix}_v_empty"}}
    %{prefix}_v_full = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = "{prefix}_v_full"}}
    %{prefix}_out_empty = aie.lock(%{prefix}, 4) {{init = 1 : i32, sym_name = "{prefix}_out_empty"}}
    %{prefix}_out_full = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = "{prefix}_out_full"}}
"""


def _memtile_buffers_and_locks(column: int) -> str:
    prefix = f"mem{column}"
    return f"""
    %{prefix}_hidden = aie.buffer(%{prefix}) {{sym_name = "{prefix}_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %{prefix}_k_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_k_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_v_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_v_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = "{prefix}_out"}} : memref<{OUTPUT_GROUP_DWORDS}xf32>

    %{prefix}_hidden_empty = aie.lock(%{prefix}, 8) {{init = {Q_HEADS_PER_GROUP} : i32, sym_name = "{prefix}_hidden_empty"}}
    %{prefix}_hidden_full = aie.lock(%{prefix}, 9) {{init = 0 : i32, sym_name = "{prefix}_hidden_full"}}
    %{prefix}_k_empty = aie.lock(%{prefix}, 0) {{init = {2 * Q_HEADS_PER_GROUP} : i32, sym_name = "{prefix}_k_empty"}}
    %{prefix}_k_full = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = "{prefix}_k_full"}}
    %{prefix}_v_empty = aie.lock(%{prefix}, 2) {{init = {2 * Q_HEADS_PER_GROUP} : i32, sym_name = "{prefix}_v_empty"}}
    %{prefix}_v_full = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = "{prefix}_v_full"}}
    %{prefix}_out_empty = aie.lock(%{prefix}, 4) {{init = {Q_HEADS_PER_GROUP} : i32, sym_name = "{prefix}_out_empty"}}
    %{prefix}_out_full = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = "{prefix}_out_full"}}
"""


def _worker_core(column: int, row: int, num_tiles: int, last_valid: int) -> str:
    prefix = f"w{column}_{row}"
    current = ""
    if row == 0:
        current = f"""
      aie.use_lock(%{prefix}_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%{prefix}_current_v_empty, AcquireGreaterEqual, 1)
      func.call @project_current_from_hidden(%{prefix}_hidden, %{prefix}_current_k, %{prefix}_current_v, %group_i32)
        : (memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, memref<{HEAD_DIM}xf32>, i32) -> ()
      aie.use_lock(%{prefix}_current_k_full, Release, 1)
      aie.use_lock(%{prefix}_current_v_full, Release, 1)
"""
    return f"""    %{prefix}_core = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %group_i32 = arith.constant {column} : i32
      %row_i32 = arith.constant {row} : i32
      %num_tiles_idx = arith.constant {num_tiles} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      aie.use_lock(%{prefix}_hidden_full, AcquireGreaterEqual, 1)
      func.call @project_query_from_hidden(%{prefix}_hidden, %{prefix}_query, %group_i32, %row_i32)
        : (memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, i32, i32) -> ()
{current}
      aie.use_lock(%{prefix}_hidden_empty, Release, 1)

      aie.use_lock(%{prefix}_out_empty, AcquireGreaterEqual, 1)
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
      aie.use_lock(%{prefix}_out_full, Release, 1)
      aie.end
    }}"""


def _worker_mem(column: int, row: int) -> str:
    prefix = f"w{column}_{row}"
    pkt_id = column * Q_HEADS_PER_GROUP + row
    current_dma = ""
    next_after_input = "^out_start"
    if row == 0:
        next_after_input = "^current_start"
        current_dma = f"""
    ^current_start:
      %current_dma = aie.dma_start(MM2S, 1, ^current_k_out, ^out_start)
    ^current_k_out:
      aie.use_lock(%{prefix}_current_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_current_k : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{prefix}_current_k_empty, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%{prefix}_current_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_current_v : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 5 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{prefix}_current_v_empty, Release, 1)
      aie.next_bd ^current_k_out
"""
    return f"""    %{prefix}_mem = aie.mem(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, {next_after_input})
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
{current_dma}
    ^out_start:
      %out_dma = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{prefix}_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {pkt_id}>}}
      aie.use_lock(%{prefix}_out_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _hidden_and_kv_out_channel(channel: int, column: int, bds: tuple[int, int, int, int, int]) -> str:
    prefix = f"mem{column}"
    hidden_bd, k_ping_bd, v_ping_bd, k_pong_bd, v_pong_bd = bds
    dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    return f"""    ^out{channel}_start:
      %out{channel}_dma = aie.dma_start(MM2S, {channel}, ^out{channel}_hidden, ^out{channel + 1}_start)
    ^out{channel}_hidden:
      aie.use_lock(%{prefix}_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = {hidden_bd} : i32, next_bd_id = {k_ping_bd} : i32}}
      aie.use_lock(%{prefix}_hidden_empty, Release, 1)
      aie.next_bd ^out{channel}_k_ping
    ^out{channel}_k_ping:
      aie.use_lock(%{prefix}_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {k_ping_bd} : i32, next_bd_id = {v_ping_bd} : i32}}
      aie.use_lock(%{prefix}_k_empty, Release, 1)
      aie.next_bd ^out{channel}_v_ping
    ^out{channel}_v_ping:
      aie.use_lock(%{prefix}_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {v_ping_bd} : i32, next_bd_id = {k_pong_bd} : i32}}
      aie.use_lock(%{prefix}_v_empty, Release, 1)
      aie.next_bd ^out{channel}_k_pong
    ^out{channel}_k_pong:
      aie.use_lock(%{prefix}_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {k_pong_bd} : i32, next_bd_id = {v_pong_bd} : i32}}
      aie.use_lock(%{prefix}_k_empty, Release, 1)
      aie.next_bd ^out{channel}_v_pong
    ^out{channel}_v_pong:
      aie.use_lock(%{prefix}_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = {v_pong_bd} : i32, next_bd_id = {k_ping_bd} : i32}}
      aie.use_lock(%{prefix}_v_empty, Release, 1)
      aie.next_bd ^out{channel}_k_ping"""


def _memtile_dma(column: int) -> str:
    prefix = f"mem{column}"
    channel_bds = {
        0: (10, 11, 12, 13, 14),
        1: (26, 27, 28, 29, 30),
        2: (15, 16, 17, 18, 19),
        3: (31, 32, 33, 34, 35),
    }
    outputs = "\n".join(_hidden_and_kv_out_channel(ch, column, channel_bds[ch]) for ch in range(Q_HEADS_PER_GROUP))
    return f"""    %{prefix}_dma = aie.memtile_dma(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^v_in_start)
    ^hidden_in:
      aie.use_lock(%{prefix}_hidden_empty, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_hidden_full, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^k_in_ping
    ^k_in_ping:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^k_in_pong
    ^k_in_pong:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^k_in_ping

    ^v_in_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in_ping, ^output_collect_start)
    ^v_in_ping:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^v_in_pong
    ^v_in_pong:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^v_in_ping

    ^output_collect_start:
      %out_collect_dma = aie.dma_start(S2MM, 2, ^out_collect0, ^out0_start)
    ^out_collect0:
      aie.use_lock(%{prefix}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUTPUT_GROUP_DWORDS}xf32>, 0, {HEAD_DIM}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{prefix}_out_full, Release, 1)
      aie.next_bd ^out_collect1
    ^out_collect1:
      aie.use_lock(%{prefix}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUTPUT_GROUP_DWORDS}xf32>, {HEAD_DIM}, {HEAD_DIM}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{prefix}_out_full, Release, 1)
      aie.next_bd ^out_collect2
    ^out_collect2:
      aie.use_lock(%{prefix}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUTPUT_GROUP_DWORDS}xf32>, {2 * HEAD_DIM}, {HEAD_DIM}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%{prefix}_out_full, Release, 1)
      aie.next_bd ^out_collect3
    ^out_collect3:
      aie.use_lock(%{prefix}_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{OUTPUT_GROUP_DWORDS}xf32>, {3 * HEAD_DIM}, {HEAD_DIM}) {{bd_id = 6 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{prefix}_out_full, Release, 1)
      aie.next_bd ^out_collect0

{outputs}
    ^out4_start:
      %output_drain_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{prefix}_out_full, AcquireGreaterEqual, {Q_HEADS_PER_GROUP})
      aie.dma_bd(%{prefix}_out : memref<{OUTPUT_GROUP_DWORDS}xf32>, 0, {OUTPUT_GROUP_DWORDS}) {{bd_id = 36 : i32}}
      aie.use_lock(%{prefix}_out_empty, Release, {Q_HEADS_PER_GROUP})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}"""


def generate_mlir(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    last_valid = last_valid_for_context(context_len)
    experiment_dir = Path(__file__).parent.resolve()

    tiles = []
    for col in range(NUM_KV_GROUPS):
        tiles.append(f"    %shim{col} = aie.tile({col}, 0)")
        tiles.append(f"    %mem{col} = aie.tile({col}, 1)")
        for row in range(Q_HEADS_PER_GROUP):
            tiles.append(f"    %w{col}_{row} = aie.tile({col}, {row + 2})")
    tile_defs = "\n".join(tiles)

    buffers = []
    for col in range(NUM_KV_GROUPS):
        buffers.append(_memtile_buffers_and_locks(col))
        for row in range(Q_HEADS_PER_GROUP):
            buffers.append(_worker_buffers_and_locks(col, row))
    buffer_defs = "\n".join(buffers)

    flows = []
    for col in range(NUM_KV_GROUPS):
        flows.append(f"    aie.flow(%shim{col}, DMA : 0, %mem{col}, DMA : 0)")
        flows.append(f"    aie.flow(%shim{col}, DMA : 1, %mem{col}, DMA : 1)")
        flows.append(f"    aie.flow(%w{col}_0, DMA : 1, %shim{col}, DMA : 0)")
        for row in range(Q_HEADS_PER_GROUP):
            flows.append(f"    aie.flow(%mem{col}, DMA : {row}, %w{col}_{row}, DMA : 0)")
            pkt_id = col * Q_HEADS_PER_GROUP + row
            flows.append(f"    aie.packet_flow({pkt_id}) {{")
            flows.append(f"      aie.packet_source<%w{col}_{row}, DMA : 0>")
            flows.append(f"      aie.packet_dest<%mem{col}, DMA : 2>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mem{col}, DMA : 5, %shim{col}, DMA : 1)")
    flow_defs = "\n".join(flows)

    cores = []
    worker_mems = []
    for col in range(NUM_KV_GROUPS):
        for row in range(Q_HEADS_PER_GROUP):
            cores.append(_worker_core(col, row, num_tiles, last_valid))
            worker_mems.append(_worker_mem(col, row))
    core_defs = "\n\n".join(cores)
    worker_mem_defs = "\n\n".join(worker_mems)
    memtile_defs = "\n\n".join(_memtile_dma(col) for col in range(NUM_KV_GROUPS))

    return f"""module {{
  aie.device(npu2) {{
{tile_defs}

{buffer_defs}

{flow_defs}

    func.func private @project_query_from_hidden(memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, i32, i32) attributes {{link_with = "{experiment_dir}/projected_attention.o"}}
    func.func private @project_current_from_hidden(memref<{HIDDEN_DIM}xbf16>, memref<{HEAD_DIM}xf32>, memref<{HEAD_DIM}xf32>, i32) attributes {{link_with = "{experiment_dir}/projected_attention.o"}}
    func.func private @init_attention_state_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/projected_attention.o"}}
    func.func private @online_attention_reshaped_head(memref<{HEAD_DIM}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/projected_attention.o"}}
    func.func private @finalize_attention_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/projected_attention.o"}}

{core_defs}

{memtile_defs}

{worker_mem_defs}

{_runtime_sequence(context_len)}
  }}
}}
"""


if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
