"""Generate raw MLIR-AIE for exp31 full GQA attention group."""

from math import ceil
from pathlib import Path

NUM_Q_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_TILE = 16

QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
KV_DWORDS = HEAD_DIM
KV_PLANE_TILE_DWORDS = KV_DWORDS * TOKENS_PER_TILE
KV_TILE_DWORDS = KV_PLANE_TILE_DWORDS * 2
OUTPUT_DWORDS = QUERY_DWORDS

HIDDEN_DIM = 4096
HIDDEN_I32 = HIDDEN_DIM // 2
Q4_K_CHUNK = 256
Q4_CHUNKS = HIDDEN_DIM // Q4_K_CHUNK
GROUP_SIZE = 32
Q4_ROWS = 32
Q_ROW_BLOCKS = QUERY_DWORDS // Q4_ROWS
KV_ROW_BLOCKS = KV_DWORDS // Q4_ROWS

CHUNK_BF16 = 2560
ROWBLOCK_BF16 = Q4_CHUNKS * CHUNK_BF16
Q_PHASE_BF16 = Q_ROW_BLOCKS * ROWBLOCK_BF16
KV_PHASE_BF16 = KV_ROW_BLOCKS * ROWBLOCK_BF16
Q_PHASE_I32 = Q_PHASE_BF16 // 2
KV_PHASE_I32 = KV_PHASE_BF16 // 2
TOTAL_WEIGHT_I32 = Q_PHASE_I32 + 2 * KV_PHASE_I32


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


def _runtime_sequence(context_len: int, num_tiles: int) -> str:
    total_kv_dwords = num_tiles * KV_TILE_DWORDS
    current_tile = (context_len - 1) // TOKENS_PER_TILE
    current_token = (context_len - 1) % TOKENS_PER_TILE
    current_tile_base = current_tile * KV_TILE_DWORDS
    current_k_bytes = (current_tile_base + current_token * KV_DWORDS) * 4
    current_v_bytes = (current_tile_base + KV_PLANE_TILE_DWORDS + current_token * KV_DWORDS) * 4

    lines = [
        f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_I32}xi32>, "
        f"%qkv_weight: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%kv_cache: memref<{total_kv_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]

    lines += [
        _npu_writebd(0, 0, HIDDEN_I32, 0),
        _npu_address_patch(0, 0, 0, 0),
        _npu_push_queue(0, "MM2S", 0, 0),
        _npu_writebd(0, 13, KV_DWORDS, current_k_bytes),
        _npu_address_patch(0, 13, 2, current_k_bytes),
        _npu_push_queue(0, "S2MM", 0, 13),
        _npu_writebd(0, 14, KV_DWORDS, current_v_bytes),
        _npu_address_patch(0, 14, 2, current_v_bytes),
        _npu_push_queue(0, "S2MM", 0, 14, issue_token=True),
        _npu_writebd(0, 1, Q_PHASE_I32, 0),
        _npu_address_patch(0, 1, 1, 0),
        _npu_push_queue(0, "MM2S", 1, 1, issue_token=True),
        _npu_writebd(0, 2, KV_PHASE_I32, Q_PHASE_BF16 * 2),
        _npu_address_patch(0, 2, 1, Q_PHASE_BF16 * 2),
        _npu_push_queue(0, "MM2S", 1, 2, issue_token=True),
        _npu_writebd(0, 3, KV_PHASE_I32, (Q_PHASE_BF16 + KV_PHASE_BF16) * 2),
        _npu_address_patch(0, 3, 1, (Q_PHASE_BF16 + KV_PHASE_BF16) * 2),
        _npu_push_queue(0, "MM2S", 1, 3, issue_token=True),
        _npu_sync(0, 0),
    ]

    for tile in range(num_tiles):
        offset = tile * KV_TILE_DWORDS * 4
        lines += [
            _npu_writebd(0, tile, KV_TILE_DWORDS, offset),
            _npu_address_patch(0, tile, 2, offset),
            _npu_push_queue(0, "MM2S", 0, tile),
        ]

    lines += [
        _npu_writebd(0, 15, OUTPUT_DWORDS, 0),
        _npu_address_patch(0, 15, 3, 0),
        _npu_push_queue(0, "S2MM", 0, 15, issue_token=True),
        _npu_sync(0, 0),
        "    }",
    ]
    return "\n".join(lines)


def _projection_loop(name: str, target: str, rowblocks: int, output_dwords: int, kernel: str) -> str:
    return f"""
      scf.for %rb_{name} = %c0 to %{name}_rowblocks_idx step %c1 {{
        %rb_{name}_i32 = arith.index_cast %rb_{name} : index to i32
        %out_{name}_offset = arith.muli %rb_{name}_i32, %c32_i32 : i32

        scf.for %chunk_{name} = %c0 to %q4_chunks_idx step %c1 {{
          %chunk_{name}_i32 = arith.index_cast %chunk_{name} : index to i32
          %act_{name}_offset = arith.muli %chunk_{name}_i32, %k_chunk_i32 : i32
          %rem_{name} = arith.remsi %chunk_{name}_i32, %c2_i32 : i32
          %is_pong_{name} = arith.cmpi eq, %rem_{name}, %c1_i32 : i32

          aie.use_lock(%w0_wt_full, AcquireGreaterEqual, 1)
          scf.if %is_pong_{name} {{
            func.call @{kernel}(%w0_wt_pong, %w0_hidden, %{target}, %out_{name}_offset, %act_{name}_offset)
              : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{output_dwords}xf32>, i32, i32) -> ()
          }} else {{
            func.call @{kernel}(%w0_wt_ping, %w0_hidden, %{target}, %out_{name}_offset, %act_{name}_offset)
              : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{output_dwords}xf32>, i32, i32) -> ()
          }}
          aie.use_lock(%w0_wt_empty, Release, 1)
        }}
      }}"""


def _worker_core(num_tiles: int, last_valid: int) -> str:
    q_loop = _projection_loop("q", "w0_query", Q_ROW_BLOCKS, QUERY_DWORDS, "q4nx_project_query_chunk")
    k_loop = _projection_loop("k", "w0_current_k", KV_ROW_BLOCKS, KV_DWORDS, "q4nx_project_kv_chunk")
    v_loop = _projection_loop("v", "w0_current_v", KV_ROW_BLOCKS, KV_DWORDS, "q4nx_project_kv_chunk")
    return f"""    %core0 = aie.core(%w0) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c32_i32 = arith.constant {Q4_ROWS} : i32
      %k_chunk_i32 = arith.constant {Q4_K_CHUNK} : i32
      %q4_chunks_idx = arith.constant {Q4_CHUNKS} : index
      %q_rowblocks_idx = arith.constant {Q_ROW_BLOCKS} : index
      %k_rowblocks_idx = arith.constant {KV_ROW_BLOCKS} : index
      %v_rowblocks_idx = arith.constant {KV_ROW_BLOCKS} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32
      %num_tiles_idx = arith.constant {num_tiles} : index

      aie.use_lock(%w0_hidden_full, AcquireGreaterEqual, 1)
      func.call @zero_query(%w0_query) : (memref<{QUERY_DWORDS}xf32>) -> ()
      func.call @zero_kv(%w0_current_k) : (memref<{KV_DWORDS}xf32>) -> ()
      func.call @zero_kv(%w0_current_v) : (memref<{KV_DWORDS}xf32>) -> ()

{q_loop}
{k_loop}
{v_loop}

      aie.use_lock(%w0_hidden_empty, Release, 1)

      aie.use_lock(%w0_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_kv(%w0_current_k, %w0_current_out)
        : (memref<{KV_DWORDS}xf32>, memref<{KV_DWORDS}xf32>) -> ()
      aie.use_lock(%w0_current_out_cons, Release, 1)

      aie.use_lock(%w0_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_kv(%w0_current_v, %w0_current_out)
        : (memref<{KV_DWORDS}xf32>, memref<{KV_DWORDS}xf32>) -> ()
      aie.use_lock(%w0_current_out_cons, Release, 1)

      aie.use_lock(%w0_out_prod, AcquireGreaterEqual, 1)
      func.call @init_attention_state(%w0_out, %w0_running_max, %w0_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        aie.use_lock(%w0_kv_full, AcquireGreaterEqual, 1)
        func.call @online_gqa_attention_pair(%w0_query, %w0_kv_tile, %w0_out, %w0_running_max, %w0_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
          : (memref<{QUERY_DWORDS}xf32>, memref<{KV_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>, i32, i32, i32) -> ()
        aie.use_lock(%w0_kv_empty, Release, 1)
      }}

      func.call @finalize_attention(%w0_out, %w0_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>) -> ()
      aie.use_lock(%w0_out_cons, Release, 1)
      aie.end
    }}"""


def _worker_mem() -> str:
    return f"""    %w0_mem = aie.mem(%w0) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^weight_start)
    ^hidden_in:
      aie.use_lock(%w0_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w0_hidden_full, Release, 1)
      aie.next_bd ^kv_in
    ^kv_in:
      aie.use_lock(%w0_kv_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_kv_tile : memref<{KV_TILE_DWORDS}xf32>, 0, {KV_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w0_kv_full, Release, 1)
      aie.next_bd ^kv_in

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%w0_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_out : memref<{KV_DWORDS}xf32>, 0, {KV_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%w0_current_out_prod, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%w0_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_out : memref<{KV_DWORDS}xf32>, 0, {KV_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%w0_current_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^out_bd:
      aie.use_lock(%w0_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_out : memref<{OUTPUT_DWORDS}xf32>, 0, {OUTPUT_DWORDS}) {{bd_id = 6 : i32}}
      aie.use_lock(%w0_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _memtile_dma() -> str:
    return f"""    %mem0_dma = aie.memtile_dma(%mem0) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^a_out_start)
    ^hidden_in:
      aie.use_lock(%mem0_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem0_hidden_full, Release, 1)
      aie.next_bd ^kv_in
    ^kv_in:
      aie.use_lock(%mem0_kv_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_kv_tile : memref<{KV_TILE_DWORDS}xf32>, 0, {KV_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem0_kv_full, Release, 1)
      aie.next_bd ^kv_in

    ^a_out_start:
      %1 = aie.dma_start(MM2S, 0, ^hidden_out, ^weight_start)
    ^hidden_out:
      aie.use_lock(%mem0_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mem0_hidden_empty, Release, 1)
      aie.next_bd ^kv_out
    ^kv_out:
      aie.use_lock(%mem0_kv_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_kv_tile : memref<{KV_TILE_DWORDS}xf32>, 0, {KV_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mem0_kv_empty, Release, 1)
      aie.next_bd ^kv_out

    ^weight_start:
      %2 = aie.dma_start(S2MM, 1, ^wt_in_ping, ^weight_out_start)
    ^wt_in_ping:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt_in_pong
    ^wt_in_pong:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt_in_ping

    ^weight_out_start:
      %3 = aie.dma_start(MM2S, 1, ^wt_out_ping, ^end)
    ^wt_out_ping:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt_out_pong
    ^wt_out_pong:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt_out_ping
    ^end:
      aie.end
    }}"""


def _memtile_buffers_and_locks() -> str:
    return f"""
    %mem0_hidden = aie.buffer(%mem0) {{sym_name = "mem0_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %mem0_kv_tile = aie.buffer(%mem0) {{sym_name = "mem0_kv_tile"}} : memref<{KV_TILE_DWORDS}xf32>
    %mem0_wt_ping = aie.buffer(%mem0) {{sym_name = "mem0_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %mem0_wt_pong = aie.buffer(%mem0) {{sym_name = "mem0_wt_pong"}} : memref<{CHUNK_BF16}xbf16>

    %mem0_hidden_empty = aie.lock(%mem0, 8) {{init = 1 : i32, sym_name = "mem0_hidden_empty"}}
    %mem0_hidden_full  = aie.lock(%mem0, 9) {{init = 0 : i32, sym_name = "mem0_hidden_full"}}
    %mem0_kv_empty = aie.lock(%mem0, 0) {{init = 1 : i32, sym_name = "mem0_kv_empty"}}
    %mem0_kv_full  = aie.lock(%mem0, 1) {{init = 0 : i32, sym_name = "mem0_kv_full"}}
    %mem0_wt_empty = aie.lock(%mem0, 10) {{init = 2 : i32, sym_name = "mem0_wt_empty"}}
    %mem0_wt_full  = aie.lock(%mem0, 11) {{init = 0 : i32, sym_name = "mem0_wt_full"}}
"""


def _worker_buffers_and_locks() -> str:
    return f"""
    %w0_hidden = aie.buffer(%w0) {{sym_name = "w0_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %w0_wt_ping = aie.buffer(%w0) {{sym_name = "w0_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %w0_wt_pong = aie.buffer(%w0) {{sym_name = "w0_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %w0_query = aie.buffer(%w0) {{sym_name = "w0_query"}} : memref<{QUERY_DWORDS}xf32>
    %w0_current_k = aie.buffer(%w0) {{sym_name = "w0_current_k"}} : memref<{KV_DWORDS}xf32>
    %w0_current_v = aie.buffer(%w0) {{sym_name = "w0_current_v"}} : memref<{KV_DWORDS}xf32>
    %w0_current_out = aie.buffer(%w0) {{sym_name = "w0_current_out"}} : memref<{KV_DWORDS}xf32>
    %w0_kv_tile = aie.buffer(%w0) {{sym_name = "w0_kv_tile"}} : memref<{KV_TILE_DWORDS}xf32>
    %w0_out = aie.buffer(%w0) {{sym_name = "w0_out"}} : memref<{OUTPUT_DWORDS}xf32>
    %w0_running_max = aie.buffer(%w0) {{sym_name = "w0_running_max"}} : memref<{NUM_Q_HEADS}xf32>
    %w0_running_sum = aie.buffer(%w0) {{sym_name = "w0_running_sum"}} : memref<{NUM_Q_HEADS}xf32>

    %w0_hidden_empty = aie.lock(%w0, 12) {{init = 1 : i32, sym_name = "w0_hidden_empty"}}
    %w0_hidden_full  = aie.lock(%w0, 13) {{init = 0 : i32, sym_name = "w0_hidden_full"}}
    %w0_wt_empty = aie.lock(%w0, 14) {{init = 2 : i32, sym_name = "w0_wt_empty"}}
    %w0_wt_full  = aie.lock(%w0, 15) {{init = 0 : i32, sym_name = "w0_wt_full"}}
    %w0_kv_empty = aie.lock(%w0, 0) {{init = 1 : i32, sym_name = "w0_kv_empty"}}
    %w0_kv_full  = aie.lock(%w0, 1) {{init = 0 : i32, sym_name = "w0_kv_full"}}
    %w0_out_prod = aie.lock(%w0, 4) {{init = 1 : i32, sym_name = "w0_out_prod"}}
    %w0_out_cons = aie.lock(%w0, 5) {{init = 0 : i32, sym_name = "w0_out_cons"}}
    %w0_current_out_prod = aie.lock(%w0, 10) {{init = 1 : i32, sym_name = "w0_current_out_prod"}}
    %w0_current_out_cons = aie.lock(%w0, 11) {{init = 0 : i32, sym_name = "w0_current_out_cons"}}
"""


def generate_mlir(context_len: int) -> str:
    num_tiles = ceil(context_len / TOKENS_PER_TILE)
    last_valid = context_len - (num_tiles - 1) * TOKENS_PER_TILE
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %w0 = aie.tile(0, 2)

{_memtile_buffers_and_locks()}
{_worker_buffers_and_locks()}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %w0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %w0, DMA : 1)
    aie.flow(%w0, DMA : 0, %shim0, DMA : 0)

    func.func private @zero_query(memref<{QUERY_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @zero_kv(memref<{KV_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @q4nx_project_query_chunk(memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{QUERY_DWORDS}xf32>, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @q4nx_project_kv_chunk(memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{KV_DWORDS}xf32>, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @copy_kv(memref<{KV_DWORDS}xf32>, memref<{KV_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @init_attention_state(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @online_gqa_attention_pair(memref<{QUERY_DWORDS}xf32>, memref<{KV_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @finalize_attention(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}

{_worker_core(num_tiles, last_valid)}

{_memtile_dma()}

{_worker_mem()}

{_runtime_sequence(context_len, num_tiles)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
