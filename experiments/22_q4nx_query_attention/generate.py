"""Generate raw MLIR-AIE for exp22 Q4NX query projection + KV attention."""

from math import ceil
from pathlib import Path

NUM_HEADS = 1
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE
OUTPUT_DWORDS = TOKEN_DWORDS
HIDDEN_DIM = 1024
HIDDEN_I32 = HIDDEN_DIM // 2
Q_CHUNK = 256
Q_CHUNKS = HIDDEN_DIM // Q_CHUNK
GROUP_SIZE = 32
Q_ROWS = TOKEN_DWORDS
CHUNK_BF16 = 2560
CHUNK_I32 = CHUNK_BF16 // 2
TOTAL_WEIGHT_I32 = Q_CHUNKS * CHUNK_I32


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


def _runtime_sequence(L: int, num_tiles: int) -> str:
    total_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    plane_bytes = total_plane_dwords * 4
    token_bytes = TOKEN_DWORDS * 4
    current_token_offset = (L - 1) * token_bytes

    lines = [
        f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_I32}xi32>, "
        f"%q_weight: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%current: memref<{2 * TOKEN_DWORDS}xf32>, "
        f"%kv_cache: memref<{2 * total_plane_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]

    lines += [
        _npu_writebd(0, 0, HIDDEN_I32, 0),
        _npu_address_patch(0, 0, 0, 0),
        _npu_push_queue(0, "MM2S", 0, 0),
    ]

    for chunk in range(Q_CHUNKS):
        bd_id = 3 + chunk
        offset = chunk * CHUNK_BF16 * 2
        lines += [
            _npu_writebd(0, bd_id, CHUNK_I32, offset),
            _npu_address_patch(0, bd_id, 1, offset),
            _npu_push_queue(0, "MM2S", 1, bd_id),
        ]

    lines += [
        _npu_writebd(0, 1, TOKEN_DWORDS, 0),
        _npu_address_patch(0, 1, 2, 0),
        _npu_push_queue(0, "MM2S", 0, 1),
        _npu_writebd(0, 7, TOKEN_DWORDS, token_bytes),
        _npu_address_patch(0, 7, 2, token_bytes),
        _npu_push_queue(0, "MM2S", 1, 7),
    ]

    lines += [
        _npu_writebd(0, 9, TOKEN_DWORDS, current_token_offset),
        _npu_address_patch(0, 9, 3, current_token_offset),
        _npu_push_queue(0, "S2MM", 0, 9),
        _npu_writebd(0, 10, TOKEN_DWORDS, plane_bytes + current_token_offset),
        _npu_address_patch(0, 10, 3, plane_bytes + current_token_offset),
        _npu_push_queue(0, "S2MM", 0, 10, issue_token=True),
        _npu_sync(0, 0),
    ]

    lines += [
        _npu_writebd(0, 2, total_plane_dwords, 0),
        _npu_address_patch(0, 2, 3, 0),
        _npu_push_queue(0, "MM2S", 0, 2),
        _npu_writebd(0, 8, total_plane_dwords, plane_bytes),
        _npu_address_patch(0, 8, 3, plane_bytes),
        _npu_push_queue(0, "MM2S", 1, 8),
        _npu_writebd(0, 11, OUTPUT_DWORDS, 0),
        _npu_address_patch(0, 11, 4, 0),
        _npu_push_queue(0, "S2MM", 0, 11, issue_token=True),
        _npu_sync(0, 0),
        "    }",
    ]
    return "\n".join(lines)


def _worker_core(num_tiles: int, last_valid: int) -> str:
    return f"""    %core0 = aie.core(%w0) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32
      %num_tiles_idx = arith.constant {num_tiles} : index
      %q_chunks = arith.constant {Q_CHUNKS} : index
      %q_chunk_i32 = arith.constant {Q_CHUNK} : i32
      %q_rows_i32 = arith.constant {Q_ROWS} : i32

      aie.use_lock(%w0_hidden_full, AcquireGreaterEqual, 1)
      scf.for %chunk = %c0 to %q_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %q_chunk_i32 : i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%w0_wt_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @q4nx_query_accum_offset(%w0_wt_pong, %w0_hidden, %act_offset, %q_rows_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, i32, i32) -> ()
        }} else {{
          func.call @q4nx_query_accum_offset(%w0_wt_ping, %w0_hidden, %act_offset, %q_rows_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, i32, i32) -> ()
        }}
        aie.use_lock(%w0_wt_empty, Release, 1)
      }}
      func.call @q4nx_query_flush(%w0_query, %q_rows_i32)
        : (memref<{TOKEN_DWORDS}xf32>, i32) -> ()
      aie.use_lock(%w0_hidden_empty, Release, 1)

      aie.use_lock(%w0_current_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%w0_current_v_full, AcquireGreaterEqual, 1)

      aie.use_lock(%w0_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%w0_current_k, %w0_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%w0_current_out_cons, Release, 1)
      aie.use_lock(%w0_current_k_empty, Release, 1)

      aie.use_lock(%w0_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%w0_current_v, %w0_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%w0_current_out_cons, Release, 1)
      aie.use_lock(%w0_current_v_empty, Release, 1)

      aie.use_lock(%w0_out_prod, AcquireGreaterEqual, 1)
      func.call @init_attention_state(%w0_out, %w0_running_max, %w0_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%w0_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%w0_v_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @online_softmax_attention(%w0_query, %w0_k_pong, %w0_v_pong, %w0_out, %w0_running_max, %w0_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
            : (memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>, i32, i32, i32) -> ()
        }} else {{
          func.call @online_softmax_attention(%w0_query, %w0_k_ping, %w0_v_ping, %w0_out, %w0_running_max, %w0_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
            : (memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>, i32, i32, i32) -> ()
        }}

        aie.use_lock(%w0_k_empty, Release, 1)
        aie.use_lock(%w0_v_empty, Release, 1)
      }}

      func.call @finalize_attention(%w0_out, %w0_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>) -> ()
      func.call @layer_epilogue(%w0_query, %w0_out, %w0_epilogue_tmp)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>) -> ()
      aie.use_lock(%w0_out_cons, Release, 1)
      aie.end
    }}"""


def _worker_mem() -> str:
    return f"""    %w0_mem = aie.mem(%w0) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^b_start)
    ^hidden_in:
      aie.use_lock(%w0_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 9 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%w0_hidden_full, Release, 1)
      aie.next_bd ^current_k
    ^current_k:
      aie.use_lock(%w0_current_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_k : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%w0_current_k_full, Release, 1)
      aie.next_bd ^k_ping
    ^k_ping:
      aie.use_lock(%w0_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w0_k_full, Release, 1)
      aie.next_bd ^k_pong
    ^k_pong:
      aie.use_lock(%w0_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%w0_k_full, Release, 1)
      aie.next_bd ^k_ping

    ^b_start:
      %1 = aie.dma_start(S2MM, 1, ^wt0, ^out_start)
    ^wt0:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 10 : i32, next_bd_id = 11 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^wt1
    ^wt1:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 11 : i32, next_bd_id = 12 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^wt2
    ^wt2:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 12 : i32, next_bd_id = 13 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^wt3
    ^wt3:
      aie.use_lock(%w0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 13 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%w0_wt_full, Release, 1)
      aie.next_bd ^current_v
    ^current_v:
      aie.use_lock(%w0_current_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_v : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 6 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w0_current_v_full, Release, 1)
      aie.next_bd ^v_ping
    ^v_ping:
      aie.use_lock(%w0_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%w0_v_full, Release, 1)
      aie.next_bd ^v_pong
    ^v_pong:
      aie.use_lock(%w0_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w0_v_full, Release, 1)
      aie.next_bd ^v_ping

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%w0_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 7 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%w0_current_out_prod, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%w0_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 8 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%w0_current_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^out_bd:
      aie.use_lock(%w0_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_out : memref<{OUTPUT_DWORDS}xf32>, 0, {OUTPUT_DWORDS}) {{bd_id = 4 : i32}}
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
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 6 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mem0_hidden_full, Release, 1)
      aie.next_bd ^current_k_in
    ^current_k_in:
      aie.use_lock(%mem0_a_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mem0_a_current_full, Release, 1)
      aie.next_bd ^a_s2mm_ping
    ^a_s2mm_ping:
      aie.use_lock(%mem0_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem0_a_full, Release, 1)
      aie.next_bd ^a_s2mm_pong
    ^a_s2mm_pong:
      aie.use_lock(%mem0_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mem0_a_full, Release, 1)
      aie.next_bd ^a_s2mm_ping

    ^a_out_start:
      %1 = aie.dma_start(MM2S, 0, ^hidden_out, ^b_start)
    ^hidden_out:
      aie.use_lock(%mem0_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 7 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%mem0_hidden_empty, Release, 1)
      aie.next_bd ^current_k_out
    ^current_k_out:
      aie.use_lock(%mem0_a_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mem0_a_current_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping
    ^a_mm2s_ping:
      aie.use_lock(%mem0_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mem0_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_pong
    ^a_mm2s_pong:
      aie.use_lock(%mem0_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_a_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mem0_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping

    ^b_start:
      %2 = aie.dma_start(S2MM, 1, ^wt0_in, ^b_out_start)
    ^wt0_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt1_in
    ^wt1_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt2_in
    ^wt2_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt3_in
    ^wt3_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^current_v_in
    ^current_v_in:
      aie.use_lock(%mem0_b_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mem0_b_current_full, Release, 1)
      aie.next_bd ^b_s2mm_ping
    ^b_s2mm_ping:
      aie.use_lock(%mem0_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 29 : i32, next_bd_id = 30 : i32}}
      aie.use_lock(%mem0_b_full, Release, 1)
      aie.next_bd ^b_s2mm_pong
    ^b_s2mm_pong:
      aie.use_lock(%mem0_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 30 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mem0_b_full, Release, 1)
      aie.next_bd ^b_s2mm_ping

    ^b_out_start:
      %3 = aie.dma_start(MM2S, 1, ^wt0_out, ^end)
    ^wt0_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 31 : i32, next_bd_id = 32 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt1_out
    ^wt1_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 32 : i32, next_bd_id = 33 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt2_out
    ^wt2_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 33 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt3_out
    ^wt3_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 34 : i32, next_bd_id = 35 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%mem0_b_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 35 : i32, next_bd_id = 36 : i32}}
      aie.use_lock(%mem0_b_current_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^b_mm2s_ping:
      aie.use_lock(%mem0_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 36 : i32, next_bd_id = 37 : i32}}
      aie.use_lock(%mem0_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_pong
    ^b_mm2s_pong:
      aie.use_lock(%mem0_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_b_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 37 : i32, next_bd_id = 36 : i32}}
      aie.use_lock(%mem0_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^end:
      aie.end
    }}"""


def _memtile_buffers_and_locks() -> str:
    return f"""
    %mem0_hidden = aie.buffer(%mem0) {{sym_name = "mem0_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %mem0_a_current = aie.buffer(%mem0) {{sym_name = "mem0_a_current"}} : memref<{TOKEN_DWORDS}xf32>
    %mem0_a_ping = aie.buffer(%mem0) {{sym_name = "mem0_a_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem0_a_pong = aie.buffer(%mem0) {{sym_name = "mem0_a_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem0_wt_ping = aie.buffer(%mem0) {{sym_name = "mem0_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %mem0_wt_pong = aie.buffer(%mem0) {{sym_name = "mem0_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %mem0_b_current = aie.buffer(%mem0) {{sym_name = "mem0_b_current"}} : memref<{TOKEN_DWORDS}xf32>
    %mem0_b_ping = aie.buffer(%mem0) {{sym_name = "mem0_b_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem0_b_pong = aie.buffer(%mem0) {{sym_name = "mem0_b_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>

    %mem0_hidden_empty = aie.lock(%mem0, 8) {{init = 1 : i32, sym_name = "mem0_hidden_empty"}}
    %mem0_hidden_full  = aie.lock(%mem0, 9) {{init = 0 : i32, sym_name = "mem0_hidden_full"}}
    %mem0_a_current_empty = aie.lock(%mem0, 4) {{init = 1 : i32, sym_name = "mem0_a_current_empty"}}
    %mem0_a_current_full  = aie.lock(%mem0, 5) {{init = 0 : i32, sym_name = "mem0_a_current_full"}}
    %mem0_a_empty = aie.lock(%mem0, 0) {{init = 2 : i32, sym_name = "mem0_a_empty"}}
    %mem0_a_full  = aie.lock(%mem0, 1) {{init = 0 : i32, sym_name = "mem0_a_full"}}
    %mem0_wt_empty = aie.lock(%mem0, 10) {{init = 2 : i32, sym_name = "mem0_wt_empty"}}
    %mem0_wt_full  = aie.lock(%mem0, 11) {{init = 0 : i32, sym_name = "mem0_wt_full"}}
    %mem0_b_current_empty = aie.lock(%mem0, 6) {{init = 1 : i32, sym_name = "mem0_b_current_empty"}}
    %mem0_b_current_full  = aie.lock(%mem0, 7) {{init = 0 : i32, sym_name = "mem0_b_current_full"}}
    %mem0_b_empty = aie.lock(%mem0, 2) {{init = 2 : i32, sym_name = "mem0_b_empty"}}
    %mem0_b_full  = aie.lock(%mem0, 3) {{init = 0 : i32, sym_name = "mem0_b_full"}}
"""


def _worker_buffers_and_locks() -> str:
    return f"""
    %w0_hidden = aie.buffer(%w0) {{sym_name = "w0_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %w0_wt_ping = aie.buffer(%w0) {{sym_name = "w0_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %w0_wt_pong = aie.buffer(%w0) {{sym_name = "w0_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %w0_current_k = aie.buffer(%w0) {{sym_name = "w0_current_k"}} : memref<{TOKEN_DWORDS}xf32>
    %w0_current_v = aie.buffer(%w0) {{sym_name = "w0_current_v"}} : memref<{TOKEN_DWORDS}xf32>
    %w0_current_out = aie.buffer(%w0) {{sym_name = "w0_current_out"}} : memref<{TOKEN_DWORDS}xf32>
    %w0_k_ping = aie.buffer(%w0) {{sym_name = "w0_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %w0_k_pong = aie.buffer(%w0) {{sym_name = "w0_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %w0_v_ping = aie.buffer(%w0) {{sym_name = "w0_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %w0_v_pong = aie.buffer(%w0) {{sym_name = "w0_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %w0_query = aie.buffer(%w0) {{sym_name = "w0_query"}} : memref<{TOKEN_DWORDS}xf32>
    %w0_out = aie.buffer(%w0) {{sym_name = "w0_out"}} : memref<{OUTPUT_DWORDS}xf32>
    %w0_epilogue_tmp = aie.buffer(%w0) {{sym_name = "w0_epilogue_tmp"}} : memref<{OUTPUT_DWORDS}xf32>
    %w0_running_max = aie.buffer(%w0) {{sym_name = "w0_running_max"}} : memref<{NUM_HEADS}xf32>
    %w0_running_sum = aie.buffer(%w0) {{sym_name = "w0_running_sum"}} : memref<{NUM_HEADS}xf32>

    %w0_hidden_empty = aie.lock(%w0, 12) {{init = 1 : i32, sym_name = "w0_hidden_empty"}}
    %w0_hidden_full  = aie.lock(%w0, 13) {{init = 0 : i32, sym_name = "w0_hidden_full"}}
    %w0_wt_empty = aie.lock(%w0, 14) {{init = 2 : i32, sym_name = "w0_wt_empty"}}
    %w0_wt_full  = aie.lock(%w0, 15) {{init = 0 : i32, sym_name = "w0_wt_full"}}
    %w0_current_k_empty = aie.lock(%w0, 6) {{init = 1 : i32, sym_name = "w0_current_k_empty"}}
    %w0_current_k_full  = aie.lock(%w0, 7) {{init = 0 : i32, sym_name = "w0_current_k_full"}}
    %w0_current_v_empty = aie.lock(%w0, 8) {{init = 1 : i32, sym_name = "w0_current_v_empty"}}
    %w0_current_v_full  = aie.lock(%w0, 9) {{init = 0 : i32, sym_name = "w0_current_v_full"}}
    %w0_current_out_prod = aie.lock(%w0, 10) {{init = 1 : i32, sym_name = "w0_current_out_prod"}}
    %w0_current_out_cons = aie.lock(%w0, 11) {{init = 0 : i32, sym_name = "w0_current_out_cons"}}
    %w0_k_empty = aie.lock(%w0, 0) {{init = 2 : i32, sym_name = "w0_k_empty"}}
    %w0_k_full  = aie.lock(%w0, 1) {{init = 0 : i32, sym_name = "w0_k_full"}}
    %w0_v_empty = aie.lock(%w0, 2) {{init = 2 : i32, sym_name = "w0_v_empty"}}
    %w0_v_full  = aie.lock(%w0, 3) {{init = 0 : i32, sym_name = "w0_v_full"}}
    %w0_out_prod = aie.lock(%w0, 4) {{init = 1 : i32, sym_name = "w0_out_prod"}}
    %w0_out_cons = aie.lock(%w0, 5) {{init = 0 : i32, sym_name = "w0_out_cons"}}
"""


def generate_mlir(L: int) -> str:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
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

    func.func private @q4nx_query_accum_offset(memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @q4nx_query_flush(memref<{TOKEN_DWORDS}xf32>, i32) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @copy_token(memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @init_attention_state(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @online_softmax_attention(memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @finalize_attention(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}
    func.func private @layer_epilogue(memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/q4_query_attention.o"}}

{_worker_core(num_tiles, last_valid)}

{_memtile_dma()}

{_worker_mem()}

{_runtime_sequence(L, num_tiles)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
