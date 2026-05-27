"""Generate raw MLIR-AIE for exp24 MyLM-style projection/edge split."""

from math import ceil
from pathlib import Path

NUM_HEADS = 1
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE
OUTPUT_DWORDS = TOKEN_DWORDS

HIDDEN_DIM = 512
HIDDEN_I32 = HIDDEN_DIM // 2
Q4_K_CHUNK = 256
Q4_CHUNKS = HIDDEN_DIM // Q4_K_CHUNK
GROUP_SIZE = 32
Q4_ROWS = 32
ROW_BLOCKS = TOKEN_DWORDS // Q4_ROWS
PROJECTIONS = 3
CHUNK_BF16 = 2560
ROWBLOCK_BF16 = Q4_CHUNKS * CHUNK_BF16
ROWBLOCK_I32 = ROWBLOCK_BF16 // 2
TOTAL_WEIGHT_BLOCKS = PROJECTIONS * ROW_BLOCKS
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BLOCKS * ROWBLOCK_I32
DEBUG_PROBE = False


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
        f"%qkv_weight: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%kv_cache: memref<{2 * total_plane_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]

    lines += [
        _npu_writebd(0, 0, HIDDEN_I32, 0),
        _npu_address_patch(0, 0, 0, 0),
        _npu_push_queue(0, "MM2S", 0, 0),
    ]

    bd_id = 1
    for block in range(TOTAL_WEIGHT_BLOCKS):
        for chunk in range(Q4_CHUNKS):
            offset = block * ROWBLOCK_BF16 * 2 + chunk * CHUNK_BF16 * 2
            lines += [
                _npu_writebd(0, bd_id, CHUNK_BF16 // 2, offset),
                _npu_address_patch(0, bd_id, 1, offset),
                _npu_push_queue(0, "MM2S", 1, bd_id),
            ]
            bd_id += 1

    lines += [
        _npu_writebd(1, 13, TOKEN_DWORDS, current_token_offset),
        _npu_address_patch(1, 13, 2, current_token_offset),
        _npu_push_queue(1, "S2MM", 0, 13),
        _npu_writebd(1, 14, TOKEN_DWORDS, plane_bytes + current_token_offset),
        _npu_address_patch(1, 14, 2, plane_bytes + current_token_offset),
        _npu_push_queue(1, "S2MM", 0, 14, issue_token=True),
        _npu_sync(1, 0),
        _npu_writebd(1, 0, total_plane_dwords, 0),
        _npu_address_patch(1, 0, 2, 0),
        _npu_push_queue(1, "MM2S", 0, 0),
        _npu_writebd(1, 1, total_plane_dwords, plane_bytes),
        _npu_address_patch(1, 1, 2, plane_bytes),
        _npu_push_queue(1, "MM2S", 1, 1),
        _npu_writebd(1, 15, OUTPUT_DWORDS, 0),
        _npu_address_patch(1, 15, 3, 0),
        _npu_push_queue(1, "S2MM", 0, 15, issue_token=True),
        _npu_sync(1, 0),
        "    }",
    ]
    return "\n".join(lines)


def _project_call(chunk_index: int, target: str, row_offset: int, act_offset: int) -> str:
    wt = "proj_wt_pong" if chunk_index % 2 else "proj_wt_ping"
    return f"""
      aie.use_lock(%proj_wt_full, AcquireGreaterEqual, 1)
      func.call @q4nx_project_chunk_accum(%{wt}, %proj_hidden, %{target}, %act{act_offset}_i32, %row{row_offset}_i32)
        : (memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{TOKEN_DWORDS}xf32>, i32, i32) -> ()
      aie.use_lock(%proj_wt_empty, Release, 1)"""


def _projection_sequence() -> str:
    lines = []
    chunk_index = 0
    for target in ["proj_query", "proj_current_k", "proj_current_v"]:
        for rowblock in range(ROW_BLOCKS):
            row_offset = rowblock * Q4_ROWS
            for chunk in range(Q4_CHUNKS):
                lines.append(_project_call(chunk_index, target, row_offset, chunk * Q4_K_CHUNK))
                chunk_index += 1
    return "\n".join(lines)


def _projection_core() -> str:
    return f"""    %proj_core = aie.core(%proj0) {{
      %row0_i32 = arith.constant 0 : i32
      %row32_i32 = arith.constant 32 : i32
      %act0_i32 = arith.constant 0 : i32
      %act256_i32 = arith.constant 256 : i32

      aie.use_lock(%proj_hidden_full, AcquireGreaterEqual, 1)
      func.call @zero_token(%proj_query) : (memref<{TOKEN_DWORDS}xf32>) -> ()
      func.call @zero_token(%proj_current_k) : (memref<{TOKEN_DWORDS}xf32>) -> ()
      func.call @zero_token(%proj_current_v) : (memref<{TOKEN_DWORDS}xf32>) -> ()

{_projection_sequence()}

      aie.use_lock(%proj_hidden_empty, Release, 1)
      aie.use_lock(%proj_query_ready, Release, 1)
      aie.use_lock(%proj_current_k_ready, Release, 1)
      aie.use_lock(%proj_current_v_ready, Release, 1)
      aie.end
    }}"""


def _projection_mem() -> str:
    return f"""    %proj_mem = aie.mem(%proj0) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^weight_start)
    ^hidden_in:
      aie.use_lock(%proj_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 0 : i32}}
      aie.use_lock(%proj_hidden_full, Release, 1)
      aie.next_bd ^hidden_in

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^wt0, ^query_out_start)
    ^wt0:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 10 : i32, next_bd_id = 11 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt1
    ^wt1:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 11 : i32, next_bd_id = 12 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt2
    ^wt2:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 12 : i32, next_bd_id = 13 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt3
    ^wt3:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 13 : i32, next_bd_id = 14 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt4
    ^wt4:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 14 : i32, next_bd_id = 15 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt5
    ^wt5:
      aie.use_lock(%proj_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 15 : i32}}
      aie.use_lock(%proj_wt_full, Release, 1)
      aie.next_bd ^wt5

    ^query_out_start:
      %2 = aie.dma_start(MM2S, 0, ^query_out, ^current_out_start)
    ^query_out:
      aie.use_lock(%proj_query_ready, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_query : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 4 : i32}}
      aie.use_lock(%proj_query_sent, Release, 1)
      aie.next_bd ^query_out

    ^current_out_start:
      %3 = aie.dma_start(MM2S, 1, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%proj_current_k_ready, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_current_k : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%proj_current_k_sent, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%proj_current_v_ready, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_current_v : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 6 : i32}}
      aie.use_lock(%proj_current_v_sent, Release, 1)
      aie.next_bd ^current_v_out
    ^end:
      aie.end
    }}"""


def _projection_memtile_dma() -> str:
    return f"""    %mem0_dma = aie.memtile_dma(%mem0) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_in, ^hidden_out_start)
    ^hidden_in:
      aie.use_lock(%mem0_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 6 : i32}}
      aie.use_lock(%mem0_hidden_full, Release, 1)
      aie.next_bd ^hidden_in

    ^hidden_out_start:
      %1 = aie.dma_start(MM2S, 0, ^hidden_out, ^weight_in_start)
    ^hidden_out:
      aie.use_lock(%mem0_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_hidden : memref<{HIDDEN_DIM}xbf16>, 0, {HIDDEN_DIM}) {{bd_id = 7 : i32}}
      aie.use_lock(%mem0_hidden_empty, Release, 1)
      aie.next_bd ^hidden_out

    ^weight_in_start:
      %2 = aie.dma_start(S2MM, 1, ^wt0_in, ^weight_out_start)
    ^wt0_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt1_in
    ^wt1_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 25 : i32, next_bd_id = 32 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt2_in
    ^wt2_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 32 : i32, next_bd_id = 33 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt3_in
    ^wt3_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 33 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt4_in
    ^wt4_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 34 : i32, next_bd_id = 35 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt5_in
    ^wt5_in:
      aie.use_lock(%mem0_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 35 : i32}}
      aie.use_lock(%mem0_wt_full, Release, 1)
      aie.next_bd ^wt5_in

    ^weight_out_start:
      %3 = aie.dma_start(MM2S, 1, ^wt0_out, ^end)
    ^wt0_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt1_out
    ^wt1_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt2_out
    ^wt2_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt3_out
    ^wt3_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 29 : i32, next_bd_id = 36 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt4_out
    ^wt4_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 36 : i32, next_bd_id = 37 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt5_out
    ^wt5_out:
      aie.use_lock(%mem0_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem0_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 37 : i32}}
      aie.use_lock(%mem0_wt_empty, Release, 1)
      aie.next_bd ^wt5_out
    ^end:
      aie.end
    }}"""


def _edge_core(num_tiles: int, last_valid: int) -> str:
    tile_calls = []
    for tile_idx in range(num_tiles):
        if DEBUG_PROBE:
            tile_calls.append(
                f"""
      %tile{tile_idx}_i32 = arith.constant {tile_idx} : i32
      aie.use_lock(%edge_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_v_full, AcquireGreaterEqual, 1)
      func.call @probe_tile(%edge_k_tile, %edge_v_tile, %edge_out, %tile{tile_idx}_i32)
        : (memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, i32) -> ()
      aie.use_lock(%edge_k_empty, Release, 1)
      aie.use_lock(%edge_v_empty, Release, 1)"""
            )
            continue
        tile_calls.append(
            f"""
      %tile{tile_idx}_i32 = arith.constant {tile_idx} : i32
      aie.use_lock(%edge_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_v_full, AcquireGreaterEqual, 1)
      func.call @online_softmax_attention(%edge_query, %edge_k_tile, %edge_v_tile, %edge_out, %edge_running_max, %edge_running_sum, %tile{tile_idx}_i32, %num_tiles_i32, %last_valid_i32)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>, i32, i32, i32) -> ()
      aie.use_lock(%edge_k_empty, Release, 1)
      aie.use_lock(%edge_v_empty, Release, 1)"""
        )
    final_block = ""
    if not DEBUG_PROBE:
        final_block = f"""
      func.call @finalize_attention(%edge_out, %edge_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>) -> ()
      func.call @layer_epilogue(%edge_query, %edge_out, %edge_epilogue_tmp)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>) -> ()"""

    return f"""    %edge_core = aie.core(%edge0) {{
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      aie.use_lock(%edge_query_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_current_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_current_v_full, AcquireGreaterEqual, 1)

      aie.use_lock(%edge_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%edge_current_k, %edge_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%edge_current_out_cons, Release, 1)
      aie.use_lock(%edge_current_k_empty, Release, 1)

      aie.use_lock(%edge_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%edge_current_v, %edge_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%edge_current_out_cons, Release, 1)
      aie.use_lock(%edge_current_v_empty, Release, 1)

      aie.use_lock(%edge_out_prod, AcquireGreaterEqual, 1)
      func.call @init_attention_state(%edge_out, %edge_running_max, %edge_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>) -> ()

{''.join(tile_calls)}

{final_block}
      aie.use_lock(%edge_query_empty, Release, 1)
      aie.use_lock(%edge_out_cons, Release, 1)
      aie.end
    }}"""


def _edge_mem() -> str:
    return f"""    %edge_mem = aie.mem(%edge0) {{
      %0 = aie.dma_start(S2MM, 0, ^query_in, ^current_start)
    ^query_in:
      aie.use_lock(%edge_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_query : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%edge_query_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%edge_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_k_tile : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%edge_k_full, Release, 1)
      aie.next_bd ^k_in

    ^current_start:
      %1 = aie.dma_start(S2MM, 1, ^current_k_in, ^out_start)
    ^current_k_in:
      aie.use_lock(%edge_current_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_current_k : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 6 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%edge_current_k_full, Release, 1)
      aie.next_bd ^current_v_in
    ^current_v_in:
      aie.use_lock(%edge_current_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_current_v : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 7 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%edge_current_v_full, Release, 1)
      aie.next_bd ^v_in
    ^v_in:
      aie.use_lock(%edge_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_v_tile : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%edge_v_full, Release, 1)
      aie.next_bd ^v_in

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%edge_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 8 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%edge_current_out_prod, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%edge_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 9 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%edge_current_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^out_bd:
      aie.use_lock(%edge_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_out : memref<{OUTPUT_DWORDS}xf32>, 0, {OUTPUT_DWORDS}) {{bd_id = 4 : i32}}
      aie.use_lock(%edge_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _edge_memtile_dma() -> str:
    return f"""    %mem1_dma = aie.memtile_dma(%mem1) {{
      %0 = aie.dma_start(S2MM, 2, ^query_bridge_in, ^current_bridge_start)
    ^query_bridge_in:
      aie.use_lock(%mem1_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_query : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 7 : i32}}
      aie.use_lock(%mem1_query_full, Release, 1)
      aie.next_bd ^query_bridge_in

    ^current_bridge_start:
      %1 = aie.dma_start(S2MM, 3, ^current_k_bridge_in, ^k_history_start)
    ^current_k_bridge_in:
      aie.use_lock(%mem1_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mem1_current_full, Release, 1)
      aie.next_bd ^current_v_bridge_in
    ^current_v_bridge_in:
      aie.use_lock(%mem1_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 27 : i32}}
      aie.use_lock(%mem1_current_full, Release, 1)
      aie.next_bd ^current_v_bridge_in

    ^k_history_start:
      %2 = aie.dma_start(S2MM, 0, ^k_in, ^v_history_start)
    ^k_in:
      aie.use_lock(%mem1_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem1_k_full, Release, 1)
      aie.next_bd ^k_pong_in
    ^k_pong_in:
      aie.use_lock(%mem1_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mem1_k_full, Release, 1)
      aie.next_bd ^k_in

    ^v_history_start:
      %3 = aie.dma_start(S2MM, 1, ^v_in, ^query_out_start)
    ^v_in:
      aie.use_lock(%mem1_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mem1_v_full, Release, 1)
      aie.next_bd ^v_pong_in
    ^v_pong_in:
      aie.use_lock(%mem1_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mem1_v_full, Release, 1)
      aie.next_bd ^v_in

    ^query_out_start:
      %4 = aie.dma_start(MM2S, 0, ^query_bridge_out, ^current_out_start)
    ^query_bridge_out:
      aie.use_lock(%mem1_query_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_query : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mem1_query_empty, Release, 1)
      aie.next_bd ^k_ping_out
    ^k_ping_out:
      aie.use_lock(%mem1_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mem1_k_empty, Release, 1)
      aie.next_bd ^k_pong_out
    ^k_pong_out:
      aie.use_lock(%mem1_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mem1_k_empty, Release, 1)
      aie.next_bd ^k_ping_out

    ^current_out_start:
      %5 = aie.dma_start(MM2S, 1, ^current_k_bridge_out, ^end)
    ^current_k_bridge_out:
      aie.use_lock(%mem1_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 29 : i32, next_bd_id = 30 : i32}}
      aie.use_lock(%mem1_current_empty, Release, 1)
      aie.next_bd ^current_v_bridge_out
    ^current_v_bridge_out:
      aie.use_lock(%mem1_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 30 : i32, next_bd_id = 31 : i32}}
      aie.use_lock(%mem1_current_empty, Release, 1)
      aie.next_bd ^v_ping_out
    ^v_ping_out:
      aie.use_lock(%mem1_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 31 : i32, next_bd_id = 32 : i32}}
      aie.use_lock(%mem1_v_empty, Release, 1)
      aie.next_bd ^v_pong_out
    ^v_pong_out:
      aie.use_lock(%mem1_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem1_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 38 : i32, next_bd_id = 31 : i32}}
      aie.use_lock(%mem1_v_empty, Release, 1)
      aie.next_bd ^v_ping_out
    ^end:
      aie.end
    }}"""


def _mem0_buffers_and_locks() -> str:
    return f"""
    %mem0_hidden = aie.buffer(%mem0) {{sym_name = "mem0_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %mem0_wt_ping = aie.buffer(%mem0) {{sym_name = "mem0_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %mem0_wt_pong = aie.buffer(%mem0) {{sym_name = "mem0_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %mem0_hidden_empty = aie.lock(%mem0, 0) {{init = 1 : i32, sym_name = "mem0_hidden_empty"}}
    %mem0_hidden_full  = aie.lock(%mem0, 1) {{init = 0 : i32, sym_name = "mem0_hidden_full"}}
    %mem0_wt_empty = aie.lock(%mem0, 2) {{init = 2 : i32, sym_name = "mem0_wt_empty"}}
    %mem0_wt_full  = aie.lock(%mem0, 3) {{init = 0 : i32, sym_name = "mem0_wt_full"}}
"""


def _projection_buffers_and_locks() -> str:
    return f"""
    %proj_hidden = aie.buffer(%proj0) {{sym_name = "proj_hidden"}} : memref<{HIDDEN_DIM}xbf16>
    %proj_wt_ping = aie.buffer(%proj0) {{sym_name = "proj_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %proj_wt_pong = aie.buffer(%proj0) {{sym_name = "proj_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %proj_query = aie.buffer(%proj0) {{sym_name = "proj_query"}} : memref<{TOKEN_DWORDS}xf32>
    %proj_current_k = aie.buffer(%proj0) {{sym_name = "proj_current_k"}} : memref<{TOKEN_DWORDS}xf32>
    %proj_current_v = aie.buffer(%proj0) {{sym_name = "proj_current_v"}} : memref<{TOKEN_DWORDS}xf32>
    %proj_hidden_empty = aie.lock(%proj0, 0) {{init = 1 : i32, sym_name = "proj_hidden_empty"}}
    %proj_hidden_full  = aie.lock(%proj0, 1) {{init = 0 : i32, sym_name = "proj_hidden_full"}}
    %proj_wt_empty = aie.lock(%proj0, 2) {{init = 2 : i32, sym_name = "proj_wt_empty"}}
    %proj_wt_full  = aie.lock(%proj0, 3) {{init = 0 : i32, sym_name = "proj_wt_full"}}
    %proj_query_ready = aie.lock(%proj0, 4) {{init = 0 : i32, sym_name = "proj_query_ready"}}
    %proj_current_k_ready = aie.lock(%proj0, 5) {{init = 0 : i32, sym_name = "proj_current_k_ready"}}
    %proj_current_v_ready = aie.lock(%proj0, 6) {{init = 0 : i32, sym_name = "proj_current_v_ready"}}
    %proj_query_sent = aie.lock(%proj0, 7) {{init = 0 : i32, sym_name = "proj_query_sent"}}
    %proj_current_k_sent = aie.lock(%proj0, 8) {{init = 0 : i32, sym_name = "proj_current_k_sent"}}
    %proj_current_v_sent = aie.lock(%proj0, 9) {{init = 0 : i32, sym_name = "proj_current_v_sent"}}
"""


def _mem1_buffers_and_locks() -> str:
    return f"""
    %mem1_query = aie.buffer(%mem1) {{sym_name = "mem1_query"}} : memref<{TOKEN_DWORDS}xf32>
    %mem1_current = aie.buffer(%mem1) {{sym_name = "mem1_current"}} : memref<{TOKEN_DWORDS}xf32>
    %mem1_k_ping = aie.buffer(%mem1) {{sym_name = "mem1_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem1_k_pong = aie.buffer(%mem1) {{sym_name = "mem1_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem1_v_ping = aie.buffer(%mem1) {{sym_name = "mem1_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem1_v_pong = aie.buffer(%mem1) {{sym_name = "mem1_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem1_query_empty = aie.lock(%mem1, 0) {{init = 1 : i32, sym_name = "mem1_query_empty"}}
    %mem1_query_full  = aie.lock(%mem1, 1) {{init = 0 : i32, sym_name = "mem1_query_full"}}
    %mem1_current_empty = aie.lock(%mem1, 2) {{init = 1 : i32, sym_name = "mem1_current_empty"}}
    %mem1_current_full  = aie.lock(%mem1, 3) {{init = 0 : i32, sym_name = "mem1_current_full"}}
    %mem1_k_empty = aie.lock(%mem1, 4) {{init = 2 : i32, sym_name = "mem1_k_empty"}}
    %mem1_k_full  = aie.lock(%mem1, 5) {{init = 0 : i32, sym_name = "mem1_k_full"}}
    %mem1_v_empty = aie.lock(%mem1, 6) {{init = 2 : i32, sym_name = "mem1_v_empty"}}
    %mem1_v_full  = aie.lock(%mem1, 7) {{init = 0 : i32, sym_name = "mem1_v_full"}}
"""


def _edge_buffers_and_locks() -> str:
    return f"""
    %edge_k_tile = aie.buffer(%edge0) {{mem_bank = 2 : i32, sym_name = "edge_k_tile"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %edge_v_tile = aie.buffer(%edge0) {{mem_bank = 3 : i32, sym_name = "edge_v_tile"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %edge_query = aie.buffer(%edge0) {{mem_bank = 1 : i32, sym_name = "edge_query"}} : memref<{TOKEN_DWORDS}xf32>
    %edge_current_k = aie.buffer(%edge0) {{mem_bank = 2 : i32, sym_name = "edge_current_k"}} : memref<{TOKEN_DWORDS}xf32>
    %edge_current_v = aie.buffer(%edge0) {{mem_bank = 3 : i32, sym_name = "edge_current_v"}} : memref<{TOKEN_DWORDS}xf32>
    %edge_current_out = aie.buffer(%edge0) {{mem_bank = 0 : i32, sym_name = "edge_current_out"}} : memref<{TOKEN_DWORDS}xf32>
    %edge_out = aie.buffer(%edge0) {{mem_bank = 1 : i32, sym_name = "edge_out"}} : memref<{OUTPUT_DWORDS}xf32>
    %edge_epilogue_tmp = aie.buffer(%edge0) {{mem_bank = 2 : i32, sym_name = "edge_epilogue_tmp"}} : memref<{OUTPUT_DWORDS}xf32>
    %edge_running_max = aie.buffer(%edge0) {{mem_bank = 3 : i32, sym_name = "edge_running_max"}} : memref<{NUM_HEADS}xf32>
    %edge_running_sum = aie.buffer(%edge0) {{mem_bank = 0 : i32, sym_name = "edge_running_sum"}} : memref<{NUM_HEADS}xf32>
    %edge_query_empty = aie.lock(%edge0, 0) {{init = 1 : i32, sym_name = "edge_query_empty"}}
    %edge_query_full  = aie.lock(%edge0, 1) {{init = 0 : i32, sym_name = "edge_query_full"}}
    %edge_current_k_empty = aie.lock(%edge0, 2) {{init = 1 : i32, sym_name = "edge_current_k_empty"}}
    %edge_current_k_full  = aie.lock(%edge0, 3) {{init = 0 : i32, sym_name = "edge_current_k_full"}}
    %edge_current_v_empty = aie.lock(%edge0, 4) {{init = 1 : i32, sym_name = "edge_current_v_empty"}}
    %edge_current_v_full  = aie.lock(%edge0, 5) {{init = 0 : i32, sym_name = "edge_current_v_full"}}
    %edge_k_empty = aie.lock(%edge0, 6) {{init = 1 : i32, sym_name = "edge_k_empty"}}
    %edge_k_full  = aie.lock(%edge0, 7) {{init = 0 : i32, sym_name = "edge_k_full"}}
    %edge_v_empty = aie.lock(%edge0, 8) {{init = 1 : i32, sym_name = "edge_v_empty"}}
    %edge_v_full  = aie.lock(%edge0, 9) {{init = 0 : i32, sym_name = "edge_v_full"}}
    %edge_current_out_prod = aie.lock(%edge0, 10) {{init = 1 : i32, sym_name = "edge_current_out_prod"}}
    %edge_current_out_cons = aie.lock(%edge0, 11) {{init = 0 : i32, sym_name = "edge_current_out_cons"}}
    %edge_out_prod = aie.lock(%edge0, 12) {{init = 1 : i32, sym_name = "edge_out_prod"}}
    %edge_out_cons = aie.lock(%edge0, 13) {{init = 0 : i32, sym_name = "edge_out_cons"}}
"""


def generate_mlir(L: int) -> str:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %proj0 = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %mem1 = aie.tile(1, 1)
    %edge0 = aie.tile(1, 2)

{_mem0_buffers_and_locks()}
{_projection_buffers_and_locks()}
{_mem1_buffers_and_locks()}
{_edge_buffers_and_locks()}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %proj0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %proj0, DMA : 1)
    aie.flow(%proj0, DMA : 0, %mem1, DMA : 2)
    aie.flow(%proj0, DMA : 1, %mem1, DMA : 3)
    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)
    aie.flow(%mem1, DMA : 0, %edge0, DMA : 0)
    aie.flow(%mem1, DMA : 1, %edge0, DMA : 1)
    aie.flow(%edge0, DMA : 0, %shim1, DMA : 0)

    func.func private @zero_token(memref<{TOKEN_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @q4nx_project_chunk_accum(memref<{CHUNK_BF16}xbf16>, memref<{HIDDEN_DIM}xbf16>, memref<{TOKEN_DWORDS}xf32>, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @copy_token(memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @init_attention_state(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @online_softmax_attention(memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>, memref<{NUM_HEADS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @finalize_attention(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @probe_tile(memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, i32) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}
    func.func private @layer_epilogue(memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/qkv_attention.o"}}

{_projection_core()}
{_edge_core(num_tiles, last_valid)}
{_projection_memtile_dma()}
{_projection_mem()}
{_edge_memtile_dma()}
{_edge_mem()}
{_runtime_sequence(L, num_tiles)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
