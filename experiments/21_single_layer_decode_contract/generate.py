"""
Experiment 21: reduced single-layer decode contract.

This keeps the raw MLIR-AIE current-write KV path from exp20, then adds two
core-local phases:
- derive a current-token query from current K/V without adding a new runtime DMA
- run a post-attention layer epilogue before the final drain
"""

from math import ceil
from pathlib import Path

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE
OUTPUT_DWORDS_PER_WORKER = NUM_KV_HEADS_PER_GROUP * HEAD_DIM


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

    k03 = 0
    v03 = plane_bytes
    k47 = plane_bytes * 2
    v47 = plane_bytes * 3

    current_k03 = 0
    current_v03 = token_bytes
    current_k47 = token_bytes * 2
    current_v47 = token_bytes * 3

    lines = [
        f"    aie.runtime_sequence(%current: memref<{4 * TOKEN_DWORDS}xf32>, "
        f"%kv_cache: memref<{4 * total_plane_dwords}xf32>, "
        f"%output: memref<{2 * OUTPUT_DWORDS_PER_WORKER}xf32>) {{"
    ]

    lines += [
        _npu_writebd(0, 0, TOKEN_DWORDS, current_k03),
        _npu_address_patch(0, 0, 0, current_k03),
        _npu_push_queue(0, "MM2S", 0, 0),
        _npu_writebd(1, 0, TOKEN_DWORDS, current_v03),
        _npu_address_patch(1, 0, 0, current_v03),
        _npu_push_queue(1, "MM2S", 0, 0),
        _npu_writebd(0, 1, TOKEN_DWORDS, current_k47),
        _npu_address_patch(0, 1, 0, current_k47),
        _npu_push_queue(0, "MM2S", 1, 1),
        _npu_writebd(1, 1, TOKEN_DWORDS, current_v47),
        _npu_address_patch(1, 1, 0, current_v47),
        _npu_push_queue(1, "MM2S", 1, 1),
    ]

    lines += [
        _npu_writebd(0, 2, TOKEN_DWORDS, k03 + current_token_offset),
        _npu_address_patch(0, 2, 1, k03 + current_token_offset),
        _npu_push_queue(0, "S2MM", 0, 2),
        _npu_writebd(0, 3, TOKEN_DWORDS, v03 + current_token_offset),
        _npu_address_patch(0, 3, 1, v03 + current_token_offset),
        _npu_push_queue(0, "S2MM", 0, 3, issue_token=True),
        _npu_writebd(1, 2, TOKEN_DWORDS, k47 + current_token_offset),
        _npu_address_patch(1, 2, 1, k47 + current_token_offset),
        _npu_push_queue(1, "S2MM", 0, 2),
        _npu_writebd(1, 3, TOKEN_DWORDS, v47 + current_token_offset),
        _npu_address_patch(1, 3, 1, v47 + current_token_offset),
        _npu_push_queue(1, "S2MM", 0, 3, issue_token=True),
        _npu_sync(0, 0),
        _npu_sync(1, 0),
    ]

    lines += [
        _npu_writebd(0, 4, total_plane_dwords, k03),
        _npu_address_patch(0, 4, 1, k03),
        _npu_push_queue(0, "MM2S", 0, 4),
        _npu_writebd(0, 5, total_plane_dwords, k47),
        _npu_address_patch(0, 5, 1, k47),
        _npu_push_queue(0, "MM2S", 1, 5),
        _npu_writebd(1, 4, total_plane_dwords, v03),
        _npu_address_patch(1, 4, 1, v03),
        _npu_push_queue(1, "MM2S", 0, 4),
        _npu_writebd(1, 5, total_plane_dwords, v47),
        _npu_address_patch(1, 5, 1, v47),
        _npu_push_queue(1, "MM2S", 1, 5),
    ]

    lines += [
        _npu_writebd(0, 6, OUTPUT_DWORDS_PER_WORKER, 0),
        _npu_address_patch(0, 6, 2, 0),
        _npu_push_queue(0, "S2MM", 0, 6, issue_token=True),
        _npu_writebd(1, 6, OUTPUT_DWORDS_PER_WORKER, OUTPUT_DWORDS_PER_WORKER * 4),
        _npu_address_patch(1, 6, 2, OUTPUT_DWORDS_PER_WORKER * 4),
        _npu_push_queue(1, "S2MM", 0, 6, issue_token=True),
        _npu_sync(0, 0),
        _npu_sync(1, 0),
        "    }",
    ]
    return "\n".join(lines)


def _worker_core(name: str, num_tiles: int, last_valid: int) -> str:
    return f"""    %{name}_core = aie.core(%{name}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32
      %num_tiles_idx = arith.constant {num_tiles} : index

      aie.use_lock(%{name}_current_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{name}_current_v_full, AcquireGreaterEqual, 1)

      func.call @derive_query(%{name}_current_k, %{name}_current_v, %{name}_query)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()

      aie.use_lock(%{name}_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%{name}_current_k, %{name}_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%{name}_current_out_cons, Release, 1)
      aie.use_lock(%{name}_current_k_empty, Release, 1)

      aie.use_lock(%{name}_current_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_token(%{name}_current_v, %{name}_current_out)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) -> ()
      aie.use_lock(%{name}_current_out_cons, Release, 1)
      aie.use_lock(%{name}_current_v_empty, Release, 1)

      aie.use_lock(%{name}_out_prod, AcquireGreaterEqual, 1)
      func.call @init_attention_state(%{name}_out, %{name}_running_max, %{name}_running_sum)
        : (memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%{name}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{name}_v_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @online_softmax_attention(%{name}_query, %{name}_k_pong, %{name}_v_pong, %{name}_out, %{name}_running_max, %{name}_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
            : (memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, i32, i32, i32) -> ()
        }} else {{
          func.call @online_softmax_attention(%{name}_query, %{name}_k_ping, %{name}_v_ping, %{name}_out, %{name}_running_max, %{name}_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
            : (memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, i32, i32, i32) -> ()
        }}

        aie.use_lock(%{name}_k_empty, Release, 1)
        aie.use_lock(%{name}_v_empty, Release, 1)
      }}

      func.call @finalize_attention(%{name}_out, %{name}_running_sum)
        : (memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>) -> ()
      func.call @layer_epilogue(%{name}_query, %{name}_out, %{name}_epilogue_tmp)
        : (memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>) -> ()
      aie.use_lock(%{name}_out_cons, Release, 1)
      aie.end
    }}"""


def _worker_mem(name: str) -> str:
    return f"""    %{name}_mem = aie.mem(%{name}) {{
      %0 = aie.dma_start(S2MM, 0, ^current_k, ^v_start)
    ^current_k:
      aie.use_lock(%{name}_current_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_current_k : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{name}_current_k_full, Release, 1)
      aie.next_bd ^k_ping
    ^k_ping:
      aie.use_lock(%{name}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{name}_k_full, Release, 1)
      aie.next_bd ^k_pong
    ^k_pong:
      aie.use_lock(%{name}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{name}_k_full, Release, 1)
      aie.next_bd ^k_ping

    ^v_start:
      %1 = aie.dma_start(S2MM, 1, ^current_v, ^out_start)
    ^current_v:
      aie.use_lock(%{name}_current_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_current_v : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 6 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{name}_current_v_full, Release, 1)
      aie.next_bd ^v_ping
    ^v_ping:
      aie.use_lock(%{name}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{name}_v_full, Release, 1)
      aie.next_bd ^v_pong
    ^v_pong:
      aie.use_lock(%{name}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{name}_v_full, Release, 1)
      aie.next_bd ^v_ping

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%{name}_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 7 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%{name}_current_out_prod, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%{name}_current_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_current_out : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 8 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{name}_current_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^out_bd:
      aie.use_lock(%{name}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_out : memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, 0, {OUTPUT_DWORDS_PER_WORKER}) {{bd_id = 4 : i32}}
      aie.use_lock(%{name}_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _memtile_dma(mem_name: str) -> str:
    return f"""    %{mem_name}_dma = aie.memtile_dma(%{mem_name}) {{
      %0 = aie.dma_start(S2MM, 0, ^a_current_in, ^a_out_start)
    ^a_current_in:
      aie.use_lock(%{mem_name}_a_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{mem_name}_a_current_full, Release, 1)
      aie.next_bd ^a_s2mm_ping
    ^a_s2mm_ping:
      aie.use_lock(%{mem_name}_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{mem_name}_a_full, Release, 1)
      aie.next_bd ^a_s2mm_pong
    ^a_s2mm_pong:
      aie.use_lock(%{mem_name}_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{mem_name}_a_full, Release, 1)
      aie.next_bd ^a_s2mm_ping

    ^a_out_start:
      %1 = aie.dma_start(MM2S, 0, ^a_current_out, ^b_start)
    ^a_current_out:
      aie.use_lock(%{mem_name}_a_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{mem_name}_a_current_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping
    ^a_mm2s_ping:
      aie.use_lock(%{mem_name}_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{mem_name}_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_pong
    ^a_mm2s_pong:
      aie.use_lock(%{mem_name}_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{mem_name}_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping

    ^b_start:
      %2 = aie.dma_start(S2MM, 1, ^b_current_in, ^b_out_start)
    ^b_current_in:
      aie.use_lock(%{mem_name}_b_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 28 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{mem_name}_b_current_full, Release, 1)
      aie.next_bd ^b_s2mm_ping
    ^b_s2mm_ping:
      aie.use_lock(%{mem_name}_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%{mem_name}_b_full, Release, 1)
      aie.next_bd ^b_s2mm_pong
    ^b_s2mm_pong:
      aie.use_lock(%{mem_name}_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{mem_name}_b_full, Release, 1)
      aie.next_bd ^b_s2mm_ping

    ^b_out_start:
      %3 = aie.dma_start(MM2S, 1, ^b_current_out, ^end)
    ^b_current_out:
      aie.use_lock(%{mem_name}_b_current_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_current : memref<{TOKEN_DWORDS}xf32>, 0, {TOKEN_DWORDS}) {{bd_id = 29 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%{mem_name}_b_current_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^b_mm2s_ping:
      aie.use_lock(%{mem_name}_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%{mem_name}_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_pong
    ^b_mm2s_pong:
      aie.use_lock(%{mem_name}_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%{mem_name}_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^end:
      aie.end
    }}"""


def _memtile_buffers_and_locks(mem_name: str) -> str:
    return f"""
    %{mem_name}_a_current = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_a_current"}} : memref<{TOKEN_DWORDS}xf32>
    %{mem_name}_a_ping = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_a_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{mem_name}_a_pong = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_a_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{mem_name}_b_current = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_b_current"}} : memref<{TOKEN_DWORDS}xf32>
    %{mem_name}_b_ping = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_b_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{mem_name}_b_pong = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_b_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>

    %{mem_name}_a_current_empty = aie.lock(%{mem_name}, 4) {{init = 1 : i32, sym_name = "{mem_name}_a_current_empty"}}
    %{mem_name}_a_current_full  = aie.lock(%{mem_name}, 5) {{init = 0 : i32, sym_name = "{mem_name}_a_current_full"}}
    %{mem_name}_a_empty = aie.lock(%{mem_name}, 0) {{init = 2 : i32, sym_name = "{mem_name}_a_empty"}}
    %{mem_name}_a_full  = aie.lock(%{mem_name}, 1) {{init = 0 : i32, sym_name = "{mem_name}_a_full"}}
    %{mem_name}_b_current_empty = aie.lock(%{mem_name}, 6) {{init = 1 : i32, sym_name = "{mem_name}_b_current_empty"}}
    %{mem_name}_b_current_full  = aie.lock(%{mem_name}, 7) {{init = 0 : i32, sym_name = "{mem_name}_b_current_full"}}
    %{mem_name}_b_empty = aie.lock(%{mem_name}, 2) {{init = 2 : i32, sym_name = "{mem_name}_b_empty"}}
    %{mem_name}_b_full  = aie.lock(%{mem_name}, 3) {{init = 0 : i32, sym_name = "{mem_name}_b_full"}}
"""


def _worker_buffers_and_locks(name: str) -> str:
    return f"""
    %{name}_current_k = aie.buffer(%{name}) {{sym_name = "{name}_current_k"}} : memref<{TOKEN_DWORDS}xf32>
    %{name}_current_v = aie.buffer(%{name}) {{sym_name = "{name}_current_v"}} : memref<{TOKEN_DWORDS}xf32>
    %{name}_current_out = aie.buffer(%{name}) {{sym_name = "{name}_current_out"}} : memref<{TOKEN_DWORDS}xf32>
    %{name}_k_ping = aie.buffer(%{name}) {{sym_name = "{name}_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{name}_k_pong = aie.buffer(%{name}) {{sym_name = "{name}_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{name}_v_ping = aie.buffer(%{name}) {{sym_name = "{name}_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{name}_v_pong = aie.buffer(%{name}) {{sym_name = "{name}_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{name}_query = aie.buffer(%{name}) {{sym_name = "{name}_query"}} : memref<{TOKEN_DWORDS}xf32>
    %{name}_out = aie.buffer(%{name}) {{sym_name = "{name}_out"}} : memref<{OUTPUT_DWORDS_PER_WORKER}xf32>
    %{name}_epilogue_tmp = aie.buffer(%{name}) {{sym_name = "{name}_epilogue_tmp"}} : memref<{OUTPUT_DWORDS_PER_WORKER}xf32>
    %{name}_running_max = aie.buffer(%{name}) {{sym_name = "{name}_running_max"}} : memref<{NUM_KV_HEADS_PER_GROUP}xf32>
    %{name}_running_sum = aie.buffer(%{name}) {{sym_name = "{name}_running_sum"}} : memref<{NUM_KV_HEADS_PER_GROUP}xf32>

    %{name}_current_k_empty = aie.lock(%{name}, 6) {{init = 1 : i32, sym_name = "{name}_current_k_empty"}}
    %{name}_current_k_full  = aie.lock(%{name}, 7) {{init = 0 : i32, sym_name = "{name}_current_k_full"}}
    %{name}_current_v_empty = aie.lock(%{name}, 8) {{init = 1 : i32, sym_name = "{name}_current_v_empty"}}
    %{name}_current_v_full  = aie.lock(%{name}, 9) {{init = 0 : i32, sym_name = "{name}_current_v_full"}}
    %{name}_current_out_prod = aie.lock(%{name}, 10) {{init = 1 : i32, sym_name = "{name}_current_out_prod"}}
    %{name}_current_out_cons = aie.lock(%{name}, 11) {{init = 0 : i32, sym_name = "{name}_current_out_cons"}}
    %{name}_k_empty = aie.lock(%{name}, 0) {{init = 2 : i32, sym_name = "{name}_k_empty"}}
    %{name}_k_full  = aie.lock(%{name}, 1) {{init = 0 : i32, sym_name = "{name}_k_full"}}
    %{name}_v_empty = aie.lock(%{name}, 2) {{init = 2 : i32, sym_name = "{name}_v_empty"}}
    %{name}_v_full  = aie.lock(%{name}, 3) {{init = 0 : i32, sym_name = "{name}_v_full"}}
    %{name}_out_prod = aie.lock(%{name}, 4) {{init = 1 : i32, sym_name = "{name}_out_prod"}}
    %{name}_out_cons = aie.lock(%{name}, 5) {{init = 0 : i32, sym_name = "{name}_out_cons"}}
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
    %shim1 = aie.tile(1, 0)
    %mem1 = aie.tile(1, 1)
    %w1 = aie.tile(1, 2)

{_memtile_buffers_and_locks("mem0")}
{_memtile_buffers_and_locks("mem1")}
{_worker_buffers_and_locks("w0")}
{_worker_buffers_and_locks("w1")}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)
    aie.flow(%mem0, DMA : 0, %w0, DMA : 0)
    aie.flow(%mem1, DMA : 0, %w0, DMA : 1)
    aie.flow(%mem0, DMA : 1, %w1, DMA : 0)
    aie.flow(%mem1, DMA : 1, %w1, DMA : 1)
    aie.flow(%w0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%w1, DMA : 0, %shim1, DMA : 0)

    func.func private @copy_token(memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @derive_query(memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>, memref<{TOKEN_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @init_attention_state(memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @online_softmax_attention(memref<{TOKEN_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{PLANE_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @finalize_attention(memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{NUM_KV_HEADS_PER_GROUP}xf32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @layer_epilogue(memref<{TOKEN_DWORDS}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>, memref<{OUTPUT_DWORDS_PER_WORKER}xf32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}

{_worker_core("w0", num_tiles, last_valid)}

{_worker_core("w1", num_tiles, last_valid)}

{_memtile_dma("mem0")}

{_memtile_dma("mem1")}

{_worker_mem("w0")}

{_worker_mem("w1")}

{_runtime_sequence(L, num_tiles)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
