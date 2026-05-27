"""Generate raw MLIR-AIE for exp33 row1 KV reshape attention."""

from math import ceil
from pathlib import Path

NUM_Q_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def last_valid_for_context(context_len: int) -> int:
    return context_len - (num_tiles_for_context(context_len) - 1) * TOKENS_PER_TILE


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
    lines = [
        f"    aie.runtime_sequence(%query: memref<{QUERY_DWORDS}xf32>, "
        f"%kv_cache: memref<{total_kv_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{",
        _npu_writebd(0, 0, QUERY_DWORDS, 0),
        _npu_address_patch(0, 0, 0, 0),
        _npu_push_queue(0, "MM2S", 0, 0),
    ]

    for tile in range(num_tiles):
        k_offset = tile * PLANE_TILE_DWORDS * 4
        v_offset = (k_plane_dwords + tile * PLANE_TILE_DWORDS) * 4
        k_bd = 1 + tile
        v_bd = 8 + tile
        lines += [
            _npu_writebd(0, k_bd, PLANE_TILE_DWORDS, k_offset),
            _npu_address_patch(0, k_bd, 1, k_offset),
            _npu_push_queue(0, "MM2S", 0, k_bd),
            _npu_writebd(0, v_bd, PLANE_TILE_DWORDS, v_offset),
            _npu_address_patch(0, v_bd, 1, v_offset),
            _npu_push_queue(0, "MM2S", 1, v_bd),
        ]

    lines += [
        _npu_writebd(0, 15, OUTPUT_DWORDS, 0),
        _npu_address_patch(0, 15, 2, 0),
        _npu_push_queue(0, "S2MM", 0, 15, issue_token=True),
        _npu_sync(0, 0),
        "    }",
    ]
    return "\n".join(lines)


def _memtile_buffers_and_locks() -> str:
    return f"""
    %mem_query = aie.buffer(%mem0) {{sym_name = "mem_query"}} : memref<{QUERY_DWORDS}xf32>
    %mem_k_ping = aie.buffer(%mem0) {{sym_name = "mem_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem_k_pong = aie.buffer(%mem0) {{sym_name = "mem_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem_v_ping = aie.buffer(%mem0) {{sym_name = "mem_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %mem_v_pong = aie.buffer(%mem0) {{sym_name = "mem_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>

    %mem_query_empty = aie.lock(%mem0, 8) {{init = 1 : i32, sym_name = "mem_query_empty"}}
    %mem_query_full = aie.lock(%mem0, 9) {{init = 0 : i32, sym_name = "mem_query_full"}}
    %mem_k_empty = aie.lock(%mem0, 0) {{init = 2 : i32, sym_name = "mem_k_empty"}}
    %mem_k_full = aie.lock(%mem0, 1) {{init = 0 : i32, sym_name = "mem_k_full"}}
    %mem_v_empty = aie.lock(%mem0, 2) {{init = 2 : i32, sym_name = "mem_v_empty"}}
    %mem_v_full = aie.lock(%mem0, 3) {{init = 0 : i32, sym_name = "mem_v_full"}}
"""


def _worker_buffers_and_locks() -> str:
    return f"""
    %w_query = aie.buffer(%worker) {{sym_name = "w_query"}} : memref<{QUERY_DWORDS}xf32>
    %w_k_tile = aie.buffer(%worker) {{sym_name = "w_k_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %w_v_tile = aie.buffer(%worker) {{sym_name = "w_v_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %w_out = aie.buffer(%worker) {{sym_name = "w_out"}} : memref<{OUTPUT_DWORDS}xf32>
    %w_running_max = aie.buffer(%worker) {{sym_name = "w_running_max"}} : memref<{NUM_Q_HEADS}xf32>
    %w_running_sum = aie.buffer(%worker) {{sym_name = "w_running_sum"}} : memref<{NUM_Q_HEADS}xf32>

    %w_query_empty = aie.lock(%worker, 8) {{init = 1 : i32, sym_name = "w_query_empty"}}
    %w_query_full = aie.lock(%worker, 9) {{init = 0 : i32, sym_name = "w_query_full"}}
    %w_k_empty = aie.lock(%worker, 0) {{init = 1 : i32, sym_name = "w_k_empty"}}
    %w_k_full = aie.lock(%worker, 1) {{init = 0 : i32, sym_name = "w_k_full"}}
    %w_v_empty = aie.lock(%worker, 2) {{init = 1 : i32, sym_name = "w_v_empty"}}
    %w_v_full = aie.lock(%worker, 3) {{init = 0 : i32, sym_name = "w_v_full"}}
    %w_out_empty = aie.lock(%worker, 4) {{init = 1 : i32, sym_name = "w_out_empty"}}
    %w_out_full = aie.lock(%worker, 5) {{init = 0 : i32, sym_name = "w_out_full"}}
"""


def _memtile_dma() -> str:
    dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    return f"""    %mem0_dma = aie.memtile_dma(%mem0) {{
      %0 = aie.dma_start(S2MM, 0, ^query_in, ^v_in_start)
    ^query_in:
      aie.use_lock(%mem_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_query : memref<{QUERY_DWORDS}xf32>, 0, {QUERY_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem_query_full, Release, 1)
      aie.next_bd ^k_in_ping
    ^k_in_ping:
      aie.use_lock(%mem_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mem_k_full, Release, 1)
      aie.next_bd ^k_in_pong
    ^k_in_pong:
      aie.use_lock(%mem_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mem_k_full, Release, 1)
      aie.next_bd ^k_in_ping

    ^v_in_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in_ping, ^query_out_start)
    ^v_in_ping:
      aie.use_lock(%mem_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mem_v_full, Release, 1)
      aie.next_bd ^v_in_pong
    ^v_in_pong:
      aie.use_lock(%mem_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mem_v_full, Release, 1)
      aie.next_bd ^v_in_ping

    ^query_out_start:
      %2 = aie.dma_start(MM2S, 0, ^query_out, ^v_out_start)
    ^query_out:
      aie.use_lock(%mem_query_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_query : memref<{QUERY_DWORDS}xf32>, 0, {QUERY_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mem_query_empty, Release, 1)
      aie.next_bd ^k_out_ping
    ^k_out_ping:
      aie.use_lock(%mem_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%mem_k_empty, Release, 1)
      aie.next_bd ^k_out_pong
    ^k_out_pong:
      aie.use_lock(%mem_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 5 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%mem_k_empty, Release, 1)
      aie.next_bd ^k_out_ping

    ^v_out_start:
      %3 = aie.dma_start(MM2S, 1, ^v_out_ping, ^end)
    ^v_out_ping:
      aie.use_lock(%mem_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mem_v_empty, Release, 1)
      aie.next_bd ^v_out_pong
    ^v_out_pong:
      aie.use_lock(%mem_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mem_v_empty, Release, 1)
      aie.next_bd ^v_out_ping
    ^end:
      aie.end
    }}"""


def _worker_mem() -> str:
    return f"""    %worker_mem = aie.mem(%worker) {{
      %0 = aie.dma_start(S2MM, 0, ^query_in, ^v_start)
    ^query_in:
      aie.use_lock(%w_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w_query : memref<{QUERY_DWORDS}xf32>, 0, {QUERY_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w_query_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%w_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w_k_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w_k_full, Release, 1)
      aie.next_bd ^k_in

    ^v_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in, ^out_start)
    ^v_in:
      aie.use_lock(%w_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w_v_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w_v_full, Release, 1)
      aie.next_bd ^v_in

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%w_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%w_out : memref<{OUTPUT_DWORDS}xf32>, 0, {OUTPUT_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%w_out_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _worker_core(num_tiles: int, last_valid: int) -> str:
    return f"""    %worker_core = aie.core(%worker) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %num_tiles_idx = arith.constant {num_tiles} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      aie.use_lock(%w_query_full, AcquireGreaterEqual, 1)
      aie.use_lock(%w_out_empty, AcquireGreaterEqual, 1)
      func.call @init_attention_state(%w_out, %w_running_max, %w_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        aie.use_lock(%w_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%w_v_full, AcquireGreaterEqual, 1)
        func.call @online_attention_reshaped(%w_query, %w_k_tile, %w_v_tile, %w_out, %w_running_max, %w_running_sum, %i_i32, %num_tiles_i32, %last_valid_i32)
          : (memref<{QUERY_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>, i32, i32, i32) -> ()
        aie.use_lock(%w_k_empty, Release, 1)
        aie.use_lock(%w_v_empty, Release, 1)
      }}

      func.call @finalize_attention(%w_out, %w_running_sum)
        : (memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>) -> ()
      aie.use_lock(%w_query_empty, Release, 1)
      aie.use_lock(%w_out_full, Release, 1)
      aie.end
    }}"""


def generate_mlir(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    last_valid = last_valid_for_context(context_len)
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %worker = aie.tile(0, 2)

{_memtile_buffers_and_locks()}
{_worker_buffers_and_locks()}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker, DMA : 1)
    aie.flow(%worker, DMA : 0, %shim0, DMA : 0)

    func.func private @init_attention_state(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}
    func.func private @online_attention_reshaped(memref<{QUERY_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>, memref<{NUM_Q_HEADS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}
    func.func private @finalize_attention(memref<{OUTPUT_DWORDS}xf32>, memref<{NUM_Q_HEADS}xf32>) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}

{_worker_core(num_tiles, last_valid)}

{_memtile_dma()}

{_worker_mem()}

{_runtime_sequence(context_len)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
