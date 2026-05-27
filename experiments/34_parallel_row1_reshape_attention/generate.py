"""Generate raw MLIR-AIE for exp34 parallel row1 reshape attention."""

from math import ceil
from pathlib import Path
import sys

NUM_Q_HEADS = 4
NUM_COLS = NUM_Q_HEADS
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

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


def _npu_sync(column: int) -> str:
    return (
        f"      aiex.npu.sync {{channel = 0 : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _runtime_sequence(context_len: int) -> str:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    total_kv_dwords = 2 * k_plane_dwords
    lines = [
        f"    aie.runtime_sequence(%query: memref<{QUERY_DWORDS}xf32>, "
        f"%kv_cache: memref<{total_kv_dwords}xf32>, "
        f"%output: memref<{OUTPUT_DWORDS}xf32>) {{"
    ]

    for column in range(NUM_COLS):
        query_offset = column * QUERY_HEAD_DWORDS * 4
        lines += [
            _npu_writebd(column, 0, QUERY_HEAD_DWORDS, query_offset),
            _npu_address_patch(column, 0, 0, query_offset),
            _npu_push_queue(column, "MM2S", 0, 0),
        ]

    for tile in range(num_tiles):
        k_offset = tile * PLANE_TILE_DWORDS * 4
        v_offset = (k_plane_dwords + tile * PLANE_TILE_DWORDS) * 4
        k_bd = 1 + tile
        v_bd = 8 + tile
        for column in range(NUM_COLS):
            lines += [
                _npu_writebd(column, k_bd, PLANE_TILE_DWORDS, k_offset),
                _npu_address_patch(column, k_bd, 1, k_offset),
                _npu_push_queue(column, "MM2S", 0, k_bd),
                _npu_writebd(column, v_bd, PLANE_TILE_DWORDS, v_offset),
                _npu_address_patch(column, v_bd, 1, v_offset),
                _npu_push_queue(column, "MM2S", 1, v_bd),
            ]

    for column in range(NUM_COLS):
        output_offset = column * HEAD_DIM * 4
        lines += [
            _npu_writebd(column, 15, HEAD_DIM, output_offset),
            _npu_address_patch(column, 15, 2, output_offset),
            _npu_push_queue(column, "S2MM", 0, 15, issue_token=True),
        ]

    for column in range(NUM_COLS):
        lines.append(_npu_sync(column))
    lines.append("    }")
    return "\n".join(lines)


def _memtile_buffers_and_locks(column: int) -> str:
    prefix = f"mem{column}"
    return f"""
    %{prefix}_query = aie.buffer(%{prefix}) {{sym_name = "{prefix}_query"}} : memref<{QUERY_HEAD_DWORDS}xf32>
    %{prefix}_k_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_k_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_v_ping = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_ping"}} : memref<{PLANE_TILE_DWORDS}xf32>
    %{prefix}_v_pong = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_pong"}} : memref<{PLANE_TILE_DWORDS}xf32>

    %{prefix}_query_empty = aie.lock(%{prefix}, 8) {{init = 1 : i32, sym_name = "{prefix}_query_empty"}}
    %{prefix}_query_full = aie.lock(%{prefix}, 9) {{init = 0 : i32, sym_name = "{prefix}_query_full"}}
    %{prefix}_k_empty = aie.lock(%{prefix}, 0) {{init = 2 : i32, sym_name = "{prefix}_k_empty"}}
    %{prefix}_k_full = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = "{prefix}_k_full"}}
    %{prefix}_v_empty = aie.lock(%{prefix}, 2) {{init = 2 : i32, sym_name = "{prefix}_v_empty"}}
    %{prefix}_v_full = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = "{prefix}_v_full"}}
"""


def _worker_buffers_and_locks(column: int) -> str:
    prefix = f"w{column}"
    return f"""
    %{prefix}_query = aie.buffer(%{prefix}) {{sym_name = "{prefix}_query"}} : memref<{QUERY_HEAD_DWORDS}xf32>
    %{prefix}_k_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_k_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_v_tile = aie.buffer(%{prefix}) {{sym_name = "{prefix}_v_tile"}} : memref<{RESHAPED_TILE_DWORDS}xf32>
    %{prefix}_out = aie.buffer(%{prefix}) {{sym_name = "{prefix}_out"}} : memref<{HEAD_DIM}xf32>
    %{prefix}_running_max = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_max"}} : memref<{STATE_DWORDS}xf32>
    %{prefix}_running_sum = aie.buffer(%{prefix}) {{sym_name = "{prefix}_running_sum"}} : memref<{STATE_DWORDS}xf32>

    %{prefix}_query_empty = aie.lock(%{prefix}, 8) {{init = 1 : i32, sym_name = "{prefix}_query_empty"}}
    %{prefix}_query_full = aie.lock(%{prefix}, 9) {{init = 0 : i32, sym_name = "{prefix}_query_full"}}
    %{prefix}_k_empty = aie.lock(%{prefix}, 0) {{init = 1 : i32, sym_name = "{prefix}_k_empty"}}
    %{prefix}_k_full = aie.lock(%{prefix}, 1) {{init = 0 : i32, sym_name = "{prefix}_k_full"}}
    %{prefix}_v_empty = aie.lock(%{prefix}, 2) {{init = 1 : i32, sym_name = "{prefix}_v_empty"}}
    %{prefix}_v_full = aie.lock(%{prefix}, 3) {{init = 0 : i32, sym_name = "{prefix}_v_full"}}
    %{prefix}_out_empty = aie.lock(%{prefix}, 4) {{init = 1 : i32, sym_name = "{prefix}_out_empty"}}
    %{prefix}_out_full = aie.lock(%{prefix}, 5) {{init = 0 : i32, sym_name = "{prefix}_out_full"}}
"""


def _memtile_dma(column: int) -> str:
    prefix = f"mem{column}"
    dims = (
        f"[<size = {DIM_GROUPS}, stride = {GROUP_DWORDS}>, "
        f"<size = {TOKENS_PER_TILE}, stride = {HEAD_DIM}>, "
        f"<size = {GROUP_DWORDS}, stride = 1>]"
    )
    return f"""    %{prefix}_dma = aie.memtile_dma(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 0, ^query_in, ^v_in_start)
    ^query_in:
      aie.use_lock(%{prefix}_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_query : memref<{QUERY_HEAD_DWORDS}xf32>, 0, {QUERY_HEAD_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_query_full, Release, 1)
      aie.next_bd ^k_in_ping
    ^k_in_ping:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, 1)
      aie.next_bd ^k_in_pong
    ^k_in_pong:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, 1)
      aie.next_bd ^k_in_ping

    ^v_in_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in_ping, ^query_out_start)
    ^v_in_ping:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, 1)
      aie.next_bd ^v_in_pong
    ^v_in_pong:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, 1)
      aie.next_bd ^v_in_ping

    ^query_out_start:
      %2 = aie.dma_start(MM2S, 0, ^query_out, ^v_out_start)
    ^query_out:
      aie.use_lock(%{prefix}_query_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_query : memref<{QUERY_HEAD_DWORDS}xf32>, 0, {QUERY_HEAD_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{prefix}_query_empty, Release, 1)
      aie.next_bd ^k_out_ping
    ^k_out_ping:
      aie.use_lock(%{prefix}_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{prefix}_k_empty, Release, 1)
      aie.next_bd ^k_out_pong
    ^k_out_pong:
      aie.use_lock(%{prefix}_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 5 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{prefix}_k_empty, Release, 1)
      aie.next_bd ^k_out_ping

    ^v_out_start:
      %3 = aie.dma_start(MM2S, 1, ^v_out_ping, ^end)
    ^v_out_ping:
      aie.use_lock(%{prefix}_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_ping : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%{prefix}_v_empty, Release, 1)
      aie.next_bd ^v_out_pong
    ^v_out_pong:
      aie.use_lock(%{prefix}_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_pong : memref<{PLANE_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}, {dims}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%{prefix}_v_empty, Release, 1)
      aie.next_bd ^v_out_ping
    ^end:
      aie.end
    }}"""


def _worker_mem(column: int) -> str:
    prefix = f"w{column}"
    return f"""    %{prefix}_mem = aie.mem(%{prefix}) {{
      %0 = aie.dma_start(S2MM, 0, ^query_in, ^v_start)
    ^query_in:
      aie.use_lock(%{prefix}_query_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_query : memref<{QUERY_HEAD_DWORDS}xf32>, 0, {QUERY_HEAD_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_query_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%{prefix}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_k_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{prefix}_k_full, Release, 1)
      aie.next_bd ^k_in

    ^v_start:
      %1 = aie.dma_start(S2MM, 1, ^v_in, ^out_start)
    ^v_in:
      aie.use_lock(%{prefix}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_v_tile : memref<{RESHAPED_TILE_DWORDS}xf32>, 0, {RESHAPED_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{prefix}_v_full, Release, 1)
      aie.next_bd ^v_in

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{prefix}_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_out : memref<{HEAD_DIM}xf32>, 0, {HEAD_DIM}) {{bd_id = 3 : i32}}
      aie.use_lock(%{prefix}_out_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _worker_core(column: int, num_tiles: int, last_valid: int) -> str:
    prefix = f"w{column}"
    return f"""    %{prefix}_core = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %num_tiles_idx = arith.constant {num_tiles} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      aie.use_lock(%{prefix}_query_full, AcquireGreaterEqual, 1)
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
      aie.use_lock(%{prefix}_query_empty, Release, 1)
      aie.use_lock(%{prefix}_out_full, Release, 1)
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
    buffers = "\n".join(
        _memtile_buffers_and_locks(column) + _worker_buffers_and_locks(column)
        for column in range(NUM_COLS)
    )
    flows = "\n".join(
        f"    aie.flow(%shim{column}, DMA : 0, %mem{column}, DMA : 0)\n"
        f"    aie.flow(%shim{column}, DMA : 1, %mem{column}, DMA : 1)\n"
        f"    aie.flow(%mem{column}, DMA : 0, %w{column}, DMA : 0)\n"
        f"    aie.flow(%mem{column}, DMA : 1, %w{column}, DMA : 1)\n"
        f"    aie.flow(%w{column}, DMA : 0, %shim{column}, DMA : 0)"
        for column in range(NUM_COLS)
    )
    cores = "\n\n".join(_worker_core(column, num_tiles, last_valid) for column in range(NUM_COLS))
    memtiles = "\n\n".join(_memtile_dma(column) for column in range(NUM_COLS))
    worker_mems = "\n\n".join(_worker_mem(column) for column in range(NUM_COLS))
    return f"""module {{
  aie.device(npu2) {{
{tile_defs}

{buffers}

{flows}

    func.func private @init_attention_state_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}
    func.func private @online_attention_reshaped_head(memref<{HEAD_DIM}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{RESHAPED_TILE_DWORDS}xf32>, memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>, memref<{STATE_DWORDS}xf32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}
    func.func private @finalize_attention_head(memref<{HEAD_DIM}xf32>, memref<{STATE_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/reshape_attention.o"}}

{cores}

{memtiles}

{worker_mems}

{_runtime_sequence(context_len)}
  }}
}}
"""


if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(length))
