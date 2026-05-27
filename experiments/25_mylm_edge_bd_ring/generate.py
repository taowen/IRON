"""Generate exp25 runnable MyLM-style edge/KV BD-ring skeleton."""

from math import ceil
from pathlib import Path

TOKENS_PER_TILE = 16
TOKEN_DWORDS = 256
PLANE_TILE_DWORDS = 0x1000
HALF_TILE_DWORDS = PLANE_TILE_DWORDS // 2
CHECKSUM_DWORDS = 4
NUM_PLANES = 4


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(
    column: int, bd_id: int, buffer_length: int, buffer_offset: int
) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 64 : i32, column = {column} : i32, "
        f"d0_size = 0 : i32, d0_stride = 0 : i32, "
        f"d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = 0 : i32, d1_stride = 0 : i32, "
        f"d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = 0 : i32, d2_stride = 0 : i32, "
        f"d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(
    column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int
) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def _npu_push_queue(
    column: int, direction: str, channel: int, bd_id: int, issue_token: bool = False
) -> str:
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
    num_tiles = ceil(context_len / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    plane_bytes = plane_dwords * 4
    out_bytes = CHECKSUM_DWORDS * 4
    offsets = {
        "k03": 0,
        "v03": plane_bytes,
        "k47": 2 * plane_bytes,
        "v47": 3 * plane_bytes,
    }
    lines = [
        f"    aie.runtime_sequence(%kv_cache: memref<{NUM_PLANES * plane_dwords}xi32>, "
        f"%output: memref<{NUM_PLANES * CHECKSUM_DWORDS}xi32>) {{"
    ]

    # History descriptors.  Each row1 ring consumes 4096-dword chunks from one
    # large runtime descriptor; L changes only this descriptor length.
    lines += [
        _npu_writebd(0, 0, plane_dwords, offsets["k03"]),
        _npu_address_patch(0, 0, 0, offsets["k03"]),
        _npu_push_queue(0, "MM2S", 0, 0),
        _npu_writebd(0, 1, plane_dwords, offsets["k47"]),
        _npu_address_patch(0, 1, 0, offsets["k47"]),
        _npu_push_queue(0, "MM2S", 1, 1),
        _npu_writebd(7, 0, plane_dwords, offsets["v03"]),
        _npu_address_patch(7, 0, 0, offsets["v03"]),
        _npu_push_queue(7, "MM2S", 0, 0),
        _npu_writebd(7, 1, plane_dwords, offsets["v47"]),
        _npu_address_patch(7, 1, 0, offsets["v47"]),
        _npu_push_queue(7, "MM2S", 1, 1),
    ]

    # Four checksum outputs: k03, k47, v03, v47.
    lines += [
        _npu_writebd(1, 2, CHECKSUM_DWORDS, 0),
        _npu_address_patch(1, 2, 1, 0),
        _npu_push_queue(1, "S2MM", 0, 2, issue_token=True),
        _npu_writebd(1, 3, CHECKSUM_DWORDS, out_bytes),
        _npu_address_patch(1, 3, 1, out_bytes),
        _npu_push_queue(1, "S2MM", 1, 3, issue_token=True),
        _npu_writebd(6, 2, CHECKSUM_DWORDS, 2 * out_bytes),
        _npu_address_patch(6, 2, 1, 2 * out_bytes),
        _npu_push_queue(6, "S2MM", 0, 2, issue_token=True),
        _npu_writebd(6, 3, CHECKSUM_DWORDS, 3 * out_bytes),
        _npu_address_patch(6, 3, 1, 3 * out_bytes),
        _npu_push_queue(6, "S2MM", 1, 3, issue_token=True),
        _npu_sync(1, 0),
        _npu_sync(1, 1),
        _npu_sync(6, 0),
        _npu_sync(6, 1),
        "    }",
    ]
    return "\n".join(lines)


def _ring_buffers_and_locks(mem_name: str, prefix: str, lock_base: int) -> str:
    return f"""
    %{mem_name}_{prefix}_ping = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_{prefix}_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %{mem_name}_{prefix}_pong = aie.buffer(%{mem_name}) {{sym_name = "{mem_name}_{prefix}_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %{mem_name}_{prefix}_empty = aie.lock(%{mem_name}, {lock_base}) {{init = 2 : i32, sym_name = "{mem_name}_{prefix}_empty"}}
    %{mem_name}_{prefix}_loaded = aie.lock(%{mem_name}, {lock_base + 1}) {{init = 0 : i32, sym_name = "{mem_name}_{prefix}_loaded"}}
    %{mem_name}_{prefix}_half0_done = aie.lock(%{mem_name}, {lock_base + 2}) {{init = 0 : i32, sym_name = "{mem_name}_{prefix}_half0_done"}}
"""


def _edge_memtile_dma(mem_name: str) -> str:
    return f"""    %{mem_name}_dma = aie.memtile_dma(%{mem_name}) {{
      // Ring A: MyLM-like 4096-dword load followed by two 2048-dword phases.
      %0 = aie.dma_start(S2MM, 0, ^a_load_ping, ^a_out_start)
    ^a_load_ping:
      aie.use_lock(%{mem_name}_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{mem_name}_a_loaded, Release, 1)
      aie.next_bd ^a_load_pong
    ^a_load_pong:
      aie.use_lock(%{mem_name}_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{mem_name}_a_loaded, Release, 1)
      aie.next_bd ^a_load_ping

    ^a_out_start:
      %1 = aie.dma_start(MM2S, 0, ^a_half0_ping, ^b_start)
    ^a_half0_ping:
      aie.use_lock(%{mem_name}_a_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%{mem_name}_a_half0_done, Release, 1)
      aie.next_bd ^a_half1_ping
    ^a_half1_ping:
      aie.use_lock(%{mem_name}_a_half0_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 8 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{mem_name}_a_empty, Release, 1)
      aie.next_bd ^a_half0_pong
    ^a_half0_pong:
      aie.use_lock(%{mem_name}_a_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%{mem_name}_a_half0_done, Release, 1)
      aie.next_bd ^a_half1_pong
    ^a_half1_pong:
      aie.use_lock(%{mem_name}_a_half0_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 9 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{mem_name}_a_empty, Release, 1)
      aie.next_bd ^a_half0_ping

      // Ring B uses the same structure on a different shim input and output route.
    ^b_start:
      %2 = aie.dma_start(S2MM, 1, ^b_load_ping, ^b_out_start)
    ^b_load_ping:
      aie.use_lock(%{mem_name}_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%{mem_name}_b_loaded, Release, 1)
      aie.next_bd ^b_load_pong
    ^b_load_pong:
      aie.use_lock(%{mem_name}_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%{mem_name}_b_loaded, Release, 1)
      aie.next_bd ^b_load_ping

    ^b_out_start:
      %3 = aie.dma_start(MM2S, 2, ^b_half0_ping, ^end)
    ^b_half0_ping:
      aie.use_lock(%{mem_name}_b_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 10 : i32}}
      aie.use_lock(%{mem_name}_b_half0_done, Release, 1)
      aie.next_bd ^b_half1_ping
    ^b_half1_ping:
      aie.use_lock(%{mem_name}_b_half0_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 10 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{mem_name}_b_empty, Release, 1)
      aie.next_bd ^b_half0_pong
    ^b_half0_pong:
      aie.use_lock(%{mem_name}_b_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 11 : i32}}
      aie.use_lock(%{mem_name}_b_half0_done, Release, 1)
      aie.next_bd ^b_half1_pong
    ^b_half1_pong:
      aie.use_lock(%{mem_name}_b_half0_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem_name}_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 11 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{mem_name}_b_empty, Release, 1)
      aie.next_bd ^b_half0_ping
    ^end:
      aie.end
    }}"""


def _worker_buffers_and_locks(name: str) -> str:
    return f"""
    %{name}_half_ping = aie.buffer(%{name}) {{sym_name = "{name}_half_ping"}} : memref<{HALF_TILE_DWORDS}xi32>
    %{name}_half_pong = aie.buffer(%{name}) {{sym_name = "{name}_half_pong"}} : memref<{HALF_TILE_DWORDS}xi32>
    %{name}_out = aie.buffer(%{name}) {{sym_name = "{name}_out"}} : memref<{CHECKSUM_DWORDS}xi32>
    %{name}_half_empty = aie.lock(%{name}, 0) {{init = 2 : i32, sym_name = "{name}_half_empty"}}
    %{name}_half_full = aie.lock(%{name}, 1) {{init = 0 : i32, sym_name = "{name}_half_full"}}
    %{name}_out_empty = aie.lock(%{name}, 2) {{init = 1 : i32, sym_name = "{name}_out_empty"}}
    %{name}_out_full = aie.lock(%{name}, 3) {{init = 0 : i32, sym_name = "{name}_out_full"}}
"""


def _worker_mem(name: str) -> str:
    return f"""    %{name}_mem = aie.mem(%{name}) {{
      %0 = aie.dma_start(S2MM, 0, ^half_ping, ^out_start)
    ^half_ping:
      aie.use_lock(%{name}_half_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_half_ping : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{name}_half_full, Release, 1)
      aie.next_bd ^half_pong
    ^half_pong:
      aie.use_lock(%{name}_half_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_half_pong : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{name}_half_full, Release, 1)
      aie.next_bd ^half_ping

    ^out_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{name}_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_out : memref<{CHECKSUM_DWORDS}xi32>, 0, {CHECKSUM_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{name}_out_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}"""


def _worker_core(name: str, worker_id: int, num_tiles: int) -> str:
    chunks = num_tiles * 2
    return f"""    %{name}_core = aie.core(%{name}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %chunks = arith.constant {chunks} : index
      %worker_id = arith.constant {worker_id} : i32
      aie.use_lock(%{name}_out_empty, AcquireGreaterEqual, 1)
      func.call @checksum_zero(%{name}_out, %worker_id)
        : (memref<{CHECKSUM_DWORDS}xi32>, i32) -> ()
      scf.for %chunk = %c0 to %chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{name}_half_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @checksum_accum(%{name}_half_pong, %{name}_out, %chunk_i32)
            : (memref<{HALF_TILE_DWORDS}xi32>, memref<{CHECKSUM_DWORDS}xi32>, i32) -> ()
        }} else {{
          func.call @checksum_accum(%{name}_half_ping, %{name}_out, %chunk_i32)
            : (memref<{HALF_TILE_DWORDS}xi32>, memref<{CHECKSUM_DWORDS}xi32>, i32) -> ()
        }}
        aie.use_lock(%{name}_half_empty, Release, 1)
      }}
      aie.use_lock(%{name}_out_full, Release, 1)
      aie.end
    }}"""


def generate_mlir(context_len: int) -> str:
    num_tiles = ceil(context_len / TOKENS_PER_TILE)
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %shim7 = aie.tile(7, 0)
    %mem7 = aie.tile(7, 1)
    %shim1 = aie.tile(1, 0)
    %c1r2 = aie.tile(1, 2)
    %c1r3 = aie.tile(1, 3)
    %shim6 = aie.tile(6, 0)
    %c6r2 = aie.tile(6, 2)
    %c6r3 = aie.tile(6, 3)

{_ring_buffers_and_locks("mem0", "a", 0)}
{_ring_buffers_and_locks("mem0", "b", 3)}
{_ring_buffers_and_locks("mem7", "a", 0)}
{_ring_buffers_and_locks("mem7", "b", 3)}
{_worker_buffers_and_locks("c1r2")}
{_worker_buffers_and_locks("c1r3")}
{_worker_buffers_and_locks("c6r2")}
{_worker_buffers_and_locks("c6r3")}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%shim7, DMA : 0, %mem7, DMA : 0)
    aie.flow(%shim7, DMA : 1, %mem7, DMA : 1)
    aie.flow(%mem0, DMA : 0, %c1r2, DMA : 0)
    aie.flow(%mem0, DMA : 2, %c1r3, DMA : 0)
    aie.flow(%mem7, DMA : 0, %c6r2, DMA : 0)
    aie.flow(%mem7, DMA : 2, %c6r3, DMA : 0)
    aie.flow(%c1r2, DMA : 0, %shim1, DMA : 0)
    aie.flow(%c1r3, DMA : 0, %shim1, DMA : 1)
    aie.flow(%c6r2, DMA : 0, %shim6, DMA : 0)
    aie.flow(%c6r3, DMA : 0, %shim6, DMA : 1)

    func.func private @checksum_zero(memref<{CHECKSUM_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/checksum.o"}}
    func.func private @checksum_accum(memref<{HALF_TILE_DWORDS}xi32>, memref<{CHECKSUM_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/checksum.o"}}

{_worker_core("c1r2", 0, num_tiles)}
{_worker_core("c1r3", 1, num_tiles)}
{_worker_core("c6r2", 2, num_tiles)}
{_worker_core("c6r3", 3, num_tiles)}

{_edge_memtile_dma("mem0")}
{_edge_memtile_dma("mem7")}
{_worker_mem("c1r2")}
{_worker_mem("c1r3")}
{_worker_mem("c6r2")}
{_worker_mem("c6r3")}

{_runtime_sequence(context_len)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    length = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    print(generate_mlir(length))
