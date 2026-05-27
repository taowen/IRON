"""Generate exp27 shape-A/shape-B single-tile attention MLIR."""

from pathlib import Path

CURRENT_DWORDS = 512
HISTORY_SOURCE_DWORDS = 4096
HISTORY_DWORDS = 2048
SIDEBAND_DWORDS = 17
SIDEBAND_DEBUG_DWORDS = 4
ATTENTION_OUT_DWORDS = 512
TOTAL_OUT_DWORDS = SIDEBAND_DEBUG_DWORDS + ATTENTION_OUT_DWORDS


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
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


def _npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
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


def _lock_pair(tile: str, prefix: str, base: int) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = 1 : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _history_ring(mem: str, output_dma: int) -> str:
    return f"""
    %{mem}_hist_ping = aie.buffer(%{mem}) {{sym_name = "{mem}_hist_ping"}} : memref<{HISTORY_SOURCE_DWORDS}xf32>
    %{mem}_hist_pong = aie.buffer(%{mem}) {{sym_name = "{mem}_hist_pong"}} : memref<{HISTORY_SOURCE_DWORDS}xf32>
    %{mem}_hist_empty = aie.lock(%{mem}, 0) {{init = 2 : i32, sym_name = "{mem}_hist_empty"}}
    %{mem}_hist_loaded = aie.lock(%{mem}, 1) {{init = 0 : i32, sym_name = "{mem}_hist_loaded"}}

    %{mem}_dma = aie.memtile_dma(%{mem}) {{
      %0 = aie.dma_start(S2MM, 0, ^load_ping, ^out_start)
    ^load_ping:
      aie.use_lock(%{mem}_hist_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_ping : memref<{HISTORY_SOURCE_DWORDS}xf32>, 0, {HISTORY_SOURCE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{mem}_hist_loaded, Release, 1)
      aie.next_bd ^load_pong
    ^load_pong:
      aie.use_lock(%{mem}_hist_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_pong : memref<{HISTORY_SOURCE_DWORDS}xf32>, 0, {HISTORY_SOURCE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{mem}_hist_loaded, Release, 1)
      aie.next_bd ^load_ping

    ^out_start:
      %1 = aie.dma_start(MM2S, {output_dma}, ^half_ping, ^end)
    ^half_ping:
      aie.use_lock(%{mem}_hist_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_ping : memref<{HISTORY_SOURCE_DWORDS}xf32>, 0, {HISTORY_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{mem}_hist_empty, Release, 1)
      aie.next_bd ^half_pong
    ^half_pong:
      aie.use_lock(%{mem}_hist_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_pong : memref<{HISTORY_SOURCE_DWORDS}xf32>, 0, {HISTORY_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{mem}_hist_empty, Release, 1)
      aie.next_bd ^half_ping
    ^end:
      aie.end
    }}
"""


def _current_source() -> str:
    return f"""
    %current_in = aie.buffer(%current_src) {{sym_name = "current_in"}} : memref<{CURRENT_DWORDS}xf32>
    %current_packet = aie.buffer(%current_src) {{sym_name = "current_packet"}} : memref<{CURRENT_DWORDS}xf32>
{_lock_pair("current_src", "in", 0)}
{_lock_pair("current_src", "packet", 2)}

    %current_src_core = aie.core(%current_src) {{
      aie.use_lock(%current_src_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%current_src_packet_empty, AcquireGreaterEqual, 1)
      func.call @copy_current_512(%current_in, %current_packet)
        : (memref<{CURRENT_DWORDS}xf32>, memref<{CURRENT_DWORDS}xf32>) -> ()
      aie.use_lock(%current_src_in_empty, Release, 1)
      aie.use_lock(%current_src_packet_full, Release, 1)
      aie.end
    }}

    %current_src_mem = aie.mem(%current_src) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^packet_start)
    ^in_bd:
      aie.use_lock(%current_src_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%current_in : memref<{CURRENT_DWORDS}xf32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%current_src_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^packet_start:
      %1 = aie.dma_start(MM2S, 0, ^packet_bd, ^end)
    ^packet_bd:
      aie.use_lock(%current_src_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%current_packet : memref<{CURRENT_DWORDS}xf32>, 0, {CURRENT_DWORDS}) {{bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 14>}}
      aie.use_lock(%current_src_packet_empty, Release, 1)
      aie.next_bd ^packet_bd
    ^end:
      aie.end
    }}
"""


def _shape_a() -> str:
    return f"""
    %shape_a_current = aie.buffer(%shape_a) {{sym_name = "shape_a_current"}} : memref<{CURRENT_DWORDS}xf32>
    %shape_a_history = aie.buffer(%shape_a) {{sym_name = "shape_a_history"}} : memref<{HISTORY_DWORDS}xf32>
    %shape_a_sideband = aie.buffer(%shape_a) {{sym_name = "shape_a_sideband"}} : memref<{SIDEBAND_DWORDS}xf32>
{_lock_pair("shape_a", "current", 0)}
{_lock_pair("shape_a", "history", 2)}
{_lock_pair("shape_a", "sideband", 4)}

    %shape_a_core = aie.core(%shape_a) {{
      aie.use_lock(%shape_a_current_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_a_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_a_sideband_empty, AcquireGreaterEqual, 1)
      func.call @shape_a_softmax_sideband(%shape_a_current, %shape_a_history, %shape_a_sideband)
        : (memref<{CURRENT_DWORDS}xf32>, memref<{HISTORY_DWORDS}xf32>, memref<{SIDEBAND_DWORDS}xf32>) -> ()
      aie.use_lock(%shape_a_current_empty, Release, 1)
      aie.use_lock(%shape_a_history_empty, Release, 1)
      aie.use_lock(%shape_a_sideband_full, Release, 1)
      aie.end
    }}

    %shape_a_mem = aie.mem(%shape_a) {{
      %0 = aie.dma_start(S2MM, 0, ^current_bd, ^history_start)
    ^current_bd:
      aie.use_lock(%shape_a_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_current : memref<{CURRENT_DWORDS}xf32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%shape_a_current_full, Release, 1)
      aie.next_bd ^current_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^sideband_start)
    ^history_bd:
      aie.use_lock(%shape_a_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_history : memref<{HISTORY_DWORDS}xf32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%shape_a_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^sideband_start:
      %2 = aie.dma_start(MM2S, 0, ^sideband_bd, ^end)
    ^sideband_bd:
      aie.use_lock(%shape_a_sideband_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_sideband : memref<{SIDEBAND_DWORDS}xf32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%shape_a_sideband_empty, Release, 1)
      aie.next_bd ^sideband_bd
    ^end:
      aie.end
    }}
"""


def _side_sink() -> str:
    return f"""
    %side_sink_in = aie.buffer(%side_sink) {{sym_name = "side_sink_in"}} : memref<{SIDEBAND_DWORDS}xf32>
    %side_sink_debug = aie.buffer(%side_sink) {{sym_name = "side_sink_debug"}} : memref<{SIDEBAND_DEBUG_DWORDS}xf32>
    %side_sink_forward = aie.buffer(%side_sink) {{sym_name = "side_sink_forward"}} : memref<{SIDEBAND_DWORDS}xf32>
{_lock_pair("side_sink", "in", 0)}
{_lock_pair("side_sink", "debug", 2)}
{_lock_pair("side_sink", "forward", 4)}

    %side_sink_core = aie.core(%side_sink) {{
      aie.use_lock(%side_sink_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%side_sink_debug_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%side_sink_forward_empty, AcquireGreaterEqual, 1)
      func.call @sideband_debug_and_forward(%side_sink_in, %side_sink_debug, %side_sink_forward)
        : (memref<{SIDEBAND_DWORDS}xf32>, memref<{SIDEBAND_DEBUG_DWORDS}xf32>, memref<{SIDEBAND_DWORDS}xf32>) -> ()
      aie.use_lock(%side_sink_in_empty, Release, 1)
      aie.use_lock(%side_sink_debug_full, Release, 1)
      aie.use_lock(%side_sink_forward_full, Release, 1)
      aie.end
    }}

    %side_sink_mem = aie.mem(%side_sink) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^debug_start)
    ^in_bd:
      aie.use_lock(%side_sink_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_in : memref<{SIDEBAND_DWORDS}xf32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%side_sink_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^debug_start:
      %1 = aie.dma_start(MM2S, 0, ^debug_bd, ^forward_start)
    ^debug_bd:
      aie.use_lock(%side_sink_debug_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_debug : memref<{SIDEBAND_DEBUG_DWORDS}xf32>, 0, {SIDEBAND_DEBUG_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%side_sink_debug_empty, Release, 1)
      aie.next_bd ^debug_bd

    ^forward_start:
      %2 = aie.dma_start(MM2S, 1, ^forward_bd, ^end)
    ^forward_bd:
      aie.use_lock(%side_sink_forward_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_forward : memref<{SIDEBAND_DWORDS}xf32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%side_sink_forward_empty, Release, 1)
      aie.next_bd ^forward_bd
    ^end:
      aie.end
    }}
"""


def _shape_b() -> str:
    return f"""
    %shape_b_sideband = aie.buffer(%shape_b) {{sym_name = "shape_b_sideband"}} : memref<{SIDEBAND_DWORDS}xf32>
    %shape_b_history = aie.buffer(%shape_b) {{sym_name = "shape_b_history"}} : memref<{HISTORY_DWORDS}xf32>
    %shape_b_output = aie.buffer(%shape_b) {{sym_name = "shape_b_output"}} : memref<{ATTENTION_OUT_DWORDS}xf32>
{_lock_pair("shape_b", "sideband", 0)}
{_lock_pair("shape_b", "history", 2)}
{_lock_pair("shape_b", "output", 4)}

    %shape_b_core = aie.core(%shape_b) {{
      aie.use_lock(%shape_b_sideband_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_b_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_b_output_empty, AcquireGreaterEqual, 1)
      func.call @shape_b_weighted_value(%shape_b_sideband, %shape_b_history, %shape_b_output)
        : (memref<{SIDEBAND_DWORDS}xf32>, memref<{HISTORY_DWORDS}xf32>, memref<{ATTENTION_OUT_DWORDS}xf32>) -> ()
      aie.use_lock(%shape_b_sideband_empty, Release, 1)
      aie.use_lock(%shape_b_history_empty, Release, 1)
      aie.use_lock(%shape_b_output_full, Release, 1)
      aie.end
    }}

    %shape_b_mem = aie.mem(%shape_b) {{
      %0 = aie.dma_start(S2MM, 0, ^sideband_bd, ^history_start)
    ^sideband_bd:
      aie.use_lock(%shape_b_sideband_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_sideband : memref<{SIDEBAND_DWORDS}xf32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%shape_b_sideband_full, Release, 1)
      aie.next_bd ^sideband_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^output_start)
    ^history_bd:
      aie.use_lock(%shape_b_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_history : memref<{HISTORY_DWORDS}xf32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%shape_b_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^output_start:
      %2 = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:
      aie.use_lock(%shape_b_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_output : memref<{ATTENTION_OUT_DWORDS}xf32>, 0, {ATTENTION_OUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%shape_b_output_empty, Release, 1)
      aie.next_bd ^output_bd
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    attention_offset = SIDEBAND_DEBUG_DWORDS * 4
    return "\n".join(
        [
            f"    aie.runtime_sequence(%current: memref<{CURRENT_DWORDS}xf32>, "
            f"%k_history: memref<{HISTORY_SOURCE_DWORDS}xf32>, "
            f"%v_history: memref<{HISTORY_SOURCE_DWORDS}xf32>, "
            f"%output: memref<{TOTAL_OUT_DWORDS}xf32>) {{",
            _npu_writebd(6, 2, SIDEBAND_DEBUG_DWORDS, 0),
            _npu_address_patch(6, 2, 3, 0),
            _npu_push_queue(6, "S2MM", 0, 2, issue_token=True),
            _npu_writebd(6, 4, ATTENTION_OUT_DWORDS, attention_offset),
            _npu_address_patch(6, 4, 3, attention_offset),
            _npu_push_queue(6, "S2MM", 1, 4, issue_token=True),
            _npu_writebd(1, 0, CURRENT_DWORDS, 0),
            _npu_address_patch(1, 0, 0, 0),
            _npu_push_queue(1, "MM2S", 0, 0),
            _npu_writebd(0, 0, HISTORY_SOURCE_DWORDS, 0),
            _npu_address_patch(0, 0, 1, 0),
            _npu_push_queue(0, "MM2S", 0, 0),
            _npu_writebd(7, 0, HISTORY_SOURCE_DWORDS, 0),
            _npu_address_patch(7, 0, 2, 0),
            _npu_push_queue(7, "MM2S", 0, 0),
            _npu_sync(6, 0),
            _npu_sync(6, 1),
            "    }",
        ]
    )


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %shape_a = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %current_src = aie.tile(1, 3)
    %shim6 = aie.tile(6, 0)
    %side_sink = aie.tile(6, 2)
    %shape_b = aie.tile(6, 3)
    %shim7 = aie.tile(7, 0)
    %mem7 = aie.tile(7, 1)

    aie.flow(%shim1, DMA : 0, %current_src, DMA : 0)
    aie.packet_flow(14) {{
      aie.packet_source<%current_src, DMA : 0>
      aie.packet_dest<%shape_a, DMA : 0>
    }}
    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem0, DMA : 0, %shape_a, DMA : 1)
    aie.flow(%shape_a, DMA : 0, %side_sink, DMA : 0)
    aie.flow(%side_sink, DMA : 0, %shim6, DMA : 0)
    aie.flow(%side_sink, DMA : 1, %shape_b, DMA : 0)
    aie.flow(%shim7, DMA : 0, %mem7, DMA : 0)
    aie.flow(%mem7, DMA : 0, %shape_b, DMA : 1)
    aie.flow(%shape_b, DMA : 0, %shim6, DMA : 1)

    func.func private @copy_current_512(memref<{CURRENT_DWORDS}xf32>, memref<{CURRENT_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/attention_kernels.o"}}
    func.func private @shape_a_softmax_sideband(memref<{CURRENT_DWORDS}xf32>, memref<{HISTORY_DWORDS}xf32>, memref<{SIDEBAND_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/attention_kernels.o"}}
    func.func private @sideband_debug_and_forward(memref<{SIDEBAND_DWORDS}xf32>, memref<{SIDEBAND_DEBUG_DWORDS}xf32>, memref<{SIDEBAND_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/attention_kernels.o"}}
    func.func private @shape_b_weighted_value(memref<{SIDEBAND_DWORDS}xf32>, memref<{HISTORY_DWORDS}xf32>, memref<{ATTENTION_OUT_DWORDS}xf32>) attributes {{link_with = "{experiment_dir}/attention_kernels.o"}}

{_history_ring("mem0", 0)}
{_history_ring("mem7", 0)}
{_current_source()}
{_shape_a()}
{_side_sink()}
{_shape_b()}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
