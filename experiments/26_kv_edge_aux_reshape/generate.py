"""Generate exp26 edge selector and sideband calibration MLIR."""

from dataclasses import dataclass
from pathlib import Path

CURRENT_DWORDS = 512
HISTORY_SOURCE_DWORDS = 4096
HISTORY_DWORDS = 2048
SIDEBAND_DWORDS = 17
SHAPE_OUT_DWORDS = 8
SIDEBAND_OUT_DWORDS = 4
SHAPE_B_OUT_DWORDS = 4
TOTAL_OUT_DWORDS = SHAPE_OUT_DWORDS + SIDEBAND_OUT_DWORDS + SHAPE_B_OUT_DWORDS


@dataclass(frozen=True)
class Variant:
    name: str
    packet_id: int
    shape_col: int

    @property
    def shape_tile(self) -> str:
        return f"c{self.shape_col}r2"

    @property
    def shim_tile(self) -> str:
        return f"shim{self.shape_col}"

    @property
    def mem_tile(self) -> str:
        return f"mem{self.shape_col}"


VARIANTS = {
    "left14": Variant(name="left14", packet_id=14, shape_col=0),
    "right15": Variant(name="right15", packet_id=15, shape_col=7),
}


def get_variant(name: str) -> Variant:
    return VARIANTS[name]


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


def _history_ring(variant: Variant) -> str:
    mem = variant.mem_tile
    return f"""
    %{mem}_hist_ping = aie.buffer(%{mem}) {{sym_name = "{mem}_hist_ping"}} : memref<{HISTORY_SOURCE_DWORDS}xi32>
    %{mem}_hist_pong = aie.buffer(%{mem}) {{sym_name = "{mem}_hist_pong"}} : memref<{HISTORY_SOURCE_DWORDS}xi32>
    %{mem}_hist_empty = aie.lock(%{mem}, 0) {{init = 2 : i32, sym_name = "{mem}_hist_empty"}}
    %{mem}_hist_loaded = aie.lock(%{mem}, 1) {{init = 0 : i32, sym_name = "{mem}_hist_loaded"}}

    %{mem}_dma = aie.memtile_dma(%{mem}) {{
      %0 = aie.dma_start(S2MM, 0, ^load_ping, ^out_start)
    ^load_ping:
      aie.use_lock(%{mem}_hist_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_ping : memref<{HISTORY_SOURCE_DWORDS}xi32>, 0, {HISTORY_SOURCE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{mem}_hist_loaded, Release, 1)
      aie.next_bd ^load_pong
    ^load_pong:
      aie.use_lock(%{mem}_hist_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_pong : memref<{HISTORY_SOURCE_DWORDS}xi32>, 0, {HISTORY_SOURCE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{mem}_hist_loaded, Release, 1)
      aie.next_bd ^load_ping

    ^out_start:
      %1 = aie.dma_start(MM2S, 0, ^half_ping, ^end)
    ^half_ping:
      aie.use_lock(%{mem}_hist_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_ping : memref<{HISTORY_SOURCE_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{mem}_hist_empty, Release, 1)
      aie.next_bd ^half_pong
    ^half_pong:
      aie.use_lock(%{mem}_hist_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{mem}_hist_pong : memref<{HISTORY_SOURCE_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{mem}_hist_empty, Release, 1)
      aie.next_bd ^half_ping
    ^end:
      aie.end
    }}
"""


def _current_source(variant: Variant) -> str:
    return f"""
    %current_src_in = aie.buffer(%current_src) {{sym_name = "current_src_in"}} : memref<{CURRENT_DWORDS}xi32>
    %current_src_packet = aie.buffer(%current_src) {{sym_name = "current_src_packet"}} : memref<{CURRENT_DWORDS}xi32>
{_lock_pair("current_src", "in", 0)}
{_lock_pair("current_src", "packet", 2)}

    %current_src_core = aie.core(%current_src) {{
      aie.use_lock(%current_src_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%current_src_packet_empty, AcquireGreaterEqual, 1)
      func.call @copy_current_512(%current_src_in, %current_src_packet)
        : (memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>) -> ()
      aie.use_lock(%current_src_in_empty, Release, 1)
      aie.use_lock(%current_src_packet_full, Release, 1)
      aie.end
    }}

    %current_src_mem = aie.mem(%current_src) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^packet_start)
    ^in_bd:
      aie.use_lock(%current_src_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%current_src_in : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%current_src_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^packet_start:
      %1 = aie.dma_start(MM2S, 0, ^packet_bd, ^end)
    ^packet_bd:
      aie.use_lock(%current_src_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%current_src_packet : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {variant.packet_id}>}}
      aie.use_lock(%current_src_packet_empty, Release, 1)
      aie.next_bd ^packet_bd
    ^end:
      aie.end
    }}
"""


def _shape_a_consumer(variant: Variant) -> str:
    shape = variant.shape_tile
    return f"""
    %shape_current = aie.buffer(%{shape}) {{sym_name = "shape_current"}} : memref<{CURRENT_DWORDS}xi32>
    %shape_history = aie.buffer(%{shape}) {{sym_name = "shape_history"}} : memref<{HISTORY_DWORDS}xi32>
    %shape_debug = aie.buffer(%{shape}) {{sym_name = "shape_debug"}} : memref<{SHAPE_OUT_DWORDS}xi32>
{_lock_pair(shape, "current", 0)}
{_lock_pair(shape, "history", 2)}
{_lock_pair(shape, "debug", 4)}

    %{shape}_core = aie.core(%{shape}) {{
      aie.use_lock(%{shape}_current_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{shape}_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{shape}_debug_empty, AcquireGreaterEqual, 1)
      func.call @shape_a_debug(%shape_current, %shape_history, %shape_debug)
        : (memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{SHAPE_OUT_DWORDS}xi32>) -> ()
      aie.use_lock(%{shape}_current_empty, Release, 1)
      aie.use_lock(%{shape}_history_empty, Release, 1)
      aie.use_lock(%{shape}_debug_full, Release, 1)
      aie.end
    }}

    %{shape}_mem = aie.mem(%{shape}) {{
      %0 = aie.dma_start(S2MM, 0, ^current_bd, ^history_start)
    ^current_bd:
      aie.use_lock(%{shape}_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_current : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{shape}_current_full, Release, 1)
      aie.next_bd ^current_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^debug_start)
    ^history_bd:
      aie.use_lock(%{shape}_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_history : memref<{HISTORY_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{shape}_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^debug_start:
      %2 = aie.dma_start(MM2S, 0, ^debug_bd, ^end)
    ^debug_bd:
      aie.use_lock(%{shape}_debug_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_debug : memref<{SHAPE_OUT_DWORDS}xi32>, 0, {SHAPE_OUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{shape}_debug_empty, Release, 1)
      aie.next_bd ^debug_bd
    ^end:
      aie.end
    }}
"""


def _sideband_source() -> str:
    return f"""
    %side_src_in = aie.buffer(%side_src) {{sym_name = "side_src_in"}} : memref<{SIDEBAND_DWORDS}xi32>
    %side_src_stream = aie.buffer(%side_src) {{sym_name = "side_src_stream"}} : memref<{SIDEBAND_DWORDS}xi32>
{_lock_pair("side_src", "in", 0)}
{_lock_pair("side_src", "stream", 2)}

    %side_src_core = aie.core(%side_src) {{
      aie.use_lock(%side_src_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%side_src_stream_empty, AcquireGreaterEqual, 1)
      func.call @copy_sideband_17(%side_src_in, %side_src_stream)
        : (memref<{SIDEBAND_DWORDS}xi32>, memref<{SIDEBAND_DWORDS}xi32>) -> ()
      aie.use_lock(%side_src_in_empty, Release, 1)
      aie.use_lock(%side_src_stream_full, Release, 1)
      aie.end
    }}

    %side_src_mem = aie.mem(%side_src) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^stream_start)
    ^in_bd:
      aie.use_lock(%side_src_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_src_in : memref<{SIDEBAND_DWORDS}xi32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%side_src_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^stream_start:
      %1 = aie.dma_start(MM2S, 0, ^stream_bd, ^end)
    ^stream_bd:
      aie.use_lock(%side_src_stream_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_src_stream : memref<{SIDEBAND_DWORDS}xi32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%side_src_stream_empty, Release, 1)
      aie.next_bd ^stream_bd
    ^end:
      aie.end
    }}
"""


def _sideband_consumer() -> str:
    return f"""
    %side_sink_in = aie.buffer(%side_sink) {{sym_name = "side_sink_in"}} : memref<{SIDEBAND_DWORDS}xi32>
    %side_sink_debug = aie.buffer(%side_sink) {{sym_name = "side_sink_debug"}} : memref<{SIDEBAND_OUT_DWORDS}xi32>
    %side_sink_forward = aie.buffer(%side_sink) {{sym_name = "side_sink_forward"}} : memref<{SIDEBAND_DWORDS}xi32>
{_lock_pair("side_sink", "in", 0)}
{_lock_pair("side_sink", "debug", 2)}
{_lock_pair("side_sink", "forward", 4)}

    %side_sink_core = aie.core(%side_sink) {{
      aie.use_lock(%side_sink_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%side_sink_debug_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%side_sink_forward_empty, AcquireGreaterEqual, 1)
      func.call @sideband_debug_and_forward(%side_sink_in, %side_sink_debug, %side_sink_forward)
        : (memref<{SIDEBAND_DWORDS}xi32>, memref<{SIDEBAND_OUT_DWORDS}xi32>, memref<{SIDEBAND_DWORDS}xi32>) -> ()
      aie.use_lock(%side_sink_in_empty, Release, 1)
      aie.use_lock(%side_sink_debug_full, Release, 1)
      aie.use_lock(%side_sink_forward_full, Release, 1)
      aie.end
    }}

    %side_sink_mem = aie.mem(%side_sink) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^debug_start)
    ^in_bd:
      aie.use_lock(%side_sink_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_in : memref<{SIDEBAND_DWORDS}xi32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%side_sink_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^debug_start:
      %1 = aie.dma_start(MM2S, 0, ^debug_bd, ^forward_start)
    ^debug_bd:
      aie.use_lock(%side_sink_debug_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_debug : memref<{SIDEBAND_OUT_DWORDS}xi32>, 0, {SIDEBAND_OUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%side_sink_debug_empty, Release, 1)
      aie.next_bd ^debug_bd

    ^forward_start:
      %2 = aie.dma_start(MM2S, 1, ^forward_bd, ^end)
    ^forward_bd:
      aie.use_lock(%side_sink_forward_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%side_sink_forward : memref<{SIDEBAND_DWORDS}xi32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%side_sink_forward_empty, Release, 1)
      aie.next_bd ^forward_bd
    ^end:
      aie.end
    }}
"""


def _shape_b_consumer() -> str:
    return f"""
    %shape_b_in = aie.buffer(%shape_b) {{sym_name = "shape_b_in"}} : memref<{SIDEBAND_DWORDS}xi32>
    %shape_b_debug = aie.buffer(%shape_b) {{sym_name = "shape_b_debug"}} : memref<{SHAPE_B_OUT_DWORDS}xi32>
{_lock_pair("shape_b", "in", 0)}
{_lock_pair("shape_b", "debug", 2)}

    %shape_b_core = aie.core(%shape_b) {{
      aie.use_lock(%shape_b_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_b_debug_empty, AcquireGreaterEqual, 1)
      func.call @sideband_debug(%shape_b_in, %shape_b_debug)
        : (memref<{SIDEBAND_DWORDS}xi32>, memref<{SHAPE_B_OUT_DWORDS}xi32>) -> ()
      aie.use_lock(%shape_b_in_empty, Release, 1)
      aie.use_lock(%shape_b_debug_full, Release, 1)
      aie.end
    }}

    %shape_b_mem = aie.mem(%shape_b) {{
      %0 = aie.dma_start(S2MM, 0, ^in_bd, ^debug_start)
    ^in_bd:
      aie.use_lock(%shape_b_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_in : memref<{SIDEBAND_DWORDS}xi32>, 0, {SIDEBAND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%shape_b_in_full, Release, 1)
      aie.next_bd ^in_bd

    ^debug_start:
      %1 = aie.dma_start(MM2S, 0, ^debug_bd, ^end)
    ^debug_bd:
      aie.use_lock(%shape_b_debug_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_debug : memref<{SHAPE_B_OUT_DWORDS}xi32>, 0, {SHAPE_B_OUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%shape_b_debug_empty, Release, 1)
      aie.next_bd ^debug_bd
    ^end:
      aie.end
    }}
"""


def _runtime_sequence(variant: Variant) -> str:
    shape_out_bytes = SHAPE_OUT_DWORDS * 4
    sideband_out_bytes = SIDEBAND_OUT_DWORDS * 4
    lines = [
        f"    aie.runtime_sequence(%current: memref<{CURRENT_DWORDS}xi32>, "
        f"%history: memref<{HISTORY_SOURCE_DWORDS}xi32>, "
        f"%sideband: memref<{SIDEBAND_DWORDS}xi32>, "
        f"%output: memref<{TOTAL_OUT_DWORDS}xi32>) {{"
    ]

    lines += [
        _npu_writebd(variant.shape_col, 2, SHAPE_OUT_DWORDS, 0),
        _npu_address_patch(variant.shape_col, 2, 3, 0),
        _npu_push_queue(variant.shape_col, "S2MM", 0, 2, issue_token=True),
        _npu_writebd(6, 2, SIDEBAND_OUT_DWORDS, shape_out_bytes),
        _npu_address_patch(6, 2, 3, shape_out_bytes),
        _npu_push_queue(6, "S2MM", 0, 2, issue_token=True),
        _npu_writebd(6, 4, SHAPE_B_OUT_DWORDS, shape_out_bytes + sideband_out_bytes),
        _npu_address_patch(6, 4, 3, shape_out_bytes + sideband_out_bytes),
        _npu_push_queue(6, "S2MM", 1, 4, issue_token=True),
        _npu_writebd(1, 0, CURRENT_DWORDS, 0),
        _npu_address_patch(1, 0, 0, 0),
        _npu_push_queue(1, "MM2S", 0, 0),
        _npu_writebd(variant.shape_col, 0, HISTORY_SOURCE_DWORDS, 0),
        _npu_address_patch(variant.shape_col, 0, 1, 0),
        _npu_push_queue(variant.shape_col, "MM2S", 0, 0),
        _npu_writebd(2, 0, SIDEBAND_DWORDS, 0),
        _npu_address_patch(2, 0, 2, 0),
        _npu_push_queue(2, "MM2S", 0, 0),
        _npu_sync(variant.shape_col, 0),
        _npu_sync(6, 0),
        _npu_sync(6, 1),
        "    }",
    ]
    return "\n".join(lines)


def _tile_declarations(variant: Variant) -> str:
    return f"""
    %{variant.shim_tile} = aie.tile({variant.shape_col}, 0)
    %{variant.mem_tile} = aie.tile({variant.shape_col}, 1)
    %{variant.shape_tile} = aie.tile({variant.shape_col}, 2)
    %shim1 = aie.tile(1, 0)
    %current_src = aie.tile(1, 3)
    %shim2 = aie.tile(2, 0)
    %side_src = aie.tile(2, 2)
    %shim6 = aie.tile(6, 0)
    %side_sink = aie.tile(6, 2)
    %shape_b = aie.tile(6, 3)
"""


def _flows(variant: Variant) -> str:
    shape = variant.shape_tile
    return f"""
    aie.flow(%shim1, DMA : 0, %current_src, DMA : 0)
    aie.packet_flow({variant.packet_id}) {{
      aie.packet_source<%current_src, DMA : 0>
      aie.packet_dest<%{shape}, DMA : 0>
    }}
    aie.flow(%{variant.shim_tile}, DMA : 0, %{variant.mem_tile}, DMA : 0)
    aie.flow(%{variant.mem_tile}, DMA : 0, %{shape}, DMA : 1)
    aie.flow(%{shape}, DMA : 0, %{variant.shim_tile}, DMA : 0)

    aie.flow(%shim2, DMA : 0, %side_src, DMA : 0)
    aie.flow(%side_src, DMA : 0, %side_sink, DMA : 0)
    aie.flow(%side_sink, DMA : 0, %shim6, DMA : 0)
    aie.flow(%side_sink, DMA : 1, %shape_b, DMA : 0)
    aie.flow(%shape_b, DMA : 0, %shim6, DMA : 1)
"""


def generate_mlir(variant_name: str = "left14") -> str:
    variant = get_variant(variant_name)
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
{_tile_declarations(variant)}
{_flows(variant)}
    func.func private @copy_current_512(memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/debug_kernels.o"}}
    func.func private @copy_sideband_17(memref<{SIDEBAND_DWORDS}xi32>, memref<{SIDEBAND_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/debug_kernels.o"}}
    func.func private @shape_a_debug(memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{SHAPE_OUT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/debug_kernels.o"}}
    func.func private @sideband_debug(memref<{SIDEBAND_DWORDS}xi32>, memref<{SIDEBAND_OUT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/debug_kernels.o"}}
    func.func private @sideband_debug_and_forward(memref<{SIDEBAND_DWORDS}xi32>, memref<{SIDEBAND_OUT_DWORDS}xi32>, memref<{SIDEBAND_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/debug_kernels.o"}}

{_history_ring(variant)}
{_current_source(variant)}
{_shape_a_consumer(variant)}
{_sideband_source()}
{_sideband_consumer()}
{_shape_b_consumer()}
{_runtime_sequence(variant)}
  }}
}}
"""


if __name__ == "__main__":
    import sys

    print(generate_mlir(sys.argv[1] if len(sys.argv) > 1 else "left14"))
