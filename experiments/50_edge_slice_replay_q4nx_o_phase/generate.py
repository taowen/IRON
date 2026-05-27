"""Generate raw MLIR-AIE for exp50 edge-slice replay -> Q4NX O phase."""

from pathlib import Path

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
O_INPUT_DIM = 1024
ACT_SLICE_BF16 = 256
K_CHUNK = 256
NUM_CHUNKS = O_INPUT_DIM // K_CHUNK
GROUP_SIZE = 32
RECORD_DWORDS = 17

CHUNK_BF16 = 2560
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COLUMN
OUT_BF16 = M_PER_TILE
COLUMN_WEIGHT_BF16 = NUM_CHUNKS * FAT_CHUNK_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


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


def _output_packet(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _main_chunk_body(tile: str) -> str:
    return f"""
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @q4nx_chunk_accum_slice(%{tile}_wt_pong, %{tile}_act_pong, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
        }} else {{
          func.call @q4nx_chunk_accum_slice(%{tile}_wt_ping, %{tile}_act_ping, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
        }}
        aie.use_lock(%{tile}_act_empty, Release, 1)
        aie.use_lock(%{tile}_wt_empty, Release, 1)
"""


def _main_tile(group: int, row: int) -> str:
    tile = f"m{group}_{row}"
    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{OUT_BF16}xbf16>
{_lock_pair(tile, "record", 0)}
{_lock_pair(tile, "act", 2, init_empty=2)}
{_lock_pair(tile, "wt", 4, init_empty=2)}
{_lock_pair(tile, "output", 6)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %num_chunks = arith.constant {NUM_CHUNKS} : index
      %m_i32 = arith.constant {M_PER_TILE} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      func.call @emit_main_sideband(%{tile}_record, %group_i32, %row_i32)
        : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_record_full, Release, 1)

      scf.for %chunk = %c0 to %num_chunks step %c1 {{
{_main_chunk_body(tile)}
      }}

      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @q4nx_flush_output(%{tile}_output, %m_i32)
        : (memref<{OUT_BF16}xbf16>, i32) -> ()
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^wt_ping, ^act_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 0 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^act_start:
      %1 = aie.dma_start(S2MM, 1, ^act_ping, ^record_start)
    ^act_ping:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 1 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 5 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_ping

    ^record_start:
      %2 = aie.dma_start(MM2S, 0, ^record_out, ^output_start)
    ^record_out:
      aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_record_empty, Release, 1)
      aie.next_bd ^record_out

    ^output_start:
      %3 = aie.dma_start(MM2S, 1, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{OUT_BF16}xbf16>, 0, {OUT_BF16}) {{bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_output_packet(group, row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _edge_tile(group: int, row: int) -> str:
    tile = f"edge{group}_{row}"
    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
{_lock_pair(tile, "record", 0)}
{_lock_pair(tile, "act", 2, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %num_chunks = arith.constant {NUM_CHUNKS} : index
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @edge_make_attention_slice(%{tile}_record, %{tile}_act_pong, %chunk_i32, %group_i32, %row_i32)
            : (memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) -> ()
        }} else {{
          func.call @edge_make_attention_slice(%{tile}_record, %{tile}_act_ping, %chunk_i32, %group_i32, %row_i32)
            : (memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_act_full, Release, 1)
      }}
      aie.use_lock(%{tile}_record_empty, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^record_in, ^act_start)
    ^record_in:
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_record_full, Release, 1)
      aie.next_bd ^record_in

    ^act_start:
      %1 = aie.dma_start(MM2S, 0, ^act_ping, ^end)
    ^act_ping:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 1 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 5 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_ping
    ^end:
      aie.end
    }}
"""


def _memtile_column(group: int) -> str:
    tile = f"mt{group}"
    output_bds = []
    for bd_index, row in enumerate(range(ROWS_PER_COLUMN)):
        next_bd_index = (bd_index + 1) % ROWS_PER_COLUMN
        output_bds.append(
            f"""    ^out_collect{bd_index}:
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, {row * OUT_BF16}, {OUT_BF16}) {{bd_id = {7 + bd_index} : i32, next_bd_id = {7 + next_bd_index} : i32}}
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.next_bd ^out_collect{next_bd_index}"""
        )

    row_channels = (
        (0, 2, 3, 0),
        (1, 24, 25, CHUNK_BF16),
        (2, 4, 5, 2 * CHUNK_BF16),
        (3, 26, 27, 3 * CHUNK_BF16),
    )
    row_streams = []
    for row, ping_bd, pong_bd, offset in row_channels:
        next_label = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^output_collect_start"
        row_streams.append(
            f"""    ^row{row}_start:
      %row{row}_dma = aie.dma_start(MM2S, {row}, ^row{row}_ping, {next_label})
    ^row{row}_ping:
      aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock(%{tile}_wt_empty, Release, 1)
      aie.next_bd ^row{row}_pong
    ^row{row}_pong:
      aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock(%{tile}_wt_empty, Release, 1)
      aie.next_bd ^row{row}_ping"""
        )

    return f"""
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{FAT_CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{FAT_CHUNK_BF16}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{COLUMN_OUTPUT_BF16}xbf16>
    %{tile}_wt_empty = aie.lock(%{tile}, 0) {{init = {ROWS_PER_COLUMN * 2} : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}
    %{tile}_output_empty = aie.lock(%{tile}, 2) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_output_empty"}}
    %{tile}_output_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_output_full"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^wt_in_ping, ^row0_start)
    ^wt_in_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_wt_ping : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^wt_in_pong
    ^wt_in_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_wt_pong : memref<{FAT_CHUNK_BF16}xbf16>, 0, {FAT_CHUNK_BF16}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^wt_in_ping

{chr(10).join(row_streams)}

    ^output_collect_start:
      %output_collect_dma = aie.dma_start(S2MM, 2, ^out_collect0, ^output_drain_start)
{chr(10).join(output_bds)}

    ^output_drain_start:
      %output_drain_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, 0, {COLUMN_OUTPUT_BF16}) {{bd_id = 34 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%output: memref<{OUT_TOTAL_I32}xi32>) {{"
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        weight_offset = group * COLUMN_WEIGHT_BF16 * 2
        output_offset = group * COLUMN_OUTPUT_BF16 * 2
        lines += [
            _npu_writebd(column, 13, COLUMN_OUTPUT_BF16 // 2, output_offset),
            _npu_address_patch(column, 13, 1, output_offset),
            _npu_push_queue(column, "S2MM", 1, 13, issue_token=True),
            _npu_writebd(column, 0, COLUMN_WEIGHT_BF16 // 2, weight_offset),
            _npu_address_patch(column, 0, 0, weight_offset),
            _npu_push_queue(column, "MM2S", 0, 0),
            _npu_sync(column, 1),
        ]
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    tile_defs = []
    for group, main_col in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({main_col}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({main_col}, 1)")
        for row in range(ROWS_PER_COLUMN):
            tile_defs.append(f"    %edge{group}_{row} = aie.tile({EDGE_COLUMNS[group]}, {row + 2})")
            tile_defs.append(f"    %m{group}_{row} = aie.tile({main_col}, {row + 2})")

    flows = []
    for group in range(len(MAIN_COLUMNS)):
        flows.append(f"    aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : 0)")
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : 0)")
            flows.append(f"    aie.flow(%m{group}_{row}, DMA : 0, %edge{group}_{row}, DMA : 0)")
            flows.append(f"    aie.flow(%edge{group}_{row}, DMA : 0, %m{group}_{row}, DMA : 1)")
            flows.append(f"    aie.packet_flow({_output_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%m{group}_{row}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : 2>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mt{group}, DMA : 5, %shim{group}, DMA : 1)")

    blocks = []
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_memtile_column(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_edge_tile(group, row))
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @emit_main_sideband(memref<{RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/q4nx_edge_o_phase.o"}}
    func.func private @edge_make_attention_slice(memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/q4nx_edge_o_phase.o"}}
    func.func private @q4nx_chunk_accum_slice(memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/q4nx_edge_o_phase.o"}}
    func.func private @q4nx_flush_output(memref<{OUT_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/q4nx_edge_o_phase.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
