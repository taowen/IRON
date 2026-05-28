"""Generate runnable MLIR-AIE for hierarchical record compaction."""

from __future__ import annotations

from pathlib import Path

from contract import (
    COMPACT_PACKET_DWORDS,
    CONSUMER_HALF_DWORDS,
    CONSUMER_INPUT_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)
from reference import (
    GLOBAL_PACKET_ID,
    MAIN_RECORD_DWORDS,
    CONSUMER_OUTPUT_DWORDS,
    column_packet,
    main_packet,
)

COLUMN_COMPACT_DWORDS = RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS
OUTPUT_DRAIN_BD = 34
BRIDGE_RECEIVE_BDS = ((0, 1), (24, 25), (2, 3), (26, 27))
COLUMN_RECEIVE_BDS = ((0, 1), (24, 25), (2, 3), (26, 27))


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
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


def npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def npu_push_queue(column: int, direction: str, channel: int, bd_id: int) -> str:
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = true, repeat_count = 0 : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _bridge_lock_defs() -> str:
    lines = [
        '    %bridge_low_full = aie.lock(%bridge, 0) {init = 0 : i32, sym_name = "bridge_low_full"}\n',
        '    %bridge_high_full = aie.lock(%bridge, 1) {init = 0 : i32, sym_name = "bridge_high_full"}\n',
        '    %bridge_drain_token = aie.lock(%bridge, 10) {init = 0 : i32, sym_name = "bridge_drain_token"}\n',
    ]
    for group in range(len(MAIN_COLUMNS)):
        low_lock = 2 + group * 2
        high_lock = low_lock + 1
        lines.append(
            f'    %bridge_low{group}_empty = aie.lock(%bridge, {low_lock}) '
            f'{{init = 1 : i32, sym_name = "bridge_low{group}_empty"}}\n'
        )
        lines.append(
            f'    %bridge_high{group}_empty = aie.lock(%bridge, {high_lock}) '
            f'{{init = 1 : i32, sym_name = "bridge_high{group}_empty"}}\n'
        )
    return "".join(lines)


def _column_lock_defs(tile: str) -> str:
    lines = [
        f'    %{tile}_low_full = aie.lock(%{tile}, 0) {{init = 0 : i32, sym_name = "{tile}_low_full"}}\n',
        f'    %{tile}_high_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_high_full"}}\n',
        f'    %{tile}_drain_token = aie.lock(%{tile}, 10) {{init = 0 : i32, sym_name = "{tile}_drain_token"}}\n',
    ]
    for row in range(ROWS_PER_COLUMN):
        low_lock = 2 + row * 2
        high_lock = low_lock + 1
        lines.append(
            f'    %{tile}_low{row}_empty = aie.lock(%{tile}, {low_lock}) '
            f'{{init = 1 : i32, sym_name = "{tile}_low{row}_empty"}}\n'
        )
        lines.append(
            f'    %{tile}_high{row}_empty = aie.lock(%{tile}, {high_lock}) '
            f'{{init = 1 : i32, sym_name = "{tile}_high{row}_empty"}}\n'
        )
    return "".join(lines)


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    if row == 0:
        source_offset = 0
        source_length = RECORD_DWORDS
    else:
        source_offset = 1
        source_length = RECORD_PAYLOAD_DWORDS
    packet = main_packet(group, row)
    return f"""
    %{tile}_records = aie.buffer(%{tile}) {{sym_name = "{tile}_records"}} : memref<{MAIN_RECORD_DWORDS}xi32>
{_lock_pair(tile, "records", 0, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 2)
      func.call @emit_paired_records(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 2)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(MM2S, 0, ^low_out, ^end)
    ^low_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 0 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^high_out
    ^high_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {RECORD_DWORDS + source_offset}, {source_length}) {{bd_id = 1 : i32, next_bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^low_out
    ^end:
      aie.end
    }}
"""


def _column_memtile(group: int) -> str:
    tile = f"mt{group}"
    packet = column_packet(group)
    source_offset = 0 if group == 0 else 1
    source_length = COLUMN_COMPACT_DWORDS if group == 0 else COLUMN_COMPACT_DWORDS - 1
    receive_starts = []
    for row in range(ROWS_PER_COLUMN):
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^out_start"
        if row == 0:
            length = RECORD_DWORDS
            dest_offset = 0
        else:
            length = RECORD_PAYLOAD_DWORDS
            dest_offset = RECORD_DWORDS + (row - 1) * RECORD_PAYLOAD_DWORDS
        low_bd, high_bd = COLUMN_RECEIVE_BDS[row]
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_low, {next_start})
    ^row{row}_low:
      aie.use_lock(%{tile}_low{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_low : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {low_bd} : i32, next_bd_id = {high_bd} : i32}}
      aie.use_lock(%{tile}_low_full, Release, 1)
      aie.next_bd ^row{row}_high
    ^row{row}_high:
      aie.use_lock(%{tile}_high{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_high : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {high_bd} : i32, next_bd_id = {low_bd} : i32}}
      aie.use_lock(%{tile}_high_full, Release, 1)
      aie.next_bd ^row{row}_low"""
        )

    return f"""
    %{tile}_low = aie.buffer(%{tile}) {{sym_name = "{tile}_low"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
    %{tile}_high = aie.buffer(%{tile}) {{sym_name = "{tile}_high"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
{_column_lock_defs(tile)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^low_out, ^end)
    ^low_out:
      aie.use_lock(%{tile}_low_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_low : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 34 : i32, next_bd_id = 35 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^high_out
    ^high_out:
      aie.use_lock(%{tile}_high_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_high : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 35 : i32, next_bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^low_out
    ^end:
      aie.end
    }}
"""


def _bridge() -> str:
    receive_starts = []
    for group in range(len(MAIN_COLUMNS)):
        start_label = "" if group == 0 else f"    ^g{group}_start:\n"
        next_start = f"^g{group + 1}_start" if group + 1 < len(MAIN_COLUMNS) else "^out_start"
        if group == 0:
            length = COLUMN_COMPACT_DWORDS
            dest_offset = 0
        else:
            length = COLUMN_COMPACT_DWORDS - 1
            dest_offset = COLUMN_COMPACT_DWORDS + (group - 1) * (COLUMN_COMPACT_DWORDS - 1)
        low_bd, high_bd = BRIDGE_RECEIVE_BDS[group]
        receive_starts.append(
            f"""{start_label}      %{group} = aie.dma_start(S2MM, {group}, ^g{group}_low, {next_start})
    ^g{group}_low:
      aie.use_lock(%bridge_low{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_low : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {low_bd} : i32, next_bd_id = {high_bd} : i32}}
      aie.use_lock(%bridge_low_full, Release, 1)
      aie.next_bd ^g{group}_high
    ^g{group}_high:
      aie.use_lock(%bridge_high{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_high : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {high_bd} : i32, next_bd_id = {low_bd} : i32}}
      aie.use_lock(%bridge_high_full, Release, 1)
      aie.next_bd ^g{group}_low"""
        )
    return f"""
    %bridge_low = aie.buffer(%bridge) {{sym_name = "bridge_low"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_high = aie.buffer(%bridge) {{sym_name = "bridge_high"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
{_bridge_lock_defs()}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out = aie.dma_start(MM2S, 5, ^low_out, ^end)
    ^low_out:
      aie.use_lock(%bridge_low_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_low : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {CONSUMER_HALF_DWORDS}) {{bd_id = 34 : i32, next_bd_id = 35 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^high_out
    ^high_out:
      aie.use_lock(%bridge_high_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_high : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {CONSUMER_HALF_DWORDS}) {{bd_id = 35 : i32, next_bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^low_out
    ^end:
      aie.end
    }}
"""


def _consumer() -> str:
    return f"""
    %consumer_input = aie.buffer(%consumer) {{sym_name = "consumer_input"}} : memref<{CONSUMER_INPUT_DWORDS}xi32>
    %consumer_output = aie.buffer(%consumer) {{sym_name = "consumer_output"}} : memref<{CONSUMER_OUTPUT_DWORDS}xi32>
{_lock_pair("consumer", "input", 0, init_empty=2)}
{_lock_pair("consumer", "output", 2)}

    %consumer_core = aie.core(%consumer) {{
      %dwords_i32 = arith.constant {CONSUMER_INPUT_DWORDS} : i32
      aie.use_lock(%consumer_input_full, AcquireGreaterEqual, 2)
      aie.use_lock(%consumer_output_empty, AcquireGreaterEqual, 1)
      func.call @merge_paired_halves(%consumer_input, %consumer_output, %dwords_i32)
        : (memref<{CONSUMER_INPUT_DWORDS}xi32>, memref<{CONSUMER_OUTPUT_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%consumer_input_empty, Release, 2)
      aie.use_lock(%consumer_output_full, Release, 1)
      aie.end
    }}

    %consumer_mem = aie.mem(%consumer) {{
      %0 = aie.dma_start(S2MM, 0, ^low_in, ^out_start)
    ^low_in:
      aie.use_lock(%consumer_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%consumer_input : memref<{CONSUMER_INPUT_DWORDS}xi32>, 0, {CONSUMER_HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%consumer_input_full, Release, 1)
      aie.next_bd ^high_in
    ^high_in:
      aie.use_lock(%consumer_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%consumer_input : memref<{CONSUMER_INPUT_DWORDS}xi32>, {CONSUMER_HALF_DWORDS}, {CONSUMER_HALF_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%consumer_input_full, Release, 1)
      aie.next_bd ^low_in

    ^out_start:
      %1 = aie.dma_start(MM2S, 1, ^out, ^end)
    ^out:
      aie.use_lock(%consumer_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%consumer_output : memref<{CONSUMER_OUTPUT_DWORDS}xi32>, 0, {CONSUMER_OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%consumer_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}
"""


def _hub() -> str:
    return f"""
    %hub_output = aie.buffer(%hub) {{sym_name = "hub_output"}} : memref<{CONSUMER_OUTPUT_DWORDS}xi32>
{_lock_pair("hub", "output", 0)}

    %hub_dma = aie.memtile_dma(%hub) {{
      %0 = aie.dma_start(S2MM, 0, ^output_in, ^output_drain_start)
    ^output_in:
      aie.use_lock(%hub_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_output : memref<{CONSUMER_OUTPUT_DWORDS}xi32>, 0, {CONSUMER_OUTPUT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_output_full, Release, 1)
      aie.next_bd ^output_in

    ^output_drain_start:
      %1 = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%hub_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_output : memref<{CONSUMER_OUTPUT_DWORDS}xi32>, 0, {CONSUMER_OUTPUT_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%hub_output_empty, Release, 1)
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        (
            f"    aie.runtime_sequence(%output: memref<{CONSUMER_OUTPUT_DWORDS}xi32>) {{",
            npu_writebd(6, 13, CONSUMER_OUTPUT_DWORDS, 0),
            npu_address_patch(6, 13, 0, 0),
            npu_push_queue(6, "S2MM", 1, 13),
            npu_sync(6, 1),
            "    }",
        )
    )


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    tile_defs = [
        "    %shim_out = aie.tile(6, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %hub = aie.tile(6, 1)",
        "    %consumer = aie.tile(6, 2)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = []
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.packet_flow({main_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%{_main_symbol(group, row)}, DMA : 0>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : {row}>")
            flows.append("    }")
        flows.append(f"    aie.packet_flow({column_packet(group)}) {{")
        flows.append(f"      aie.packet_source<%mt{group}, DMA : 5>")
        flows.append(f"      aie.packet_dest<%bridge, DMA : {group}>")
        flows.append("    }")
    flows.extend(
        (
            f"    aie.packet_flow({GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%consumer, DMA : 0>",
            "    }",
            "    aie.flow(%consumer, DMA : 1, %hub, DMA : 0)",
            "    aie.flow(%hub, DMA : 5, %shim_out, DMA : 1)",
        )
    )

    blocks = [_bridge(), _consumer(), _hub()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @emit_paired_records(memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/dataflow_kernels.o"}}
    func.func private @merge_paired_halves(memref<{CONSUMER_INPUT_DWORDS}xi32>, memref<{CONSUMER_OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/dataflow_kernels.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        "aie.tile(6, 2)",
        "aie.tile(1, 1)",
        f"aie.packet_flow({GLOBAL_PACKET_ID})",
        "aie.packet_source<%bridge, DMA : 5>",
        "aie.packet_dest<%consumer, DMA : 0>",
        f"memref<{COMPACT_PACKET_DWORDS}xi32>",
        f"memref<{CONSUMER_INPUT_DWORDS}xi32>",
        f"memref<{CONSUMER_OUTPUT_DWORDS}xi32>",
        "emit_paired_records",
        "merge_paired_halves",
        "dataflow_kernels.o",
    )
    errors = [f"missing compaction marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count("aie.packet_flow(") != len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 1:
        errors.append("packet flow count mismatch")
    if mlir.count("aie.tile(") != 4 + len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1):
        errors.append("tile count mismatch")
    if mlir.count("aie.dma_start(S2MM,") < len(MAIN_COLUMNS) + 2:
        errors.append("missing S2MM starts for compact route")
    if f"packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>" not in mlir:
        errors.append("global compact -> consumer packet marker missing")
    return errors
