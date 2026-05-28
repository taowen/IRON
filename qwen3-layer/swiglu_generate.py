"""Generate runnable MLIR-AIE for main16 up/gate compact route into c6r2."""

from __future__ import annotations

from pathlib import Path

from contract import (
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)
from swiglu_reference import (
    CASE_NAME,
    COLUMN_PACKET_BASE,
    GLOBAL_PACKET_ID,
    MAIN_RECORD_DWORDS,
    SWIGLU_OUTPUT_DWORDS,
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
        '    %bridge_up_full = aie.lock(%bridge, 0) {init = 0 : i32, sym_name = "bridge_up_full"}\n',
        '    %bridge_gate_full = aie.lock(%bridge, 1) {init = 0 : i32, sym_name = "bridge_gate_full"}\n',
        '    %bridge_drain_token = aie.lock(%bridge, 10) {init = 0 : i32, sym_name = "bridge_drain_token"}\n',
    ]
    for group in range(len(MAIN_COLUMNS)):
        up_lock = 2 + group * 2
        gate_lock = up_lock + 1
        lines.append(
            f'    %bridge_up{group}_empty = aie.lock(%bridge, {up_lock}) '
            f'{{init = 1 : i32, sym_name = "bridge_up{group}_empty"}}\n'
        )
        lines.append(
            f'    %bridge_gate{group}_empty = aie.lock(%bridge, {gate_lock}) '
            f'{{init = 1 : i32, sym_name = "bridge_gate{group}_empty"}}\n'
        )
    return "".join(lines)


def _column_lock_defs(tile: str) -> str:
    lines = [
        f'    %{tile}_up_full = aie.lock(%{tile}, 0) {{init = 0 : i32, sym_name = "{tile}_up_full"}}\n',
        f'    %{tile}_gate_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_gate_full"}}\n',
        f'    %{tile}_drain_token = aie.lock(%{tile}, 10) {{init = 0 : i32, sym_name = "{tile}_drain_token"}}\n',
    ]
    for row in range(ROWS_PER_COLUMN):
        up_lock = 2 + row * 2
        gate_lock = up_lock + 1
        lines.append(
            f'    %{tile}_up{row}_empty = aie.lock(%{tile}, {up_lock}) '
            f'{{init = 1 : i32, sym_name = "{tile}_up{row}_empty"}}\n'
        )
        lines.append(
            f'    %{tile}_gate{row}_empty = aie.lock(%{tile}, {gate_lock}) '
            f'{{init = 1 : i32, sym_name = "{tile}_gate{row}_empty"}}\n'
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
      func.call @ffn_emit_records(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 2)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(MM2S, 0, ^up_out, ^end)
    ^up_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 0 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {RECORD_DWORDS + source_offset}, {source_length}) {{bd_id = 1 : i32, next_bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^up_out
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
        up_bd, gate_bd = COLUMN_RECEIVE_BDS[row]
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_up, {next_start})
    ^row{row}_up:
      aie.use_lock(%{tile}_up{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_up : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {up_bd} : i32, next_bd_id = {gate_bd} : i32}}
      aie.use_lock(%{tile}_up_full, Release, 1)
      aie.next_bd ^row{row}_gate
    ^row{row}_gate:
      aie.use_lock(%{tile}_gate{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_gate : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {gate_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%{tile}_gate_full, Release, 1)
      aie.next_bd ^row{row}_up"""
        )

    return f"""
    %{tile}_up = aie.buffer(%{tile}) {{sym_name = "{tile}_up"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
    %{tile}_gate = aie.buffer(%{tile}) {{sym_name = "{tile}_gate"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
{_column_lock_defs(tile)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^up_out, ^end)
    ^up_out:
      aie.use_lock(%{tile}_up_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_up : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 34 : i32, next_bd_id = 35 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%{tile}_gate_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_gate : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = 35 : i32, next_bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^up_out
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
        up_bd, gate_bd = BRIDGE_RECEIVE_BDS[group]
        receive_starts.append(
            f"""{start_label}      %{group} = aie.dma_start(S2MM, {group}, ^g{group}_up, {next_start})
    ^g{group}_up:
      aie.use_lock(%bridge_up{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_up : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {up_bd} : i32, next_bd_id = {gate_bd} : i32}}
      aie.use_lock(%bridge_up_full, Release, 1)
      aie.next_bd ^g{group}_gate
    ^g{group}_gate:
      aie.use_lock(%bridge_gate{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_gate : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {gate_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%bridge_gate_full, Release, 1)
      aie.next_bd ^g{group}_up"""
        )
    return f"""
    %bridge_up = aie.buffer(%bridge) {{sym_name = "bridge_up"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_gate = aie.buffer(%bridge) {{sym_name = "bridge_gate"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
{_bridge_lock_defs()}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out = aie.dma_start(MM2S, 5, ^up_out, ^end)
    ^up_out:
      aie.use_lock(%bridge_up_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_up : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {C6R2_HALF_DWORDS}) {{bd_id = 34 : i32, next_bd_id = 35 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%bridge_gate_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_gate : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {C6R2_HALF_DWORDS}) {{bd_id = 35 : i32, next_bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^up_out
    ^end:
      aie.end
    }}
"""


def _swiglu() -> str:
    return f"""
    %swiglu_input = aie.buffer(%swiglu) {{sym_name = "swiglu_input"}} : memref<{C6R2_INPUT_DWORDS}xi32>
    %swiglu_output = aie.buffer(%swiglu) {{sym_name = "swiglu_output"}} : memref<{SWIGLU_OUTPUT_DWORDS}xi32>
{_lock_pair("swiglu", "input", 0, init_empty=2)}
{_lock_pair("swiglu", "output", 2)}

    %swiglu_core = aie.core(%swiglu) {{
      %dwords_i32 = arith.constant {C6R2_INPUT_DWORDS} : i32
      aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)
      aie.use_lock(%swiglu_output_empty, AcquireGreaterEqual, 1)
      func.call @ffn_swiglu_contract(%swiglu_input, %swiglu_output, %dwords_i32)
        : (memref<{C6R2_INPUT_DWORDS}xi32>, memref<{SWIGLU_OUTPUT_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%swiglu_input_empty, Release, 2)
      aie.use_lock(%swiglu_output_full, Release, 1)
      aie.end
    }}

    %swiglu_mem = aie.mem(%swiglu) {{
      %0 = aie.dma_start(S2MM, 0, ^up_in, ^out_start)
    ^up_in:
      aie.use_lock(%swiglu_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_input : memref<{C6R2_INPUT_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%swiglu_input_full, Release, 1)
      aie.next_bd ^gate_in
    ^gate_in:
      aie.use_lock(%swiglu_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_input : memref<{C6R2_INPUT_DWORDS}xi32>, {C6R2_HALF_DWORDS}, {C6R2_HALF_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%swiglu_input_full, Release, 1)
      aie.next_bd ^up_in

    ^out_start:
      %1 = aie.dma_start(MM2S, 1, ^out, ^end)
    ^out:
      aie.use_lock(%swiglu_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_output : memref<{SWIGLU_OUTPUT_DWORDS}xi32>, 0, {SWIGLU_OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%swiglu_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}
"""


def _hub() -> str:
    return f"""
    %hub_output = aie.buffer(%hub) {{sym_name = "hub_output"}} : memref<{SWIGLU_OUTPUT_DWORDS}xi32>
{_lock_pair("hub", "output", 0)}

    %hub_dma = aie.memtile_dma(%hub) {{
      %0 = aie.dma_start(S2MM, 0, ^output_in, ^output_drain_start)
    ^output_in:
      aie.use_lock(%hub_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_output : memref<{SWIGLU_OUTPUT_DWORDS}xi32>, 0, {SWIGLU_OUTPUT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_output_full, Release, 1)
      aie.next_bd ^output_in

    ^output_drain_start:
      %1 = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%hub_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_output : memref<{SWIGLU_OUTPUT_DWORDS}xi32>, 0, {SWIGLU_OUTPUT_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%hub_output_empty, Release, 1)
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        (
            f"    aie.runtime_sequence(%output: memref<{SWIGLU_OUTPUT_DWORDS}xi32>) {{",
            npu_writebd(6, 13, SWIGLU_OUTPUT_DWORDS, 0),
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
        "    %swiglu = aie.tile(6, 2)",
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
            "      aie.packet_dest<%swiglu, DMA : 0>",
            "    }",
            "    aie.flow(%swiglu, DMA : 1, %hub, DMA : 0)",
            "    aie.flow(%hub, DMA : 5, %shim_out, DMA : 1)",
        )
    )

    blocks = [_bridge(), _swiglu(), _hub()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @ffn_emit_records(memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @ffn_swiglu_contract(memref<{C6R2_INPUT_DWORDS}xi32>, memref<{SWIGLU_OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}

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
        "aie.packet_dest<%swiglu, DMA : 0>",
        f"memref<{COMPACT_PACKET_DWORDS}xi32>",
        f"memref<{C6R2_INPUT_DWORDS}xi32>",
        f"memref<{SWIGLU_OUTPUT_DWORDS}xi32>",
        "ffn_emit_records",
        "ffn_swiglu_contract",
        "qwen3_bridge.o",
    )
    errors = [f"missing swiglu marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count("aie.packet_flow(") != len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 1:
        errors.append("packet flow count mismatch")
    if mlir.count("aie.tile(") != 4 + len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1):
        errors.append("tile count mismatch")
    if mlir.count("aie.dma_start(S2MM,") < len(MAIN_COLUMNS) + 2:
        errors.append("missing S2MM starts for compact route")
    if f"packet = #aie.packet_info<pkt_type = 0, pkt_id = {GLOBAL_PACKET_ID}>" not in mlir:
        errors.append("global c1r1 -> c6r2 packet marker missing")
    if CASE_NAME not in (CASE_NAME,):
        errors.append("unreachable case marker")
    return errors
