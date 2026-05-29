"""Generate runnable MLIR-AIE for c6r1 -> c1r1 -> main16 bridge cases."""

from __future__ import annotations

from pathlib import Path

from bridge_reference import (
    BRIDGE_QUANTUM_DWORDS,
    COLUMN_SUMMARY_DWORDS,
    MAIN_CHUNK_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    ROWS_PER_COLUMN,
    SUMMARY_DWORDS,
    TOTAL_SUMMARY_DWORDS,
    BridgeCase,
)

OUTPUT_COLLECT_BD_BASE = 10
OUTPUT_DRAIN_BD = 34


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


def _main_packet(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _hub(case: BridgeCase) -> str:
    return f"""
    %hub_payload = aie.buffer(%hub) {{sym_name = "hub_payload"}} : memref<{case.payload_dwords}xi32>
{_lock_pair("hub", "payload", 0)}

    %hub_dma = aie.memtile_dma(%hub) {{
      %0 = aie.dma_start(S2MM, 0, ^payload_in, ^payload_out_start)
    ^payload_in:
      aie.use_lock(%hub_payload_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_payload : memref<{case.payload_dwords}xi32>, 0, {case.payload_dwords}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_payload_full, Release, 1)
      aie.next_bd ^payload_in

    ^payload_out_start:
      %1 = aie.dma_start(MM2S, 5, ^payload_out, ^end)
    ^payload_out:
      aie.use_lock(%hub_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_payload : memref<{case.payload_dwords}xi32>, 0, {case.payload_dwords}) {{bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {case.packet_id}>}}
      aie.use_lock(%hub_payload_empty, Release, 1)
      aie.next_bd ^payload_out
    ^end:
      aie.end
    }}
"""


def _bridge() -> str:
    return f"""
    %bridge_ping = aie.buffer(%bridge) {{sym_name = "bridge_ping"}} : memref<{BRIDGE_QUANTUM_DWORDS}xi32>
    %bridge_pong = aie.buffer(%bridge) {{sym_name = "bridge_pong"}} : memref<{BRIDGE_QUANTUM_DWORDS}xi32>
{_lock_pair("bridge", "buf", 0, init_empty=2)}

    %bridge_dma = aie.memtile_dma(%bridge) {{
      %0 = aie.dma_start(S2MM, 4, ^in_ping, ^out_start)
    ^in_ping:
      aie.use_lock(%bridge_buf_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_ping : memref<{BRIDGE_QUANTUM_DWORDS}xi32>, 0, {BRIDGE_QUANTUM_DWORDS}) {{bd_id = 6 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%bridge_buf_full, Release, 1)
      aie.next_bd ^in_pong
    ^in_pong:
      aie.use_lock(%bridge_buf_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_pong : memref<{BRIDGE_QUANTUM_DWORDS}xi32>, 0, {BRIDGE_QUANTUM_DWORDS}) {{bd_id = 7 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%bridge_buf_full, Release, 1)
      aie.next_bd ^in_ping

    ^out_start:
      %1 = aie.dma_start(MM2S, 1, ^out_ping, ^end)
    ^out_ping:
      aie.use_lock(%bridge_buf_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_ping : memref<{BRIDGE_QUANTUM_DWORDS}xi32>, 0, {BRIDGE_QUANTUM_DWORDS}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%bridge_buf_empty, Release, 1)
      aie.next_bd ^out_pong
    ^out_pong:
      aie.use_lock(%bridge_buf_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_pong : memref<{BRIDGE_QUANTUM_DWORDS}xi32>, 0, {BRIDGE_QUANTUM_DWORDS}) {{bd_id = 29 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%bridge_buf_empty, Release, 1)
      aie.next_bd ^out_ping
    ^end:
      aie.end
    }}
"""


def _main_tile(group: int, row: int, case: BridgeCase) -> str:
    tile = _main_symbol(group, row)
    return f"""
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_summary = aie.buffer(%{tile}) {{sym_name = "{tile}_summary"}} : memref<{SUMMARY_DWORDS}xi32>
{_lock_pair(tile, "chunk", 0, init_empty=2)}
{_lock_pair(tile, "output", 2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %chunks = arith.constant {case.main_chunks} : index
      %dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      func.call @bridge_init_summary(%{tile}_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      scf.for %chunk = %c0 to %chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @bridge_accum_chunk(%{tile}_chunk_pong, %{tile}_summary, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32) -> ()
        }} else {{
          func.call @bridge_accum_chunk(%{tile}_chunk_ping, %{tile}_summary, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^chunk_ping, ^output_start)
    ^chunk_ping:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_ping : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_pong
    ^chunk_pong:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_pong : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_ping

    ^output_start:
      %1 = aie.dma_start(MM2S, 1, ^summary_out, ^end)
    ^summary_out:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_summary : memref<{SUMMARY_DWORDS}xi32>, 0, {SUMMARY_DWORDS}) {{bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_main_packet(group, row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^summary_out
    ^end:
      aie.end
    }}
"""


def _column_memtile(group: int) -> str:
    tile = f"mt{group}"
    output_bds = []
    for row in range(ROWS_PER_COLUMN):
        next_row = (row + 1) % ROWS_PER_COLUMN
        bd_id = OUTPUT_COLLECT_BD_BASE + row
        next_bd_id = OUTPUT_COLLECT_BD_BASE + next_row
        output_bds.append(f"""    ^out{row}:
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{COLUMN_SUMMARY_DWORDS}xi32>, {row * SUMMARY_DWORDS}, {SUMMARY_DWORDS}) {{bd_id = {bd_id} : i32, next_bd_id = {next_bd_id} : i32}}
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.next_bd ^out{next_row}""")

    return f"""
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{COLUMN_SUMMARY_DWORDS}xi32>
    %{tile}_output_empty = aie.lock(%{tile}, 0) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_output_empty"}}
    %{tile}_output_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_output_full"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %0 = aie.dma_start(S2MM, 2, ^out0, ^output_drain_start)
{chr(10).join(output_bds)}

    ^output_drain_start:
      %1 = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_output : memref<{COLUMN_SUMMARY_DWORDS}xi32>, 0, {COLUMN_SUMMARY_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%{tile}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence(case: BridgeCase) -> str:
    lines = [
        f"    aie.runtime_sequence(%payload: memref<{case.payload_dwords}xi32>, "
        f"%output: memref<{TOTAL_SUMMARY_DWORDS}xi32>) {{"
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        output_offset = group * COLUMN_SUMMARY_DWORDS * 4
        lines.extend(
            (
                npu_writebd(column, 13, COLUMN_SUMMARY_DWORDS, output_offset),
                npu_address_patch(column, 13, 1, output_offset),
                npu_push_queue(column, "S2MM", 1, 13),
            )
        )

    lines.extend(
        (
            npu_writebd(6, 0, case.payload_dwords, 0),
            npu_address_patch(6, 0, 0, 0),
            npu_push_queue(6, "MM2S", 0, 0),
            npu_sync(6, 0, direction=1),
        )
    )
    for column in MAIN_COLUMNS:
        lines.append(npu_sync(column, 1))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir(case: BridgeCase) -> str:
    experiment_dir = Path(__file__).parent.resolve()
    tile_defs = [
        "    %shim_src = aie.tile(6, 0)",
        "    %hub = aie.tile(6, 1)",
        "    %bridge = aie.tile(1, 1)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = [
        "    aie.flow(%shim_src, DMA : 0, %hub, DMA : 0)",
        f"    aie.packet_flow({case.packet_id}) {{",
        "      aie.packet_source<%hub, DMA : 5>",
        "      aie.packet_dest<%bridge, DMA : 4>",
        "    }",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.flow(%bridge, DMA : 1, %{_main_symbol(group, row)}, DMA : 0)")
            flows.append(f"    aie.packet_flow({_main_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%{_main_symbol(group, row)}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : 2>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mt{group}, DMA : 5, %shim{group}, DMA : 1)")

    blocks = [_hub(case), _bridge()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row, case))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @bridge_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @bridge_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}

{chr(10).join(blocks)}
{_runtime_sequence(case)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, case: BridgeCase) -> list[str]:
    required = (
        "aie.tile(6, 1)",
        "aie.tile(1, 1)",
        f"aie.packet_flow({case.packet_id})",
        "aie.packet_source<%hub, DMA : 5>",
        "aie.packet_dest<%bridge, DMA : 4>",
        "aie.flow(%bridge, DMA : 1, %m0_0, DMA : 0)",
        f"memref<{case.payload_dwords}xi32>",
        f"memref<{TOTAL_SUMMARY_DWORDS}xi32>",
        "debug_contract.o",
    )
    errors = [f"missing bridge marker: {marker}" for marker in required if marker not in mlir]
    expected_main_flows = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    actual_main_flows = mlir.count("aie.flow(%bridge, DMA : 1")
    if actual_main_flows != expected_main_flows:
        errors.append(f"main multicast flow count: {actual_main_flows} != {expected_main_flows}")
    return errors
