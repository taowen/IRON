"""Generate MLIR-AIE for Q4NX Q body from host hidden through c1r2 replay."""

from __future__ import annotations

from pathlib import Path

from contract import C1R2_PACKET_DWORDS, CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, RECORD_DWORDS, ROWS_PER_COLUMN
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_absent_markers,
    require_count,
    require_disjoint_bd_ids,
    require_dma_bd_lock_balance,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_bd_ids,
)
from compact_dataflow import (
    BODY_RECORD_SLOTS,
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    FULL_REPLAY_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    Q_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _bd_dimensions,
    _bridge,
    _compact_phase,
    _main_record_transfer,
    _main_symbol,
    _phase_trace_marker,
    column_packet,
    main_packet,
    q4nx_weight_column_memtile,
)
from projection_schedule import Q_BODY_RECORDS, Q_CHUNKS_PER_RECORD
from cases.q4nx_q_body_reference import (
    CASE_NAME,
    COLUMN_WEIGHT_BF16,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    PATCH_WEIGHT_BF16,
    TOTAL_WEIGHT_I32,
)

Q_BODY_PHASE_TRACE = (_compact_phase("q", "q", BODY_RECORD_SLOTS[0], Q_BODY_RECORDS),)
Q_MAIN_RECORD_DWORDS = Q_BODY_RECORDS * RECORD_DWORDS


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_DWORDS}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%output: memref<{OUTPUT_DWORDS}xi32>) {{"
    ]
    lines.extend(
        (
            npu_writebd(1, 0, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 0, 0, 0),
            npu_push_queue(1, "MM2S", 0, 0),
            npu_writebd(1, 13, OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 2, 0),
            npu_push_queue(1, "S2MM", 1, 13),
        )
    )
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = group * COLUMN_WEIGHT_BF16 * 2
        patch_bytes = PATCH_WEIGHT_BF16 * 2
        lines.extend(
            (
                npu_writebd(column, 0, PATCH_WEIGHT_BF16 // 2, column_base),
                npu_address_patch(column, 0, 1, column_base),
                npu_writebd(column, 1, PATCH_WEIGHT_BF16 // 2, column_base + patch_bytes),
                npu_address_patch(column, 1, 1, column_base + patch_bytes),
                npu_push_queue(column, "MM2S", 0, 0),
                npu_push_queue(column, "MM2S", 1, 1),
            )
        )
    lines.append(npu_sync(1, 1))
    lines.append(npu_sync(1, 0, direction=1))
    for column in MAIN_COLUMNS:
        lines.extend((npu_sync(column, 0, direction=1), npu_sync(column, 1, direction=1)))
    lines.append("    }")
    return "\n".join(lines)


def _full_hidden_replay() -> str:
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}

    %full_mem = aie.mem(%full) {{
      %hidden_dma = aie.dma_start(S2MM, 0, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_hidden_full, Release, {Q_BODY_RECORDS})
      aie.next_bd ^hidden_in

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^end)
    ^replay_out:
      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
      aie.use_lock(%full_hidden_empty, Release, 1)
      aie.next_bd ^replay_out
    ^end:
      aie.end
    }}
"""


def _post_drain() -> str:
    return f"""
    %post_q_payload = aie.buffer(%post) {{sym_name = "post_q_payload"}} : memref<{OUTPUT_DWORDS}xi32>
    %post_q_empty = aie.lock(%post, 0) {{init = 1 : i32, sym_name = "post_q_empty"}}
    %post_q_full = aie.lock(%post, 1) {{init = 0 : i32, sym_name = "post_q_full"}}

    %post_mem = aie.mem(%post) {{
      %q_in_dma = aie.dma_start(S2MM, 0, ^q_in, ^q_out_start)
    ^q_in:
      aie.use_lock(%post_q_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%post_q_full, Release, 1)
      aie.next_bd ^q_in

    ^q_out_start:
      %q_out_dma = aie.dma_start(MM2S, 0, ^q_out, ^end)
    ^q_out:
      aie.use_lock(%post_q_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%post_q_empty, Release, 1)
      aie.next_bd ^q_out
    ^end:
      aie.end
    }}
"""


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    packet = main_packet(group, row)
    buffer_name, offset, length, dimensions = _main_record_transfer(Q_BODY_PHASE_TRACE[0], row)
    if buffer_name != "q_records":
        raise RuntimeError(f"bad q body record buffer: {buffer_name}")
    return f"""
    %{tile}_q_records = aie.buffer(%{tile}) {{sym_name = "{tile}_q_records"}} : memref<{Q_MAIN_RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_q4nx_output = aie.buffer(%{tile}) {{sym_name = "{tile}_q4nx_output"}} : memref<32xbf16>
    %{tile}_records_empty = aie.lock(%{tile}, 0) {{init = 1 : i32, sym_name = "{tile}_records_empty"}}
    %{tile}_records_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_records_full"}}
    %{tile}_chunk_empty = aie.lock(%{tile}, 2) {{init = 2 : i32, sym_name = "{tile}_chunk_empty"}}
    %{tile}_chunk_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_chunk_full"}}
    %{tile}_wt_empty = aie.lock(%{tile}, 4) {{init = 2 : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, 5) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %q_records = arith.constant {Q_BODY_RECORDS} : index
      %chunks_per_record = arith.constant {Q_CHUNKS_PER_RECORD} : index
      %m_i32 = arith.constant 32 : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      scf.for %block = %c0 to %q_records step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        func.call @clear_summary(%{tile}_q4nx_output, %m_i32)
          : (memref<32xbf16>, i32) -> ()
        scf.for %chunk = %c0 to %chunks_per_record step %c1 {{
          %base_chunk = arith.muli %block, %chunks_per_record : index
          %global_chunk = arith.addi %base_chunk, %chunk : index
          %global_chunk_i32 = arith.index_cast %global_chunk : index to i32
          %rem = arith.remsi %global_chunk_i32, %c2_i32 : i32
          %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
          aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
          aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
          scf.if %is_pong {{
            func.call @q4nx_chunk_accum_slice_i32(%{tile}_wt_pong, %{tile}_chunk_pong, %m_i32)
              : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) -> ()
          }} else {{
            func.call @q4nx_chunk_accum_slice_i32(%{tile}_wt_ping, %{tile}_chunk_ping, %m_i32)
              : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) -> ()
          }}
          aie.use_lock(%{tile}_chunk_empty, Release, 1)
          aie.use_lock(%{tile}_wt_empty, Release, 1)
        }}
        func.call @q4nx_flush_output(%{tile}_q4nx_output, %m_i32)
          : (memref<32xbf16>, i32) -> ()
        func.call @q4nx_emit_q_body_record(%{tile}_q_records, %{tile}_q4nx_output, %group_i32, %row_i32, %block_i32, %m_i32)
          : (memref<{Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) -> ()
      }}
      aie.use_lock(%{tile}_records_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^wt_start)
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

    ^wt_start:
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 8 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 9 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^q_out, ^end)
    ^q_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_q_records : memref<{Q_MAIN_RECORD_DWORDS}xi32>, {offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^q_out
    ^end:
      aie.end
    }}
"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    flows.append(flow("shim", 0, "full", 0))
    flows.append(packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4))
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(packet_flow(main_packet(group, row), _main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, _main_symbol(group, row), 1))
        flows.append(packet_flow(column_packet(group), f"mt{group}", 5, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
    flows.append(packet_flow(Q_GLOBAL_PACKET_ID, "bridge", 5, "post", 0))
    flows.append(flow("post", 0, "shim", 1))

    blocks = [_full_hidden_replay(), _bridge(Q_BODY_PHASE_TRACE), _post_drain()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, Q_BODY_PHASE_TRACE))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @clear_summary(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_chunk_accum_slice_i32(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_flush_output(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_q_body_record(memref<{Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(Q_BODY_PHASE_TRACE)}",
        "q4nx_emit_q_body_record",
        "q4nx_chunk_accum_slice_i32",
        "main_projection_q4nx.o",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"aie.packet_flow({FULL_REPLAY_PACKET_ID})",
        f"aie.packet_flow({Q_GLOBAL_PACKET_ID})",
        "aie.flow(%bridge, DMA : 1",
        "aie.dma_start(S2MM, 4, ^packet_in_ping, ^packet_out_start)",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping, ^wt_row0_start)",
        "aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)",
        "arg_idx = 0 : i32",
        "arg_idx = 1 : i32",
        "arg_idx = 2 : i32",
    )
    errors = [f"missing q4nx q-body marker: {marker}" for marker in required if marker not in mlir]
    if "qwen3_layer.o" in mlir:
        errors.append("q4nx-q-body must not link the old mixed qwen3_layer object")
    if "qwen3_bridge.o" in mlir:
        errors.append("q4nx-q-body must not link the old mixed qwen3_bridge object")
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 2
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "main activation bridge flows", mlir.count("aie.flow(%bridge, DMA : 1"), len(MAIN_COLUMNS) * ROWS_PER_COLUMN))
    errors.extend(require_count(CASE_NAME, "q4nx q emit calls", mlir.count("func.call @q4nx_emit_q_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx chunk call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32"), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 2))
    errors.extend(require_count(CASE_NAME, "hidden arg0 address patches", mlir.count("arg_idx = 0 : i32"), 1))
    errors.extend(require_count(CASE_NAME, "weight arg1 address patches", mlir.count("arg_idx = 1 : i32"), 8))
    errors.extend(require_count(CASE_NAME, "output arg2 address patches", mlir.count("arg_idx = 2 : i32"), 1))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 2))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, BRIDGE_PACKET_IN_BDS, BRIDGE_PACKET_OUT_BDS))
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids[:1]))
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact receive row {row}", bd_ids[:1]))
    errors.extend(require_unique_bd_ids(CASE_NAME, COMPACT_OUT_BDS[:1]))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    compact_bds = tuple(row_bds[0] for row_bds in COLUMN_RECEIVE_BDS) + COMPACT_OUT_BDS[:1]
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qkv_emit_qkv_body_records",
                "qkv_emit_first_q_body_record",
                "currentkv_postprocess_body_payload",
            ),
        )
    )
    return errors


if __name__ == "__main__":
    print(generate_mlir())
