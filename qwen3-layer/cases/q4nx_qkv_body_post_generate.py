"""Generate MLIR-AIE for Q4NX Q/K/V body handoff into c1r3 postprocess."""

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
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    Q_GLOBAL_PACKET_ID,
    V_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _bd_dimensions,
    _bridge,
    _compact_phase,
    _main_record_transfer,
    _main_symbol,
    _next_phase,
    _phase_trace_marker,
    column_packet,
    main_packet,
    q4nx_weight_column_memtile,
)
from projection_schedule import (
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    V_CHUNKS_PER_RECORD,
    V_WEIGHT_CHUNK_BASE,
)
from cases.q4nx_qkv_body_post_reference import (
    CASE_NAME,
    COLUMN_WEIGHT_BF16,
    CURRENT_DWORDS,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    PATCH_WEIGHT_BF16,
    Q_DWORDS,
    TOTAL_WEIGHT_I32,
)

QKV_PHASE_TRACE = (
    _compact_phase("q", "q", BODY_RECORD_SLOTS[0], Q_BODY_RECORDS),
    _compact_phase("k", "k", BODY_RECORD_SLOTS[1], KV_BODY_RECORDS),
    _compact_phase("v", "v", BODY_RECORD_SLOTS[2], KV_BODY_RECORDS),
)
Q_MAIN_RECORD_DWORDS = Q_BODY_RECORDS * RECORD_DWORDS
KV_MAIN_RECORD_DWORDS = KV_BODY_RECORDS * RECORD_DWORDS


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
    total_replays = Q_BODY_RECORDS + KV_BODY_RECORDS + KV_BODY_RECORDS
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}

    %full_mem = aie.mem(%full) {{
      %hidden_dma = aie.dma_start(S2MM, 0, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_hidden_full, Release, {total_replays})
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


def _postprocess_drain() -> str:
    return f"""
    %post_q_body = aie.buffer(%post) {{sym_name = "post_q_body"}} : memref<{Q_DWORDS}xi32>
    %post_k_body = aie.buffer(%post) {{sym_name = "post_k_body"}} : memref<{CURRENT_DWORDS}xi32>
    %post_v_body = aie.buffer(%post) {{sym_name = "post_v_body"}} : memref<{CURRENT_DWORDS}xi32>
    %post_q_payload = aie.buffer(%post) {{sym_name = "post_q_payload"}} : memref<{Q_DWORDS}xi32>
    %post_current_k = aie.buffer(%post) {{sym_name = "post_current_k"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_v = aie.buffer(%post) {{sym_name = "post_current_v"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_token = aie.buffer(%post) {{sym_name = "post_current_token"}} : memref<1xi32>
    %post_q_body_empty = aie.lock(%post, 0) {{init = 1 : i32, sym_name = "post_q_body_empty"}}
    %post_q_body_full = aie.lock(%post, 1) {{init = 0 : i32, sym_name = "post_q_body_full"}}
    %post_k_body_empty = aie.lock(%post, 2) {{init = 1 : i32, sym_name = "post_k_body_empty"}}
    %post_k_body_full = aie.lock(%post, 3) {{init = 0 : i32, sym_name = "post_k_body_full"}}
    %post_v_body_empty = aie.lock(%post, 4) {{init = 1 : i32, sym_name = "post_v_body_empty"}}
    %post_v_body_full = aie.lock(%post, 5) {{init = 0 : i32, sym_name = "post_v_body_full"}}
    %post_q_payload_empty = aie.lock(%post, 6) {{init = 1 : i32, sym_name = "post_q_payload_empty"}}
    %post_q_payload_full = aie.lock(%post, 7) {{init = 0 : i32, sym_name = "post_q_payload_full"}}
    %post_current_k_empty = aie.lock(%post, 8) {{init = 1 : i32, sym_name = "post_current_k_empty"}}
    %post_current_k_full = aie.lock(%post, 9) {{init = 0 : i32, sym_name = "post_current_k_full"}}
    %post_current_v_empty = aie.lock(%post, 10) {{init = 1 : i32, sym_name = "post_current_v_empty"}}
    %post_current_v_full = aie.lock(%post, 11) {{init = 0 : i32, sym_name = "post_current_v_full"}}

    %post_core = aie.core(%post) {{
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      %current_dwords_i32 = arith.constant {CURRENT_DWORDS} : i32
      aie.use_lock(%post_q_body_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_k_body_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_v_body_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_v_empty, AcquireGreaterEqual, 1)
      func.call @currentkv_postprocess_body_payload(%post_q_body, %post_k_body, %post_v_body, %post_q_payload, %post_current_k, %post_current_v, %post_current_token, %q_dwords_i32, %current_dwords_i32)
        : (memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) -> ()
      aie.use_lock(%post_q_body_empty, Release, 1)
      aie.use_lock(%post_k_body_empty, Release, 1)
      aie.use_lock(%post_v_body_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.use_lock(%post_current_k_full, Release, 1)
      aie.use_lock(%post_current_v_full, Release, 1)
      aie.end
    }}

    %post_mem = aie.mem(%post) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^q_in, ^out_start)
    ^q_in:
      aie.use_lock(%post_q_body_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_body : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%post_q_body_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%post_k_body_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_k_body : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%post_k_body_full, Release, 1)
      aie.next_bd ^v_in
    ^v_in:
      aie.use_lock(%post_v_body_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_v_body : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%post_v_body_full, Release, 1)
      aie.next_bd ^v_in

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 0, ^q_out, ^end)
    ^q_out:
      aie.use_lock(%post_q_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%post_q_payload_empty, Release, 1)
      aie.next_bd ^k_out
    ^k_out:
      aie.use_lock(%post_current_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_k : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%post_current_k_empty, Release, 1)
      aie.next_bd ^v_out
    ^v_out:
      aie.use_lock(%post_current_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_v : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 5 : i32}}
      aie.use_lock(%post_current_v_empty, Release, 1)
      aie.next_bd ^v_out
    ^end:
      aie.end
    }}
"""


def _record_buffer_ref(tile: str, buffer_name: str) -> str:
    if buffer_name == "q_records":
        return f"%{tile}_q_records : memref<{Q_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "k_records":
        return f"%{tile}_k_records : memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "v_records":
        return f"%{tile}_v_records : memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    raise RuntimeError(f"bad body record buffer: {buffer_name}")


def _phase_kernel(
    tile: str,
    phase_label: str,
    emit_name: str,
    records: int,
    chunks_per_record: int,
    weight_base: int,
    buffer_name: str,
    buffer_type: str,
) -> str:
    return f"""
      scf.for %{phase_label}_block = %c0 to %c{records} step %c1 {{
        %{phase_label}_block_i32 = arith.index_cast %{phase_label}_block : index to i32
        func.call @clear_summary(%{tile}_q4nx_output, %m_i32)
          : (memref<32xbf16>, i32) -> ()
        scf.for %{phase_label}_chunk = %c0 to %c{chunks_per_record} step %c1 {{
          %{phase_label}_base = arith.muli %{phase_label}_block, %c{chunks_per_record} : index
          %{phase_label}_local_chunk = arith.addi %{phase_label}_base, %{phase_label}_chunk : index
          %{phase_label}_local_chunk_i32 = arith.index_cast %{phase_label}_local_chunk : index to i32
          %{phase_label}_weight_chunk = arith.addi %{phase_label}_local_chunk_i32, %c{weight_base}_i32 : i32
          %{phase_label}_rem = arith.remsi %{phase_label}_weight_chunk, %c2_i32 : i32
          %{phase_label}_is_pong = arith.cmpi eq, %{phase_label}_rem, %c1_i32 : i32
          aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
          aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
          scf.if %{phase_label}_is_pong {{
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
        func.call @{emit_name}({buffer_name}, %{tile}_q4nx_output, %group_i32, %row_i32, %{phase_label}_block_i32, %m_i32)
          : ({buffer_type}, memref<32xbf16>, i32, i32, i32, i32) -> ()
      }}"""


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    packet = main_packet(group, row)
    record_blocks = []
    for stage_idx, phase in enumerate(QKV_PHASE_TRACE):
        buffer_name, offset, length, dimensions = _main_record_transfer(phase, row)
        buffer_ref = _record_buffer_ref(tile, buffer_name)
        next_label = f"^{_next_phase(QKV_PHASE_TRACE, stage_idx).label}_out"
        record_blocks.append(f"""    ^{phase.label}_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer_ref}, {offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = {2 + stage_idx} : i32, next_bd_id = {2 + ((stage_idx + 1) % len(QKV_PHASE_TRACE))} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd {next_label}""")

    q_name = f"%{tile}_q_records"
    q_type = f"memref<{Q_MAIN_RECORD_DWORDS}xi32>"
    k_name = f"%{tile}_k_records"
    k_type = f"memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    v_name = f"%{tile}_v_records"
    v_type = f"memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    return f"""
    %{tile}_q_records = aie.buffer(%{tile}) {{sym_name = "{tile}_q_records"}} : memref<{Q_MAIN_RECORD_DWORDS}xi32>
    %{tile}_k_records = aie.buffer(%{tile}) {{sym_name = "{tile}_k_records"}} : memref<{KV_MAIN_RECORD_DWORDS}xi32>
    %{tile}_v_records = aie.buffer(%{tile}) {{sym_name = "{tile}_v_records"}} : memref<{KV_MAIN_RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_q4nx_output = aie.buffer(%{tile}) {{sym_name = "{tile}_q4nx_output"}} : memref<32xbf16>
    %{tile}_records_empty = aie.lock(%{tile}, 0) {{init = {len(QKV_PHASE_TRACE)} : i32, sym_name = "{tile}_records_empty"}}
    %{tile}_records_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_records_full"}}
    %{tile}_chunk_empty = aie.lock(%{tile}, 2) {{init = 2 : i32, sym_name = "{tile}_chunk_empty"}}
    %{tile}_chunk_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_chunk_full"}}
    %{tile}_wt_empty = aie.lock(%{tile}, 4) {{init = 2 : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, 5) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c{Q_WEIGHT_CHUNK_BASE}_i32 = arith.constant {Q_WEIGHT_CHUNK_BASE} : i32
      %c{K_WEIGHT_CHUNK_BASE}_i32 = arith.constant {K_WEIGHT_CHUNK_BASE} : i32
      %c{V_WEIGHT_CHUNK_BASE}_i32 = arith.constant {V_WEIGHT_CHUNK_BASE} : i32
      %m_i32 = arith.constant 32 : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, {len(QKV_PHASE_TRACE)})
{_phase_kernel(tile, "q", "q4nx_emit_q_body_record", Q_BODY_RECORDS, Q_CHUNKS_PER_RECORD, Q_WEIGHT_CHUNK_BASE, q_name, q_type)}
{_phase_kernel(tile, "k", "q4nx_emit_k_body_record", KV_BODY_RECORDS, K_CHUNKS_PER_RECORD, K_WEIGHT_CHUNK_BASE, k_name, k_type)}
{_phase_kernel(tile, "v", "q4nx_emit_v_body_record", KV_BODY_RECORDS, V_CHUNKS_PER_RECORD, V_WEIGHT_CHUNK_BASE, v_name, v_type)}
      aie.use_lock(%{tile}_records_full, Release, {len(QKV_PHASE_TRACE)})
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
{chr(10).join(record_blocks)}
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
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(packet_flow(packet, "bridge", 5, "post", 0))
    flows.append(flow("post", 0, "shim", 1))

    blocks = [_full_hidden_replay(), _bridge(QKV_PHASE_TRACE), _postprocess_drain()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, QKV_PHASE_TRACE))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @currentkv_postprocess_body_payload(memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @clear_summary(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_chunk_accum_slice_i32(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_flush_output(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_q_body_record(memref<{Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_k_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_v_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(QKV_PHASE_TRACE)}",
        "currentkv_postprocess_body_payload",
        "q4nx_emit_q_body_record",
        "q4nx_emit_k_body_record",
        "q4nx_emit_v_body_record",
        "main_projection_q4nx.o",
        "postprocess_qkv.o",
        f"aie.dma_bd(%post_q_body : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS})",
        f"aie.dma_bd(%post_k_body : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS})",
        f"aie.dma_bd(%post_current_v : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS})",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        "aie.dma_start(S2MM, 4, ^packet_in_ping, ^packet_out_start)",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping, ^wt_row0_start)",
        "aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)",
        "arg_idx = 0 : i32",
        "arg_idx = 1 : i32",
        "arg_idx = 2 : i32",
    )
    errors = [f"missing q4nx qkv-body-post marker: {marker}" for marker in required if marker not in mlir]
    if "qwen3_layer.o" in mlir:
        errors.append("q4nx-qkv-body-post must not link the old mixed qwen3_layer object")
    if "qwen3_bridge.o" in mlir:
        errors.append("q4nx-qkv-body-post must not link the old mixed qwen3_bridge object")
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 4
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "main activation bridge flows", mlir.count("aie.flow(%bridge, DMA : 1"), len(MAIN_COLUMNS) * ROWS_PER_COLUMN))
    errors.extend(require_count(CASE_NAME, "currentkv body postprocess", mlir.count("currentkv_postprocess_body_payload"), 2))
    errors.extend(require_count(CASE_NAME, "q4nx q emit calls", mlir.count("func.call @q4nx_emit_q_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx k emit calls", mlir.count("func.call @q4nx_emit_k_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx v emit calls", mlir.count("func.call @q4nx_emit_v_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx chunk call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32"), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 6))
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
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids[:3]))
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact receive row {row}", bd_ids[:3]))
    errors.extend(require_unique_bd_ids(CASE_NAME, COMPACT_OUT_BDS[:3]))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    compact_bds = tuple(bd for row_bds in COLUMN_RECEIVE_BDS for bd in row_bds[:3]) + COMPACT_OUT_BDS[:3]
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qkv_emit_qkv_body_records",
                "qkv_emit_first_q_body_record",
                "qkv_postprocess_payload",
            ),
        )
    )
    return errors


if __name__ == "__main__":
    print(generate_mlir())
