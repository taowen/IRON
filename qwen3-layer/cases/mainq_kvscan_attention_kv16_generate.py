"""Generate MLIR-AIE for main16-produced Q into streaming KV-scan attention."""

from __future__ import annotations

from pathlib import Path

from contract import (
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    SHAPE_CARRIER_DWORDS,
)
from mlir_utils import (
    flow,
    lock_pair,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_absent_markers,
    require_count,
    require_kv16_attention_shapes,
    require_max_address_patch_arg,
    require_no_compute_kv_materialization,
)
from shape_generate import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    _hub,
    _shape_a_symbol,
    _shape_b_symbol,
)
from shape_reference import MAIN_CHUNK_DWORDS, PACKET_ID, SUMMARY_DWORDS
from cases.attention_kv16_generate import _shape_a, _shape_b
from cases.kvscan_attention_kv16_generate import _kv_scan_memtile, _push_kv_scan_side
from cases.mainq_kvscan_attention_kv16_reference import (
    CASE_NAME,
    COLUMN_COMPACT_DWORDS,
    COLUMN_PACKET_BASE,
    HOST_OUTPUT_DWORDS,
    K_CACHE_SIDE_DWORDS,
    K_WINDOW_DWORDS,
    KV_CACHE_SIDE_DWORDS,
    KV_SIDE_DWORDS,
    MAIN_PACKET_BASE,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    OUTPUT_DWORDS,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    SCALAR_DWORDS,
    STAGES,
    V_CACHE_SIDE_DWORDS,
    V_WINDOW_DWORDS,
    WEIGHT_DWORDS,
    WINDOW_DWORDS,
    column_packet,
    main_packet,
)

ROWS_PER_COLUMN = len(MAIN_ROWS)
COLUMN_RECEIVE_BDS = (
    (0, 1),
    (24, 25),
    (4, 5),
    (28, 29),
)
BRIDGE_RECEIVE_BDS = (
    (0, 1),
    (24, 25),
    (4, 5),
    (30, 31),
)
COMPACT_OUT_BDS = (34, 35)
BRIDGE_PACKET_IN_BDS = (6, 7)
BRIDGE_PACKET_OUT_BDS = (28, 29)


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def _segment(row: int) -> tuple[int, int]:
    if row == 0:
        return 0, RECORD_DWORDS
    return RECORD_DWORDS + (row - 1) * RECORD_PAYLOAD_DWORDS, RECORD_PAYLOAD_DWORDS


def _source_segment(stage: int, row: int) -> tuple[int, int]:
    base = stage * RECORD_DWORDS
    if row == 0:
        return base, RECORD_DWORDS
    return base + 1, RECORD_PAYLOAD_DWORDS


def _column_lock_defs(tile: str) -> str:
    lines = []
    for stage_idx, stage in enumerate(STAGES):
        lines.append(
            f'    %{tile}_{stage}_full = aie.lock(%{tile}, {stage_idx}) '
            f'{{init = 0 : i32, sym_name = "{tile}_{stage}_full"}}\n'
        )
    for row in range(ROWS_PER_COLUMN):
        lines.append(
            f'    %{tile}_row{row}_empty = aie.lock(%{tile}, {len(STAGES) + row}) '
            f'{{init = {len(STAGES)} : i32, sym_name = "{tile}_row{row}_empty"}}\n'
        )
    lines.append(
        f'    %{tile}_drain_token = aie.lock(%{tile}, {len(STAGES) + ROWS_PER_COLUMN}) '
        f'{{init = 0 : i32, sym_name = "{tile}_drain_token"}}\n'
    )
    return "".join(lines)


def _bridge_lock_defs() -> str:
    lines = []
    for stage_idx, stage in enumerate(STAGES):
        lines.append(
            f'    %bridge_{stage}_full = aie.lock(%bridge, {stage_idx}) '
            f'{{init = 0 : i32, sym_name = "bridge_{stage}_full"}}\n'
        )
    lines.append(lock_pair("bridge", "packet", len(STAGES), init_empty=2))
    group_base = len(STAGES) + 2
    for group in range(len(MAIN_COLUMNS)):
        lines.append(
            f'    %bridge_g{group}_empty = aie.lock(%bridge, {group_base + group}) '
            f'{{init = {len(STAGES)} : i32, sym_name = "bridge_g{group}_empty"}}\n'
        )
    lines.append(
        f'    %bridge_drain_token = aie.lock(%bridge, {group_base + len(MAIN_COLUMNS)}) '
        '{init = 0 : i32, sym_name = "bridge_drain_token"}\n'
    )
    return "".join(lines)


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    chunks = Q_DWORDS // MAIN_CHUNK_DWORDS
    packet = main_packet(group, row)
    record_bds = (2, 3)
    record_blocks = []
    for stage_idx, stage in enumerate(STAGES):
        offset, length = _source_segment(stage_idx, row)
        next_label = f"^{STAGES[(stage_idx + 1) % len(STAGES)]}_out"
        record_blocks.append(f"""    ^{stage}_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {offset}, {length}) {{bd_id = {record_bds[stage_idx]} : i32, next_bd_id = {record_bds[(stage_idx + 1) % len(STAGES)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd {next_label}""")

    return f"""
    %{tile}_records = aie.buffer(%{tile}) {{sym_name = "{tile}_records"}} : memref<{MAIN_RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_summary = aie.buffer(%{tile}) {{sym_name = "{tile}_summary"}} : memref<{SUMMARY_DWORDS}xi32>
{lock_pair(tile, "records", 0, init_empty=len(STAGES))}
{lock_pair(tile, "chunk", 2, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %chunks = arith.constant {chunks} : index
      %dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      func.call @shape_init_summary(%{tile}_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      func.call @mainq_emit_q_record(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 1)

      scf.for %chunk = %c0 to %chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @shape_accum_chunk(%{tile}_chunk_pong, %{tile}_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @shape_accum_chunk(%{tile}_chunk_ping, %{tile}_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      func.call @mainq_emit_o_record(%{tile}_records, %{tile}_summary, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^record_start)
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

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^q_out, ^end)
{chr(10).join(record_blocks)}
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
        dest_offset, length = _segment(row)
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^out_start"
        bds = COLUMN_RECEIVE_BDS[row]
        stage_blocks = []
        for stage_idx, stage in enumerate(STAGES):
            next_stage = STAGES[(stage_idx + 1) % len(STAGES)]
            stage_blocks.append(f"""    ^row{row}_{stage}:
      aie.use_lock(%{tile}_row{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_{stage} : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {bds[stage_idx]} : i32, next_bd_id = {bds[(stage_idx + 1) % len(STAGES)]} : i32}}
      aie.use_lock(%{tile}_{stage}_full, Release, 1)
      aie.next_bd ^row{row}_{next_stage}""")
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_q, {next_start})
{chr(10).join(stage_blocks)}"""
        )

    out_blocks = []
    for stage_idx, stage in enumerate(STAGES):
        next_stage = STAGES[(stage_idx + 1) % len(STAGES)]
        out_blocks.append(f"""    ^{stage}_out:
      aie.use_lock(%{tile}_{stage}_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_{stage} : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(STAGES)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^{next_stage}_out""")

    buffers = "\n".join(
        f'    %{tile}_{stage} = aie.buffer(%{tile}) {{sym_name = "{tile}_{stage}"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>'
        for stage in STAGES
    )
    return f"""
{buffers}
{_column_lock_defs(tile)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^q_out, ^end)
{chr(10).join(out_blocks)}
    ^end:
      aie.end
    }}
"""


def _bridge_receive_starts() -> str:
    starts = []
    for group in range(len(MAIN_COLUMNS)):
        if group == 0:
            length = COLUMN_COMPACT_DWORDS
            dest_offset = 0
        else:
            length = COLUMN_COMPACT_DWORDS - 1
            dest_offset = COLUMN_COMPACT_DWORDS + (group - 1) * (COLUMN_COMPACT_DWORDS - 1)
        start_label = "" if group == 0 else f"    ^g{group}_start:\n"
        next_start = f"^g{group + 1}_start" if group + 1 < len(MAIN_COLUMNS) else "^compact_out_start"
        bds = BRIDGE_RECEIVE_BDS[group]
        stage_blocks = []
        for stage_idx, stage in enumerate(STAGES):
            next_stage = STAGES[(stage_idx + 1) % len(STAGES)]
            stage_blocks.append(f"""    ^g{group}_{stage}:
      aie.use_lock(%bridge_g{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_{stage} : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {bds[stage_idx]} : i32, next_bd_id = {bds[(stage_idx + 1) % len(STAGES)]} : i32}}
      aie.use_lock(%bridge_{stage}_full, Release, 1)
      aie.next_bd ^g{group}_{next_stage}""")
        starts.append(
            f"""{start_label}      %g{group}_dma = aie.dma_start(S2MM, {group}, ^g{group}_q, {next_start})
{chr(10).join(stage_blocks)}"""
        )
    return "\n".join(starts)


def _bridge() -> str:
    buffers = "\n".join(
        f'    %bridge_{stage} = aie.buffer(%bridge) {{sym_name = "bridge_{stage}"}} : memref<{COMPACT_PACKET_DWORDS}xi32>'
        for stage in STAGES
    )
    packet_ids = (Q_GLOBAL_PACKET_ID, O_GLOBAL_PACKET_ID)
    compact_out_blocks = []
    for stage_idx, stage in enumerate(STAGES):
        next_stage = STAGES[(stage_idx + 1) % len(STAGES)]
        compact_out_blocks.append(f"""    ^{stage}_out:
      aie.use_lock(%bridge_{stage}_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_{stage} : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(STAGES)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet_ids[stage_idx]}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^{next_stage}_out""")

    return f"""
{buffers}
    %bridge_packet_ping = aie.buffer(%bridge) {{sym_name = "bridge_packet_ping"}} : memref<{MAIN_CHUNK_DWORDS * 2}xi32>
    %bridge_packet_pong = aie.buffer(%bridge) {{sym_name = "bridge_packet_pong"}} : memref<{MAIN_CHUNK_DWORDS * 2}xi32>
{_bridge_lock_defs()}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{_bridge_receive_starts()}

    ^compact_out_start:
      %compact_out_dma = aie.dma_start(MM2S, 5, ^q_out, ^packet_in_start)
{chr(10).join(compact_out_blocks)}

    ^packet_in_start:
      %packet_in_dma = aie.dma_start(S2MM, 4, ^packet_in_ping, ^packet_out_start)
    ^packet_in_ping:
      aie.use_lock(%bridge_packet_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_ping : memref<{MAIN_CHUNK_DWORDS * 2}xi32>, 0, {MAIN_CHUNK_DWORDS * 2}) {{bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_pong
    ^packet_in_pong:
      aie.use_lock(%bridge_packet_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{MAIN_CHUNK_DWORDS * 2}xi32>, 0, {MAIN_CHUNK_DWORDS * 2}) {{bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_ping

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 1, ^packet_out_ping, ^end)
    ^packet_out_ping:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_ping : memref<{MAIN_CHUNK_DWORDS * 2}xi32>, 0, {MAIN_CHUNK_DWORDS * 2}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_pong
    ^packet_out_pong:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{MAIN_CHUNK_DWORDS * 2}xi32>, 0, {MAIN_CHUNK_DWORDS * 2}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_ping
    ^end:
      aie.end
    }}
"""


def _postprocess() -> str:
    return f"""
    %post_q_compact = aie.buffer(%post) {{sym_name = "post_q_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %post_q_payload = aie.buffer(%post) {{sym_name = "post_q_payload"}} : memref<{Q_DWORDS}xi32>
{lock_pair("post", "q_compact", 0)}
{lock_pair("post", "q_payload", 2)}

    %post_core = aie.core(%post) {{
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      aie.use_lock(%post_q_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      func.call @mainq_postprocess_payload(%post_q_compact, %post_q_payload, %q_dwords_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%post_q_compact_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.end
    }}

    %post_mem = aie.mem(%post) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^q_in, ^q_out_start)
    ^q_in:
      aie.use_lock(%post_q_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%post_q_compact_full, Release, 1)
      aie.next_bd ^q_in

    ^q_out_start:
      %q_dma = aie.dma_start(MM2S, 0, ^q_out, ^end)
    ^q_out:
      aie.use_lock(%post_q_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%post_q_payload_empty, Release, 1)
      aie.next_bd ^q_out
    ^end:
      aie.end
    }}
"""


def _full_vector() -> str:
    return f"""
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %full_summary = aie.buffer(%full) {{sym_name = "full_summary"}} : memref<{HOST_OUTPUT_DWORDS}xi32>
{lock_pair("full", "compact", 0)}
{lock_pair("full", "summary", 2)}

    %full_core = aie.core(%full) {{
      %dwords_i32 = arith.constant {COMPACT_PACKET_DWORDS} : i32
      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_summary_empty, AcquireGreaterEqual, 1)
      func.call @qkv_c1r2_summarize_compact(%full_compact, %full_summary, %dwords_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{HOST_OUTPUT_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%full_compact_empty, Release, 1)
      aie.use_lock(%full_summary_full, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^compact_in, ^summary_out_start)
    ^compact_in:
      aie.use_lock(%full_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_compact_full, Release, 1)
      aie.next_bd ^compact_in

    ^summary_out_start:
      %summary_dma = aie.dma_start(MM2S, 1, ^summary_out, ^end)
    ^summary_out:
      aie.use_lock(%full_summary_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_summary : memref<{HOST_OUTPUT_DWORDS}xi32>, 0, {HOST_OUTPUT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%full_summary_empty, Release, 1)
      aie.next_bd ^summary_out
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%kv_left_arg: memref<{KV_CACHE_SIDE_DWORDS}xi32>, "
        f"%kv_right_arg: memref<{KV_CACHE_SIDE_DWORDS}xi32>, "
        f"%output: memref<{HOST_OUTPUT_DWORDS}xi32>) {{"
    ]
    lines.extend(
        (
            npu_writebd(1, 13, HOST_OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 2, 0),
            npu_push_queue(1, "S2MM", 1, 13),
        )
    )
    lines.extend(_push_kv_scan_side(0, 0))
    lines.extend(_push_kv_scan_side(7, 1))
    lines.extend(
        (
            npu_sync(1, 1),
            npu_sync(0, 0, direction=1),
            npu_sync(7, 0, direction=1),
        )
    )
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_left = aie.tile(0, 0)",
        "    %kv_left = aie.tile(0, 1)",
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
        "    %hub = aie.tile(6, 1)",
        "    %shim_right = aie.tile(7, 0)",
        "    %kv_right = aie.tile(7, 1)",
    ]
    for window, (column, row) in enumerate(SHAPE_A_TILES):
        tile_defs.append(f"    %{_shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{_shape_b_symbol(window)} = aie.tile({column}, {row})")
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(packet_flow(main_packet(group, row), _main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
        flows.append(packet_flow(column_packet(group), f"mt{group}", 5, "bridge", group))
    flows.extend(
        (
            packet_flow(Q_GLOBAL_PACKET_ID, "bridge", 5, "post", 0),
            packet_flow(O_GLOBAL_PACKET_ID, "bridge", 5, "full", 0),
            flow("post", 0, "hub", 0),
            flow("shim_left", 0, "kv_left", 0),
            flow("shim_right", 0, "kv_right", 0),
        )
    )
    for window in range(4):
        kv_tile = "kv_left" if window < 2 else "kv_right"
        kv_k_channel = 0 if window in (0, 2) else 2
        kv_v_channel = 1 if window in (0, 2) else 3
        flows.append(flow("hub", window, _shape_a_symbol(window), 0))
        flows.append(flow(kv_tile, kv_k_channel, _shape_a_symbol(window), 1))
        flows.append(flow(kv_tile, kv_v_channel, _shape_b_symbol(window), 0))
        flows.append(flow(_shape_a_symbol(window), 0, _shape_b_symbol(window), 1))
        flows.append(flow(_shape_b_symbol(window), 0, "hub", window + 1))
    flows.extend(
        (
            packet_flow(PACKET_ID, "hub", 5, "bridge", 4),
            flow("full", 1, "shim_out", 1),
        )
    )

    blocks = [_bridge(), _postprocess(), _hub(), _kv_scan_memtile(0), _kv_scan_memtile(1), _full_vector()]
    for window in range(4):
        blocks.append(_shape_a(window))
        blocks.append(_shape_b(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @mainq_emit_q_record(memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @mainq_postprocess_payload(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @mainq_emit_o_record(memref<{MAIN_RECORD_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @attention_kv16_make_carrier(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @attention_kv16_make_return(memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_c1r2_summarize_compact(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{HOST_OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "mainq_emit_q_record",
        "mainq_postprocess_payload",
        "mainq_emit_o_record",
        "attention_kv16_make_carrier",
        "attention_kv16_make_return",
        f"memref<{COMPACT_PACKET_DWORDS}xi32>",
        f"memref<{Q_DWORDS}xi32>",
        f"memref<{KV_CACHE_SIDE_DWORDS}xi32>",
        f"memref<{KV_SIDE_DWORDS}xi32>",
        f"memref<{K_WINDOW_DWORDS}xi32>",
        f"memref<{V_WINDOW_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"aie.packet_flow({Q_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({PACKET_ID})",
        f"aie.packet_flow({O_GLOBAL_PACKET_ID})",
        "slot0_empty",
        "slot3_full",
        "issue_token = false",
    )
    errors = [f"missing mainq kvscan marker: {marker}" for marker in required if marker not in mlir]
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 3
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(
        require_count(
            CASE_NAME,
            "main activation bridge flow",
            mlir.count("aie.flow(%bridge, DMA : 1"),
            len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        )
    )
    errors.extend(require_count(CASE_NAME, "mainq Q producer", mlir.count("mainq_emit_q_record"), 17))
    errors.extend(require_count(CASE_NAME, "mainq O producer", mlir.count("mainq_emit_o_record"), 17))
    errors.extend(require_count(CASE_NAME, "mainq postprocess", mlir.count("mainq_postprocess_payload"), 2))
    errors.extend(require_count(CASE_NAME, "attention_kv16_make_carrier", mlir.count("attention_kv16_make_carrier"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_make_return", mlir.count("attention_kv16_make_return"), 5))
    errors.extend(
        require_kv16_attention_shapes(
            CASE_NAME,
            K_WINDOW_DWORDS,
            V_WINDOW_DWORDS,
            KV_SIDE_DWORDS,
            K_CACHE_SIDE_DWORDS,
            V_CACHE_SIDE_DWORDS,
            SHAPE_CARRIER_DWORDS,
            WEIGHT_DWORDS,
            SCALAR_DWORDS,
            OUTPUT_DWORDS,
        )
    )
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 2))
    errors.extend(
        require_no_compute_kv_materialization(
            CASE_NAME,
            mlir,
            KV_SIDE_DWORDS,
            KV_SIDE_DWORDS * 2,
        )
    )
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "shim_q",
                "qkv_postprocess_payload",
                "qkv_split_kv_payload",
                "post_kv_payload",
                "kv_split",
            ),
        )
    )
    if MAIN_PACKET_BASE != 16 or COLUMN_PACKET_BASE != 4:
        errors.append("packet base mismatch")
    if KV_OUT_BDS != (2, 24, 4, 26):
        errors.append("KV memtile output BD contract mismatch")
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    if BRIDGE_PACKET_IN_BDS != (6, 7) or BRIDGE_PACKET_OUT_BDS != (28, 29):
        errors.append("bridge packet BD contract mismatch")
    return errors
