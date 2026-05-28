"""Generate MLIR-AIE for main16 Q/K/V -> Shape-A/B -> O compact -> c1r2."""

from __future__ import annotations

from pathlib import Path

from contract import (
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
    SHAPE_CARRIER_DWORDS,
)
from mlir_utils import lock_pair, npu_address_patch, npu_push_queue, npu_sync, npu_writebd
from shape_generate import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    _hub as shape_hub,
    _kv_memtile as shape_kv_memtile,
    _shape_a as shape_a,
    _shape_a_symbol,
    _shape_b as shape_b,
    _shape_b_symbol,
)

from cases.qkv_shape_o_c1r2_reference import (
    CASE_NAME,
    COLUMN_COMPACT_DWORDS,
    COLUMN_PACKET_BASE,
    K_GLOBAL_PACKET_ID,
    KV_SIDE_DWORDS,
    MAIN_CHUNK_DWORDS,
    MAIN_PACKET_BASE,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    OUTPUT_DWORDS,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    SUMMARY_DWORDS,
    V_GLOBAL_PACKET_ID,
    WINDOW_DWORDS,
    column_packet,
    main_packet,
)

COLUMN_RECEIVE_BDS = (
    (0, 1, 2, 3),
    (24, 25, 26, 27),
    (4, 5, 6, 7),
    (28, 29, 30, 31),
)
BRIDGE_RECEIVE_BDS = (
    (0, 1, 2, 3),
    (24, 25, 26, 27),
    (4, 5, 12, 13),
    (30, 31, 32, 33),
)
COMPACT_OUT_BDS = (34, 35, 36, 37)
BRIDGE_PACKET_IN_BDS = (6, 7)
BRIDGE_PACKET_OUT_BDS = (28, 29)
OUTPUT_DRAIN_BD = 1
STAGES = ("q", "k", "v", "o")
KV_ALL_DWORDS = KV_SIDE_DWORDS * 2


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
            f'    %{tile}_row{row}_empty = aie.lock(%{tile}, {4 + row}) '
            f'{{init = {len(STAGES)} : i32, sym_name = "{tile}_row{row}_empty"}}\n'
        )
    lines.append(
        f'    %{tile}_drain_token = aie.lock(%{tile}, 8) '
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
    lines.append(lock_pair("bridge", "packet", 4, init_empty=2))
    for group in range(len(MAIN_COLUMNS)):
        lines.append(
            f'    %bridge_g{group}_empty = aie.lock(%bridge, {6 + group}) '
            f'{{init = {len(STAGES)} : i32, sym_name = "bridge_g{group}_empty"}}\n'
        )
    lines.append('    %bridge_drain_token = aie.lock(%bridge, 10) {init = 0 : i32, sym_name = "bridge_drain_token"}\n')
    return "".join(lines)


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    chunks = Q_DWORDS // MAIN_CHUNK_DWORDS
    packet = main_packet(group, row)
    record_bds = (2, 3, 4, 5)
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
      func.call @qkv_main_init_summary(%{tile}_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 3)
      func.call @qkv_emit_qkv_records(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 3)

      scf.for %chunk = %c0 to %chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @qkv_main_accum_chunk(%{tile}_chunk_pong, %{tile}_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @qkv_main_accum_chunk(%{tile}_chunk_ping, %{tile}_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      func.call @qkv_main_emit_o_record(%{tile}_records, %{tile}_summary, %group_i32, %row_i32)
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
    packet_ids = (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID, O_GLOBAL_PACKET_ID)
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
    %post_k_compact = aie.buffer(%post) {{sym_name = "post_k_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %post_v_compact = aie.buffer(%post) {{sym_name = "post_v_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %post_q_payload = aie.buffer(%post) {{sym_name = "post_q_payload"}} : memref<{Q_DWORDS}xi32>
    %post_kv_payload = aie.buffer(%post) {{sym_name = "post_kv_payload"}} : memref<{KV_ALL_DWORDS}xi32>
{lock_pair("post", "q_compact", 0)}
{lock_pair("post", "k_compact", 2)}
{lock_pair("post", "v_compact", 4)}
{lock_pair("post", "q_payload", 6)}
{lock_pair("post", "kv_payload", 8)}

    %post_core = aie.core(%post) {{
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      %kv_side_i32 = arith.constant {KV_SIDE_DWORDS} : i32
      %window_i32 = arith.constant {WINDOW_DWORDS} : i32
      aie.use_lock(%post_q_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_k_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_v_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_kv_payload_empty, AcquireGreaterEqual, 1)
      func.call @qkv_postprocess_payload(%post_q_compact, %post_k_compact, %post_v_compact, %post_q_payload, %post_kv_payload, %q_dwords_i32, %kv_side_i32, %window_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{KV_ALL_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%post_q_compact_empty, Release, 1)
      aie.use_lock(%post_k_compact_empty, Release, 1)
      aie.use_lock(%post_v_compact_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.use_lock(%post_kv_payload_full, Release, 1)
      aie.end
    }}

    %post_mem = aie.mem(%post) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^q_in, ^q_out_start)
    ^q_in:
      aie.use_lock(%post_q_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%post_q_compact_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%post_k_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_k_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%post_k_compact_full, Release, 1)
      aie.next_bd ^v_in
    ^v_in:
      aie.use_lock(%post_v_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_v_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%post_v_compact_full, Release, 1)
      aie.next_bd ^v_in

    ^q_out_start:
      %q_dma = aie.dma_start(MM2S, 0, ^q_out, ^kv_left_start)
    ^q_out:
      aie.use_lock(%post_q_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%post_q_payload_empty, Release, 1)
      aie.next_bd ^q_out

    ^kv_left_start:
      %kv_left_dma = aie.dma_start(MM2S, 1, ^kv_left_out, ^end)
    ^kv_left_out:
      aie.use_lock(%post_kv_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_kv_payload : memref<{KV_ALL_DWORDS}xi32>, 0, {KV_ALL_DWORDS}) {{bd_id = 4 : i32}}
      aie.use_lock(%post_kv_payload_empty, Release, 1)
      aie.next_bd ^kv_left_out
    ^end:
      aie.end
    }}
"""


def _kv_split() -> str:
    return f"""
    %split_kv_payload = aie.buffer(%kv_split) {{sym_name = "split_kv_payload"}} : memref<{KV_ALL_DWORDS}xi32>
    %split_kv_left = aie.buffer(%kv_split) {{sym_name = "split_kv_left"}} : memref<{KV_SIDE_DWORDS}xi32>
    %split_kv_right = aie.buffer(%kv_split) {{sym_name = "split_kv_right"}} : memref<{KV_SIDE_DWORDS}xi32>
{lock_pair("kv_split", "payload", 0)}
{lock_pair("kv_split", "left", 2)}
{lock_pair("kv_split", "right", 4)}

    %kv_split_core = aie.core(%kv_split) {{
      %kv_side_i32 = arith.constant {KV_SIDE_DWORDS} : i32
      aie.use_lock(%kv_split_payload_full, AcquireGreaterEqual, 1)
      aie.use_lock(%kv_split_left_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%kv_split_right_empty, AcquireGreaterEqual, 1)
      func.call @qkv_split_kv_payload(%split_kv_payload, %split_kv_left, %split_kv_right, %kv_side_i32)
        : (memref<{KV_ALL_DWORDS}xi32>, memref<{KV_SIDE_DWORDS}xi32>, memref<{KV_SIDE_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%kv_split_payload_empty, Release, 1)
      aie.use_lock(%kv_split_left_full, Release, 1)
      aie.use_lock(%kv_split_right_full, Release, 1)
      aie.end
    }}

    %kv_split_mem = aie.mem(%kv_split) {{
      %payload_dma = aie.dma_start(S2MM, 0, ^payload_in, ^left_start)
    ^payload_in:
      aie.use_lock(%kv_split_payload_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%split_kv_payload : memref<{KV_ALL_DWORDS}xi32>, 0, {KV_ALL_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%kv_split_payload_full, Release, 1)
      aie.next_bd ^payload_in

    ^left_start:
      %left_dma = aie.dma_start(MM2S, 0, ^left_out, ^right_start)
    ^left_out:
      aie.use_lock(%kv_split_left_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%split_kv_left : memref<{KV_SIDE_DWORDS}xi32>, 0, {KV_SIDE_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%kv_split_left_empty, Release, 1)
      aie.next_bd ^left_out

    ^right_start:
      %right_dma = aie.dma_start(MM2S, 1, ^right_out, ^end)
    ^right_out:
      aie.use_lock(%kv_split_right_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%split_kv_right : memref<{KV_SIDE_DWORDS}xi32>, 0, {KV_SIDE_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%kv_split_right_empty, Release, 1)
      aie.next_bd ^right_out
    ^end:
      aie.end
    }}
"""


def _full_vector() -> str:
    return f"""
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %full_summary = aie.buffer(%full) {{sym_name = "full_summary"}} : memref<{OUTPUT_DWORDS}xi32>
{lock_pair("full", "compact", 0)}
{lock_pair("full", "summary", 2)}

    %full_core = aie.core(%full) {{
      %dwords_i32 = arith.constant {COMPACT_PACKET_DWORDS} : i32
      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_summary_empty, AcquireGreaterEqual, 1)
      func.call @qkv_c1r2_summarize_compact(%full_compact, %full_summary, %dwords_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) -> ()
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
      aie.dma_bd(%full_summary : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%full_summary_empty, Release, 1)
      aie.next_bd ^summary_out
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        (
            f"    aie.runtime_sequence(%output: memref<{OUTPUT_DWORDS}xi32>) {{",
            npu_writebd(1, 13, OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 0, 0),
            npu_push_queue(1, "S2MM", 1, 13),
            npu_sync(1, 1),
            "    }",
        )
    )


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
        "    %kv_split = aie.tile(1, 4)",
        "    %kv_left = aie.tile(0, 1)",
        "    %hub = aie.tile(6, 1)",
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

    flows = []
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.packet_flow({main_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%{_main_symbol(group, row)}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : {row}>")
            flows.append("    }")
            flows.append(f"    aie.flow(%bridge, DMA : 1, %{_main_symbol(group, row)}, DMA : 0)")
        flows.append(f"    aie.packet_flow({column_packet(group)}) {{")
        flows.append(f"      aie.packet_source<%mt{group}, DMA : 5>")
        flows.append(f"      aie.packet_dest<%bridge, DMA : {group}>")
        flows.append("    }")
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(f"    aie.packet_flow({packet}) {{")
        flows.append("      aie.packet_source<%bridge, DMA : 5>")
        flows.append("      aie.packet_dest<%post, DMA : 0>")
        flows.append("    }")
    flows.extend(
        (
            "    aie.flow(%post, DMA : 0, %hub, DMA : 0)",
            "    aie.flow(%post, DMA : 1, %kv_split, DMA : 0)",
            "    aie.flow(%kv_split, DMA : 0, %kv_left, DMA : 0)",
            "    aie.flow(%kv_split, DMA : 1, %kv_right, DMA : 0)",
        )
    )
    for window in range(4):
        kv_tile = "kv_left" if window < 2 else "kv_right"
        kv_k_channel = 0 if window in (0, 2) else 2
        kv_v_channel = 1 if window in (0, 2) else 3
        flows.append(f"    aie.flow(%hub, DMA : {window}, %{_shape_a_symbol(window)}, DMA : 0)")
        flows.append(f"    aie.flow(%{kv_tile}, DMA : {kv_k_channel}, %{_shape_a_symbol(window)}, DMA : 1)")
        flows.append(f"    aie.flow(%{kv_tile}, DMA : {kv_v_channel}, %{_shape_b_symbol(window)}, DMA : 0)")
        flows.append(f"    aie.flow(%{_shape_a_symbol(window)}, DMA : 0, %{_shape_b_symbol(window)}, DMA : 1)")
        flows.append(f"    aie.flow(%{_shape_b_symbol(window)}, DMA : 0, %hub, DMA : {window + 1})")
    flows.extend(
        (
            f"    aie.packet_flow({PACKET_ID_ATTENTION}) {{",
            "      aie.packet_source<%hub, DMA : 5>",
            "      aie.packet_dest<%bridge, DMA : 4>",
            "    }",
            f"    aie.packet_flow({O_GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%full, DMA : 0>",
            "    }",
            "    aie.flow(%full, DMA : 1, %shim_out, DMA : 1)",
        )
    )

    blocks = [
        _bridge(),
        _postprocess(),
        _kv_split(),
        shape_hub(),
        shape_kv_memtile(0),
        shape_kv_memtile(1),
        _full_vector(),
    ]
    for window in range(4):
        blocks.append(shape_a(window))
        blocks.append(shape_b(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
    // case marker {CASE_NAME}
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @qkv_emit_qkv_records(memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_postprocess_payload(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{KV_ALL_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_split_kv_payload(memref<{KV_ALL_DWORDS}xi32>, memref<{KV_SIDE_DWORDS}xi32>, memref<{KV_SIDE_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_main_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_main_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_main_emit_o_record(memref<{MAIN_RECORD_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @qkv_c1r2_summarize_compact(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_make_carrier(memref<{WINDOW_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_make_return(memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "aie.tile(1, 3)",
        "aie.tile(1, 4)",
        "aie.tile(1, 2)",
        "aie.tile(6, 1)",
        "aie.tile(0, 2)",
        "aie.tile(7, 5)",
        f"aie.packet_flow({Q_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({K_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({V_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({PACKET_ID_ATTENTION})",
        f"aie.packet_flow({O_GLOBAL_PACKET_ID})",
        "qkv_postprocess_payload",
        "qkv_split_kv_payload",
        "qkv_main_emit_o_record",
        "qkv_c1r2_summarize_compact",
        "shape_make_carrier",
        "shape_make_return",
        f"memref<{COMPACT_PACKET_DWORDS}xi32>",
        f"memref<{Q_DWORDS}xi32>",
        f"memref<{KV_SIDE_DWORDS}xi32>",
        f"memref<{KV_ALL_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
    )
    errors = [f"missing qkv marker: {marker}" for marker in required if marker not in mlir]
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 5
    if mlir.count("aie.packet_flow(") != expected_packets:
        errors.append("packet flow count mismatch")
    if mlir.count("aie.flow(%bridge, DMA : 1") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN:
        errors.append("main activation bridge flow count mismatch")
    if mlir.count("qkv_emit_qkv_records") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN + 1:
        errors.append("qkv producer call count mismatch")
    if mlir.count("shape_make_carrier") != 5 or mlir.count("shape_make_return") != 5:
        errors.append("shape function declaration/call count mismatch")
    if MAIN_PACKET_BASE != 16 or COLUMN_PACKET_BASE != 4:
        errors.append("packet base mismatch")
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    if KV_OUT_BDS != (2, 24, 4, 26):
        errors.append("kv BD contract mismatch")
    if BRIDGE_PACKET_IN_BDS != (6, 7) or BRIDGE_PACKET_OUT_BDS != (28, 29):
        errors.append("bridge packet BD contract mismatch")
    return errors
