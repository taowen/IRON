"""Generate MLIR-AIE for the deterministic full qwen3-layer bridge contract."""

from __future__ import annotations

from pathlib import Path

from c1r2_generate import _swiglu as swiglu_station
from contract import (
    C1R2_PACKET_DWORDS,
    C1R2_UPGATE_REPLAYS,
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
    SHAPE_CARRIER_DWORDS,
)
from mlir_utils import (
    lock_pair,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    require_c1r1_s2mm3_high_bds,
    require_compact_payload_slice,
    require_disjoint_bd_ids,
    require_unique_bd_ids,
)
from shape_generate import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    _kv_memtile as shape_kv_memtile,
    _shape_a as shape_a,
    _shape_a_symbol,
    _shape_b as shape_b,
    _shape_b_symbol,
)
from cases.qkv_shape_o_c1r2_generate import _kv_split as qkv_kv_split
from cases.qkv_shape_o_c1r2_generate import _postprocess as qkv_postprocess
from cases.full_layer_contract_reference import (
    CASE_NAME,
    COLUMN_COMPACT_DWORDS,
    DOWN_ACT_PACKET_ID,
    DOWN_CHUNKS,
    DOWN_GLOBAL_PACKET_ID,
    FFN_GLOBAL_PACKET_ID,
    FULL_REPLAY_PACKET_ID,
    K_GLOBAL_PACKET_ID,
    KV_SIDE_DWORDS,
    MAIN_CHUNK_DWORDS,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    OUTPUT_DWORDS,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    SUMMARY_DWORDS,
    STAGES,
    TOTAL_MAIN_CHUNKS,
    V_GLOBAL_PACKET_ID,
    WINDOW_DWORDS,
    column_packet,
    main_packet,
)

C1R2_REPLAY_PAYLOAD_DWORDS = C1R2_PACKET_DWORDS - 1
KV_ALL_DWORDS = KV_SIDE_DWORDS * 2
OUTPUT_DRAIN_BD = 2
HUB_FFN_IN_BD = 36
HUB_ATTENTION_OUT_BD = 34
HUB_FFN_OUT_BD = 35

COLUMN_RECEIVE_BDS = (
    (0, 1, 2, 3, 4, 5, 6),
    (24, 25, 26, 27, 28, 29, 30),
    (7, 8, 9, 10, 11, 12, 13),
    (31, 32, 33, 41, 42, 43, 44),
)
BRIDGE_RECEIVE_BDS = (
    (0, 1, 2, 3, 4, 5, 8),
    (24, 25, 26, 27, 30, 31, 32),
    (9, 10, 11, 12, 13, 14, 15),
    (41, 42, 43, 44, 45, 46, 47),
)
COMPACT_OUT_BDS = (34, 35, 36, 37, 38, 39, 40)
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
    group_lock_base = len(STAGES) + 2
    for group in range(len(MAIN_COLUMNS)):
        lines.append(
            f'    %bridge_g{group}_empty = aie.lock(%bridge, {group_lock_base + group}) '
            f'{{init = {len(STAGES)} : i32, sym_name = "bridge_g{group}_empty"}}\n'
        )
    lines.append(
        f'    %bridge_drain_token = aie.lock(%bridge, {group_lock_base + len(MAIN_COLUMNS)}) '
        '{init = 0 : i32, sym_name = "bridge_drain_token"}\n'
    )
    return "".join(lines)


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    packet = main_packet(group, row)
    record_bds = (2, 3, 4, 5, 6, 7, 8)
    record_blocks = []
    for stage_idx, stage in enumerate(STAGES):
        offset, length = _source_segment(stage_idx, row)
        next_label = f"^{STAGES[(stage_idx + 1) % len(STAGES)]}_out"
        record_blocks.append(f"""    ^{stage}_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {offset}, {length}) {{bd_id = {record_bds[stage_idx]} : i32, next_bd_id = {record_bds[(stage_idx + 1) % len(STAGES)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd {next_label}""")

    attention_chunks = Q_DWORDS // MAIN_CHUNK_DWORDS
    return f"""
    %{tile}_records = aie.buffer(%{tile}) {{sym_name = "{tile}_records"}} : memref<{MAIN_RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_attention_summary = aie.buffer(%{tile}) {{sym_name = "{tile}_attention_summary"}} : memref<{SUMMARY_DWORDS}xi32>
    %{tile}_upgate_accum = aie.buffer(%{tile}) {{sym_name = "{tile}_upgate_accum"}} : memref<32xi32>
    %{tile}_down_summary = aie.buffer(%{tile}) {{sym_name = "{tile}_down_summary"}} : memref<{SUMMARY_DWORDS}xi32>
{lock_pair(tile, "records", 0, init_empty=len(STAGES))}
{lock_pair(tile, "chunk", 2, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %attention_chunks = arith.constant {attention_chunks} : index
      %upgate_chunks = arith.constant {TOTAL_MAIN_CHUNKS} : index
      %down_chunks = arith.constant {DOWN_CHUNKS} : index
      %dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      func.call @qkv_main_init_summary(%{tile}_attention_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 3)
      func.call @qkv_emit_qkv_records(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 3)

      scf.for %chunk = %c0 to %attention_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @qkv_main_accum_chunk(%{tile}_chunk_pong, %{tile}_attention_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @qkv_main_accum_chunk(%{tile}_chunk_ping, %{tile}_attention_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      func.call @qkv_main_emit_o_record(%{tile}_records, %{tile}_attention_summary, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 1)

      func.call @c1r2_main_init_accum(%{tile}_upgate_accum, %group_i32, %row_i32)
        : (memref<32xi32>, i32, i32) -> ()
      scf.for %chunk = %c0 to %upgate_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @c1r2_main_accum_chunk(%{tile}_chunk_pong, %{tile}_upgate_accum, %chunk_i32, %group_i32, %row_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<32xi32>, i32, i32, i32, i32) -> ()
        }} else {{
          func.call @c1r2_main_accum_chunk(%{tile}_chunk_ping, %{tile}_upgate_accum, %chunk_i32, %group_i32, %row_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<32xi32>, i32, i32, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 2)
      func.call @full_main_emit_upgate_records(%{tile}_records, %{tile}_upgate_accum, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, memref<32xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_records_full, Release, 2)

      func.call @full_main_init_down_summary(%{tile}_down_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
      scf.for %chunk = %c0 to %down_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @full_main_accum_down_chunk(%{tile}_chunk_pong, %{tile}_down_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @full_main_accum_down_chunk(%{tile}_chunk_ping, %{tile}_down_summary, %chunk_i32, %dwords_i32)
            : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      func.call @full_main_emit_down_record(%{tile}_records, %{tile}_down_summary, %group_i32, %row_i32)
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
    packet_ids = (
        Q_GLOBAL_PACKET_ID,
        K_GLOBAL_PACKET_ID,
        V_GLOBAL_PACKET_ID,
        O_GLOBAL_PACKET_ID,
        FFN_GLOBAL_PACKET_ID,
        FFN_GLOBAL_PACKET_ID,
        DOWN_GLOBAL_PACKET_ID,
    )
    compact_out_blocks = []
    for stage_idx, stage in enumerate(STAGES):
        next_stage = STAGES[(stage_idx + 1) % len(STAGES)]
        source_offset = 1 if stage in ("up", "gate") else 0
        source_length = C6R2_HALF_DWORDS if stage in ("up", "gate") else COMPACT_PACKET_DWORDS
        compact_out_blocks.append(f"""    ^{stage}_out:
      aie.use_lock(%bridge_{stage}_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_{stage} : memref<{COMPACT_PACKET_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(STAGES)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet_ids[stage_idx]}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^{next_stage}_out""")

    buffers = "\n".join(
        f'    %bridge_{stage} = aie.buffer(%bridge) {{sym_name = "bridge_{stage}"}} : memref<{COMPACT_PACKET_DWORDS}xi32>'
        for stage in STAGES
    )
    return f"""
{buffers}
    %bridge_packet_ping = aie.buffer(%bridge) {{sym_name = "bridge_packet_ping"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %bridge_packet_pong = aie.buffer(%bridge) {{sym_name = "bridge_packet_pong"}} : memref<{C6R2_HALF_DWORDS}xi32>
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
      aie.dma_bd(%bridge_packet_ping : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_pong
    ^packet_in_pong:
      aie.use_lock(%bridge_packet_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_ping

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 1, ^packet_out_ping, ^end)
    ^packet_out_ping:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_ping : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_pong
    ^packet_out_pong:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_ping
    ^end:
      aie.end
    }}
"""


def _full_vector() -> str:
    return f"""
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{C1R2_PACKET_DWORDS}xi32>
    %full_summary = aie.buffer(%full) {{sym_name = "full_summary"}} : memref<{OUTPUT_DWORDS}xi32>
{lock_pair("full", "compact", 0)}
{lock_pair("full", "replay", 2)}
{lock_pair("full", "summary", 4)}

    %full_core = aie.core(%full) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %replays = arith.constant {C1R2_UPGATE_REPLAYS} : index
      %payload_i32 = arith.constant {C1R2_REPLAY_PAYLOAD_DWORDS} : i32
      %compact_i32 = arith.constant {COMPACT_PACKET_DWORDS} : i32
      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      scf.for %replay = %c0 to %replays step %c1 {{
        %replay_i32 = arith.index_cast %replay : index to i32
        aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
        func.call @c1r2_make_replay(%full_compact, %full_replay, %replay_i32, %payload_i32)
          : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32) -> ()
        aie.use_lock(%full_replay_full, Release, 1)
      }}
      aie.use_lock(%full_compact_empty, Release, 1)

      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_summary_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_summarize_down(%full_compact, %full_summary, %compact_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%full_compact_empty, Release, 1)
      aie.use_lock(%full_summary_full, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^compact_in, ^replay_out_start)
    ^compact_in:
      aie.use_lock(%full_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_compact_full, Release, 1)
      aie.next_bd ^compact_in

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^summary_out_start)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{C1R2_PACKET_DWORDS}xi32>, 1, {C1R2_REPLAY_PAYLOAD_DWORDS}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^replay_out

    ^summary_out_start:
      %summary_dma = aie.dma_start(MM2S, 0, ^summary_out, ^end)
    ^summary_out:
      aie.use_lock(%full_summary_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_summary : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%full_summary_empty, Release, 1)
      aie.next_bd ^summary_out
    ^end:
      aie.end
    }}
"""


def _hub() -> str:
    q_outs = []
    for window, bd_id in enumerate(HUB_Q_OUT_BDS):
        next_start = f"^q{window + 1}_start" if window + 1 < 4 else "^return0_start"
        q_outs.append(f"""    ^q{window}_start:
      %q{window}_dma = aie.dma_start(MM2S, {window}, ^q{window}_out, {next_start})
    ^q{window}_out:
      aie.use_lock(%hub_q_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_q_empty, Release, 1)
      aie.next_bd ^q{window}_out""")

    return_ins = []
    for window, bd_id in enumerate(HUB_RETURN_IN_BDS):
        next_start = f"^return{window + 1}_start" if window + 1 < 4 else "^ffn_in_start"
        return_ins.append(f"""    ^return{window}_start:
      %return{window}_dma = aie.dma_start(S2MM, {window + 1}, ^return{window}_in, {next_start})
    ^return{window}_in:
      aie.use_lock(%hub_return_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_return_full, Release, 1)
      aie.next_bd ^return{window}_in""")

    return f"""
    %hub_q = aie.buffer(%hub) {{sym_name = "hub_q"}} : memref<{Q_DWORDS}xi32>
    %hub_return = aie.buffer(%hub) {{sym_name = "hub_return"}} : memref<{Q_DWORDS}xi32>
    %hub_ffn = aie.buffer(%hub) {{sym_name = "hub_ffn"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %hub_q_empty = aie.lock(%hub, 0) {{init = 4 : i32, sym_name = "hub_q_empty"}}
    %hub_q_full = aie.lock(%hub, 1) {{init = 0 : i32, sym_name = "hub_q_full"}}
    %hub_return_empty = aie.lock(%hub, 2) {{init = 4 : i32, sym_name = "hub_return_empty"}}
    %hub_return_full = aie.lock(%hub, 3) {{init = 0 : i32, sym_name = "hub_return_full"}}
{lock_pair("hub", "ffn", 4)}

    %hub_dma = aie.memtile_dma(%hub) {{
      %q_in_dma = aie.dma_start(S2MM, 0, ^q_in, ^q0_start)
    ^q_in:
      aie.use_lock(%hub_q_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_q_full, Release, 4)
      aie.next_bd ^q_in

{chr(10).join(q_outs)}

{chr(10).join(return_ins)}

    ^ffn_in_start:
      %ffn_in_dma = aie.dma_start(S2MM, 5, ^ffn_in, ^packet_out_start)
    ^ffn_in:
      aie.use_lock(%hub_ffn_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_ffn : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {HUB_FFN_IN_BD} : i32}}
      aie.use_lock(%hub_ffn_full, Release, 1)
      aie.next_bd ^ffn_in

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 5, ^attention_out, ^end)
    ^attention_out:
      aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = {HUB_ATTENTION_OUT_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_ATTENTION}>}}
      aie.use_lock(%hub_return_empty, Release, 4)
      aie.next_bd ^ffn_out
    ^ffn_out:
      aie.use_lock(%hub_ffn_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_ffn : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {HUB_FFN_OUT_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {DOWN_ACT_PACKET_ID}>}}
      aie.use_lock(%hub_ffn_empty, Release, 1)
      aie.next_bd ^ffn_out
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
        "    %swiglu = aie.tile(6, 2)",
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
            f"    aie.packet_flow({FULL_REPLAY_PACKET_ID}) {{",
            "      aie.packet_source<%full, DMA : 1>",
            "      aie.packet_dest<%bridge, DMA : 4>",
            "    }",
            f"    aie.packet_flow({FFN_GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%swiglu, DMA : 0>",
            "    }",
            "    aie.flow(%swiglu, DMA : 1, %hub, DMA : 5)",
            f"    aie.packet_flow({DOWN_ACT_PACKET_ID}) {{",
            "      aie.packet_source<%hub, DMA : 5>",
            "      aie.packet_dest<%bridge, DMA : 4>",
            "    }",
            f"    aie.packet_flow({DOWN_GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%full, DMA : 0>",
            "    }",
            "    aie.flow(%full, DMA : 0, %shim_out, DMA : 1)",
        )
    )

    blocks = [
        _bridge(),
        qkv_postprocess(),
        qkv_kv_split(),
        _hub(),
        shape_kv_memtile(0),
        shape_kv_memtile(1),
        _full_vector(),
        swiglu_station(),
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
    func.func private @c1r2_make_replay(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @c1r2_main_init_accum(memref<32xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @c1r2_main_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<32xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @full_main_emit_upgate_records(memref<{MAIN_RECORD_DWORDS}xi32>, memref<32xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @full_main_init_down_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @full_main_accum_down_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @full_main_emit_down_record(memref<{MAIN_RECORD_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @full_c1r2_summarize_down(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @ffn_swiglu_contract(memref<{C6R2_INPUT_DWORDS}xi32>, memref<{C6R2_HALF_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
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
        "aie.tile(1, 2)",
        "aie.tile(1, 3)",
        "aie.tile(6, 1)",
        "aie.tile(6, 2)",
        f"aie.packet_flow({FULL_REPLAY_PACKET_ID})",
        f"aie.packet_flow({DOWN_ACT_PACKET_ID})",
        f"aie.packet_flow({DOWN_GLOBAL_PACKET_ID})",
        "full_main_emit_upgate_records",
        "full_main_emit_down_record",
        "full_c1r2_summarize_down",
        "ffn_swiglu_contract",
        f"memref<{C1R2_PACKET_DWORDS}xi32>",
        f"memref<{C6R2_INPUT_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
    )
    errors = [f"missing full-layer marker: {marker}" for marker in required if marker not in mlir]
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 9
    if mlir.count("aie.packet_flow(") != expected_packets:
        errors.append("packet flow count mismatch")
    if mlir.count("aie.flow(%bridge, DMA : 1") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN:
        errors.append("main activation bridge flow count mismatch")
    if f"%upgate_chunks = arith.constant {TOTAL_MAIN_CHUNKS} : index" not in mlir:
        errors.append("upgate replay chunk count missing")
    if f"%down_chunks = arith.constant {DOWN_CHUNKS} : index" not in mlir:
        errors.append("down chunk count missing")
    if BRIDGE_PACKET_IN_BDS != (6, 7) or BRIDGE_PACKET_OUT_BDS != (28, 29):
        errors.append("bridge packet BD contract mismatch")
    errors.extend(require_unique_bd_ids("full c1r2 MM2S", (1, OUTPUT_DRAIN_BD)))
    errors.extend(
        require_disjoint_bd_ids(
            "bridge packet ping-pong",
            BRIDGE_PACKET_IN_BDS,
            BRIDGE_PACKET_OUT_BDS,
        )
    )
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"bridge compact receive group {group}", bd_ids))
    errors.extend(
        require_c1r1_s2mm3_high_bds("bridge compact receive group 3", BRIDGE_RECEIVE_BDS[3])
    )
    errors.extend(
        require_compact_payload_slice(
            "bridge up compact to c6r2",
            1,
            C6R2_HALF_DWORDS,
            1,
            C6R2_HALF_DWORDS,
        )
    )
    errors.extend(
        require_compact_payload_slice(
            "bridge gate compact to c6r2",
            1,
            C6R2_HALF_DWORDS,
            1,
            C6R2_HALF_DWORDS,
        )
    )
    return errors
