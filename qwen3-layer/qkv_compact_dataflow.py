"""Shared four-phase Q/K/V/O compact dataflow for attention integration cases."""

from __future__ import annotations

from contract import COMPACT_PACKET_DWORDS, MAIN_COLUMNS, RECORD_DWORDS, RECORD_PAYLOAD_DWORDS, ROWS_PER_COLUMN
from mlir_utils import lock_pair
from qkv_compact_reference import (
    COLUMN_COMPACT_DWORDS,
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    OUTPUT_DWORDS,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    SUMMARY_DWORDS,
    V_GLOBAL_PACKET_ID,
    column_packet,
    main_packet,
)

STAGES = ("q", "k", "v", "o")
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


def main_symbol(group: int, row: int) -> str:
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
    lines: list[str] = []
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
    lines: list[str] = []
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


def main_tile(group: int, row: int) -> str:
    tile = main_symbol(group, row)
    chunks = Q_DWORDS // MAIN_CHUNK_DWORDS
    packet = main_packet(group, row)
    record_bds = (2, 3, 4, 5)
    record_blocks: list[str] = []
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


def column_memtile(group: int) -> str:
    tile = f"mt{group}"
    packet = column_packet(group)
    source_offset = 0 if group == 0 else 1
    source_length = COLUMN_COMPACT_DWORDS if group == 0 else COLUMN_COMPACT_DWORDS - 1
    receive_starts: list[str] = []
    for row in range(ROWS_PER_COLUMN):
        dest_offset, length = _segment(row)
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^out_start"
        bds = COLUMN_RECEIVE_BDS[row]
        stage_blocks: list[str] = []
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

    out_blocks: list[str] = []
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
    starts: list[str] = []
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
        stage_blocks: list[str] = []
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


def bridge() -> str:
    buffers = "\n".join(
        f'    %bridge_{stage} = aie.buffer(%bridge) {{sym_name = "bridge_{stage}"}} : memref<{COMPACT_PACKET_DWORDS}xi32>'
        for stage in STAGES
    )
    packet_ids = (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID, O_GLOBAL_PACKET_ID)
    compact_out_blocks: list[str] = []
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


def full_vector() -> str:
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
