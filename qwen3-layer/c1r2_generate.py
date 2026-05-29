"""Generate runnable MLIR-AIE for the c1r2 full-vector replay bridge."""

from __future__ import annotations

from pathlib import Path

from c1r2_reference import (
    CASE_NAME,
    COLUMN_PACKET_BASE,
    FFN_GLOBAL_PACKET_ID,
    MAIN_ACCUM_DWORDS,
    MAIN_CHUNKS_PER_REPLAY,
    MAIN_CHUNK_DWORDS,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    OUTPUT_DWORDS,
    SWIGLU_OUTPUT_DWORDS,
    SWIGLU_SLICES,
    TOTAL_MAIN_CHUNKS,
    column_packet,
    main_packet,
)
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
)
from swiglu_generate import npu_address_patch, npu_push_queue, npu_sync, npu_writebd

COLUMN_COMPACT_DWORDS = RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS
C1R2_REPLAY_PAYLOAD_DWORDS = C1R2_PACKET_DWORDS - 1
COLUMN_RECEIVE_BDS = ((0, 1, 2), (24, 25, 26), (3, 4, 5), (27, 28, 29))
BRIDGE_COMPACT_BDS = ((0, 1, 2), (24, 25, 26), (3, 4, 5), (30, 31, 32))
BRIDGE_PACKET_IN_BDS = (6, 7)
BRIDGE_PACKET_OUT_BDS = (28, 29)
COMPACT_OUT_BDS = (34, 35, 36)
OUTPUT_DRAIN_BD = 34


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _segment(row: int) -> tuple[int, int]:
    if row == 0:
        return 0, RECORD_DWORDS
    return RECORD_DWORDS + (row - 1) * RECORD_PAYLOAD_DWORDS, RECORD_PAYLOAD_DWORDS


def _source_segment(index: int) -> tuple[int, int]:
    offset = index * RECORD_DWORDS
    return offset, RECORD_DWORDS


def _source_segment_for_compact(index: int, row: int) -> tuple[int, int]:
    offset, length = _source_segment(index)
    if row == 0:
        return offset, length
    return offset + 1, RECORD_PAYLOAD_DWORDS


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    o_offset, o_length = _source_segment_for_compact(0, row)
    up_offset, up_length = _source_segment_for_compact(1, row)
    gate_offset, gate_length = _source_segment_for_compact(2, row)
    packet = main_packet(group, row)
    return f"""
    %{tile}_records = aie.buffer(%{tile}) {{sym_name = "{tile}_records"}} : memref<{MAIN_RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_accum = aie.buffer(%{tile}) {{sym_name = "{tile}_accum"}} : memref<{MAIN_ACCUM_DWORDS}xi32>
{_lock_pair(tile, "o", 0)}
{_lock_pair(tile, "up", 2)}
{_lock_pair(tile, "gate", 4)}
{_lock_pair(tile, "chunk", 6, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %replays = arith.constant {C1R2_UPGATE_REPLAYS} : index
      %chunks_per_replay = arith.constant {MAIN_CHUNKS_PER_REPLAY} : index
      %dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_o_empty, AcquireGreaterEqual, 1)
      func.call @c1r2_emit_o_record(%{tile}_records, %group_i32, %row_i32)
        : (memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_o_full, Release, 1)

      scf.for %replay = %c0 to %replays step %c1 {{
        %replay_i32 = arith.index_cast %replay : index to i32
        %is_gate_value = arith.remsi %replay_i32, %c2_i32 : i32
        %is_gate = arith.cmpi eq, %is_gate_value, %c1_i32 : i32
        func.call @c1r2_main_init_accum(%{tile}_accum, %group_i32, %row_i32)
          : (memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32) -> ()
        scf.for %chunk = %c0 to %chunks_per_replay step %c1 {{
          %base_chunk = arith.muli %replay, %chunks_per_replay : index
          %global_chunk = arith.addi %base_chunk, %chunk : index
          %global_chunk_i32 = arith.index_cast %global_chunk : index to i32
          %rem = arith.remsi %global_chunk_i32, %c2_i32 : i32
          %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
          aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
          scf.if %is_pong {{
            func.call @c1r2_main_accum_chunk(%{tile}_chunk_pong, %{tile}_accum, %global_chunk_i32, %group_i32, %row_i32, %dwords_i32)
              : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32, i32) -> ()
          }} else {{
            func.call @c1r2_main_accum_chunk(%{tile}_chunk_ping, %{tile}_accum, %global_chunk_i32, %group_i32, %row_i32, %dwords_i32)
              : (memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32, i32) -> ()
          }}
          aie.use_lock(%{tile}_chunk_empty, Release, 1)
        }}
        scf.if %is_gate {{
          aie.use_lock(%{tile}_gate_empty, AcquireGreaterEqual, 1)
          func.call @c1r2_main_emit_gate_record(%{tile}_records, %{tile}_accum, %group_i32, %row_i32, %replay_i32)
            : (memref<{MAIN_RECORD_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32) -> ()
          aie.use_lock(%{tile}_gate_full, Release, 1)
        }} else {{
          aie.use_lock(%{tile}_up_empty, AcquireGreaterEqual, 1)
          func.call @c1r2_main_emit_up_record(%{tile}_records, %{tile}_accum, %group_i32, %row_i32, %replay_i32)
            : (memref<{MAIN_RECORD_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32) -> ()
          aie.use_lock(%{tile}_up_full, Release, 1)
        }}
      }}
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
      %record_dma = aie.dma_start(MM2S, 1, ^o_out, ^end)
    ^o_out:
      aie.use_lock(%{tile}_o_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {o_offset}, {o_length}) {{bd_id = 2 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_o_empty, Release, 1)
      aie.next_bd ^up_out
    ^up_out:
      aie.use_lock(%{tile}_up_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {up_offset}, {up_length}) {{bd_id = 3 : i32, next_bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_up_empty, Release, 1)
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%{tile}_gate_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>, {gate_offset}, {gate_length}) {{bd_id = 4 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_gate_empty, Release, 1)
      aie.next_bd ^up_out
    ^end:
      aie.end
    }}
"""


def _column_lock_defs(tile: str) -> str:
    lines = [
        f'    %{tile}_o_full = aie.lock(%{tile}, 0) {{init = 0 : i32, sym_name = "{tile}_o_full"}}\n',
        f'    %{tile}_up_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_up_full"}}\n',
        f'    %{tile}_gate_full = aie.lock(%{tile}, 2) {{init = 0 : i32, sym_name = "{tile}_gate_full"}}\n',
        f'    %{tile}_o_empty = aie.lock(%{tile}, 3) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_o_empty"}}\n',
        f'    %{tile}_up_empty = aie.lock(%{tile}, 4) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_up_empty"}}\n',
        f'    %{tile}_gate_empty = aie.lock(%{tile}, 5) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_gate_empty"}}\n',
    ]
    return "".join(lines)


def _column_stage_empty_releases(tile: str, stage: str) -> str:
    return f"      aie.use_lock(%{tile}_{stage}_empty, Release, {ROWS_PER_COLUMN})"


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
        o_bd, up_bd, gate_bd = COLUMN_RECEIVE_BDS[row]
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_o, {next_start})
    ^row{row}_o:
      aie.use_lock(%{tile}_o_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_o : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {o_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%{tile}_o_full, Release, 1)
      aie.next_bd ^row{row}_up
    ^row{row}_up:
      aie.use_lock(%{tile}_up_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_up : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {up_bd} : i32, next_bd_id = {gate_bd} : i32}}
      aie.use_lock(%{tile}_up_full, Release, 1)
      aie.next_bd ^row{row}_gate
    ^row{row}_gate:
      aie.use_lock(%{tile}_gate_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_gate : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {gate_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%{tile}_gate_full, Release, 1)
      aie.next_bd ^row{row}_up"""
        )

    return f"""
    %{tile}_o = aie.buffer(%{tile}) {{sym_name = "{tile}_o"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
    %{tile}_up = aie.buffer(%{tile}) {{sym_name = "{tile}_up"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
    %{tile}_gate = aie.buffer(%{tile}) {{sym_name = "{tile}_gate"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>
{_column_lock_defs(tile)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^o_out, ^end)
    ^o_out:
      aie.use_lock(%{tile}_o_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_o : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {COMPACT_OUT_BDS[0]} : i32, next_bd_id = {COMPACT_OUT_BDS[1]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
{_column_stage_empty_releases(tile, "o")}
      aie.next_bd ^up_out
    ^up_out:
      aie.use_lock(%{tile}_up_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_up : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {COMPACT_OUT_BDS[1]} : i32, next_bd_id = {COMPACT_OUT_BDS[2]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
{_column_stage_empty_releases(tile, "up")}
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%{tile}_gate_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_gate : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {COMPACT_OUT_BDS[2]} : i32, next_bd_id = {COMPACT_OUT_BDS[1]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
{_column_stage_empty_releases(tile, "gate")}
      aie.next_bd ^up_out
    ^end:
      aie.end
    }}
"""


def _bridge_lock_defs() -> str:
    lines = [
        '    %bridge_o_full = aie.lock(%bridge, 0) {init = 0 : i32, sym_name = "bridge_o_full"}\n',
        '    %bridge_up_full = aie.lock(%bridge, 1) {init = 0 : i32, sym_name = "bridge_up_full"}\n',
        '    %bridge_gate_full = aie.lock(%bridge, 2) {init = 0 : i32, sym_name = "bridge_gate_full"}\n',
    ]
    lines.append(_lock_pair("bridge", "packet", 3, init_empty=2))
    lines.extend(
        (
            f'    %bridge_o_empty = aie.lock(%bridge, 5) {{init = {len(MAIN_COLUMNS)} : i32, sym_name = "bridge_o_empty"}}\n',
            f'    %bridge_up_empty = aie.lock(%bridge, 6) {{init = {len(MAIN_COLUMNS)} : i32, sym_name = "bridge_up_empty"}}\n',
            f'    %bridge_gate_empty = aie.lock(%bridge, 7) {{init = {len(MAIN_COLUMNS)} : i32, sym_name = "bridge_gate_empty"}}\n',
        )
    )
    return "".join(lines)


def _bridge_stage_empty_releases(stage: str) -> str:
    return f"      aie.use_lock(%bridge_{stage}_empty, Release, {len(MAIN_COLUMNS)})"


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
        o_bd, up_bd, gate_bd = BRIDGE_COMPACT_BDS[group]
        starts.append(
            f"""{start_label}      %g{group}_dma = aie.dma_start(S2MM, {group}, ^g{group}_o, {next_start})
    ^g{group}_o:
      aie.use_lock(%bridge_o_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_o : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {o_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%bridge_o_full, Release, 1)
      aie.next_bd ^g{group}_up
    ^g{group}_up:
      aie.use_lock(%bridge_up_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_up : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {up_bd} : i32, next_bd_id = {gate_bd} : i32}}
      aie.use_lock(%bridge_up_full, Release, 1)
      aie.next_bd ^g{group}_gate
    ^g{group}_gate:
      aie.use_lock(%bridge_gate_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_gate : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {gate_bd} : i32, next_bd_id = {up_bd} : i32}}
      aie.use_lock(%bridge_gate_full, Release, 1)
      aie.next_bd ^g{group}_up"""
        )
    return "\n".join(starts)


def _bridge() -> str:
    return f"""
    %bridge_o = aie.buffer(%bridge) {{sym_name = "bridge_o"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_up = aie.buffer(%bridge) {{sym_name = "bridge_up"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_gate = aie.buffer(%bridge) {{sym_name = "bridge_gate"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_packet_ping = aie.buffer(%bridge) {{sym_name = "bridge_packet_ping"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %bridge_packet_pong = aie.buffer(%bridge) {{sym_name = "bridge_packet_pong"}} : memref<{C6R2_HALF_DWORDS}xi32>
{_bridge_lock_defs()}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{_bridge_receive_starts()}

    ^compact_out_start:
      %compact_out_dma = aie.dma_start(MM2S, 5, ^o_out, ^packet_in_start)
    ^o_out:
      aie.use_lock(%bridge_o_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_o : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = {COMPACT_OUT_BDS[0]} : i32, next_bd_id = {COMPACT_OUT_BDS[1]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {O_GLOBAL_PACKET_ID}>}}
{_bridge_stage_empty_releases("o")}
      aie.next_bd ^up_out
    ^up_out:
      aie.use_lock(%bridge_up_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_up : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {C6R2_HALF_DWORDS}) {{bd_id = {COMPACT_OUT_BDS[1]} : i32, next_bd_id = {COMPACT_OUT_BDS[2]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FFN_GLOBAL_PACKET_ID}>}}
{_bridge_stage_empty_releases("up")}
      aie.next_bd ^gate_out
    ^gate_out:
      aie.use_lock(%bridge_gate_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_gate : memref<{COMPACT_PACKET_DWORDS}xi32>, 1, {C6R2_HALF_DWORDS}) {{bd_id = {COMPACT_OUT_BDS[2]} : i32, next_bd_id = {COMPACT_OUT_BDS[1]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FFN_GLOBAL_PACKET_ID}>}}
{_bridge_stage_empty_releases("gate")}
      aie.next_bd ^up_out

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
{_lock_pair("full", "compact", 0)}
{_lock_pair("full", "replay", 2)}

    %full_core = aie.core(%full) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %replays = arith.constant {C1R2_UPGATE_REPLAYS} : index
      %payload_i32 = arith.constant {C1R2_REPLAY_PAYLOAD_DWORDS} : i32
      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      scf.for %replay = %c0 to %replays step %c1 {{
        %replay_i32 = arith.index_cast %replay : index to i32
        aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
        func.call @c1r2_make_replay(%full_compact, %full_replay, %replay_i32, %payload_i32)
          : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32) -> ()
        aie.use_lock(%full_replay_full, Release, 1)
      }}
      aie.use_lock(%full_compact_empty, Release, 1)
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
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^end)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{C1R2_PACKET_DWORDS}xi32>, 1, {C1R2_REPLAY_PAYLOAD_DWORDS}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^replay_out
    ^end:
      aie.end
    }}
"""


def _swiglu(repeats: int = 1, consume_input_per_repeat: bool = False) -> str:
    input_acquire_before_loop = "" if consume_input_per_repeat else "      aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)"
    input_acquire_in_loop = "        aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)" if consume_input_per_repeat else ""
    input_release_in_loop = "        aie.use_lock(%swiglu_input_empty, Release, 2)" if consume_input_per_repeat else ""
    input_release_after_loop = "" if consume_input_per_repeat else "      aie.use_lock(%swiglu_input_empty, Release, 2)"
    return f"""
    %swiglu_input = aie.buffer(%swiglu) {{sym_name = "swiglu_input"}} : memref<{C6R2_INPUT_DWORDS}xi32>
    %swiglu_output = aie.buffer(%swiglu) {{sym_name = "swiglu_output"}} : memref<{SWIGLU_OUTPUT_DWORDS}xi32>
{_lock_pair("swiglu", "input", 0, init_empty=2)}
{_lock_pair("swiglu", "output", 2)}

    %swiglu_core = aie.core(%swiglu) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %repeats = arith.constant {repeats} : index
      %dwords_i32 = arith.constant {C6R2_INPUT_DWORDS} : i32
{input_acquire_before_loop}
      scf.for %repeat = %c0 to %repeats step %c1 {{
        %slice_i32 = arith.index_cast %repeat : index to i32
{input_acquire_in_loop}
        aie.use_lock(%swiglu_output_empty, AcquireGreaterEqual, 1)
        func.call @ffn_swiglu_slice_contract(%swiglu_input, %swiglu_output, %dwords_i32, %slice_i32)
          : (memref<{C6R2_INPUT_DWORDS}xi32>, memref<{SWIGLU_OUTPUT_DWORDS}xi32>, i32, i32) -> ()
        aie.use_lock(%swiglu_output_full, Release, 1)
{input_release_in_loop}
      }}
{input_release_after_loop}
      aie.end
    }}

    %swiglu_mem = aie.mem(%swiglu) {{
      %input_dma = aie.dma_start(S2MM, 0, ^up_in, ^out_start)
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
      %output_dma = aie.dma_start(MM2S, 1, ^out, ^end)
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
      %input_dma = aie.dma_start(S2MM, 0, ^output_in, ^output_drain_start)
    ^output_in:
      aie.use_lock(%hub_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_output : memref<{SWIGLU_OUTPUT_DWORDS}xi32>, 0, {SWIGLU_OUTPUT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_output_full, Release, 1)
      aie.next_bd ^output_in

    ^output_drain_start:
      %output_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
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
            f"    aie.runtime_sequence(%output: memref<{OUTPUT_DWORDS}xi32>) {{",
            npu_writebd(6, 13, OUTPUT_DWORDS, 0),
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
        "    %full = aie.tile(1, 2)",
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
            flows.append(f"      aie.packet_source<%{_main_symbol(group, row)}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : {row}>")
            flows.append("    }")
            flows.append(f"    aie.flow(%bridge, DMA : 1, %{_main_symbol(group, row)}, DMA : 0)")
        flows.append(f"    aie.packet_flow({column_packet(group)}) {{")
        flows.append(f"      aie.packet_source<%mt{group}, DMA : 5>")
        flows.append(f"      aie.packet_dest<%bridge, DMA : {group}>")
        flows.append("    }")
    flows.extend(
        (
            f"    aie.packet_flow({O_GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%full, DMA : 0>",
            "    }",
            "    aie.packet_flow(0) {",
            "      aie.packet_source<%full, DMA : 1>",
            "      aie.packet_dest<%bridge, DMA : 4>",
            "    }",
            f"    aie.packet_flow({FFN_GLOBAL_PACKET_ID}) {{",
            "      aie.packet_source<%bridge, DMA : 5>",
            "      aie.packet_dest<%swiglu, DMA : 0>",
            "    }",
            "    aie.flow(%swiglu, DMA : 1, %hub, DMA : 0)",
            "    aie.flow(%hub, DMA : 5, %shim_out, DMA : 1)",
        )
    )

    blocks = [_bridge(), _full_vector(), _swiglu(SWIGLU_SLICES, consume_input_per_repeat=True), _hub()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @c1r2_emit_o_record(memref<{MAIN_RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @c1r2_make_replay(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @c1r2_main_init_accum(memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @c1r2_main_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @c1r2_main_emit_up_record(memref<{MAIN_RECORD_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @c1r2_main_emit_gate_record(memref<{MAIN_RECORD_DWORDS}xi32>, memref<{MAIN_ACCUM_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @ffn_swiglu_slice_contract(memref<{C6R2_INPUT_DWORDS}xi32>, memref<{SWIGLU_OUTPUT_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/swiglu.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "aie.tile(1, 2)",
        "aie.tile(1, 1)",
        "aie.tile(6, 2)",
        f"aie.packet_flow({O_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({FFN_GLOBAL_PACKET_ID})",
        "aie.packet_flow(0)",
        f"memref<{COMPACT_PACKET_DWORDS}xi32>",
        f"memref<{C1R2_PACKET_DWORDS}xi32>",
        f"memref<{C6R2_INPUT_DWORDS}xi32>",
        f"memref<{SWIGLU_OUTPUT_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        "c1r2_make_replay",
        "c1r2_main_accum_chunk",
        "c1r2_main_emit_up_record",
        "c1r2_main_emit_gate_record",
        "ffn_swiglu_slice_contract",
        "debug_contract.o",
    )
    markers = tuple(marker for marker in required if not marker.startswith("case marker"))
    errors = [f"missing c1r2 marker: {marker}" for marker in markers if marker not in mlir]
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 3
    if mlir.count("aie.packet_flow(") != expected_packets:
        errors.append("packet flow count mismatch")
    if mlir.count("aie.flow(%bridge, DMA : 1") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN:
        errors.append("main activation bridge flow count mismatch")
    if f"%replays = arith.constant {C1R2_UPGATE_REPLAYS} : index" not in mlir:
        errors.append("c1r2 replay count marker missing")
    if f"%chunks_per_replay = arith.constant {MAIN_CHUNKS_PER_REPLAY} : index" not in mlir:
        errors.append("c1r2 per-replay chunk count marker missing")
    if f"%repeats = arith.constant {SWIGLU_SLICES} : index" not in mlir:
        errors.append("c1r2 swiglu slice count marker missing")
    if COLUMN_PACKET_BASE != 4:
        errors.append("unreachable packet base mismatch")
    return errors
