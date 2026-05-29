"""Generate runnable MLIR-AIE for Shape-A/B attention-to-O bridge."""

from __future__ import annotations

from pathlib import Path

from bridge_generate import npu_address_patch, npu_push_queue, npu_sync, npu_writebd
from contract import MAIN_COLUMNS, MAIN_ROWS, SHAPE_CARRIER_DWORDS
from shape_reference import (
    CASE_NAME,
    COLUMN_SUMMARY_DWORDS,
    KV_SIDE_DWORDS,
    MAIN_CHUNK_DWORDS,
    PACKET_ID,
    Q_DWORDS,
    SUMMARY_DWORDS,
    TOTAL_SUMMARY_DWORDS,
    WINDOW_DWORDS,
)

OUTPUT_COLLECT_BD_BASE = 10
OUTPUT_DRAIN_BD = 34
BRIDGE_QUANTUM_DWORDS = 256
ROWS_PER_COLUMN = len(MAIN_ROWS)
SHAPE_A_TILES = ((0, 2), (0, 4), (7, 2), (7, 4))
SHAPE_B_TILES = ((0, 3), (0, 5), (7, 3), (7, 5))
HUB_Q_OUT_BDS = (2, 24, 4, 26)
HUB_RETURN_IN_BDS = (25, 6, 27, 8)
KV_OUT_BDS = (2, 24, 4, 26)


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def _main_packet(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _shape_a_symbol(window: int) -> str:
    return f"shape_a{window}"


def _shape_b_symbol(window: int) -> str:
    return f"shape_b{window}"


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
        next_start = f"^return{window + 1}_start" if window + 1 < 4 else "^packet_out_start"
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
    %hub_q_empty = aie.lock(%hub, 0) {{init = 4 : i32, sym_name = "hub_q_empty"}}
    %hub_q_full = aie.lock(%hub, 1) {{init = 0 : i32, sym_name = "hub_q_full"}}
    %hub_return_empty = aie.lock(%hub, 2) {{init = 4 : i32, sym_name = "hub_return_empty"}}
    %hub_return_full = aie.lock(%hub, 3) {{init = 0 : i32, sym_name = "hub_return_full"}}

    %hub_dma = aie.memtile_dma(%hub) {{
      %q_in_dma = aie.dma_start(S2MM, 0, ^q_in, ^q0_start)
    ^q_in:
      aie.use_lock(%hub_q_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_q_full, Release, 4)
      aie.next_bd ^q_in

{chr(10).join(q_outs)}

{chr(10).join(return_ins)}

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 5, ^packet_out, ^end)
    ^packet_out:
      aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID}>}}
      aie.use_lock(%hub_return_empty, Release, 4)
      aie.next_bd ^packet_out
    ^end:
      aie.end
    }}
"""


def _kv_memtile(side: int) -> str:
    tile = "kv_left" if side == 0 else "kv_right"
    starts = []
    for slot, bd_id in enumerate(KV_OUT_BDS):
        next_start = f"^out{slot + 1}_start" if slot + 1 < 4 else "^end"
        starts.append(f"""    ^out{slot}_start:
      %out{slot}_dma = aie.dma_start(MM2S, {slot}, ^out{slot}, {next_start})
    ^out{slot}:
      aie.use_lock(%{tile}_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_payload : memref<{KV_SIDE_DWORDS}xi32>, {slot * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_payload_empty, Release, 1)
      aie.next_bd ^out{slot}""")

    return f"""
    %{tile}_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_payload"}} : memref<{KV_SIDE_DWORDS}xi32>
    %{tile}_payload_empty = aie.lock(%{tile}, 0) {{init = 4 : i32, sym_name = "{tile}_payload_empty"}}
    %{tile}_payload_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_payload_full"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %input_dma = aie.dma_start(S2MM, 0, ^input, ^out0_start)
    ^input:
      aie.use_lock(%{tile}_payload_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%{tile}_payload : memref<{KV_SIDE_DWORDS}xi32>, 0, {KV_SIDE_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_payload_full, Release, 4)
      aie.next_bd ^input

{chr(10).join(starts)}
    ^end:
      aie.end
    }}
"""


def _shape_a(window: int) -> str:
    tile = _shape_a_symbol(window)
    return f"""
    %{tile}_q = aie.buffer(%{tile}) {{sym_name = "{tile}_q"}} : memref<{WINDOW_DWORDS}xi32>
    %{tile}_k = aie.buffer(%{tile}) {{sym_name = "{tile}_k"}} : memref<{WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
{_lock_pair(tile, "q", 0)}
{_lock_pair(tile, "k", 2)}
{_lock_pair(tile, "carrier", 4)}

    %{tile}_core = aie.core(%{tile}) {{
      %window_i32 = arith.constant {window} : i32
      %window_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_q_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
      func.call @shape_make_carrier(%{tile}_q, %{tile}_k, %{tile}_carrier, %window_i32, %window_dwords_i32, %carrier_dwords_i32)
        : (memref<{WINDOW_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_q_empty, Release, 1)
      aie.use_lock(%{tile}_k_empty, Release, 1)
      aie.use_lock(%{tile}_carrier_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %q_dma = aie.dma_start(S2MM, 0, ^q_in, ^k_start)
    ^q_in:
      aie.use_lock(%{tile}_q_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_q : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_q_full, Release, 1)
      aie.next_bd ^q_in

    ^k_start:
      %k_dma = aie.dma_start(S2MM, 1, ^k_in, ^carrier_start)
    ^k_in:
      aie.use_lock(%{tile}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_k_full, Release, 1)
      aie.next_bd ^k_in

    ^carrier_start:
      %carrier_dma = aie.dma_start(MM2S, 0, ^carrier_out, ^end)
    ^carrier_out:
      aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_carrier : memref<{SHAPE_CARRIER_DWORDS}xi32>, 0, {SHAPE_CARRIER_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_carrier_empty, Release, 1)
      aie.next_bd ^carrier_out
    ^end:
      aie.end
    }}
"""


def _shape_b(window: int) -> str:
    tile = _shape_b_symbol(window)
    return f"""
    %{tile}_v = aie.buffer(%{tile}) {{sym_name = "{tile}_v"}} : memref<{WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{WINDOW_DWORDS}xi32>
{_lock_pair(tile, "v", 0)}
{_lock_pair(tile, "carrier", 2)}
{_lock_pair(tile, "output", 4)}

    %{tile}_core = aie.core(%{tile}) {{
      %window_i32 = arith.constant {window} : i32
      %window_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_v_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @shape_make_return(%{tile}_v, %{tile}_carrier, %{tile}_output, %window_i32, %window_dwords_i32, %carrier_dwords_i32)
        : (memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_v_empty, Release, 1)
      aie.use_lock(%{tile}_carrier_empty, Release, 1)
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %v_dma = aie.dma_start(S2MM, 0, ^v_in, ^carrier_start)
    ^v_in:
      aie.use_lock(%{tile}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_v_full, Release, 1)
      aie.next_bd ^v_in

    ^carrier_start:
      %carrier_dma = aie.dma_start(S2MM, 1, ^carrier_in, ^output_start)
    ^carrier_in:
      aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_carrier : memref<{SHAPE_CARRIER_DWORDS}xi32>, 0, {SHAPE_CARRIER_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_carrier_full, Release, 1)
      aie.next_bd ^carrier_in

    ^output_start:
      %output_dma = aie.dma_start(MM2S, 0, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
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
      %input_dma = aie.dma_start(S2MM, 4, ^in_ping, ^out_start)
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
      %output_dma = aie.dma_start(MM2S, 1, ^out_ping, ^end)
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


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    main_chunks = Q_DWORDS // MAIN_CHUNK_DWORDS
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
      %chunks = arith.constant {main_chunks} : index
      %dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      func.call @shape_init_summary(%{tile}_summary, %group_i32, %row_i32)
        : (memref<{SUMMARY_DWORDS}xi32>, i32, i32) -> ()
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
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %input_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^output_start)
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
      %output_dma = aie.dma_start(MM2S, 1, ^summary_out, ^end)
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
      %input_dma = aie.dma_start(S2MM, 2, ^out0, ^output_drain_start)
{chr(10).join(output_bds)}

    ^output_drain_start:
      %output_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_output : memref<{COLUMN_SUMMARY_DWORDS}xi32>, 0, {COLUMN_SUMMARY_DWORDS}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%{tile}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%q: memref<{Q_DWORDS}xi32>, "
        f"%kv_left_arg: memref<{KV_SIDE_DWORDS}xi32>, "
        f"%kv_right_arg: memref<{KV_SIDE_DWORDS}xi32>, "
        f"%output: memref<{TOTAL_SUMMARY_DWORDS}xi32>) {{"
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        output_offset = group * COLUMN_SUMMARY_DWORDS * 4
        lines.extend(
            (
                npu_writebd(column, 13, COLUMN_SUMMARY_DWORDS, output_offset),
                npu_address_patch(column, 13, 3, output_offset),
                npu_push_queue(column, "S2MM", 1, 13),
            )
        )
    lines.extend(
        (
            npu_writebd(6, 0, Q_DWORDS, 0),
            npu_address_patch(6, 0, 0, 0),
            npu_push_queue(6, "MM2S", 0, 0),
            npu_writebd(0, 0, KV_SIDE_DWORDS, 0),
            npu_address_patch(0, 0, 1, 0),
            npu_push_queue(0, "MM2S", 0, 0),
            npu_writebd(7, 0, KV_SIDE_DWORDS, 0),
            npu_address_patch(7, 0, 2, 0),
            npu_push_queue(7, "MM2S", 0, 0),
            npu_sync(6, 0, direction=1),
            npu_sync(0, 0, direction=1),
            npu_sync(7, 0, direction=1),
        )
    )
    for column in MAIN_COLUMNS:
        lines.append(npu_sync(column, 1))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    tile_defs = [
        "    %shim_left = aie.tile(0, 0)",
        "    %kv_left = aie.tile(0, 1)",
        "    %bridge = aie.tile(1, 1)",
        "    %shim_q = aie.tile(6, 0)",
        "    %hub = aie.tile(6, 1)",
        "    %shim_right = aie.tile(7, 0)",
        "    %kv_right = aie.tile(7, 1)",
    ]
    for window, (column, row) in enumerate(SHAPE_A_TILES):
        tile_defs.append(f"    %{_shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{_shape_b_symbol(window)} = aie.tile({column}, {row})")
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = [
        "    aie.flow(%shim_q, DMA : 0, %hub, DMA : 0)",
        "    aie.flow(%shim_left, DMA : 0, %kv_left, DMA : 0)",
        "    aie.flow(%shim_right, DMA : 0, %kv_right, DMA : 0)",
    ]
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
            f"    aie.packet_flow({PACKET_ID}) {{",
            "      aie.packet_source<%hub, DMA : 5>",
            "      aie.packet_dest<%bridge, DMA : 4>",
            "    }",
        )
    )
    for group, column in enumerate(MAIN_COLUMNS):
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.flow(%bridge, DMA : 1, %{_main_symbol(group, row)}, DMA : 0)")
            flows.append(f"    aie.packet_flow({_main_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%{_main_symbol(group, row)}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : 2>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mt{group}, DMA : 5, %shim{group}, DMA : 1)")

    blocks = [_hub(), _kv_memtile(0), _kv_memtile(1), _bridge()]
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

    func.func private @shape_make_carrier(memref<{WINDOW_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @shape_make_return(memref<{WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{WINDOW_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @shape_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @shape_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        "aie.tile(0, 2)",
        "aie.tile(7, 5)",
        "aie.tile(6, 1)",
        "aie.tile(1, 1)",
        f"aie.packet_flow({PACKET_ID})",
        "aie.packet_source<%hub, DMA : 5>",
        "aie.packet_dest<%bridge, DMA : 4>",
        f"memref<{Q_DWORDS}xi32>",
        f"memref<{KV_SIDE_DWORDS}xi32>",
        f"memref<{WINDOW_DWORDS}xi32>",
        f"memref<{SHAPE_CARRIER_DWORDS}xi32>",
        "shape_make_carrier",
        "shape_make_return",
        "shape_accum_chunk",
        "debug_contract.o",
    )
    errors = [f"missing shape marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count("shape_make_carrier") != 5:
        errors.append("Shape-A function declaration/call count mismatch")
    if mlir.count("shape_make_return") != 5:
        errors.append("Shape-B function declaration/call count mismatch")
    if mlir.count("aie.flow(%bridge, DMA : 1") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN:
        errors.append("main activation bridge flow count mismatch")
    if mlir.count("aie.packet_flow(") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN + 1:
        errors.append("packet flow count mismatch")
    if CASE_NAME not in (CASE_NAME,):
        errors.append("unreachable case marker")
    return errors
