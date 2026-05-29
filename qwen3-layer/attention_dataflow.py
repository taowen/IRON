"""Shared Shape-A/B attention fabric layout for qwen3-layer."""

from __future__ import annotations

from contract import MAIN_ROWS
from qkv_compact_reference import PACKET_ID_ATTENTION, Q_DWORDS, WINDOW_DWORDS

ROWS_PER_COLUMN = len(MAIN_ROWS)
SHAPE_A_TILES = ((0, 2), (0, 4), (7, 2), (7, 4))
SHAPE_B_TILES = ((0, 3), (0, 5), (7, 3), (7, 5))
HUB_Q_OUT_BDS = (2, 24, 4, 26)
HUB_RETURN_IN_BDS = (25, 6, 27, 8)
KV_OUT_BDS = (2, 24, 4, 26)


def shape_a_symbol(window: int) -> str:
    return f"shape_a{window}"


def shape_b_symbol(window: int) -> str:
    return f"shape_b{window}"


def attention_hub() -> str:
    q_outs: list[str] = []
    for window, bd_id in enumerate(HUB_Q_OUT_BDS):
        next_start = f"^q{window + 1}_start" if window + 1 < 4 else "^return0_start"
        q_outs.append(f"""    ^q{window}_start:
      %q{window}_dma = aie.dma_start(MM2S, {window}, ^q{window}_out, {next_start})
    ^q{window}_out:
      aie.use_lock(%hub_q_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_q_empty, Release, 1)
      aie.next_bd ^q{window}_out""")

    return_ins: list[str] = []
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
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 34 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_ATTENTION}>}}
      aie.use_lock(%hub_return_empty, Release, 4)
      aie.next_bd ^packet_out
    ^end:
      aie.end
    }}
"""
