"""Generate MLIR-AIE for current K/V cache writeback before KV-scan attention."""

from __future__ import annotations

from pathlib import Path

from attention_dataflow import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    attention_hub,
    shape_a_symbol,
    shape_b_symbol,
)
from contract import COMPACT_PACKET_DWORDS, MAIN_COLUMNS, MAIN_ROWS, RECORD_DWORDS, SHAPE_CARRIER_DWORDS
from mlir_utils import (
    flow,
    lock_pair,
    npu_address_patch,
    npu_push_queue,
    npu_rtp_write,
    npu_set_lock,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_absent_markers,
    require_count,
    require_disjoint_bd_ids,
    require_dma_bd_lock_balance,
    require_kv16_attention_shapes,
    require_marker_order,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_no_compute_kv_materialization,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_bd_ids,
)
from qkv_compact_dataflow import (
    bridge as qkv_compact_bridge,
    column_memtile as qkv_compact_column_memtile,
    full_vector as qkv_compact_full_vector,
    main_symbol,
    main_tile as qkv_compact_main_tile,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    ACCUM_LANES,
    CASE_NAME,
    CACHE_BLOCK_DWORDS,
    CURRENT_DWORDS,
    CURRENT_PACKET_K,
    CURRENT_PACKET_V,
    DEFAULT_SCHEDULE,
    DecodeSchedule,
    HEAD_DWORDS,
    HOST_OUTPUT_DWORDS,
    K_CACHE_SIDE_DWORDS,
    K_WINDOW_DWORDS,
    KV_HEADS,
    KV_SIDE_DWORDS,
    V_CACHE_SIDE_DWORDS,
    V_WINDOW_DWORDS,
    WINDOW_HEAD_DWORDS,
)
from cases.kvscan_attention_kv16_reference import (
    OUTPUT_DWORDS,
    Q_DWORDS,
    SCALAR_DWORDS,
    WEIGHT_DWORDS,
    WINDOW_DWORDS,
)
from qkv_compact_reference import (
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    MAIN_PACKET_BASE,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_GLOBAL_PACKET_ID,
    SUMMARY_DWORDS,
    V_GLOBAL_PACKET_ID,
    column_packet,
    main_packet,
)

ROWS_PER_COLUMN = len(MAIN_ROWS)
K_SCAN_BD = 0
V_SCAN_BD = 1
KV_SCAN_BDS = (K_SCAN_BD, V_SCAN_BD)
CURRENT_WRITE_BDS = (2, 3)
CURRENT_WRITE_CHANNEL = 1
KV_SPLIT_K_IN_BDS = (0, 1)
KV_SPLIT_V_IN_BDS = (28, 29)
K_SIDE_SCAN_DWORDS = K_WINDOW_DWORDS * 2
V_SIDE_SCAN_DWORDS = V_WINDOW_DWORDS * 2


def _shape_blocks_name(tile: str) -> str:
    return f"{tile}_blocks"


def _shape_tail_tokens_name(tile: str) -> str:
    return f"{tile}_tail_tokens"


def _shape_runtime_start_name(tile: str) -> str:
    return f"{tile}_runtime_start"


def _kv_split_scan_memtile(side: int) -> str:
    tile = "kv_left" if side == 0 else "kv_right"
    logical_slots = (
        ("k", 0, 0, K_WINDOW_DWORDS, KV_SPLIT_K_IN_BDS[0]),
        ("v", 1, 0, V_WINDOW_DWORDS, KV_SPLIT_V_IN_BDS[0]),
        ("k", 2, K_WINDOW_DWORDS, K_WINDOW_DWORDS, KV_SPLIT_K_IN_BDS[1]),
        ("v", 3, V_WINDOW_DWORDS, V_WINDOW_DWORDS, KV_SPLIT_V_IN_BDS[1]),
    )
    locks = []
    for _, slot, _, _, _ in logical_slots:
        locks.append(
            f"""    %{tile}_slot{slot}_empty = aie.lock(%{tile}, {slot * 2}) {{init = 1 : i32, sym_name = "{tile}_slot{slot}_empty"}}
    %{tile}_slot{slot}_full = aie.lock(%{tile}, {slot * 2 + 1}) {{init = 0 : i32, sym_name = "{tile}_slot{slot}_full"}}"""
        )

    outputs = []
    for slot, bd_id in enumerate(KV_OUT_BDS):
        next_start = f"^out{slot + 1}_start" if slot + 1 < len(KV_OUT_BDS) else "^end"
        plane, _, offset, length, _ = logical_slots[slot]
        buffer = f"%{tile}_{plane}_payload"
        payload_dwords = K_SIDE_SCAN_DWORDS if plane == "k" else V_SIDE_SCAN_DWORDS
        outputs.append(f"""    ^out{slot}_start:
      %out{slot}_dma = aie.dma_start(MM2S, {slot}, ^out{slot}, {next_start})
    ^out{slot}:
      aie.use_lock(%{tile}_slot{slot}_full, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer} : memref<{payload_dwords}xi32>, {offset}, {length}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_slot{slot}_empty, Release, 1)
      aie.next_bd ^out{slot}""")

    return f"""
    %{tile}_k_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_k_payload"}} : memref<{K_SIDE_SCAN_DWORDS}xi32>
    %{tile}_v_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_v_payload"}} : memref<{V_SIDE_SCAN_DWORDS}xi32>
{chr(10).join(locks)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %k_dma = aie.dma_start(S2MM, 0, ^k0_in, ^v_start)
    ^k0_in:
      aie.use_lock(%{tile}_slot0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k_payload : memref<{K_SIDE_SCAN_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_K_IN_BDS[0]} : i32, next_bd_id = {KV_SPLIT_K_IN_BDS[1]} : i32}}
      aie.use_lock(%{tile}_slot0_full, Release, 1)
      aie.next_bd ^k1_in
    ^k1_in:
      aie.use_lock(%{tile}_slot2_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k_payload : memref<{K_SIDE_SCAN_DWORDS}xi32>, {K_WINDOW_DWORDS}, {K_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_K_IN_BDS[1]} : i32, next_bd_id = {KV_SPLIT_K_IN_BDS[0]} : i32}}
      aie.use_lock(%{tile}_slot2_full, Release, 1)
      aie.next_bd ^k0_in

    ^v_start:
      %v_dma = aie.dma_start(S2MM, 1, ^v0_in, ^out0_start)
    ^v0_in:
      aie.use_lock(%{tile}_slot1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v_payload : memref<{V_SIDE_SCAN_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_V_IN_BDS[0]} : i32, next_bd_id = {KV_SPLIT_V_IN_BDS[1]} : i32}}
      aie.use_lock(%{tile}_slot1_full, Release, 1)
      aie.next_bd ^v1_in
    ^v1_in:
      aie.use_lock(%{tile}_slot3_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v_payload : memref<{V_SIDE_SCAN_DWORDS}xi32>, {V_WINDOW_DWORDS}, {V_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_V_IN_BDS[1]} : i32, next_bd_id = {KV_SPLIT_V_IN_BDS[0]} : i32}}
      aie.use_lock(%{tile}_slot3_full, Release, 1)
      aie.next_bd ^v0_in

{chr(10).join(outputs)}
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
    %post_current_k = aie.buffer(%post) {{sym_name = "post_current_k"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_v = aie.buffer(%post) {{sym_name = "post_current_v"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_token = aie.buffer(%post) {{sym_name = "post_current_token"}} : memref<1xi32>
{lock_pair("post", "q_compact", 0)}
{lock_pair("post", "k_compact", 2)}
{lock_pair("post", "v_compact", 4)}
{lock_pair("post", "q_payload", 6)}
{lock_pair("post", "current_k", 8)}
{lock_pair("post", "current_v", 10)}
    %post_runtime_start = aie.lock(%post, 12) {{init = 0 : i32, sym_name = "post_runtime_start"}}

    %post_core = aie.core(%post) {{
      aie.use_lock(%post_runtime_start, Acquire, 1)
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      %current_dwords_i32 = arith.constant {CURRENT_DWORDS} : i32
      aie.use_lock(%post_q_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_k_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_v_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_v_empty, AcquireGreaterEqual, 1)
      func.call @currentkv_postprocess_payload(%post_q_compact, %post_k_compact, %post_v_compact, %post_q_payload, %post_current_k, %post_current_v, %post_current_token, %q_dwords_i32, %current_dwords_i32)
        : (memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) -> ()
      aie.use_lock(%post_q_compact_empty, Release, 1)
      aie.use_lock(%post_k_compact_empty, Release, 1)
      aie.use_lock(%post_v_compact_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.use_lock(%post_current_k_full, Release, 1)
      aie.use_lock(%post_current_v_full, Release, 1)
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
      %q_dma = aie.dma_start(MM2S, 0, ^q_out, ^current_out_start)
    ^q_out:
      aie.use_lock(%post_q_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%post_q_payload_empty, Release, 1)
      aie.next_bd ^q_out

    ^current_out_start:
      %current_dma = aie.dma_start(MM2S, 1, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%post_current_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_k : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 5 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {CURRENT_PACKET_K}>}}
      aie.use_lock(%post_current_k_empty, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%post_current_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_v : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {CURRENT_PACKET_V}>}}
      aie.use_lock(%post_current_v_empty, Release, 1)
      aie.next_bd ^current_k_out
    ^end:
      aie.end
    }}
"""


def _push_current_cache_write(column: int, cache_arg: int, schedule: DecodeSchedule) -> list[str]:
    lines: list[str] = []
    base_offset = schedule.current_write_byte_offset
    half_current_dwords = CURRENT_DWORDS // 2
    half_head_dwords = HEAD_DWORDS // 2
    for parity, bd_id in enumerate(CURRENT_WRITE_BDS):
        has_next = parity + 1 < len(CURRENT_WRITE_BDS)
        next_bd = CURRENT_WRITE_BDS[parity + 1] if has_next else 0
        offset = base_offset + parity * 4
        lines.extend(
            (
                npu_writebd(
                    column,
                    bd_id,
                    half_current_dwords,
                    offset,
                    next_bd=next_bd,
                    use_next_bd=has_next,
                    d0_size=half_head_dwords,
                    d0_stride=1,
                    d1_size=KV_HEADS,
                    d1_stride=WINDOW_HEAD_DWORDS - 1,
                ),
                npu_address_patch(column, bd_id, cache_arg, offset),
            )
        )
    lines.append(npu_push_queue(column, "S2MM", CURRENT_WRITE_CHANNEL, CURRENT_WRITE_BDS[0]))
    return lines


def _push_kv_scan_from_cache(
    column: int,
    k_arg: int,
    v_arg: int,
    head_base: int,
    schedule: DecodeSchedule,
) -> list[str]:
    k_offset = head_base * WINDOW_HEAD_DWORDS * 4
    v_offset = head_base * WINDOW_HEAD_DWORDS * 4
    return [
        npu_writebd(
            column,
            K_SCAN_BD,
            K_SIDE_SCAN_DWORDS,
            k_offset,
            iteration_size=schedule.kv_blocks,
            iteration_stride=CACHE_BLOCK_DWORDS - 1,
        ),
        npu_address_patch(column, K_SCAN_BD, k_arg, k_offset),
        npu_push_queue(column, "MM2S", 0, K_SCAN_BD, repeat_count=schedule.kv_blocks - 1),
        npu_writebd(
            column,
            V_SCAN_BD,
            V_SIDE_SCAN_DWORDS,
            v_offset,
            iteration_size=schedule.kv_blocks,
            iteration_stride=CACHE_BLOCK_DWORDS - 1,
        ),
        npu_address_patch(column, V_SCAN_BD, v_arg, v_offset),
        npu_push_queue(column, "MM2S", 1, V_SCAN_BD, repeat_count=schedule.kv_blocks - 1),
    ]


def _shape_a_multiblock(window: int) -> str:
    tile = shape_a_symbol(window)
    blocks_name = _shape_blocks_name(tile)
    tail_tokens_name = _shape_tail_tokens_name(tile)
    runtime_start = _shape_runtime_start_name(tile)
    return f"""
    %{tile}_q = aie.buffer(%{tile}) {{sym_name = "{tile}_q"}} : memref<{WINDOW_DWORDS}xi32>
    %{tile}_k = aie.buffer(%{tile}) {{sym_name = "{tile}_k"}} : memref<{K_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{blocks_name} = aie.buffer(%{tile}) {{sym_name = "{blocks_name}"}} : memref<1xi32>
    %{tail_tokens_name} = aie.buffer(%{tile}) {{sym_name = "{tail_tokens_name}"}} : memref<1xi32>
{lock_pair(tile, "q", 0)}
{lock_pair(tile, "k", 2)}
{lock_pair(tile, "carrier", 4)}
    %{runtime_start} = aie.lock(%{tile}, 6) {{init = 0 : i32, sym_name = "{runtime_start}"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%{runtime_start}, Acquire, 1)
      %blocks_i32 = memref.load %{blocks_name}[%c0] : memref<1xi32>
      %tail_tokens_i32 = memref.load %{tail_tokens_name}[%c0] : memref<1xi32>
      %blocks = arith.index_cast %blocks_i32 : i32 to index
      %window_i32 = arith.constant {window} : i32
      %q_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %k_dwords_i32 = arith.constant {K_WINDOW_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_q_full, AcquireGreaterEqual, 1)
      scf.for %block = %c0 to %blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%{tile}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
        func.call @attention_kv16_make_carrier_masked(%{tile}_q, %{tile}_k, %{tile}_carrier, %window_i32, %block_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
          : (memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%{tile}_k_empty, Release, 1)
        aie.use_lock(%{tile}_carrier_full, Release, 1)
      }}
      aie.use_lock(%{tile}_q_empty, Release, 1)
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
      aie.dma_bd(%{tile}_k : memref<{K_WINDOW_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = 1 : i32}}
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


def _shape_b_multiblock(window: int) -> str:
    tile = shape_b_symbol(window)
    blocks_name = _shape_blocks_name(tile)
    runtime_start = _shape_runtime_start_name(tile)
    return f"""
    %{tile}_v = aie.buffer(%{tile}) {{sym_name = "{tile}_v"}} : memref<{V_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{tile}_accum = aie.buffer(%{tile}) {{sym_name = "{tile}_accum"}} : memref<{ACCUM_LANES}xi32>
    %{tile}_state = aie.buffer(%{tile}) {{sym_name = "{tile}_state"}} : memref<{SCALAR_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{OUTPUT_DWORDS}xi32>
    %{blocks_name} = aie.buffer(%{tile}) {{sym_name = "{blocks_name}"}} : memref<1xi32>
{lock_pair(tile, "v", 0)}
{lock_pair(tile, "carrier", 2)}
{lock_pair(tile, "output", 4)}
    %{runtime_start} = aie.lock(%{tile}, 6) {{init = 0 : i32, sym_name = "{runtime_start}"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%{runtime_start}, Acquire, 1)
      %blocks_i32 = memref.load %{blocks_name}[%c0] : memref<1xi32>
      %blocks = arith.index_cast %blocks_i32 : i32 to index
      %v_dwords_i32 = arith.constant {V_WINDOW_DWORDS} : i32
      %out_dwords_i32 = arith.constant {OUTPUT_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      %accum_lanes_i32 = arith.constant {ACCUM_LANES} : i32
      %state_dwords_i32 = arith.constant {SCALAR_DWORDS} : i32
      func.call @attention_kv16_init_accum(%{tile}_accum, %{tile}_state, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) -> ()
      scf.for %block = %c0 to %blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%{tile}_v_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
        func.call @attention_kv16_accum_block(%{tile}_v, %{tile}_carrier, %{tile}_accum, %{tile}_state, %block_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
          : (memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%{tile}_v_empty, Release, 1)
        aie.use_lock(%{tile}_carrier_empty, Release, 1)
      }}
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @attention_kv16_finish_accum(%{tile}_accum, %{tile}_state, %{tile}_output, %out_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %v_dma = aie.dma_start(S2MM, 0, ^v_in, ^carrier_start)
    ^v_in:
      aie.use_lock(%{tile}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v : memref<{V_WINDOW_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = 0 : i32}}
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
      aie.dma_bd(%{tile}_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%k_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%v_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%output: memref<{HOST_OUTPUT_DWORDS}xi32>) {{"
    ]
    lines.append(npu_rtp_write("post_current_token", 0, schedule.current_token))
    for window in range(4):
        lines.append(npu_rtp_write(_shape_blocks_name(shape_a_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(_shape_blocks_name(shape_b_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(_shape_tail_tokens_name(shape_a_symbol(window)), 0, schedule.tail_tokens))
    lines.append(npu_set_lock("post_runtime_start", 1))
    for window in range(4):
        lines.append(npu_set_lock(_shape_runtime_start_name(shape_a_symbol(window)), 1))
        lines.append(npu_set_lock(_shape_runtime_start_name(shape_b_symbol(window)), 1))
    lines.extend(_push_current_cache_write(0, 0, schedule))
    lines.extend(_push_current_cache_write(7, 1, schedule))
    lines.extend(
        (
            npu_writebd(1, 13, HOST_OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 2, 0),
            npu_push_queue(1, "S2MM", 1, 13),
        )
    )
    lines.extend(
        (
            npu_sync(0, CURRENT_WRITE_CHANNEL),
            npu_sync(7, CURRENT_WRITE_CHANNEL),
        )
    )
    lines.extend(_push_kv_scan_from_cache(0, 0, 1, 0, schedule))
    lines.extend(_push_kv_scan_from_cache(7, 0, 1, 4, schedule))
    lines.extend(
        (
            npu_sync(0, 0, direction=1),
            npu_sync(0, 1, direction=1),
            npu_sync(7, 0, direction=1),
            npu_sync(7, 1, direction=1),
            npu_sync(1, 1),
        )
    )
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> str:
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
        tile_defs.append(f"    %{shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{shape_b_symbol(window)} = aie.tile({column}, {row})")
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{main_symbol(group, row_idx)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(packet_flow(main_packet(group, row), main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, main_symbol(group, row), 0))
        flows.append(packet_flow(column_packet(group), f"mt{group}", 5, "bridge", group))
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(packet_flow(packet, "bridge", 5, "post", 0))
    flows.extend(
        (
            packet_flow(CURRENT_PACKET_K, "post", 1, "shim_left", 1),
            packet_flow(CURRENT_PACKET_V, "post", 1, "shim_right", 1),
            packet_flow(O_GLOBAL_PACKET_ID, "bridge", 5, "full", 0),
            flow("post", 0, "hub", 0),
            flow("shim_left", 0, "kv_left", 0),
            flow("shim_left", 1, "kv_left", 1),
            flow("shim_right", 0, "kv_right", 0),
            flow("shim_right", 1, "kv_right", 1),
        )
    )
    for window in range(4):
        kv_tile = "kv_left" if window < 2 else "kv_right"
        kv_k_channel = 0 if window in (0, 2) else 2
        kv_v_channel = 1 if window in (0, 2) else 3
        flows.append(flow("hub", window, shape_a_symbol(window), 0))
        flows.append(flow(kv_tile, kv_k_channel, shape_a_symbol(window), 1))
        flows.append(flow(kv_tile, kv_v_channel, shape_b_symbol(window), 0))
        flows.append(flow(shape_a_symbol(window), 0, shape_b_symbol(window), 1))
        flows.append(flow(shape_b_symbol(window), 0, "hub", window + 1))
    flows.extend(
        (
            packet_flow(PACKET_ID_ATTENTION, "hub", 5, "bridge", 4),
            flow("full", 1, "shim_out", 1),
        )
    )

    blocks = [
        qkv_compact_bridge(),
        _postprocess(),
        attention_hub(),
        _kv_split_scan_memtile(0),
        _kv_split_scan_memtile(1),
        qkv_compact_full_vector(),
    ]
    for window in range(4):
        blocks.append(_shape_a_multiblock(window))
        blocks.append(_shape_b_multiblock(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(qkv_compact_column_memtile(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(qkv_compact_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @qkv_emit_qkv_records(memref<{RECORD_DWORDS * 4}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @currentkv_postprocess_payload(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @qkv_main_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @qkv_main_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @qkv_main_emit_o_record(memref<{RECORD_DWORDS * 4}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/debug_contract.o"}}
    func.func private @qkv_c1r2_summarize_compact(memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{HOST_OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @attention_kv16_make_carrier_masked(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_init_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_accum_block(memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_finish_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "currentkv_postprocess_payload",
        "qkv_emit_qkv_records",
        "qkv_main_emit_o_record",
        "attention_kv16_make_carrier_masked",
        "attention_kv16_init_accum",
        "attention_kv16_accum_block",
        "attention_kv16_finish_accum",
        f"aie.packet_flow({CURRENT_PACKET_K})",
        f"aie.packet_flow({CURRENT_PACKET_V})",
        f"pkt_id = {CURRENT_PACKET_K}",
        f"pkt_id = {CURRENT_PACKET_V}",
        f"memref<{CURRENT_DWORDS}xi32>",
        "memref<1xi32>",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{K_SIDE_SCAN_DWORDS}xi32>",
        f"memref<{V_SIDE_SCAN_DWORDS}xi32>",
        "post_current_token",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        f"aiex.npu.rtp_write(@shape_a0_blocks, 0, {schedule.kv_blocks})",
        f"aiex.npu.rtp_write(@shape_b0_blocks, 0, {schedule.kv_blocks})",
        f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
        "aiex.set_lock(%post_runtime_start, 1)",
        "aiex.set_lock(%shape_a0_runtime_start, 1)",
        "aiex.set_lock(%shape_b0_runtime_start, 1)",
        "aie.use_lock(%post_runtime_start, Acquire, 1)",
        "aie.use_lock(%shape_a0_runtime_start, Acquire, 1)",
        "aie.use_lock(%shape_b0_runtime_start, Acquire, 1)",
        "memref.load %shape_a0_blocks[%c0] : memref<1xi32>",
        "memref.load %shape_a0_tail_tokens[%c0] : memref<1xi32>",
        "memref.load %shape_b0_blocks[%c0] : memref<1xi32>",
        "arith.index_cast %blocks_i32 : i32 to index",
        "aie.packet_dest<%shim_left, DMA : 1>",
        "aie.packet_dest<%shim_right, DMA : 1>",
        "aie.flow(%shim_left, DMA : 1, %kv_left, DMA : 1)",
        "aie.flow(%shim_right, DMA : 1, %kv_right, DMA : 1)",
        f"buffer_length = {K_SIDE_SCAN_DWORDS} : i32",
        f"iteration_size = {schedule.kv_blocks} : i32",
        f"iteration_stride = {CACHE_BLOCK_DWORDS - 1} : i32",
        f"repeat_count = {schedule.kv_blocks - 1} : i32",
        f"d0_size = {HEAD_DWORDS // 2} : i32",
        "d0_stride = 1 : i32",
        f"d1_size = {KV_HEADS} : i32",
        f"d1_stride = {WINDOW_HEAD_DWORDS - 1} : i32",
        "use_next_bd = 1",
        "slot0_empty",
        "slot3_full",
    )
    errors = [f"missing currentkv kvscan marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                "aie.use_lock(%post_runtime_start, Acquire, 1)",
                "func.call @currentkv_postprocess_payload",
            ),
        )
    )
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
                f"aiex.npu.rtp_write(@shape_a0_blocks, 0, {schedule.kv_blocks})",
                f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
                "aiex.set_lock(%post_runtime_start, 1)",
                "aiex.set_lock(%shape_a0_runtime_start, 1)",
            ),
        )
    )
    if f"%blocks = arith.constant {schedule.kv_blocks} : index" in mlir:
        errors.append("Shape block loop count must be runtime RTP, not a core constant")
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 7
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(
        require_count(
            CASE_NAME,
            "main activation bridge flow",
            mlir.count("aie.flow(%bridge, DMA : 1"),
            len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        )
    )
    errors.extend(require_count(CASE_NAME, "qkv producer", mlir.count("qkv_emit_qkv_records"), 17))
    errors.extend(require_count(CASE_NAME, "qkv O producer", mlir.count("qkv_main_emit_o_record"), 17))
    errors.extend(require_count(CASE_NAME, "currentkv postprocess", mlir.count("currentkv_postprocess_payload"), 2))
    errors.extend(
        require_count(
            CASE_NAME,
            "attention_kv16_make_carrier_masked",
            mlir.count("attention_kv16_make_carrier_masked"),
            5,
        )
    )
    errors.extend(require_count(CASE_NAME, "attention_kv16_init_accum", mlir.count("attention_kv16_init_accum"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_accum_block", mlir.count("attention_kv16_accum_block"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_finish_accum", mlir.count("attention_kv16_finish_accum"), 5))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
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
    errors.extend(require_no_compute_kv_materialization(CASE_NAME, mlir, KV_SIDE_DWORDS, schedule.kv_cache_dwords * 2))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qkv_postprocess_payload",
                "qkv_split_kv_payload",
                "post_kv_payload",
                "current_token_i32",
                "aie.packet_flow(14)",
                "aie.packet_flow(15)",
                "pkt_id = 14",
                "pkt_id = 15",
            ),
        )
    )
    if MAIN_PACKET_BASE != 16:
        errors.append("main packet base mismatch")
    if KV_OUT_BDS != (2, 24, 4, 26):
        errors.append("KV memtile output BD contract mismatch")
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SCAN_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, CURRENT_WRITE_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SPLIT_K_IN_BDS + KV_SPLIT_V_IN_BDS + KV_OUT_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, KV_SCAN_BDS, CURRENT_WRITE_BDS))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    return errors
