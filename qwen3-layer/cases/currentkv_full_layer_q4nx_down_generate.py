"""Generate MLIR-AIE for current K/V full-layer tail with Q4NX up/gate and down."""

from __future__ import annotations

from pathlib import Path

from attention_dataflow import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    shape_a_symbol,
    shape_b_symbol,
)
from contract import (
    CHUNK_BF16,
    C1R2_QKV_REPLAYS,
    C1R2_PACKET_DWORDS,
    C1R2_UPGATE_REPLAYS,
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    SHAPE_CARRIER_DWORDS,
)
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
from compact_dataflow import (
    BODY_RECORD_SLOTS,
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    DOWN_ACT_PACKET_ID,
    DOWN_GLOBAL_PACKET_ID,
    FFN_GLOBAL_PACKET_ID,
    FULL_REPLAY_PACKET_ID,
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    MAIN_RECORD_DWORDS,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    TOTAL_MAIN_CHUNKS,
    UPGATE_MAIN_RECORD_DWORDS,
    V_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _bd_dimensions,
    _bridge,
    _compact_phase,
    _hub,
    _main_record_transfer,
    _main_symbol,
    _next_phase,
    _phase_trace_errors,
    _phase_trace_marker,
    column_packet,
    main_packet,
    q4nx_weight_column_memtile,
)
from physical_contract import validate_q4nx_down_full_layer_ownership
from projection_schedule import (
    DOWN_CHUNKS,
    DOWN_BODY_RECORDS,
    DOWN_WEIGHT_CHUNKS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_TOTAL_WEIGHT_CHUNKS,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    O_BODY_RECORDS,
    O_CHUNKS_PER_RECORD,
    O_WEIGHT_CHUNKS,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    UPGATE_CHUNKS_PER_REPLAY,
    UPGATE_WEIGHT_CHUNKS,
    V_CHUNKS_PER_RECORD,
    V_WEIGHT_CHUNK_BASE,
)
from cases.currentkv_full_layer_q4nx_down_reference import (
    CASE_NAME,
    COLUMN_WEIGHT_BF16,
    DEFAULT_SCHEDULE,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    PATCH_WEIGHT_BF16,
    TOTAL_WEIGHT_I32,
)
from cases.currentkv_kvscan_attention_kv16_generate import (
    CURRENT_WRITE_BDS,
    CURRENT_WRITE_CHANNEL,
    K_SCAN_BD,
    K_SIDE_SCAN_DWORDS,
    KV_SCAN_BDS,
    KV_SPLIT_K_IN_BDS,
    KV_SPLIT_V_IN_BDS,
    V_SCAN_BD,
    V_SIDE_SCAN_DWORDS,
    _kv_split_scan_memtile,
    _push_current_cache_write,
    _push_kv_scan_from_cache,
    _shape_a_multiblock,
    _shape_blocks_name,
    _shape_runtime_start_name,
    _shape_tail_tokens_name,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    ACCUM_LANES,
    CACHE_BLOCK_DWORDS,
    CURRENT_DWORDS,
    CURRENT_PACKET_K,
    CURRENT_PACKET_V,
    DecodeSchedule,
    K_CACHE_SIDE_DWORDS,
    K_WINDOW_DWORDS,
    KV_SIDE_DWORDS,
    SCALAR_DWORDS,
    V_CACHE_SIDE_DWORDS,
    V_WINDOW_DWORDS,
)
from cases.kvscan_attention_kv16_reference import OUTPUT_DWORDS as ATTENTION_OUTPUT_DWORDS
from cases.kvscan_attention_kv16_reference import WEIGHT_DWORDS, WINDOW_DWORDS

QKV_BODY_PHASE_TRACE = (
    _compact_phase("q", "q", BODY_RECORD_SLOTS[0], Q_BODY_RECORDS),
    _compact_phase("k", "k", BODY_RECORD_SLOTS[1], KV_BODY_RECORDS),
    _compact_phase("v", "v", BODY_RECORD_SLOTS[2], KV_BODY_RECORDS),
    _compact_phase("o", "o", BODY_RECORD_SLOTS[3], O_BODY_RECORDS),
    _compact_phase("upgate", "upgate", BODY_RECORD_SLOTS[4]),
    _compact_phase("down", "down", BODY_RECORD_SLOTS[5], DOWN_BODY_RECORDS),
)
Q_MAIN_RECORD_DWORDS = Q_BODY_RECORDS * RECORD_DWORDS
KV_MAIN_RECORD_DWORDS = KV_BODY_RECORDS * RECORD_DWORDS


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%k_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%v_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%output: memref<{OUTPUT_DWORDS}xi32>, "
        f"%hidden: memref<{HIDDEN_DWORDS}xi32>) {{"
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
            npu_writebd(1, 13, OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 3, 0),
            npu_push_queue(1, "S2MM", 1, 13),
            npu_writebd(1, 12, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 12, 4, 0),
            npu_push_queue(1, "MM2S", 0, 12),
        )
    )
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = group * COLUMN_WEIGHT_BF16 * 2
        patch_bytes = PATCH_WEIGHT_BF16 * 2
        lines.extend(
            (
                npu_writebd(column, 0, PATCH_WEIGHT_BF16 // 2, column_base),
                npu_address_patch(column, 0, 2, column_base),
                npu_writebd(column, 1, PATCH_WEIGHT_BF16 // 2, column_base + patch_bytes),
                npu_address_patch(column, 1, 2, column_base + patch_bytes),
                npu_push_queue(column, "MM2S", 0, 0),
                npu_push_queue(column, "MM2S", 1, 1),
            )
        )
    lines.extend((npu_sync(0, CURRENT_WRITE_CHANNEL), npu_sync(7, CURRENT_WRITE_CHANNEL)))
    lines.extend(_push_kv_scan_from_cache(0, 0, 1, 0, schedule))
    lines.extend(_push_kv_scan_from_cache(7, 0, 1, 4, schedule))
    lines.extend(
        (
            npu_sync(0, 0, direction=1),
            npu_sync(0, 1, direction=1),
            npu_sync(7, 0, direction=1),
            npu_sync(7, 1, direction=1),
        )
    )
    for column in MAIN_COLUMNS:
        lines.extend((npu_sync(column, 0, direction=1), npu_sync(column, 1, direction=1)))
    lines.append(npu_sync(1, 1))
    lines.append(npu_sync(1, 0, direction=1))
    lines.append("    }")
    return "\n".join(lines)


def _shape_b_multiblock_bf16(window: int) -> str:
    tile = shape_b_symbol(window)
    blocks_name = _shape_blocks_name(tile)
    runtime_start = _shape_runtime_start_name(tile)
    return f"""
    %{tile}_v = aie.buffer(%{tile}) {{sym_name = "{tile}_v"}} : memref<{V_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{tile}_accum = aie.buffer(%{tile}) {{sym_name = "{tile}_accum"}} : memref<{ACCUM_LANES}xi32>
    %{tile}_state = aie.buffer(%{tile}) {{sym_name = "{tile}_state"}} : memref<{SCALAR_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{ATTENTION_OUTPUT_DWORDS}xi32>
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
      %out_dwords_i32 = arith.constant {ATTENTION_OUTPUT_DWORDS} : i32
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
      func.call @attention_kv16_finish_accum_bf16(%{tile}_accum, %{tile}_state, %{tile}_output, %out_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{ATTENTION_OUTPUT_DWORDS}xi32>, i32, i32, i32) -> ()
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
      aie.dma_bd(%{tile}_output : memref<{ATTENTION_OUTPUT_DWORDS}xi32>, 0, {ATTENTION_OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _postprocess_qkv_body() -> str:
    return f"""
    %post_q_compact = aie.buffer(%post) {{sym_name = "post_q_compact"}} : memref<{Q_DWORDS}xi32>
    %post_k_compact = aie.buffer(%post) {{sym_name = "post_k_compact"}} : memref<{CURRENT_DWORDS}xi32>
    %post_v_compact = aie.buffer(%post) {{sym_name = "post_v_compact"}} : memref<{CURRENT_DWORDS}xi32>
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
      func.call @currentkv_postprocess_q4nx_body_payload(%post_q_compact, %post_k_compact, %post_v_compact, %post_q_payload, %post_current_k, %post_current_v, %post_current_token, %q_dwords_i32, %current_dwords_i32)
        : (memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) -> ()
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
      aie.dma_bd(%post_q_compact : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%post_q_compact_full, Release, 1)
      aie.next_bd ^k_in
    ^k_in:
      aie.use_lock(%post_k_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_k_compact : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%post_k_compact_full, Release, 1)
      aie.next_bd ^v_in
    ^v_in:
      aie.use_lock(%post_v_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_v_compact : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 2 : i32}}
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


def _record_buffer_ref(tile: str, buffer_name: str) -> str:
    if buffer_name == "q_records":
        return f"%{tile}_q_records : memref<{Q_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "k_records":
        return f"%{tile}_k_records : memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "v_records":
        return f"%{tile}_v_records : memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "o_records":
        return f"%{tile}_o_records : memref<{O_BODY_RECORDS * RECORD_DWORDS}xi32>"
    if buffer_name == "upgate_records":
        return f"%{tile}_upgate_records : memref<{UPGATE_MAIN_RECORD_DWORDS}xi32>"
    if buffer_name == "down_records":
        return f"%{tile}_down_records : memref<{DOWN_BODY_RECORDS * RECORD_DWORDS}xi32>"
    return f"%{tile}_records : memref<{MAIN_RECORD_DWORDS}xi32>"


def _q4nx_body_phase_kernel(
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


def _q4nx_multiblock_phase_kernel(
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
      %{phase_label}_mb_blocks = arith.constant {records} : index
      %{phase_label}_mb_chunks = arith.constant {chunks_per_record} : index
      %{phase_label}_mb_blocks_i32 = arith.constant {records} : i32
      %{phase_label}_mb_weight_base_i32 = arith.constant {weight_base} : i32
      func.call @q4nx_clear_block_summaries(%{phase_label}_mb_blocks_i32, %m_i32)
        : (i32, i32) -> ()
      scf.for %{phase_label}_chunk = %c0 to %{phase_label}_mb_chunks step %c1 {{
        %{phase_label}_chunk_i32 = arith.index_cast %{phase_label}_chunk : index to i32
        %{phase_label}_chunk_rem = arith.remsi %{phase_label}_chunk_i32, %c2_i32 : i32
        %{phase_label}_activation_is_pong = arith.cmpi eq, %{phase_label}_chunk_rem, %c1_i32 : i32
        aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
        scf.for %{phase_label}_block = %c0 to %{phase_label}_mb_blocks step %c1 {{
          %{phase_label}_block_i32 = arith.index_cast %{phase_label}_block : index to i32
          %{phase_label}_stream_base = arith.muli %{phase_label}_chunk, %{phase_label}_mb_blocks : index
          %{phase_label}_stream_local = arith.addi %{phase_label}_stream_base, %{phase_label}_block : index
          %{phase_label}_stream_i32 = arith.index_cast %{phase_label}_stream_local : index to i32
          %{phase_label}_weight_chunk = arith.addi %{phase_label}_stream_i32, %{phase_label}_mb_weight_base_i32 : i32
          %{phase_label}_weight_rem = arith.remsi %{phase_label}_weight_chunk, %c2_i32 : i32
          %{phase_label}_weight_is_pong = arith.cmpi eq, %{phase_label}_weight_rem, %c1_i32 : i32
          aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
          scf.if %{phase_label}_activation_is_pong {{
            scf.if %{phase_label}_weight_is_pong {{
              func.call @q4nx_chunk_accum_block_slice_i32(%{tile}_wt_pong, %{tile}_chunk_pong, %{phase_label}_block_i32, %m_i32)
                : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) -> ()
            }} else {{
              func.call @q4nx_chunk_accum_block_slice_i32(%{tile}_wt_ping, %{tile}_chunk_pong, %{phase_label}_block_i32, %m_i32)
                : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) -> ()
            }}
          }} else {{
            scf.if %{phase_label}_weight_is_pong {{
              func.call @q4nx_chunk_accum_block_slice_i32(%{tile}_wt_pong, %{tile}_chunk_ping, %{phase_label}_block_i32, %m_i32)
                : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) -> ()
            }} else {{
              func.call @q4nx_chunk_accum_block_slice_i32(%{tile}_wt_ping, %{tile}_chunk_ping, %{phase_label}_block_i32, %m_i32)
                : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) -> ()
            }}
          }}
          aie.use_lock(%{tile}_wt_empty, Release, 1)
        }}
        aie.use_lock(%{tile}_chunk_empty, Release, 1)
      }}
      scf.for %{phase_label}_emit_block = %c0 to %{phase_label}_mb_blocks step %c1 {{
        %{phase_label}_emit_block_i32 = arith.index_cast %{phase_label}_emit_block : index to i32
        func.call @q4nx_flush_block_output(%{tile}_q4nx_output, %{phase_label}_emit_block_i32, %m_i32)
          : (memref<32xbf16>, i32, i32) -> ()
        func.call @{emit_name}({buffer_name}, %{tile}_q4nx_output, %group_i32, %row_i32, %{phase_label}_emit_block_i32, %m_i32)
          : ({buffer_type}, memref<32xbf16>, i32, i32, i32, i32) -> ()
      }}"""


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    packet = main_packet(group, row)
    record_blocks = []
    for stage_idx, phase in enumerate(QKV_BODY_PHASE_TRACE):
        buffer_name, offset, length, dimensions = _main_record_transfer(phase, row)
        buffer_ref = _record_buffer_ref(tile, buffer_name)
        next_label = f"^{_next_phase(QKV_BODY_PHASE_TRACE, stage_idx).label}_out"
        record_blocks.append(f"""    ^{phase.label}_out:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer_ref}, {offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = {2 + stage_idx} : i32, next_bd_id = {2 + ((stage_idx + 1) % len(QKV_BODY_PHASE_TRACE))} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd {next_label}""")

    q_name = f"%{tile}_q_records"
    q_type = f"memref<{Q_MAIN_RECORD_DWORDS}xi32>"
    k_name = f"%{tile}_k_records"
    k_type = f"memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    v_name = f"%{tile}_v_records"
    v_type = f"memref<{KV_MAIN_RECORD_DWORDS}xi32>"
    o_name = f"%{tile}_o_records"
    o_type = f"memref<{O_BODY_RECORDS * RECORD_DWORDS}xi32>"
    down_name = f"%{tile}_down_records"
    down_type = f"memref<{DOWN_BODY_RECORDS * RECORD_DWORDS}xi32>"
    return f"""
    %{tile}_q_records = aie.buffer(%{tile}) {{sym_name = "{tile}_q_records"}} : memref<{Q_MAIN_RECORD_DWORDS}xi32>
    %{tile}_k_records = aie.buffer(%{tile}) {{sym_name = "{tile}_k_records"}} : memref<{KV_MAIN_RECORD_DWORDS}xi32>
    %{tile}_v_records = aie.buffer(%{tile}) {{sym_name = "{tile}_v_records"}} : memref<{KV_MAIN_RECORD_DWORDS}xi32>
    %{tile}_o_records = aie.buffer(%{tile}) {{sym_name = "{tile}_o_records"}} : memref<{O_BODY_RECORDS * RECORD_DWORDS}xi32>
    %{tile}_records = aie.buffer(%{tile}) {{sym_name = "{tile}_records"}} : memref<{MAIN_RECORD_DWORDS}xi32>
    %{tile}_upgate_records = aie.buffer(%{tile}) {{sym_name = "{tile}_upgate_records"}} : memref<{UPGATE_MAIN_RECORD_DWORDS}xi32>
    %{tile}_down_records = aie.buffer(%{tile}) {{sym_name = "{tile}_down_records"}} : memref<{DOWN_BODY_RECORDS * RECORD_DWORDS}xi32>
    %{tile}_chunk_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_ping"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_chunk_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk_pong"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_q4nx_output = aie.buffer(%{tile}) {{sym_name = "{tile}_q4nx_output"}} : memref<32xbf16>
    %{tile}_records_empty = aie.lock(%{tile}, 0) {{init = {len(QKV_BODY_PHASE_TRACE)} : i32, sym_name = "{tile}_records_empty"}}
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
      %upgate_replays = arith.constant {C1R2_UPGATE_REPLAYS} : index
      %chunks_per_replay = arith.constant {UPGATE_CHUNKS_PER_REPLAY} : index
      %m_i32 = arith.constant 32 : i32
      %upgate_weight_chunk_base_i32 = arith.constant {FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 3)
{_q4nx_body_phase_kernel(tile, "q", "q4nx_emit_q_body_record", Q_BODY_RECORDS, Q_CHUNKS_PER_RECORD, Q_WEIGHT_CHUNK_BASE, q_name, q_type)}
{_q4nx_body_phase_kernel(tile, "k", "q4nx_emit_k_body_record", KV_BODY_RECORDS, K_CHUNKS_PER_RECORD, K_WEIGHT_CHUNK_BASE, k_name, k_type)}
{_q4nx_body_phase_kernel(tile, "v", "q4nx_emit_v_body_record", KV_BODY_RECORDS, V_CHUNKS_PER_RECORD, V_WEIGHT_CHUNK_BASE, v_name, v_type)}
      aie.use_lock(%{tile}_records_full, Release, 3)

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
{_q4nx_multiblock_phase_kernel(tile, "o", "q4nx_emit_o_body_record", O_BODY_RECORDS, O_CHUNKS_PER_RECORD, FULL_LAYER_O_WEIGHT_CHUNK_BASE, o_name, o_type)}
      aie.use_lock(%{tile}_records_full, Release, 1)

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
      scf.for %replay = %c0 to %upgate_replays step %c1 {{
        %replay_i32 = arith.index_cast %replay : index to i32
        func.call @clear_summary(%{tile}_q4nx_output, %m_i32)
          : (memref<32xbf16>, i32) -> ()
        scf.for %chunk = %c0 to %chunks_per_replay step %c1 {{
          %base_chunk = arith.muli %replay, %chunks_per_replay : index
          %global_chunk = arith.addi %base_chunk, %chunk : index
          %global_chunk_i32 = arith.index_cast %global_chunk : index to i32
          %weight_chunk = arith.addi %global_chunk_i32, %upgate_weight_chunk_base_i32 : i32
          %rem = arith.remsi %weight_chunk, %c2_i32 : i32
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
        func.call @q4nx_emit_upgate_record(%{tile}_upgate_records, %{tile}_q4nx_output, %group_i32, %row_i32, %replay_i32, %m_i32)
          : (memref<{UPGATE_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) -> ()
      }}
      aie.use_lock(%{tile}_records_full, Release, 1)

      aie.use_lock(%{tile}_records_empty, AcquireGreaterEqual, 1)
{_q4nx_multiblock_phase_kernel(tile, "down", "q4nx_emit_down_body_record", DOWN_BODY_RECORDS, DOWN_CHUNKS, FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE, down_name, down_type)}
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
{chr(10).join(record_blocks)}
    ^end:
      aie.end
    }}
"""


def _full_vector_q4nx_output() -> str:
    replay_payload_dwords = C1R2_PACKET_DWORDS - 1
    compact_dwords = DOWN_BODY_RECORDS * COMPACT_PACKET_DWORDS
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{compact_dwords}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{C1R2_PACKET_DWORDS}xi32>
    %full_output = aie.buffer(%full) {{sym_name = "full_output"}} : memref<{OUTPUT_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}
    %full_compact_empty = aie.lock(%full, 2) {{init = 1 : i32, sym_name = "full_compact_empty"}}
    %full_compact_full = aie.lock(%full, 3) {{init = 0 : i32, sym_name = "full_compact_full"}}
    %full_replay_empty = aie.lock(%full, 4) {{init = 1 : i32, sym_name = "full_replay_empty"}}
    %full_replay_full = aie.lock(%full, 5) {{init = 0 : i32, sym_name = "full_replay_full"}}
    %full_output_empty = aie.lock(%full, 6) {{init = 1 : i32, sym_name = "full_output_empty"}}
    %full_output_full = aie.lock(%full, 7) {{init = 0 : i32, sym_name = "full_output_full"}}

    %full_core = aie.core(%full) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %qkv_replays = arith.constant {C1R2_QKV_REPLAYS} : index
      %replays = arith.constant {C1R2_UPGATE_REPLAYS} : index
      %payload_i32 = arith.constant {replay_payload_dwords} : i32
      %compact_i32 = arith.constant {compact_dwords} : i32
      %blocks_i32 = arith.constant {DOWN_BODY_RECORDS} : i32

      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      scf.for %replay = %c0 to %qkv_replays step %c1 {{
        aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
        func.call @full_c1r2_copy_hidden_replay(%full_hidden, %full_replay, %payload_i32)
          : (memref<{HIDDEN_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32) -> ()
        aie.use_lock(%full_replay_full, Release, 1)
      }}
      aie.use_lock(%full_hidden_empty, Release, 1)

      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      scf.for %replay = %c0 to %replays step %c1 {{
        %replay_i32 = arith.index_cast %replay : index to i32
        aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
        func.call @full_c1r2_make_replay_from_o_compacts(%full_compact, %full_replay, %replay_i32, %payload_i32, %blocks_i32)
          : (memref<{compact_dwords}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32, i32) -> ()
        aie.use_lock(%full_replay_full, Release, 1)
      }}
      aie.use_lock(%full_compact_empty, Release, 1)

      aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_output_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_output_from_down_compacts(%full_compact, %full_output, %compact_i32, %blocks_i32)
        : (memref<{compact_dwords}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%full_compact_empty, Release, 1)
      aie.use_lock(%full_output_full, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^compact_in, ^hidden_in_start)
    ^compact_in:
      aie.use_lock(%full_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_compact : memref<{compact_dwords}xi32>, 0, {compact_dwords}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_compact_full, Release, 1)
      aie.next_bd ^compact_in

    ^hidden_in_start:
      %hidden_dma = aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%full_hidden_full, Release, 1)
      aie.next_bd ^hidden_in

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^output_out_start)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{C1R2_PACKET_DWORDS}xi32>, 1, {replay_payload_dwords}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^replay_out

    ^output_out_start:
      %output_dma = aie.dma_start(MM2S, 0, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%full_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%full_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _swiglu_bf16() -> str:
    return f"""
    %swiglu_input = aie.buffer(%swiglu) {{sym_name = "swiglu_input"}} : memref<{C6R2_INPUT_DWORDS}xi32>
    %swiglu_output = aie.buffer(%swiglu) {{sym_name = "swiglu_output"}} : memref<{C6R2_HALF_DWORDS * 2}xbf16>
    %swiglu_input_empty = aie.lock(%swiglu, 0) {{init = 2 : i32, sym_name = "swiglu_input_empty"}}
    %swiglu_input_full = aie.lock(%swiglu, 1) {{init = 0 : i32, sym_name = "swiglu_input_full"}}
    %swiglu_output_empty = aie.lock(%swiglu, 2) {{init = 1 : i32, sym_name = "swiglu_output_empty"}}
    %swiglu_output_full = aie.lock(%swiglu, 3) {{init = 0 : i32, sym_name = "swiglu_output_full"}}

    %swiglu_core = aie.core(%swiglu) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %repeats = arith.constant {C1R2_UPGATE_REPLAYS // 2} : index
      %dwords_i32 = arith.constant {C6R2_INPUT_DWORDS} : i32
      scf.for %repeat = %c0 to %repeats step %c1 {{
        %slice_i32 = arith.index_cast %repeat : index to i32
        aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)
        aie.use_lock(%swiglu_output_empty, AcquireGreaterEqual, 1)
        func.call @ffn_swiglu_slice_bf16_inputs(%swiglu_input, %swiglu_output, %dwords_i32, %slice_i32)
          : (memref<{C6R2_INPUT_DWORDS}xi32>, memref<{C6R2_HALF_DWORDS * 2}xbf16>, i32, i32) -> ()
        aie.use_lock(%swiglu_output_full, Release, 1)
        aie.use_lock(%swiglu_input_empty, Release, 2)
      }}
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
      aie.dma_bd(%swiglu_output : memref<{C6R2_HALF_DWORDS * 2}xbf16>, 0, {C6R2_HALF_DWORDS * 2}) {{bd_id = 2 : i32}}
      aie.use_lock(%swiglu_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}
"""


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
        "    %swiglu = aie.tile(6, 2)",
        "    %shim_right = aie.tile(7, 0)",
        "    %kv_right = aie.tile(7, 1)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_A_TILES):
        tile_defs.append(f"    %{shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{shape_b_symbol(window)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            flows.append(packet_flow(main_packet(group, row), _main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, _main_symbol(group, row), 1))
        flows.append(packet_flow(column_packet(group), f"mt{group}", 5, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
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
            packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
            packet_flow(FFN_GLOBAL_PACKET_ID, "bridge", 5, "swiglu", 0),
            flow("swiglu", 1, "hub", 5),
            packet_flow(DOWN_ACT_PACKET_ID, "hub", 5, "bridge", 4),
            packet_flow(DOWN_GLOBAL_PACKET_ID, "bridge", 5, "full", 0),
            flow("shim_out", 0, "full", 1),
            flow("full", 0, "shim_out", 1),
        )
    )

    blocks = [
        _bridge(QKV_BODY_PHASE_TRACE),
        _postprocess_qkv_body(),
        _hub(),
        _kv_split_scan_memtile(0),
        _kv_split_scan_memtile(1),
        _full_vector_q4nx_output(),
        _swiglu_bf16(),
    ]
    for window in range(4):
        blocks.append(_shape_a_multiblock(window))
        blocks.append(_shape_b_multiblock_bf16(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, QKV_BODY_PHASE_TRACE))
        for row in range(len(MAIN_ROWS)):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @currentkv_postprocess_q4nx_body_payload(memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @full_c1r2_make_replay_from_o_compacts(memref<{O_BODY_RECORDS * COMPACT_PACKET_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_output_from_down_compacts(memref<{DOWN_BODY_RECORDS * COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_copy_hidden_replay(memref<{HIDDEN_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @ffn_swiglu_slice_bf16_inputs(memref<{C6R2_INPUT_DWORDS}xi32>, memref<{C6R2_HALF_DWORDS * 2}xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/swiglu.o"}}
    func.func private @attention_kv16_make_carrier_masked(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_init_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_accum_block(memref<{V_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @attention_kv16_finish_accum_bf16(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{ATTENTION_OUTPUT_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @clear_summary(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_chunk_accum_slice_i32(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_clear_block_summaries(i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_chunk_accum_block_slice_i32(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_flush_block_output(memref<32xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_flush_output(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_q_body_record(memref<{Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_k_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_v_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_o_body_record(memref<{O_BODY_RECORDS * RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_upgate_record(memref<{UPGATE_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}
    func.func private @q4nx_emit_down_body_record(memref<{DOWN_BODY_RECORDS * RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx.o"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(QKV_BODY_PHASE_TRACE)}",
        "currentkv_postprocess_q4nx_body_payload",
        f"aie.dma_bd(%post_q_compact : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS})",
        f"aie.dma_bd(%post_k_compact : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS})",
        "attention_kv16_make_carrier_masked",
        "attention_kv16_finish_accum_bf16",
        "full_c1r2_copy_hidden_replay",
        "full_c1r2_make_replay_from_o_compacts",
        "q4nx_emit_q_body_record",
        "q4nx_emit_k_body_record",
        "q4nx_emit_v_body_record",
        "q4nx_emit_o_body_record",
        "q4nx_emit_upgate_record",
        "ffn_swiglu_slice_bf16_inputs",
        "full_c1r2_output_from_down_compacts",
        "q4nx_chunk_accum_slice_i32",
        "q4nx_clear_block_summaries",
        "q4nx_chunk_accum_block_slice_i32",
        "q4nx_flush_block_output",
        "q4nx_emit_down_body_record",
        "main_projection_q4nx.o",
        f"%c{Q_WEIGHT_CHUNK_BASE}_i32 = arith.constant {Q_WEIGHT_CHUNK_BASE} : i32",
        f"%c{K_WEIGHT_CHUNK_BASE}_i32 = arith.constant {K_WEIGHT_CHUNK_BASE} : i32",
        f"%c{V_WEIGHT_CHUNK_BASE}_i32 = arith.constant {V_WEIGHT_CHUNK_BASE} : i32",
        f"%o_mb_weight_base_i32 = arith.constant {FULL_LAYER_O_WEIGHT_CHUNK_BASE} : i32",
        f"%upgate_weight_chunk_base_i32 = arith.constant {FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE} : i32",
        f"%down_mb_weight_base_i32 = arith.constant {FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE} : i32",
        f"aie.packet_flow({CURRENT_PACKET_K})",
        f"aie.packet_flow({CURRENT_PACKET_V})",
        f"aie.packet_flow({FFN_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({DOWN_GLOBAL_PACKET_ID})",
        f"pkt_id = {CURRENT_PACKET_K}",
        f"pkt_id = {CURRENT_PACKET_V}",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        f"aiex.npu.rtp_write(@shape_a0_blocks, 0, {schedule.kv_blocks})",
        f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
        "main_projection_q4nx.o",
        "postprocess_qkv.o",
        "full_vector_station.o",
        "swiglu.o",
        "edge_attention.o",
        "aiex.set_lock(%post_runtime_start, 1)",
        "aiex.set_lock(%shape_a0_runtime_start, 1)",
        f"iteration_size = {schedule.kv_blocks} : i32",
        f"iteration_stride = {CACHE_BLOCK_DWORDS - 1} : i32",
        f"repeat_count = {schedule.kv_blocks - 1} : i32",
        "aie.dma_start(S2MM, 4, ^patch0_q4nx_ping, ^patch1_start)",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping, ^wt_row0_start)",
        "aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)",
        "aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)",
        "arg_idx = 3 : i32",
        "arg_idx = 2 : i32",
        "arg_idx = 4 : i32",
    )
    errors = [f"missing currentkv full-layer q4nx marker: {marker}" for marker in required if marker not in mlir]
    if "qwen3_layer.o" in mlir:
        errors.append("currentkv full-layer must not link the old mixed qwen3_layer object")
    if "qwen3_bridge.o" in mlir:
        errors.append("currentkv full-layer must not link the old mixed qwen3_bridge object")
    if "debug_contract.o" in mlir:
        errors.append("currentkv full-layer must not link debug_contract.o")
    errors.extend(_phase_trace_errors(QKV_BODY_PHASE_TRACE))
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
                f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
                "aiex.set_lock(%post_runtime_start, 1)",
                "aiex.set_lock(%shape_a0_runtime_start, 1)",
            ),
        )
    )
    expected_packets = len(MAIN_COLUMNS) * (len(MAIN_ROWS) + 1) + 11
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "q4nx q emit calls", mlir.count("func.call @q4nx_emit_q_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx k emit calls", mlir.count("func.call @q4nx_emit_k_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx v emit calls", mlir.count("func.call @q4nx_emit_v_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "qkv q-loop index constants", mlir.count("%c8 = arith.constant 8 : index"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "qkv chunk-loop index constants", mlir.count("%c16 = arith.constant 16 : index"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "currentkv q4nx body postprocess", mlir.count("currentkv_postprocess_q4nx_body_payload"), 2))
    errors.extend(require_count(CASE_NAME, "attention_kv16_make_carrier_masked", mlir.count("attention_kv16_make_carrier_masked"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_init_accum", mlir.count("attention_kv16_init_accum"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_accum_block", mlir.count("attention_kv16_accum_block"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_finish_accum_bf16", mlir.count("attention_kv16_finish_accum_bf16"), 5))
    errors.extend(require_count(CASE_NAME, "q4nx o emit calls", mlir.count("func.call @q4nx_emit_o_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx upgate emit calls", mlir.count("func.call @q4nx_emit_upgate_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx single-accum chunk call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32"), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 8))
    errors.extend(require_count(CASE_NAME, "q4nx block-accum chunk call sites", mlir.count("func.call @q4nx_chunk_accum_block_slice_i32"), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 8))
    errors.extend(require_count(CASE_NAME, "q4nx down emit calls", mlir.count("func.call @q4nx_emit_down_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "weight arg2 address patches", mlir.count("arg_idx = 2 : i32"), 8))
    errors.extend(require_count(CASE_NAME, "output arg3 address patches", mlir.count("arg_idx = 3 : i32"), 1))
    errors.extend(require_count(CASE_NAME, "hidden arg4 address patches", mlir.count("arg_idx = 4 : i32"), 1))
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
            SCALAR_DWORDS + WEIGHT_DWORDS,
            WEIGHT_DWORDS,
            SCALAR_DWORDS,
            WEIGHT_DWORDS * 8,
        )
    )
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 4))
    errors.extend(require_no_compute_kv_materialization(CASE_NAME, mlir, KV_SIDE_DWORDS, schedule.kv_cache_dwords * 2))
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SCAN_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, CURRENT_WRITE_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SPLIT_K_IN_BDS + KV_SPLIT_V_IN_BDS + KV_OUT_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, KV_SCAN_BDS, CURRENT_WRITE_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, BRIDGE_PACKET_IN_BDS, BRIDGE_PACKET_OUT_BDS))
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qkv_postprocess_payload",
                "qkv_split_kv_payload",
                "currentkv_postprocess_payload",
                "currentkv_postprocess_body_payload",
                "qkv_emit_qkv_body_records",
                "qkv_emit_qkv_records",
                "c1r2_main_accum_chunk",
                "func.call @attention_kv16_finish_accum(",
                "qkv_main_init_summary",
                "qkv_main_accum_chunk",
                "qkv_main_emit_o_record",
                "ffn_swiglu_slice_bf16_inputs_contract",
                "full_main_emit_upgate_slice_record",
                "full_main_init_down_accum",
                "full_main_emit_down_record",
                "aie.packet_flow(3)",
            ),
        )
    )
    errors.extend(validate_q4nx_down_full_layer_ownership(CASE_NAME, mlir))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    compact_bds = tuple(bd for row_bds in COLUMN_RECEIVE_BDS for bd in row_bds) + COMPACT_OUT_BDS
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    if CURRENT_PACKET_K != 8 or CURRENT_PACKET_V != 9:
        errors.append("current K/V packets must stay at 8/9 to avoid full-layer packet14/15 traffic")
    if DOWN_CHUNKS != 48 or TOTAL_MAIN_CHUNKS != 768:
        errors.append("full-layer replay/down chunk contract mismatch")
    if (
        FULL_LAYER_O_WEIGHT_CHUNK_BASE != QKV_BODY_WEIGHT_CHUNKS
        or FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE != FULL_LAYER_O_WEIGHT_CHUNK_BASE + O_WEIGHT_CHUNKS
        or FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE != FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE + UPGATE_WEIGHT_CHUNKS
        or FULL_LAYER_TOTAL_WEIGHT_CHUNKS != QKV_BODY_WEIGHT_CHUNKS + O_WEIGHT_CHUNKS + UPGATE_WEIGHT_CHUNKS + DOWN_WEIGHT_CHUNKS
    ):
        errors.append("full-layer Q4NX weight chunk schedule mismatch")
    if BODY_RECORD_SLOTS != (0, 1, 2, 3, -1, 6):
        errors.append("full-layer q4nx body record slots changed unexpectedly")
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
