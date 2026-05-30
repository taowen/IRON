"""Generate MLIR-AIE for phase-local Qwen3 Q/K/V current-cache writeback."""

from __future__ import annotations

from pathlib import Path

from compact_dataflow import (
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
    _bridge,
    _main_symbol,
    _phase_trace_marker,
    column_packet,
    compact_phase_trace,
    main_packet,
    q4nx_weight_column_memtile,
)
from contract import (
    C1R2_QKV_REPLAYS,
    C1R2_PACKET_DWORDS,
    CHUNK_BF16,
    MAIN_COLUMNS,
    MAIN_ROWS,
    ROWS_PER_COLUMN,
    ROWS_PER_PATCH,
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
    require_dma_bd_next_ids,
    require_dma_bd_lock_balance,
    require_dma_next_bd_labels,
    require_main_record_phase_barrier,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_packet_flows,
    require_unique_bd_ids,
)
from projection_schedule import (
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    Q_BODY_RECORDS,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    V_WEIGHT_CHUNK_BASE,
)
from cases import full_layer_engine_generate as full
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    COLUMN_WEIGHT_BF16,
    HIDDEN_DWORDS,
    PATCH_WEIGHT_BF16,
    QK_ROPE_DWORDS,
    RMS_NORM_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
)
from cases.currentkv_cache_dataflow import (
    CURRENT_WRITE_BDS,
    CURRENT_WRITE_CHANNEL,
    push_current_cache_write,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    CURRENT_DWORDS,
    CURRENT_PACKET_K,
    CURRENT_PACKET_V,
    DecodeSchedule,
    DEFAULT_SCHEDULE,
)
from qkv_compact_reference import Q_DWORDS, RECORD_DWORDS

CASE_NAME = "qwen3-8b-qkv-cache-write-bridge"
QKV_CACHE_PHASE_TRACE = compact_phase_trace(("q", "k", "v"))
QKV_PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * QKV_BODY_WEIGHT_CHUNKS * CHUNK_BF16
Q_MAIN_RECORD_DWORDS = Q_BODY_RECORDS * RECORD_DWORDS
KV_MAIN_RECORD_DWORDS = KV_BODY_RECORDS * RECORD_DWORDS


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%k_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%v_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>, "
        f"%hidden: memref<{HIDDEN_DWORDS}xi32>) {{"
    ]
    lines.append(npu_rtp_write("post_current_token", 0, schedule.current_token))
    lines.extend(
        (
            npu_writebd(1, 10, QK_ROPE_DWORDS, RMS_NORM_DWORDS * 4),
            npu_address_patch(1, 10, 2, RMS_NORM_DWORDS * 4),
            npu_push_queue(1, "MM2S", 1, 10),
        )
    )
    lines.extend(push_current_cache_write(0, 0, schedule))
    lines.extend(push_current_cache_write(7, 1, schedule))
    lines.extend(
        (
            npu_writebd(1, 12, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 12, 3, 0),
            npu_push_queue(1, "MM2S", 0, 12),
            npu_writebd(1, 14, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 14, 2, 0),
            npu_push_queue(1, "MM2S", 0, 14),
        )
    )
    full_patch_bytes = PATCH_WEIGHT_BF16 * 2
    qkv_patch_dwords = QKV_PATCH_WEIGHT_BF16 // 2
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = AUX_DWORDS * 4 + group * COLUMN_WEIGHT_BF16 * 2
        patch1_base = column_base + full_patch_bytes
        lines.extend(
            (
                npu_writebd(column, 0, qkv_patch_dwords, 0),
                npu_address_patch(column, 0, 2, column_base),
                npu_writebd(column, 1, qkv_patch_dwords, 0),
                npu_address_patch(column, 1, 2, patch1_base),
                npu_push_queue(column, "MM2S", 0, 0),
                npu_push_queue(column, "MM2S", 1, 1),
            )
        )
    lines.append(npu_set_lock("post_runtime_start", 1))
    lines.extend((npu_sync(0, CURRENT_WRITE_CHANNEL), npu_sync(7, CURRENT_WRITE_CHANNEL)))
    lines.append(npu_sync(1, 0, direction=1))
    lines.append(npu_sync(1, 1, direction=1))
    for column in MAIN_COLUMNS:
        lines.extend((npu_sync(column, 0, direction=1), npu_sync(column, 1, direction=1)))
    lines.append("    }")
    return "\n".join(lines)


def _input_norm_replay() -> str:
    payload_dwords = C1R2_PACKET_DWORDS - 1
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_norm = aie.buffer(%full) {{sym_name = "full_norm"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{payload_dwords}xi32>
{lock_pair("full", "hidden", 0)}
{lock_pair("full", "norm", 2)}
{lock_pair("full", "replay", 4)}

    %full_core = aie.core(%full) {{
      %payload_i32 = arith.constant {payload_dwords} : i32
      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_norm_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_make_input_norm_payload(%full_hidden, %full_norm, %full_replay, %payload_i32)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{payload_dwords}xi32>, i32) -> ()
      aie.use_lock(%full_hidden_empty, Release, 1)
      aie.use_lock(%full_norm_empty, Release, 1)
      aie.use_lock(%full_replay_full, Release, {C1R2_QKV_REPLAYS})
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %input_dma = aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%full_hidden_full, Release, 1)
      aie.next_bd ^norm_in
    ^norm_in:
      aie.use_lock(%full_norm_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_norm : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%full_norm_full, Release, 1)
      aie.next_bd ^input_end
    ^input_end:
      aie.end

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^end)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{payload_dwords}xi32>, 0, {payload_dwords}) {{bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^replay_out
    ^end:
      aie.end
    }}
"""


def _postprocess_qkv_body() -> str:
    return f"""
    %post_q_compact = aie.buffer(%post) {{sym_name = "post_q_compact"}} : memref<{Q_DWORDS}xi32>
    %post_k_compact = aie.buffer(%post) {{sym_name = "post_k_compact"}} : memref<{CURRENT_DWORDS}xi32>
    %post_v_compact = aie.buffer(%post) {{sym_name = "post_v_compact"}} : memref<{CURRENT_DWORDS}xi32>
    %post_qk_rope_side = aie.buffer(%post) {{sym_name = "post_qk_rope_side"}} : memref<{QK_ROPE_DWORDS}xi32>
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
{lock_pair("post", "qk_rope_side", 13)}
    %post_runtime_start = aie.lock(%post, 12) {{init = 0 : i32, sym_name = "post_runtime_start"}}

    %post_core = aie.core(%post) {{
      aie.use_lock(%post_runtime_start, Acquire, 1)
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      %current_dwords_i32 = arith.constant {CURRENT_DWORDS} : i32
      aie.use_lock(%post_q_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_k_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_v_compact_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_qk_rope_side_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_v_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_postprocess_q4nx_body_payload(%post_q_compact, %post_k_compact, %post_v_compact, %post_qk_rope_side, %post_q_payload, %post_current_k, %post_current_v, %post_current_token, %q_dwords_i32, %current_dwords_i32)
        : (memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) -> ()
      aie.use_lock(%post_q_compact_empty, Release, 1)
      aie.use_lock(%post_k_compact_empty, Release, 1)
      aie.use_lock(%post_v_compact_empty, Release, 1)
      aie.use_lock(%post_qk_rope_side_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.use_lock(%post_current_k_full, Release, 1)
      aie.use_lock(%post_current_v_full, Release, 1)
      aie.end
    }}

    %post_mem = aie.mem(%post) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^q_in, ^side_start)
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

    ^side_start:
      %side_dma = aie.dma_start(S2MM, 1, ^side_in, ^q_out_start)
    ^side_in:
      aie.use_lock(%post_qk_rope_side_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_qk_rope_side : memref<{QK_ROPE_DWORDS}xi32>, 0, {QK_ROPE_DWORDS}) {{bd_id = 6 : i32}}
      aie.use_lock(%post_qk_rope_side_full, Release, 1)
      aie.next_bd ^side_in

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


def generate_mlir(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_left = aie.tile(0, 0)",
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
        "    %shim_right = aie.tile(7, 0)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row_value in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row_value})")

    flows = [f"    // case marker {CASE_NAME}"]
    flows.extend(
        (
            flow("shim_out", 0, "full", 1),
            flow("shim_out", 1, "post", 1),
            packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
        )
    )
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
    flows.extend(
        (
            packet_flow(CURRENT_PACKET_K, "post", 1, "shim_left", 1),
            packet_flow(CURRENT_PACKET_V, "post", 1, "shim_right", 1),
        )
    )

    blocks = [_input_norm_replay(), _bridge(QKV_CACHE_PHASE_TRACE), _postprocess_qkv_body()]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, QKV_CACHE_PHASE_TRACE))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(full.main16_qkv_prefix_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @full_c1r2_make_input_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @qwen3_postprocess_q4nx_body_payload(memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @clear_summary_fast(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}
    func.func private @q4nx_chunk_accum_slice_i32_fast(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}
    func.func private @q4nx_flush_output_fast(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}
    func.func private @q4nx_emit_q_body_record(memref<{Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}
    func.func private @q4nx_emit_k_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}
    func.func private @q4nx_emit_v_body_record(memref<{KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(QKV_CACHE_PHASE_TRACE)}",
        "full_c1r2_make_input_norm_payload",
        "qwen3_postprocess_q4nx_body_payload",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        f"aie.packet_flow({CURRENT_PACKET_K})",
        f"aie.packet_flow({CURRENT_PACKET_V})",
        f"aie.dma_bd(%post_qk_rope_side : memref<{QK_ROPE_DWORDS}xi32>, 0, {QK_ROPE_DWORDS})",
        f"aiex.npu.push_queue(0, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
        f"aiex.npu.push_queue(7, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
        f"%c{Q_WEIGHT_CHUNK_BASE}_i32 = arith.constant {Q_WEIGHT_CHUNK_BASE} : i32",
        f"%c{K_WEIGHT_CHUNK_BASE}_i32 = arith.constant {K_WEIGHT_CHUNK_BASE} : i32",
        f"%c{V_WEIGHT_CHUNK_BASE}_i32 = arith.constant {V_WEIGHT_CHUNK_BASE} : i32",
        "main_projection_q4nx_fast.o",
    )
    errors = [f"missing qwen3 qkv cache-write marker: {marker}" for marker in required if marker not in mlir]
    expected_packets = len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 1) + 6
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "q4nx fast chunk call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32_fast("), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 6))
    errors.extend(require_count(CASE_NAME, "aux-prefixed weight arg2 address patches", mlir.count("arg_idx = 2 : i32"), 10))
    errors.extend(require_count(CASE_NAME, "hidden arg3 address patches", mlir.count("arg_idx = 3 : i32"), 1))
    errors.extend(require_unique_packet_flows(CASE_NAME, mlir))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_main_record_phase_barrier(CASE_NAME, mlir, len(QKV_CACHE_PHASE_TRACE)))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 3))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, BRIDGE_PACKET_IN_BDS, BRIDGE_PACKET_OUT_BDS))
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids))
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact receive row {row}", bd_ids))
    errors.extend(require_unique_bd_ids(CASE_NAME, COMPACT_OUT_BDS))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    compact_bds = tuple(bd for row_bds in COLUMN_RECEIVE_BDS for bd in row_bds) + COMPACT_OUT_BDS
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    if QKV_BODY_WEIGHT_CHUNKS != 192:
        errors.append("Q/K/V phase-local prefix weight chunk count changed")
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "q4nx_chunk_accum_block_slice_i32_fast",
                "q4nx_emit_o_body_record",
                "q4nx_emit_down_body_record",
                "qwen3_attention_bf16",
                "ffn_swiglu",
                "full_c1r2_make_post_norm_replay",
            ),
        )
    )
    return errors


if __name__ == "__main__":
    print(generate_mlir())
