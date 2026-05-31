"""Generate the full-layer physical Q/K/V prefix slice.

This case reuses the full-layer c1r2, bridge, row1 weight, main16, and c1r3
blocks. The runtime only starts the Q/K/V prefix and waits for packet8/9 cache
writeback, so failures classify before the KV scan/attention/O/down tail.
"""

from __future__ import annotations

from pathlib import Path

from cases import full_layer_engine_generate as full
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    COLUMN_WEIGHT_BF16,
    DEFAULT_SCHEDULE,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    PATCH_WEIGHT_BF16,
    QK_ROPE_DWORDS,
    RMS_NORM_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
)
from cases.kv_cache_dataflow import (
    CURRENT_WRITE_BDS,
    CURRENT_WRITE_CHANNEL,
    push_current_cache_write,
)
from cases.decode_cache_reference import DecodeSchedule
from compact_dataflow import (
    BRIDGE_COMPACT_OUT_CHANNEL,
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_OUT_CHANNEL,
    COLUMN_OUT_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    FULL_REPLAY_PACKET_ID,
    K_GLOBAL_PACKET_ID,
    O_GLOBAL_PACKET_ID,
    Q_GLOBAL_PACKET_ID,
    V_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _phase_trace_marker,
)
from contract import CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, ROWS_PER_PATCH
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_rtp_write,
    npu_set_lock,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_absent_markers,
    require_compact_record_packet_granularity,
    require_count,
    require_disjoint_bd_ids,
    require_dma_bd_next_ids,
    require_dma_bd_lock_balance,
    require_dma_next_bd_labels,
    require_main_record_pingpong,
    require_marker_order,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_bd_ids,
    require_unique_packet_flows,
)
from projection_schedule import (
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
)
from resource_manifest import ResourceManifest, validate_manifest_matches_mlir, validate_resource_manifest

CASE_NAME = "full-layer-qkv-prefix"
MAIN16_KERNEL_OBJECT = full.MAIN16_KERNEL_OBJECT


def _q_payload_sink() -> str:
    return f"""
    %q_sink_payload = aie.buffer(%q_sink) {{sym_name = "q_sink_payload"}} : memref<{full.Q_DWORDS}xi32>

    %q_sink_mem = aie.memtile_dma(%q_sink) {{
      %q_sink_dma = aie.dma_start(S2MM, 0, ^q_in, ^end)
    ^q_in:
      aie.dma_bd(%q_sink_payload : memref<{full.Q_DWORDS}xi32>, 0, {full.Q_DWORDS}) {{bd_id = 0 : i32}}
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def resource_manifest(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> ResourceManifest:
    return full.resource_manifest_for_case(CASE_NAME, full.QKV_PREFIX_PHASE_TRACE)


def _push_qkv_weight_prefix() -> list[str]:
    lines: list[str] = []
    chunk_base, chunk_count = full._full_weight_spans()[0]
    chunk_pair_bytes = ROWS_PER_PATCH * CHUNK_BF16 * 2
    dwords = chunk_count * ROWS_PER_PATCH * CHUNK_BF16 // 2
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = AUX_DWORDS * 4 + group * COLUMN_WEIGHT_BF16 * 2
        push_lines: list[str] = []
        for patch, bd_ids in enumerate(full.WEIGHT_PATCH_BD_IDS):
            bd_id = bd_ids[0]
            patch_base = column_base + patch * PATCH_WEIGHT_BF16 * 2
            byte_offset = patch_base + chunk_base * chunk_pair_bytes
            lines.extend(
                (
                    npu_writebd(column, bd_id, dwords, 0),
                    npu_address_patch(column, bd_id, 2, byte_offset),
                )
            )
            push_lines.append(npu_push_queue(column, "MM2S", patch, bd_id))
        lines.extend(push_lines)
    return lines


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
    lines.extend(_push_qkv_weight_prefix())
    lines.append(npu_set_lock("post_runtime_start", 1))
    lines.extend((npu_sync(0, CURRENT_WRITE_CHANNEL), npu_sync(7, CURRENT_WRITE_CHANNEL)))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_left = aie.tile(0, 0)",
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
        "    %q_sink = aie.tile(6, 1)",
        "    %shim_right = aie.tile(7, 0)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row_value in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{full._main_symbol(group, row_idx)} = aie.tile({column}, {row_value})")

    flows = [f"    // case marker {CASE_NAME}"]
    flows.extend(
        (
            flow("shim_out", 0, "full", 1),
            flow("shim_out", 1, "post", 1),
            packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
        )
    )
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            flows.append(flow(full._main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, full._main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, full._main_symbol(group, row), 1))
        flows.append(flow(f"mt{group}", COLUMN_OUT_CHANNEL, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(packet_flow(packet, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "post", 0))
    flows.extend(
        (
            flow("post", 0, "q_sink", 0),
            packet_flow(O_GLOBAL_PACKET_ID, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "full", 0, keep_pkt_header=True),
            packet_flow(full.CURRENT_PACKET_K, "post", 1, "shim_left", 1),
            packet_flow(full.CURRENT_PACKET_V, "post", 1, "shim_right", 1),
        )
    )

    blocks = [
        full._bridge(full.QKV_PREFIX_PHASE_TRACE),
        full._postprocess_qkv_body(),
        full._full_vector_q4nx_output(),
        _q_payload_sink(),
    ]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(full.q4nx_weight_column_memtile(group, full.QKV_PREFIX_PHASE_TRACE))
        for row in range(len(MAIN_ROWS)):
            blocks.append(full.main16_qkv_prefix_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @qwen3_postprocess_absorb_qkv_payload_record(memref<{full.COMPACT_PACKET_DWORDS - 1}xi32>, memref<{full.Q_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @qwen3_postprocess_q4nx_body_payload(memref<{full.Q_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{full.Q_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, memref<{full.CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @full_c1r2_make_input_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_add_o_compact_to_residual(memref<{HIDDEN_DWORDS}xi32>, memref<{full.COMPACT_PACKET_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_make_post_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_write_down_block(memref<{HIDDEN_DWORDS}xi32>, memref<{full.COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @{full.MAIN16_LAYER_SCHEDULER}(memref<{CHUNK_BF16}xbf16>, memref<{CHUNK_BF16}xbf16>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, memref<{full.MAIN_RECORD_PINGPONG_DWORDS}xi32>, memref<{full.MAIN_RECORD_PINGPONG_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    ownership = resource_manifest(schedule)
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(full.QKV_PREFIX_PHASE_TRACE)}",
        "qwen3_postprocess_absorb_qkv_payload_record",
        "qwen3_postprocess_q4nx_body_payload",
        "full_c1r2_make_input_norm_payload",
        "full_c1r2_add_o_compact_to_residual",
        full.MAIN16_LAYER_SCHEDULER,
        f"aie.packet_flow({full.CURRENT_PACKET_K})",
        f"aie.packet_flow({full.CURRENT_PACKET_V})",
        f"pkt_id = {full.CURRENT_PACKET_K}",
        f"pkt_id = {full.CURRENT_PACKET_V}",
        "aie.flow(%post, DMA : 0, %q_sink, DMA : 0)",
        f"memref<{full.Q_DWORDS}xi32>",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        "aiex.set_lock(%post_runtime_start, 1)",
        "aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)",
        "aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)",
        MAIN16_KERNEL_OBJECT,
        "postprocess_qkv.o",
        "full_vector_station.o",
        "arg_idx = 2 : i32",
        "arg_idx = 3 : i32",
    )
    errors = [f"missing full-layer qkv-prefix marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(validate_resource_manifest(CASE_NAME, ownership))
    errors.extend(validate_manifest_matches_mlir(CASE_NAME, ownership, mlir))
    if tuple(phase.label for phase in full.QKV_PREFIX_PHASE_TRACE) != ("q", "k", "v"):
        errors.append("full-layer qkv-prefix phase trace must be q,k,v")
    if mlir.count(MAIN16_KERNEL_OBJECT) != 1:
        errors.append(f"full-layer qkv-prefix expected 1 declaration linked with {MAIN16_KERNEL_OBJECT}")
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
                f"aiex.npu.push_queue(0, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                f"aiex.npu.push_queue(7, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                "aiex.set_lock(%post_runtime_start, 1)",
                f"aiex.npu.sync {{channel = {CURRENT_WRITE_CHANNEL} : i32, column = 0 : i32",
            ),
        )
    )
    expected_packets = 7
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "qwen3 compact record absorb", mlir.count("qwen3_postprocess_absorb_qkv_payload_record"), 2))
    errors.extend(require_count(CASE_NAME, "qwen3 q4nx body postprocess", mlir.count("qwen3_postprocess_q4nx_body_payload"), 2))
    main_tile_count = len(MAIN_COLUMNS) * len(MAIN_ROWS)
    errors.extend(require_count(CASE_NAME, "q4nx main16 layer scheduler calls", mlir.count(f"func.call @{full.MAIN16_LAYER_SCHEDULER}"), main_tile_count))
    errors.extend(require_count(CASE_NAME, "main16 qkv phase limit constants", mlir.count(f"%main16_phase_limit_i32 = arith.constant {full.MAIN16_PHASE_LIMIT_QKV} : i32"), main_tile_count))
    errors.extend(require_count(CASE_NAME, "old q4nx q emit calls", mlir.count("func.call @q4nx_emit_q_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "old q4nx k emit calls", mlir.count("func.call @q4nx_emit_k_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "old q4nx v emit calls", mlir.count("func.call @q4nx_emit_v_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "Q/K/V weight arg2 address patches", mlir.count("arg_idx = 2 : i32"), 10))
    errors.extend(require_count(CASE_NAME, "hidden arg3 address patch", mlir.count("arg_idx = 3 : i32"), 1))
    errors.extend(require_unique_packet_flows(CASE_NAME, mlir))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_main_record_pingpong(CASE_NAME, mlir))
    errors.extend(require_compact_record_packet_granularity(CASE_NAME, mlir))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(full.validate_main_buffer_residency(CASE_NAME, mlir))
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 3))
    errors.extend(require_unique_bd_ids(CASE_NAME, CURRENT_WRITE_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, BRIDGE_PACKET_IN_BDS, BRIDGE_PACKET_OUT_BDS))
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids))
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact receive row {row}", bd_ids))
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact output", COLUMN_OUT_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, COMPACT_OUT_BDS))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    compact_bds = tuple(bd for row_bds in COLUMN_RECEIVE_BDS for bd in row_bds) + COLUMN_OUT_BDS
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qwen3_attention_bf16",
                "ffn_swiglu",
                "q4nx_chunk_accum_slice_i32_fast",
                "q4nx_clear_block_summaries_fast",
                "q4nx_chunk_accum_block_slice_i32_fast",
                "q4nx_flush_output_fast",
                "q4nx_flush_block_output_fast",
                "q4nx_emit_q_body_record",
                "q4nx_emit_k_body_record",
                "q4nx_emit_v_body_record",
                "q4nx_emit_o_body_record",
                "q4nx_emit_upgate_record",
                "q4nx_emit_down_body_record",
                f"%o_mb_weight_base_i32 = arith.constant {FULL_LAYER_O_WEIGHT_CHUNK_BASE} : i32",
                f"%upgate_weight_chunk_base_i32 = arith.constant {FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE} : i32",
                f"%down_mb_weight_base_i32 = arith.constant {FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE} : i32",
                "aiex.npu.push_queue(0, 0, MM2S : 0)",
                "aiex.npu.push_queue(7, 0, MM2S : 1)",
                "aiex.set_lock(%shape_a0_runtime_start, 1)",
                "arg_idx = 4 : i32",
                "debug_contract.o",
                "qwen3_layer.o",
                "qwen3_bridge.o",
            ),
        )
    )
    if f"buffer_length = {PATCH_WEIGHT_BF16 // 2} : i32" in mlir:
        errors.append("full-layer qkv-prefix weight ingress must use the bounded Q/K/V span")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
