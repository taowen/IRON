"""Generate the full-layer physical Q/K/V -> bf16 attention -> O slice."""

from __future__ import annotations

from pathlib import Path

from attention_dataflow import SHAPE_A_TILES, SHAPE_B_TILES, shape_a_symbol, shape_b_symbol
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
from cases.currentkv_cache_dataflow import (
    CURRENT_WRITE_BDS,
    CURRENT_WRITE_CHANNEL,
    K_SCAN_BD,
    KV_SCAN_BDS,
    KV_SPLIT_K_IN_BDS,
    KV_SPLIT_V_IN_BDS,
    V_SCAN_BD,
    kv_split_scan_memtile,
    push_current_cache_write,
    push_kv_scan_from_cache,
    shape_blocks_name,
    shape_runtime_start_name,
    shape_tail_tokens_name,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    ACCUM_LANES,
    CACHE_BLOCK_DWORDS,
    CURRENT_DWORDS,
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
from compact_dataflow import (
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    FULL_REPLAY_PACKET_ID,
    HUB_Q_IN_CHANNEL,
    HUB_Q_OUT_CHANNELS,
    HUB_RETURN_IN_CHANNELS,
    K_GLOBAL_PACKET_ID,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_GLOBAL_PACKET_ID,
    UPGATE_MAIN_RECORD_DWORDS,
    V_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _phase_trace_marker,
    column_packet,
    main_packet,
)
from contract import C1R2_QKV_REPLAYS, C1R2_PACKET_DWORDS, CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, RECORD_DWORDS
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
    require_count,
    require_disjoint_bd_ids,
    require_dma_bd_next_ids,
    require_dma_bd_lock_balance,
    require_dma_next_bd_labels,
    require_kv16_attention_shapes,
    require_main_record_phase_barrier,
    require_marker_order,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_no_compute_kv_materialization,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_bd_ids,
    require_unique_packet_flows,
)
from projection_schedule import (
    DOWN_BODY_RECORDS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    K_WEIGHT_CHUNK_BASE,
    O_BODY_RECORDS,
    Q_WEIGHT_CHUNK_BASE,
    V_WEIGHT_CHUNK_BASE,
)
from resource_manifest import ResourceManifest, validate_manifest_matches_mlir, validate_resource_manifest

CASE_NAME = "full-layer-attention-o-bf16"
MAIN16_KERNEL_OBJECT = full.MAIN16_KERNEL_OBJECT


def resource_manifest(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> ResourceManifest:
    return full.resource_manifest_for_case(CASE_NAME, full.QKVO_PHASE_TRACE)


def _push_qkv_o_weights() -> list[str]:
    lines: list[str] = []
    spans = full._full_weight_spans()[:2]
    chunk_pair_bytes = full.ROWS_PER_PATCH * CHUNK_BF16 * 2
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = AUX_DWORDS * 4 + group * COLUMN_WEIGHT_BF16 * 2
        for patch, bd_ids in enumerate(full.WEIGHT_PATCH_BD_IDS):
            patch_base = column_base + patch * PATCH_WEIGHT_BF16 * 2
            for span_idx, (chunk_base, chunk_count) in enumerate(spans):
                bd_id = bd_ids[span_idx]
                next_bd = bd_ids[span_idx + 1] if span_idx + 1 < len(spans) else 0
                byte_offset = patch_base + chunk_base * chunk_pair_bytes
                dwords = chunk_count * full.ROWS_PER_PATCH * CHUNK_BF16 // 2
                lines.extend(
                    (
                        npu_writebd(
                            column,
                            bd_id,
                            dwords,
                            0,
                            next_bd=next_bd,
                            use_next_bd=span_idx + 1 < len(spans),
                        ),
                        npu_address_patch(column, bd_id, 2, byte_offset),
                    )
                )
            lines.append(npu_push_queue(column, "MM2S", patch, bd_ids[0]))
    return lines


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%k_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%v_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>, "
        f"%output: memref<{OUTPUT_DWORDS}xi32>, "
        f"%hidden: memref<{HIDDEN_DWORDS}xi32>) {{"
    ]
    lines.append(npu_rtp_write("post_current_token", 0, schedule.current_token))
    for window in range(4):
        lines.append(npu_rtp_write(shape_blocks_name(shape_a_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(shape_blocks_name(shape_b_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(shape_tail_tokens_name(shape_a_symbol(window)), 0, schedule.tail_tokens))
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
            npu_writebd(1, 13, OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 3, 0),
            npu_push_queue(1, "S2MM", 1, 13),
            npu_writebd(1, 12, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 12, 4, 0),
            npu_push_queue(1, "MM2S", 0, 12),
            npu_writebd(1, 14, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 14, 2, 0),
            npu_push_queue(1, "MM2S", 0, 14),
        )
    )
    lines.extend(_push_qkv_o_weights())
    lines.append(npu_set_lock("post_runtime_start", 1))
    lines.extend((npu_sync(0, CURRENT_WRITE_CHANNEL), npu_sync(7, CURRENT_WRITE_CHANNEL)))
    lines.extend(push_kv_scan_from_cache(0, 0, 1, 0, schedule))
    lines.extend(push_kv_scan_from_cache(7, 0, 1, 4, schedule))
    for window in range(4):
        lines.append(npu_set_lock(shape_runtime_start_name(shape_a_symbol(window)), 1))
        lines.append(npu_set_lock(shape_runtime_start_name(shape_b_symbol(window)), 1))
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


def _full_vector_attention_o_output() -> str:
    replay_payload_dwords = C1R2_PACKET_DWORDS - 1
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_vector = aie.buffer(%full) {{sym_name = "full_vector"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{full.COMPACT_PACKET_DWORDS}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{replay_payload_dwords}xi32>
    %full_output = aie.buffer(%full) {{sym_name = "full_output"}} : memref<{OUTPUT_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}
    %full_vector_empty = aie.lock(%full, 2) {{init = 1 : i32, sym_name = "full_vector_empty"}}
    %full_vector_full = aie.lock(%full, 3) {{init = 0 : i32, sym_name = "full_vector_full"}}
    %full_compact_empty = aie.lock(%full, 4) {{init = 1 : i32, sym_name = "full_compact_empty"}}
    %full_compact_full = aie.lock(%full, 5) {{init = 0 : i32, sym_name = "full_compact_full"}}
    %full_replay_empty = aie.lock(%full, 6) {{init = 1 : i32, sym_name = "full_replay_empty"}}
    %full_replay_full = aie.lock(%full, 7) {{init = 0 : i32, sym_name = "full_replay_full"}}
    %full_output_empty = aie.lock(%full, 8) {{init = 1 : i32, sym_name = "full_output_empty"}}
    %full_output_full = aie.lock(%full, 9) {{init = 0 : i32, sym_name = "full_output_full"}}

    %full_core = aie.core(%full) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %o_blocks = arith.constant {O_BODY_RECORDS} : index
      %payload_i32 = arith.constant {replay_payload_dwords} : i32

      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_vector_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_make_input_norm_payload(%full_hidden, %full_vector, %full_replay, %payload_i32)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{replay_payload_dwords}xi32>, i32) -> ()
      aie.use_lock(%full_replay_full, Release, {C1R2_QKV_REPLAYS})
      aie.use_lock(%full_vector_empty, Release, 1)

      aie.use_lock(%full_output_empty, AcquireGreaterEqual, 1)
      scf.for %block = %c0 to %o_blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
        func.call @full_c1r2_write_o_block(%full_compact, %full_output, %block_i32)
          : (memref<{full.COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) -> ()
        aie.use_lock(%full_compact_empty, Release, 1)
      }}
      aie.use_lock(%full_output_full, Release, 1)
      aie.use_lock(%full_hidden_empty, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^compact_in, ^hidden_in_start)
    ^compact_in:
      aie.use_lock(%full_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_compact : memref<{full.COMPACT_PACKET_DWORDS}xi32>, 0, {full.COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_compact_full, Release, 1)
      aie.next_bd ^compact_in

    ^hidden_in_start:
      %hidden_dma = aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%full_hidden_full, Release, 1)
      aie.next_bd ^input_norm_in
    ^input_norm_in:
      aie.use_lock(%full_vector_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_vector : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 4 : i32}}
      aie.use_lock(%full_vector_full, Release, 1)
      aie.next_bd ^input_end
    ^input_end:
      aie.end

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^output_out_start)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{replay_payload_dwords}xi32>, 0, {replay_payload_dwords}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
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
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{full._main_symbol(group, row_idx)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_A_TILES):
        tile_defs.append(f"    %{shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{shape_b_symbol(window)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            flows.append(packet_flow(main_packet(group, row), full._main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, full._main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, full._main_symbol(group, row), 1))
        flows.append(packet_flow(column_packet(group), f"mt{group}", 5, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(packet_flow(packet, "bridge", 5, "post", 0))
    flows.extend(
        (
            packet_flow(full.CURRENT_PACKET_K, "post", 1, "shim_left", 1),
            packet_flow(full.CURRENT_PACKET_V, "post", 1, "shim_right", 1),
            packet_flow(O_GLOBAL_PACKET_ID, "bridge", 5, "full", 0),
            flow("post", 0, "hub", HUB_Q_IN_CHANNEL),
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
        flows.append(flow("hub", HUB_Q_OUT_CHANNELS[window], shape_a_symbol(window), 0))
        flows.append(flow(kv_tile, kv_k_channel, shape_a_symbol(window), 1))
        flows.append(flow(kv_tile, kv_v_channel, shape_b_symbol(window), 0))
        flows.append(flow(shape_a_symbol(window), 0, shape_b_symbol(window), 1))
        flows.append(flow(shape_b_symbol(window), 0, "hub", HUB_RETURN_IN_CHANNELS[window]))
    flows.extend(
        (
            packet_flow(PACKET_ID_ATTENTION, "hub", 5, "bridge", 4),
            packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
            flow("shim_out", 0, "full", 1),
            flow("shim_out", 1, "post", 1),
            flow("full", 0, "shim_out", 1),
        )
    )

    blocks = [
        full._bridge(full.QKVO_PHASE_TRACE),
        full._postprocess_qkv_body(),
        full._hub(),
        kv_split_scan_memtile(0),
        kv_split_scan_memtile(1),
        _full_vector_attention_o_output(),
    ]
    for window in range(4):
        blocks.append(full._shape_a_multiblock_bf16(window))
        blocks.append(full._shape_b_multiblock_bf16(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(full.q4nx_weight_column_memtile(group, full.QKVO_PHASE_TRACE))
        for row in range(len(MAIN_ROWS)):
            blocks.append(full.main16_qkvo_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @qwen3_postprocess_q4nx_body_payload(memref<{full.Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{full.Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @full_c1r2_make_input_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_write_o_block(memref<{full.COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @qwen3_attention_bf16_make_carrier_masked(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_init_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_accum_block(memref<{V_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_finish_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{ATTENTION_OUTPUT_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @clear_summary_fast(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_chunk_accum_slice_i32_fast(memref<{CHUNK_BF16}xbf16>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_clear_block_summaries_fast(i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_chunk_accum_block_slice_i32_fast(memref<{CHUNK_BF16}xbf16>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_flush_block_output_fast(memref<32xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_flush_output_fast(memref<32xbf16>, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_emit_q_body_record(memref<{full.Q_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_emit_k_body_record(memref<{full.KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_emit_v_body_record(memref<{full.KV_MAIN_RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_emit_o_body_record(memref<{full.O_BODY_RECORDS * RECORD_DWORDS}xi32>, memref<32xbf16>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    ownership = resource_manifest(schedule)
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(full.QKVO_PHASE_TRACE)}",
        "qwen3_postprocess_q4nx_body_payload",
        "full_c1r2_make_input_norm_payload",
        "full_c1r2_write_o_block",
        "qwen3_attention_bf16_make_carrier_masked",
        "qwen3_attention_bf16_finish_accum",
        "q4nx_emit_q_body_record",
        "q4nx_emit_k_body_record",
        "q4nx_emit_v_body_record",
        "q4nx_emit_o_body_record",
        f"%c{Q_WEIGHT_CHUNK_BASE}_i32 = arith.constant {Q_WEIGHT_CHUNK_BASE} : i32",
        f"%c{K_WEIGHT_CHUNK_BASE}_i32 = arith.constant {K_WEIGHT_CHUNK_BASE} : i32",
        f"%c{V_WEIGHT_CHUNK_BASE}_i32 = arith.constant {V_WEIGHT_CHUNK_BASE} : i32",
        f"%o_mb_weight_base_i32 = arith.constant {FULL_LAYER_O_WEIGHT_CHUNK_BASE} : i32",
        f"aie.packet_flow({full.CURRENT_PACKET_K})",
        f"aie.packet_flow({full.CURRENT_PACKET_V})",
        f"aie.packet_flow({PACKET_ID_ATTENTION})",
        f"pkt_id = {full.CURRENT_PACKET_K}",
        f"pkt_id = {full.CURRENT_PACKET_V}",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        f"aiex.npu.rtp_write(@shape_a0_blocks, 0, {schedule.kv_blocks})",
        f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
        "aiex.set_lock(%post_runtime_start, 1)",
        "aiex.set_lock(%shape_a0_runtime_start, 1)",
        f"iteration_size = {schedule.kv_blocks} : i32",
        f"iteration_stride = {CACHE_BLOCK_DWORDS - 1} : i32",
        f"repeat_count = {schedule.kv_blocks - 1} : i32",
        MAIN16_KERNEL_OBJECT,
        "postprocess_qkv.o",
        "full_vector_station.o",
        "edge_attention.o",
        "aie.dma_start(S2MM, 4, ^patch0_q4nx_ping, ^patch1_start)",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping, ^wt_row0_start)",
        "arg_idx = 2 : i32",
        "arg_idx = 3 : i32",
        "arg_idx = 4 : i32",
    )
    errors = [f"missing full-layer attention-o bf16 marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(validate_resource_manifest(CASE_NAME, ownership))
    errors.extend(validate_manifest_matches_mlir(CASE_NAME, ownership, mlir))
    if tuple(phase.label for phase in full.QKVO_PHASE_TRACE) != ("q", "k", "v", "o"):
        errors.append("full-layer attention-o phase trace must be q,k,v,o")
    if mlir.count(MAIN16_KERNEL_OBJECT) != 10:
        errors.append(f"full-layer attention-o expected 10 declarations linked with {MAIN16_KERNEL_OBJECT}")
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
                f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
                f"aiex.npu.push_queue(0, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                f"aiex.npu.push_queue(7, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                "aiex.npu.push_queue(1, 0, S2MM : 1) {bd_id = 13 : i32",
                "aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 12 : i32",
                "aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 14 : i32",
                "aiex.set_lock(%post_runtime_start, 1)",
                f"aiex.npu.sync {{channel = {CURRENT_WRITE_CHANNEL} : i32, column = 0 : i32",
                f"aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = {K_SCAN_BD} : i32",
                f"aiex.npu.push_queue(7, 0, MM2S : 1) {{bd_id = {V_SCAN_BD} : i32",
                "aiex.set_lock(%shape_a0_runtime_start, 1)",
            ),
        )
    )
    expected_packets = len(MAIN_COLUMNS) * (len(MAIN_ROWS) + 1) + 8
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    errors.extend(require_count(CASE_NAME, "qwen3 q4nx body postprocess", mlir.count("qwen3_postprocess_q4nx_body_payload"), 2))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_make_carrier_masked", mlir.count("qwen3_attention_bf16_make_carrier_masked"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_init_accum", mlir.count("qwen3_attention_bf16_init_accum"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_accum_block", mlir.count("qwen3_attention_bf16_accum_block"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_finish_accum", mlir.count("qwen3_attention_bf16_finish_accum"), 5))
    errors.extend(require_count(CASE_NAME, "q4nx q emit calls", mlir.count("func.call @q4nx_emit_q_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx k emit calls", mlir.count("func.call @q4nx_emit_k_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx v emit calls", mlir.count("func.call @q4nx_emit_v_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx o emit calls", mlir.count("func.call @q4nx_emit_o_body_record"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "q4nx fast single-accum chunk call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32_fast("), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 6))
    errors.extend(require_count(CASE_NAME, "q4nx fast block-accum chunk call sites", mlir.count("func.call @q4nx_chunk_accum_block_slice_i32_fast("), len(MAIN_COLUMNS) * len(MAIN_ROWS) * 4))
    errors.extend(require_count(CASE_NAME, "weight arg2 address patches", mlir.count("arg_idx = 2 : i32"), 18))
    errors.extend(require_count(CASE_NAME, "output arg3 address patch", mlir.count("arg_idx = 3 : i32"), 1))
    errors.extend(require_count(CASE_NAME, "hidden arg4 address patch", mlir.count("arg_idx = 4 : i32"), 1))
    errors.extend(require_unique_packet_flows(CASE_NAME, mlir))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_main_record_phase_barrier(CASE_NAME, mlir, 3))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(full.validate_main_buffer_residency(CASE_NAME, mlir))
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
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SPLIT_K_IN_BDS + KV_SPLIT_V_IN_BDS + full.KV_OUT_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, KV_SCAN_BDS, CURRENT_WRITE_BDS))
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
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "attention_kv16_make_carrier_masked",
                "attention_kv16_finish_accum",
                "debug_contract.o",
                "qwen3_layer.o",
                "qwen3_bridge.o",
                "ffn_swiglu_slice_bf16_inputs",
                "full_c1r2_make_post_norm_payload",
                "full_c1r2_write_down_block",
                "q4nx_emit_upgate_record",
                "q4nx_emit_down_body_record",
                f"%upgate_weight_chunk_base_i32 = arith.constant {FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE} : i32",
                f"%down_mb_weight_base_i32 = arith.constant {FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE} : i32",
                "aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 15 : i32",
                "aie.packet_flow(14)",
                "aie.packet_flow(15)",
            ),
        )
    )
    if f"buffer_length = {PATCH_WEIGHT_BF16 // 2} : i32" in mlir:
        errors.append("attention-o weight ingress must use bounded Q/K/V and O spans")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
