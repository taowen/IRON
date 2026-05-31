"""Generate MLIR-AIE for the Qwen3 QKV compact-output integration slice."""

from __future__ import annotations

from pathlib import Path

from cases import full_layer_engine_generate as full
from cases import qwen3_8b_qkv_cache_write_generate as qkv
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    COLUMN_WEIGHT_BF16,
    HIDDEN_DWORDS,
    PATCH_WEIGHT_BF16,
    RMS_NORM_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
)
from cases.decode_cache_reference import DEFAULT_SCHEDULE, DecodeSchedule
from compact_dataflow import (
    BRIDGE_COMPACT_OUT_CHANNEL,
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_OUT_CHANNEL,
    COLUMN_OUT_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    COMPACT_PACKET_DWORDS,
    FULL_REPLAY_PACKET_ID,
    Q_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _main_symbol,
    _phase_trace_marker,
    q4nx_weight_column_memtile,
)
from contract import CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, ROWS_PER_COLUMN, ROWS_PER_PATCH
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
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
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_unique_bd_ids,
    require_unique_packet_flows,
)
from projection_schedule import QKV_BODY_WEIGHT_CHUNKS
from projection_schedule import KV_BODY_RECORDS
from projection_schedule import Q_BODY_RECORDS

CASE_NAME = "qwen3-8b-qkv-compact-output"
COMPACT_OUT_COLUMN = 0
COMPACT_OUT_CHANNEL = 1
COMPACT_OUT_BD = 2
QKV_COMPACT_OUT_RECORDS = Q_BODY_RECORDS + KV_BODY_RECORDS * 2
COMPACT_OUT_DWORDS = QKV_COMPACT_OUT_RECORDS * COMPACT_PACKET_DWORDS
QKV_PATCH_WEIGHT_BF16 = ROWS_PER_PATCH * QKV_BODY_WEIGHT_CHUNKS * CHUNK_BF16


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%compact_out: memref<{COMPACT_OUT_DWORDS}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>, "
        f"%hidden: memref<{HIDDEN_DWORDS}xi32>) {{"
    ]
    lines.extend(
        (
            npu_writebd(COMPACT_OUT_COLUMN, COMPACT_OUT_BD, COMPACT_OUT_DWORDS, 0),
            npu_address_patch(COMPACT_OUT_COLUMN, COMPACT_OUT_BD, 0, 0),
            npu_push_queue(COMPACT_OUT_COLUMN, "S2MM", COMPACT_OUT_CHANNEL, COMPACT_OUT_BD),
            npu_writebd(1, 12, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 12, 2, 0),
            npu_push_queue(1, "MM2S", 0, 12),
            npu_writebd(1, 14, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 14, 1, 0),
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
                npu_address_patch(column, 0, 1, column_base),
                npu_writebd(column, 1, qkv_patch_dwords, 0),
                npu_address_patch(column, 1, 1, patch1_base),
                npu_push_queue(column, "MM2S", 0, 0),
                npu_push_queue(column, "MM2S", 1, 1),
            )
        )
    lines.append(npu_sync(COMPACT_OUT_COLUMN, COMPACT_OUT_CHANNEL))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_compact = aie.tile(0, 0)",
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row_value in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row_value})")

    flows = [
        f"    // case marker {CASE_NAME}",
        flow("shim_out", 0, "full", 1),
        packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
        flow("bridge", BRIDGE_COMPACT_OUT_CHANNEL, "shim_compact", COMPACT_OUT_CHANNEL),
    ]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            flows.append(flow(_main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, _main_symbol(group, row), 1))
        flows.append(flow(f"mt{group}", COLUMN_OUT_CHANNEL, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))

    blocks = [qkv._input_norm_replay(), full._bridge(qkv.QKV_CACHE_PHASE_TRACE)]
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, qkv.QKV_CACHE_PHASE_TRACE))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(full.main16_qkv_prefix_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @full_c1r2_make_input_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @{full.MAIN16_LAYER_SCHEDULER}(memref<{CHUNK_BF16}xbf16>, memref<{CHUNK_BF16}xbf16>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, memref<{full.MAIN_CHUNK_DWORDS}xi32>, memref<{full.MAIN_RECORD_PINGPONG_DWORDS}xi32>, memref<{full.MAIN_RECORD_PINGPONG_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/main_projection_q4nx_fast.o"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(qkv.QKV_CACHE_PHASE_TRACE)}",
        "full_c1r2_make_input_norm_payload",
        full.MAIN16_LAYER_SCHEDULER,
        f"aie.flow(%bridge, DMA : {BRIDGE_COMPACT_OUT_CHANNEL}, %shim_compact, DMA : {COMPACT_OUT_CHANNEL})",
        f"memref<{COMPACT_OUT_DWORDS}xi32>",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.push_queue({COMPACT_OUT_COLUMN}, 0, S2MM : {COMPACT_OUT_CHANNEL}) {{bd_id = {COMPACT_OUT_BD} : i32",
        f"aiex.npu.sync {{channel = {COMPACT_OUT_CHANNEL} : i32, column = {COMPACT_OUT_COLUMN} : i32",
        "memref<257xi32>, 0, 257",
        "main_projection_q4nx_fast.o",
    )
    errors = [f"missing qwen3 qkv compact-output marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), 1))
    errors.extend(require_count(CASE_NAME, "q4nx main16 layer scheduler calls", mlir.count(f"func.call @{full.MAIN16_LAYER_SCHEDULER}"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "main16 qkv phase limit constants", mlir.count(f"%main16_phase_limit_i32 = arith.constant {full.MAIN16_PHASE_LIMIT_QKV} : i32"), len(MAIN_COLUMNS) * len(MAIN_ROWS)))
    errors.extend(require_count(CASE_NAME, "aux-prefixed weight arg1 address patches", mlir.count("arg_idx = 1 : i32"), 9))
    errors.extend(require_count(CASE_NAME, "hidden arg2 address patch", mlir.count("arg_idx = 2 : i32"), 1))
    errors.extend(require_unique_packet_flows(CASE_NAME, mlir))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_main_record_pingpong(CASE_NAME, mlir))
    errors.extend(require_compact_record_packet_granularity(CASE_NAME, mlir))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(full.validate_main_buffer_residency(CASE_NAME, mlir))
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 2))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
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
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qwen3_postprocess_absorb_qkv_payload_record",
                "qwen3_postprocess_q4nx_body_payload",
                "qwen3_attention_bf16",
                "ffn_swiglu",
                "debug_contract.o",
                "qwen3_layer.o",
            ),
        )
    )
    if QKV_BODY_WEIGHT_CHUNKS != 192:
        errors.append("Q/K/V compact-output prefix weight chunk count changed")
    if QKV_COMPACT_OUT_RECORDS != 12:
        errors.append("Q/K/V compact-output must drain all 12 Q/K/V global records")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
