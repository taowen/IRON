"""Generate MLIR-AIE for streaming KV scan into attention-kv16."""

from __future__ import annotations

from pathlib import Path

from contract import MAIN_COLUMNS, MAIN_ROWS, SHAPE_CARRIER_DWORDS
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_count,
    require_kv16_attention_shapes,
    require_max_address_patch_arg,
)
from shape_generate import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    _bridge,
    _column_memtile,
    _hub,
    _main_packet,
    _main_symbol,
    _main_tile,
    _shape_a_symbol,
    _shape_b_symbol,
)
from shape_reference import MAIN_CHUNK_DWORDS, PACKET_ID, SUMMARY_DWORDS, TOTAL_SUMMARY_DWORDS
from cases.attention_kv16_generate import _shape_a, _shape_b
from cases.kvscan_attention_kv16_reference import (
    CASE_NAME,
    CONTEXT,
    HEAD_DIM,
    HEADS_PER_WINDOW,
    K_CACHE_SIDE_DWORDS,
    K_WINDOW_DWORDS,
    KV_CACHE_SIDE_DWORDS,
    KV_HEADS_PER_WINDOW,
    KV_SIDE_DWORDS,
    KV_SLOT_DWORDS,
    OUTPUT_DWORDS,
    Q_DWORDS,
    SCALAR_DWORDS,
    V_CACHE_SIDE_DWORDS,
    V_WINDOW_DWORDS,
    WEIGHT_DWORDS,
    WINDOW_DWORDS,
)

ROWS_PER_COLUMN = len(MAIN_ROWS)
KV_SCAN_IN_BDS = (0, 1, 3, 5)


def _kv_scan_memtile(side: int) -> str:
    tile = "kv_left" if side == 0 else "kv_right"
    input_slots = (
        (0, K_WINDOW_DWORDS),
        (K_WINDOW_DWORDS, V_WINDOW_DWORDS),
        (KV_SLOT_DWORDS, K_WINDOW_DWORDS),
        (KV_SLOT_DWORDS + K_WINDOW_DWORDS, V_WINDOW_DWORDS),
    )
    locks = []
    for slot in range(len(input_slots)):
        locks.append(
            f"""    %{tile}_slot{slot}_empty = aie.lock(%{tile}, {slot * 2}) {{init = 1 : i32, sym_name = "{tile}_slot{slot}_empty"}}
    %{tile}_slot{slot}_full = aie.lock(%{tile}, {slot * 2 + 1}) {{init = 0 : i32, sym_name = "{tile}_slot{slot}_full"}}"""
        )

    inputs = []
    for slot, (offset, length) in enumerate(input_slots):
        next_bd = f"^input{slot + 1}" if slot + 1 < len(input_slots) else "^input0"
        inputs.append(f"""    ^input{slot}:
      aie.use_lock(%{tile}_slot{slot}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_payload : memref<{KV_SIDE_DWORDS}xi32>, {offset}, {length}) {{bd_id = {KV_SCAN_IN_BDS[slot]} : i32}}
      aie.use_lock(%{tile}_slot{slot}_full, Release, 1)
      aie.next_bd {next_bd}""")

    outputs = []
    for slot, bd_id in enumerate(KV_OUT_BDS):
        next_start = f"^out{slot + 1}_start" if slot + 1 < len(KV_OUT_BDS) else "^end"
        offset, length = input_slots[slot]
        outputs.append(f"""    ^out{slot}_start:
      %out{slot}_dma = aie.dma_start(MM2S, {slot}, ^out{slot}, {next_start})
    ^out{slot}:
      aie.use_lock(%{tile}_slot{slot}_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_payload : memref<{KV_SIDE_DWORDS}xi32>, {offset}, {length}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_slot{slot}_empty, Release, 1)
      aie.next_bd ^out{slot}""")

    return f"""
    %{tile}_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_payload"}} : memref<{KV_SIDE_DWORDS}xi32>
{chr(10).join(locks)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %input_dma = aie.dma_start(S2MM, 0, ^input0, ^out0_start)
{chr(10).join(inputs)}

{chr(10).join(outputs)}
    ^end:
      aie.end
    }}
"""


def _push_kv_scan_side(column: int, kv_arg: int) -> list[str]:
    v_plane_offset = K_CACHE_SIDE_DWORDS * 4
    return [
        npu_writebd(column, 0, K_WINDOW_DWORDS, 0),
        npu_address_patch(column, 0, kv_arg, 0),
        npu_push_queue(column, "MM2S", 0, 0, issue_token=False),
        npu_writebd(column, 1, V_WINDOW_DWORDS, v_plane_offset),
        npu_address_patch(column, 1, kv_arg, v_plane_offset),
        npu_push_queue(column, "MM2S", 0, 1, issue_token=False),
        npu_writebd(column, 2, K_WINDOW_DWORDS, K_WINDOW_DWORDS * 4),
        npu_address_patch(column, 2, kv_arg, K_WINDOW_DWORDS * 4),
        npu_push_queue(column, "MM2S", 0, 2, issue_token=False),
        npu_writebd(column, 3, V_WINDOW_DWORDS, v_plane_offset + V_WINDOW_DWORDS * 4),
        npu_address_patch(column, 3, kv_arg, v_plane_offset + V_WINDOW_DWORDS * 4),
        npu_push_queue(column, "MM2S", 0, 3),
    ]


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%q: memref<{Q_DWORDS}xi32>, "
        f"%kv_left_arg: memref<{KV_CACHE_SIDE_DWORDS}xi32>, "
        f"%kv_right_arg: memref<{KV_CACHE_SIDE_DWORDS}xi32>, "
        f"%output: memref<{TOTAL_SUMMARY_DWORDS}xi32>) {{"
    ]
    column_summary_dwords = TOTAL_SUMMARY_DWORDS // len(MAIN_COLUMNS)
    for group, column in enumerate(MAIN_COLUMNS):
        output_offset = group * column_summary_dwords * 4
        lines.extend(
            (
                npu_writebd(column, 13, column_summary_dwords, output_offset),
                npu_address_patch(column, 13, 3, output_offset),
                npu_push_queue(column, "S2MM", 1, 13),
            )
        )
    lines.extend(
        (
            npu_writebd(6, 0, Q_DWORDS, 0),
            npu_address_patch(6, 0, 0, 0),
            npu_push_queue(6, "MM2S", 0, 0),
        )
    )
    lines.extend(_push_kv_scan_side(0, 1))
    lines.extend(_push_kv_scan_side(7, 2))
    lines.extend(
        (
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
    experiment_dir = Path(__file__).parent.parent.resolve()
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
        f"    // case marker {CASE_NAME}",
        flow("shim_q", 0, "hub", 0),
        flow("shim_left", 0, "kv_left", 0),
        flow("shim_right", 0, "kv_right", 0),
    ]
    for window in range(4):
        kv_tile = "kv_left" if window < 2 else "kv_right"
        kv_k_channel = 0 if window in (0, 2) else 2
        kv_v_channel = 1 if window in (0, 2) else 3
        flows.append(flow("hub", window, _shape_a_symbol(window), 0))
        flows.append(flow(kv_tile, kv_k_channel, _shape_a_symbol(window), 1))
        flows.append(flow(kv_tile, kv_v_channel, _shape_b_symbol(window), 0))
        flows.append(flow(_shape_a_symbol(window), 0, _shape_b_symbol(window), 1))
        flows.append(flow(_shape_b_symbol(window), 0, "hub", window + 1))
    flows.append(packet_flow(PACKET_ID, "hub", 5, "bridge", 4))
    for group, column in enumerate(MAIN_COLUMNS):
        for row in range(ROWS_PER_COLUMN):
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
            flows.append(packet_flow(_main_packet(group, row), _main_symbol(group, row), 1, f"mt{group}", 2))
        flows.append(flow(f"mt{group}", 5, f"shim{group}", 1))

    blocks = [_hub(), _kv_scan_memtile(0), _kv_scan_memtile(1), _bridge()]
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

    func.func private @attention_kv16_make_carrier(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @attention_kv16_make_return(memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_init_summary(memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}
    func.func private @shape_accum_chunk(memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{SUMMARY_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/qwen3_bridge.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "attention_kv16_make_carrier",
        "attention_kv16_make_return",
        f"memref<{Q_DWORDS}xi32>",
        f"memref<{KV_CACHE_SIDE_DWORDS}xi32>",
        f"memref<{KV_SIDE_DWORDS}xi32>",
        f"memref<{K_WINDOW_DWORDS}xi32>",
        f"memref<{V_WINDOW_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"aie.packet_flow({PACKET_ID})",
        "aie.packet_source<%hub, DMA : 5>",
        "aie.packet_dest<%bridge, DMA : 4>",
        "slot0_empty",
        "slot3_full",
        "issue_token = false",
    )
    errors = [f"missing kvscan attention marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(require_count(CASE_NAME, "attention_kv16_make_carrier", mlir.count("attention_kv16_make_carrier"), 5))
    errors.extend(require_count(CASE_NAME, "attention_kv16_make_return", mlir.count("attention_kv16_make_return"), 5))
    errors.extend(
        require_count(
            CASE_NAME,
            "main activation bridge flow",
            mlir.count("aie.flow(%bridge, DMA : 1"),
            len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        )
    )
    errors.extend(
        require_count(
            CASE_NAME,
            "packet flow",
            mlir.count("aie.packet_flow("),
            len(MAIN_COLUMNS) * ROWS_PER_COLUMN + 1,
        )
    )
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
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 4))
    if KV_SCAN_IN_BDS != (0, 1, 3, 5):
        errors.append("KV scan input BD contract mismatch")
    if KV_OUT_BDS != (2, 24, 4, 26):
        errors.append("KV memtile output BD contract mismatch")
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    return errors
