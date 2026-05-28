"""Generate MLIR-AIE for attention-kv16-o-bridge."""

from __future__ import annotations

from pathlib import Path

from contract import MAIN_COLUMNS, MAIN_ROWS, SHAPE_CARRIER_DWORDS
from mlir_utils import flow, npu_address_patch, npu_push_queue, npu_sync, npu_writebd, packet_flow
from shape_generate import (
    HUB_Q_OUT_BDS,
    HUB_RETURN_IN_BDS,
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    _bridge,
    _column_memtile,
    _hub,
    _lock_pair,
    _main_packet,
    _main_symbol,
    _main_tile,
    _shape_a_symbol,
    _shape_b_symbol,
)
from shape_reference import MAIN_CHUNK_DWORDS, PACKET_ID, SUMMARY_DWORDS, TOTAL_SUMMARY_DWORDS
from cases.attention_kv16_reference import (
    CASE_NAME,
    CONTEXT,
    HEAD_DIM,
    HEADS_PER_WINDOW,
    K_WINDOW_DWORDS,
    KV_HEADS_PER_WINDOW,
    KV_SIDE_DWORDS,
    KV_SLOT_DWORDS,
    OUTPUT_DWORDS,
    Q_DWORDS,
    SCALAR_DWORDS,
    V_WINDOW_DWORDS,
    WEIGHT_DWORDS,
    WINDOW_DWORDS,
)

ROWS_PER_COLUMN = len(MAIN_ROWS)


def _kv_memtile(side: int) -> str:
    tile = "kv_left" if side == 0 else "kv_right"
    offsets = (
        0,
        K_WINDOW_DWORDS,
        KV_SLOT_DWORDS,
        KV_SLOT_DWORDS + K_WINDOW_DWORDS,
    )
    starts = []
    for slot, bd_id in enumerate(KV_OUT_BDS):
        next_start = f"^out{slot + 1}_start" if slot + 1 < len(KV_OUT_BDS) else "^end"
        length = K_WINDOW_DWORDS if slot % 2 == 0 else V_WINDOW_DWORDS
        starts.append(f"""    ^out{slot}_start:
      %out{slot}_dma = aie.dma_start(MM2S, {slot}, ^out{slot}, {next_start})
    ^out{slot}:
      aie.use_lock(%{tile}_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_payload : memref<{KV_SIDE_DWORDS}xi32>, {offsets[slot]}, {length}) {{bd_id = {bd_id} : i32}}
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
    %{tile}_k = aie.buffer(%{tile}) {{sym_name = "{tile}_k"}} : memref<{K_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
{_lock_pair(tile, "q", 0)}
{_lock_pair(tile, "k", 2)}
{_lock_pair(tile, "carrier", 4)}

    %{tile}_core = aie.core(%{tile}) {{
      %window_i32 = arith.constant {window} : i32
      %q_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %k_dwords_i32 = arith.constant {K_WINDOW_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_q_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
      func.call @attention_kv16_make_carrier(%{tile}_q, %{tile}_k, %{tile}_carrier, %window_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
        : (memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32) -> ()
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


def _shape_b(window: int) -> str:
    tile = _shape_b_symbol(window)
    return f"""
    %{tile}_v = aie.buffer(%{tile}) {{sym_name = "{tile}_v"}} : memref<{V_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{OUTPUT_DWORDS}xi32>
{_lock_pair(tile, "v", 0)}
{_lock_pair(tile, "carrier", 2)}
{_lock_pair(tile, "output", 4)}

    %{tile}_core = aie.core(%{tile}) {{
      %window_i32 = arith.constant {window} : i32
      %v_dwords_i32 = arith.constant {V_WINDOW_DWORDS} : i32
      %out_dwords_i32 = arith.constant {OUTPUT_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_v_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @attention_kv16_make_return(%{tile}_v, %{tile}_carrier, %{tile}_output, %window_i32, %v_dwords_i32, %out_dwords_i32, %carrier_dwords_i32)
        : (memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_v_empty, Release, 1)
      aie.use_lock(%{tile}_carrier_empty, Release, 1)
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


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%q: memref<{Q_DWORDS}xi32>, "
        f"%kv_left_arg: memref<{KV_SIDE_DWORDS}xi32>, "
        f"%kv_right_arg: memref<{KV_SIDE_DWORDS}xi32>, "
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
        f"memref<{KV_SIDE_DWORDS}xi32>",
        f"memref<{K_WINDOW_DWORDS}xi32>",
        f"memref<{V_WINDOW_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"aie.packet_flow({PACKET_ID})",
        "aie.packet_source<%hub, DMA : 5>",
        "aie.packet_dest<%bridge, DMA : 4>",
    )
    errors = [f"missing attention-kv16 marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count("attention_kv16_make_carrier") != 5:
        errors.append("attention-kv16 Shape-A function declaration/call count mismatch")
    if mlir.count("attention_kv16_make_return") != 5:
        errors.append("attention-kv16 Shape-B function declaration/call count mismatch")
    if mlir.count("aie.flow(%bridge, DMA : 1") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN:
        errors.append("main activation bridge flow count mismatch")
    if mlir.count("aie.packet_flow(") != len(MAIN_COLUMNS) * ROWS_PER_COLUMN + 1:
        errors.append("packet flow count mismatch")
    if K_WINDOW_DWORDS != KV_HEADS_PER_WINDOW * CONTEXT * HEAD_DIM // 2:
        errors.append("K window shape mismatch")
    if OUTPUT_DWORDS != HEADS_PER_WINDOW * HEAD_DIM // 2:
        errors.append("attention output shape mismatch")
    if WEIGHT_DWORDS + SCALAR_DWORDS != SHAPE_CARRIER_DWORDS:
        errors.append("carrier ABI shape mismatch")
    if KV_OUT_BDS != (2, 24, 4, 26):
        errors.append("KV memtile BD contract mismatch")
    if HUB_Q_OUT_BDS != (2, 24, 4, 26) or HUB_RETURN_IN_BDS != (25, 6, 27, 8):
        errors.append("hub BD contract mismatch")
    return errors
