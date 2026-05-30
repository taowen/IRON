"""Generate a row1 weight fanout performance slice."""

from __future__ import annotations

from pathlib import Path

from compact_dataflow import WEIGHT_PATCH_INPUT_BDS, WEIGHT_ROW_BDS, _main_symbol
from contract import CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, ROWS_PER_PATCH
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    COLUMN_WEIGHT_BF16,
    PATCH_WEIGHT_BF16,
    TOTAL_WEIGHT_AND_AUX_I32,
)
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_count,
    require_dma_bd_lock_balance,
    require_dma_bd_next_ids,
    require_dma_next_bd_labels,
    require_memtile_dma_bd_bank,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
)
from qkv_compact_reference import column_packet, main_packet
from projection_schedule import (
    DOWN_WEIGHT_CHUNKS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_TOTAL_WEIGHT_CHUNKS,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    O_WEIGHT_CHUNKS,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    UPGATE_WEIGHT_CHUNKS,
)
from weight_stream import (
    WeightStreamConfig,
    weight_stream_buffers,
    weight_stream_input_rings,
    weight_stream_lock_defs,
    weight_stream_row_streams,
)

CASE_NAME = "row1-weight-stream-perf"
DONE_DWORDS = len(MAIN_COLUMNS) * len(MAIN_ROWS)
DONE_SHIM_BD = 0
DONE_SHIM_CHANNEL = 0
DONE_SHIM_COLUMN = 1
ROW_DONE_BDS = (0, 24, 2, 26)
COLUMN_DONE_OUT_BD = 34
COLUMN_DONE_OUT_CHANNEL = 5
BRIDGE_DONE_BDS = (0, 24, 2, 26)
BRIDGE_DONE_OUT_BD = 4
BRIDGE_DONE_OUT_CHANNEL = 0
WEIGHT_PATCH_BD_IDS = (
    (0, 2, 4, 6, 8, 10, 12, 14),
    (1, 3, 5, 7, 9, 11, 13, 15),
)
WEIGHT_SPAN_CHUNKS = QKV_BODY_WEIGHT_CHUNKS
MAIN_SINK_WEIGHT_BDS = (0, 1)
MAIN_DONE_BD = 2
MAIN_DONE_CHANNEL = 1


def _done_index(group: int, row: int) -> int:
    return group * len(MAIN_ROWS) + row


def _split_weight_span(chunk_base: int, chunks: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for offset in range(0, chunks, WEIGHT_SPAN_CHUNKS):
        spans.append((chunk_base + offset, min(WEIGHT_SPAN_CHUNKS, chunks - offset)))
    return tuple(spans)


def full_weight_spans() -> tuple[tuple[int, int], ...]:
    spans = (
        (Q_WEIGHT_CHUNK_BASE, QKV_BODY_WEIGHT_CHUNKS),
        (FULL_LAYER_O_WEIGHT_CHUNK_BASE, O_WEIGHT_CHUNKS),
        *_split_weight_span(FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE, UPGATE_WEIGHT_CHUNKS),
        *_split_weight_span(FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE, DOWN_WEIGHT_CHUNKS),
    )
    if len(spans) != len(WEIGHT_PATCH_BD_IDS[0]):
        raise RuntimeError(f"bad row1 perf weight span count: {len(spans)}")
    return spans


def _weight_only_column(group: int) -> str:
    tile = f"mt{group}"
    weight_config = WeightStreamConfig(
        group=group,
        input_channels=(4, 5),
        patch_input_bds=WEIGHT_PATCH_INPUT_BDS,
        row_bds=WEIGHT_ROW_BDS,
        row_block_prefix="wt_row",
        row_terminal_label="^done_row0_start",
        input_block_prefix="q4nx_",
    )
    return f"""
{weight_stream_buffers(tile)}
{weight_stream_lock_defs(tile, 0)}
{_column_done_locks(tile)}
    %{tile}_done = aie.buffer(%{tile}) {{sym_name = "{tile}_done"}} : memref<{len(MAIN_ROWS)}xi32>

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{weight_stream_input_rings(weight_config)}

{weight_stream_row_streams(weight_config)}
{_column_done_gather(group)}
    ^end:
      aie.end
    }}
"""


def _bridge_done() -> str:
    row_blocks: list[str] = []
    for group, bd_id in enumerate(BRIDGE_DONE_BDS):
        next_start = f"^bridge_g{group + 1}_start" if group + 1 < len(MAIN_COLUMNS) else "^bridge_done_out_start"
        row_blocks.append(
            f"""    ^bridge_g{group}_start:
      %bridge_g{group}_dma = aie.dma_start(S2MM, {group}, ^bridge_g{group}_in, {next_start})
    ^bridge_g{group}_in:
      aie.use_lock(%bridge_g{group}_done_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_done : memref<{DONE_DWORDS}xi32>, {_done_index(group, 0)}, {len(MAIN_ROWS)}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%bridge_done_full, Release, 1)
      aie.next_bd ^bridge_g{group}_end
    ^bridge_g{group}_end:
      aie.end"""
        )
    lock_defs = "\n".join(
        f'    %bridge_g{group}_done_empty = aie.lock(%bridge, {group}) '
        f'{{init = 1 : i32, sym_name = "bridge_g{group}_done_empty"}}'
        for group in range(len(MAIN_COLUMNS))
    )
    return f"""
    %bridge_done = aie.buffer(%bridge) {{sym_name = "bridge_done"}} : memref<{DONE_DWORDS}xi32>
{lock_defs}
    %bridge_done_full = aie.lock(%bridge, 4) {{init = 0 : i32, sym_name = "bridge_done_full"}}
    %bridge_done_drain = aie.lock(%bridge, 5) {{init = 0 : i32, sym_name = "bridge_done_drain"}}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{chr(10).join(row_blocks)}

    ^bridge_done_out_start:
      %bridge_done_out_dma = aie.dma_start(MM2S, {BRIDGE_DONE_OUT_CHANNEL}, ^bridge_done_out, ^end)
    ^bridge_done_out:
      aie.use_lock(%bridge_done_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd(%bridge_done : memref<{DONE_DWORDS}xi32>, 0, {DONE_DWORDS}) {{bd_id = {BRIDGE_DONE_OUT_BD} : i32}}
      aie.use_lock(%bridge_done_drain, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def _column_done_locks(tile: str) -> str:
    lines = []
    for row in range(len(MAIN_ROWS)):
        lines.append(
            f'    %{tile}_row{row}_done_empty = aie.lock(%{tile}, {8 + row}) '
            f'{{init = 1 : i32, sym_name = "{tile}_row{row}_done_empty"}}'
        )
    lines.extend(
        (
            f'    %{tile}_done_full = aie.lock(%{tile}, 12) '
            f'{{init = 0 : i32, sym_name = "{tile}_done_full"}}',
            f'    %{tile}_done_drain = aie.lock(%{tile}, 13) '
            f'{{init = 0 : i32, sym_name = "{tile}_done_drain"}}',
        )
    )
    return "\n".join(lines)


def _column_done_gather(group: int) -> str:
    tile = f"mt{group}"
    row_blocks: list[str] = []
    for row, bd_id in enumerate(ROW_DONE_BDS):
        next_start = f"^done_row{row + 1}_start" if row + 1 < len(MAIN_ROWS) else "^done_out_start"
        row_blocks.append(
            f"""    ^done_row{row}_start:
      %done_row{row}_dma = aie.dma_start(S2MM, {row}, ^done_row{row}, {next_start})
    ^done_row{row}:
      aie.use_lock(%{tile}_row{row}_done_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_done : memref<{len(MAIN_ROWS)}xi32>, {row}, 1) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_done_full, Release, 1)
      aie.next_bd ^done_row{row}_end
    ^done_row{row}_end:
      aie.end"""
        )
    return f"""
{chr(10).join(row_blocks)}

    ^done_out_start:
      %done_out_dma = aie.dma_start(MM2S, {COLUMN_DONE_OUT_CHANNEL}, ^done_out, ^end)
    ^done_out:
      aie.use_lock(%{tile}_done_full, AcquireGreaterEqual, {len(MAIN_ROWS)})
      aie.dma_bd(%{tile}_done : memref<{len(MAIN_ROWS)}xi32>, 0, {len(MAIN_ROWS)}) {{bd_id = {COLUMN_DONE_OUT_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {column_packet(group)}>}}
      aie.use_lock(%{tile}_done_drain, Release, 1)
      aie.next_bd ^end"""


def _main_weight_sink(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    return f"""
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_done = aie.buffer(%{tile}) {{sym_name = "{tile}_done"}} : memref<1xi32>
    %{tile}_wt_empty = aie.lock(%{tile}, 0) {{init = 2 : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}
    %{tile}_done_empty = aie.lock(%{tile}, 2) {{init = 1 : i32, sym_name = "{tile}_done_empty"}}
    %{tile}_done_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_done_full"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %chunks = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : index
      %chunks_i32 = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : i32
      scf.for %chunk = %c0 to %chunks step %c1 {{
        aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_wt_empty, Release, 1)
      }}
      aie.use_lock(%{tile}_done_empty, AcquireGreaterEqual, 1)
      memref.store %chunks_i32, %{tile}_done[%c0] : memref<1xi32>
      aie.use_lock(%{tile}_done_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^done_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_SINK_WEIGHT_BDS[0]} : i32, next_bd_id = {MAIN_SINK_WEIGHT_BDS[1]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_SINK_WEIGHT_BDS[1]} : i32, next_bd_id = {MAIN_SINK_WEIGHT_BDS[0]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping
    ^done_start:
      %done_dma = aie.dma_start(MM2S, {MAIN_DONE_CHANNEL}, ^done_out, ^end)
    ^done_out:
      aie.use_lock(%{tile}_done_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_done : memref<1xi32>, 0, 1) {{bd_id = {MAIN_DONE_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {main_packet(group, row)}>}}
      aie.use_lock(%{tile}_done_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>, "
        f"%done: memref<{DONE_DWORDS}xi32>) {{"
    ]
    weight_spans = full_weight_spans()
    chunk_pair_bytes = ROWS_PER_PATCH * CHUNK_BF16 * 2
    lines.extend(
        (
            npu_writebd(DONE_SHIM_COLUMN, DONE_SHIM_BD, DONE_DWORDS, 0),
            npu_address_patch(DONE_SHIM_COLUMN, DONE_SHIM_BD, 1, 0),
            npu_push_queue(DONE_SHIM_COLUMN, "S2MM", DONE_SHIM_CHANNEL, DONE_SHIM_BD),
        )
    )
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = AUX_DWORDS * 4 + group * COLUMN_WEIGHT_BF16 * 2
        for patch, bd_ids in enumerate(WEIGHT_PATCH_BD_IDS):
            patch_base = column_base + patch * PATCH_WEIGHT_BF16 * 2
            for span_idx, (chunk_base, chunk_count) in enumerate(weight_spans):
                bd_id = bd_ids[span_idx]
                has_next = span_idx + 1 < len(weight_spans)
                next_bd = bd_ids[span_idx + 1] if has_next else 0
                byte_offset = patch_base + chunk_base * chunk_pair_bytes
                dwords = chunk_count * ROWS_PER_PATCH * CHUNK_BF16 // 2
                lines.extend(
                    (
                        npu_writebd(
                            column,
                            bd_id,
                            dwords,
                            0,
                            next_bd=next_bd,
                            use_next_bd=has_next,
                        ),
                        npu_address_patch(column, bd_id, 0, byte_offset),
                    )
                )
            lines.append(npu_push_queue(column, "MM2S", patch, bd_ids[0]))
    for column in MAIN_COLUMNS:
        lines.extend((npu_sync(column, 0, direction=1), npu_sync(column, 1, direction=1)))
    lines.append(npu_sync(DONE_SHIM_COLUMN, DONE_SHIM_CHANNEL))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    tile_defs: list[str] = [
        f"    // case marker {CASE_NAME}",
        f"    %shim_out = aie.tile({DONE_SHIM_COLUMN}, 0)",
        "    %bridge = aie.tile(1, 1)",
    ]
    flows: list[str] = [flow("bridge", BRIDGE_DONE_OUT_CHANNEL, "shim_out", DONE_SHIM_CHANNEL)]
    blocks: list[str] = [_bridge_done()]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
        flows.append(packet_flow(column_packet(group), f"mt{group}", COLUMN_DONE_OUT_CHANNEL, "bridge", group))
        blocks.append(_weight_only_column(group))
        for row, row_value in enumerate(MAIN_ROWS):
            tile = _main_symbol(group, row)
            tile_defs.append(f"    %{tile} = aie.tile({column}, {row_value})")
            flows.append(flow(f"mt{group}", row, tile, 1))
            flows.append(packet_flow(main_packet(group, row), tile, MAIN_DONE_CHANNEL, f"mt{group}", row))
            blocks.append(_main_weight_sink(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{DONE_DWORDS}xi32>",
        "aie.dma_start(S2MM, 4, ^patch0_q4nx_ping",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping",
        f"%chunks = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : index",
        f"aie.dma_start(MM2S, {COLUMN_DONE_OUT_CHANNEL}, ^done_out",
        f"aie.dma_start(MM2S, {BRIDGE_DONE_OUT_CHANNEL}, ^bridge_done_out",
    )
    errors = [f"missing row1 weight-stream perf marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(require_count(CASE_NAME, "main16 sink tiles", mlir.count("_wt_ping = aie.buffer"), 16))
    errors.extend(require_count(CASE_NAME, "done buffers", mlir.count("_done = aie.buffer"), 21))
    errors.extend(require_count(CASE_NAME, "row1 input queues", mlir.count("MM2S : 0) {bd_id = 0"), 4))
    errors.extend(require_count(CASE_NAME, "row1 second patch queues", mlir.count("MM2S : 1) {bd_id = 1"), 4))
    errors.extend(require_count(CASE_NAME, "shim done output queues", mlir.count(f"S2MM : {DONE_SHIM_CHANNEL}) {{bd_id = {DONE_SHIM_BD}"), 1))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    return errors


def write_build_inputs(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(generate_mlir())
    return mlir_path
