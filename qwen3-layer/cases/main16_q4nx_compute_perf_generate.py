"""Generate a main16 Q4NX compute-only performance slice."""

from __future__ import annotations

from pathlib import Path

from compact_dataflow import MAIN_CHUNK_DWORDS, _main_symbol
from contract import CHUNK_BF16, MAIN_COLUMNS, MAIN_ROWS, M_PER_TILE
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    require_count,
    require_dma_bd_lock_balance,
    require_dma_bd_next_ids,
    require_dma_next_bd_labels,
    require_memtile_dma_bd_bank,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
)
from projection_schedule import FULL_LAYER_TOTAL_WEIGHT_CHUNKS

CASE_NAME = "main16-q4nx-compute-perf"
MAIN16_KERNEL_OBJECT = "main_projection_q4nx_fast.o"
MAIN16_TILES = len(MAIN_COLUMNS) * len(MAIN_ROWS)
DONE_DWORDS = MAIN16_TILES
ROW_DONE_BDS = (0, 24, 1, 25)
COLUMN_DONE_OUT_BD = 2
COLUMN_DONE_OUT_CHANNEL = 4


def _done_index(group: int, row: int) -> int:
    return group * len(MAIN_ROWS) + row


def _main_compute_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    return f"""
    %{tile}_wt = aie.buffer(%{tile}) {{sym_name = "{tile}_wt"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_chunk = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk"}} : memref<{MAIN_CHUNK_DWORDS}xi32>
    %{tile}_done = aie.buffer(%{tile}) {{sym_name = "{tile}_done"}} : memref<1xi32>
    %{tile}_done_empty = aie.lock(%{tile}, 0) {{init = 1 : i32, sym_name = "{tile}_done_empty"}}
    %{tile}_done_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_done_full"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %chunks = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : index
      %chunks_i32 = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : i32
      %chunk_bf16_i32 = arith.constant {CHUNK_BF16} : i32
      %act_dwords_i32 = arith.constant {MAIN_CHUNK_DWORDS} : i32
      %m_i32 = arith.constant {M_PER_TILE} : i32
      func.call @q4nx_fill_perf_inputs(%{tile}_wt, %{tile}_chunk, %chunk_bf16_i32, %act_dwords_i32)
        : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) -> ()
      func.call @q4nx_clear_accum_fast(%m_i32)
        : (i32) -> ()
      scf.for %chunk = %c0 to %chunks step %c1 {{
        func.call @q4nx_chunk_accum_slice_i32_fast(%{tile}_wt, %{tile}_chunk, %m_i32)
          : (memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) -> ()
      }}
      aie.use_lock(%{tile}_done_empty, AcquireGreaterEqual, 1)
      memref.store %chunks_i32, %{tile}_done[%c0] : memref<1xi32>
      aie.use_lock(%{tile}_done_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %done_dma = aie.dma_start(MM2S, 0, ^done_out, ^end)
    ^done_out:
      aie.use_lock(%{tile}_done_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_done : memref<1xi32>, 0, 1) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_done_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def _column_done_gather(group: int) -> str:
    tile = f"mt{group}"
    row_blocks: list[str] = []
    for row, bd_id in enumerate(ROW_DONE_BDS):
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < len(MAIN_ROWS) else "^out_start"
        row_blocks.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_done, {next_start})
    ^row{row}_done:
      aie.use_lock(%{tile}_row{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_done : memref<{len(MAIN_ROWS)}xi32>, {row}, 1) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_done_full, Release, 1)
      aie.next_bd ^row{row}_end
    ^row{row}_end:
      aie.end"""
        )
    lock_defs = "\n".join(
        f'    %{tile}_row{row}_empty = aie.lock(%{tile}, {row}) '
        f'{{init = 1 : i32, sym_name = "{tile}_row{row}_empty"}}'
        for row in range(len(MAIN_ROWS))
    )
    return f"""
    %{tile}_done = aie.buffer(%{tile}) {{sym_name = "{tile}_done"}} : memref<{len(MAIN_ROWS)}xi32>
{lock_defs}
    %{tile}_done_full = aie.lock(%{tile}, {len(MAIN_ROWS)}) {{init = 0 : i32, sym_name = "{tile}_done_full"}}
    %{tile}_done_drain = aie.lock(%{tile}, {len(MAIN_ROWS) + 1}) {{init = 0 : i32, sym_name = "{tile}_done_drain"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(row_blocks)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, {COLUMN_DONE_OUT_CHANNEL}, ^done_out, ^end)
    ^done_out:
      aie.use_lock(%{tile}_done_full, AcquireGreaterEqual, {len(MAIN_ROWS)})
      aie.dma_bd(%{tile}_done : memref<{len(MAIN_ROWS)}xi32>, 0, {len(MAIN_ROWS)}) {{bd_id = {COLUMN_DONE_OUT_BD} : i32}}
      aie.use_lock(%{tile}_done_drain, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [f"    aie.runtime_sequence(%done: memref<{DONE_DWORDS}xi32>) {{"]
    for group, column in enumerate(MAIN_COLUMNS):
        offset = _done_index(group, 0) * 4
        lines.extend(
            (
                npu_writebd(column, 0, len(MAIN_ROWS), offset),
                npu_address_patch(column, 0, 0, offset),
                npu_push_queue(column, "S2MM", 0, 0),
            )
        )
    for column in MAIN_COLUMNS:
        lines.append(npu_sync(column, 0))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs: list[str] = [f"    // case marker {CASE_NAME}"]
    flows: list[str] = []
    blocks: list[str] = []
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        flows.append(flow(f"mt{group}", COLUMN_DONE_OUT_CHANNEL, f"shim{group}", 0))
        blocks.append(_column_done_gather(group))
        for row, row_value in enumerate(MAIN_ROWS):
            tile = _main_symbol(group, row)
            tile_defs.append(f"    %{tile} = aie.tile({column}, {row_value})")
            flows.append(flow(tile, 0, f"mt{group}", row))
            blocks.append(_main_compute_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @q4nx_fill_perf_inputs(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_clear_accum_fast(i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}
    func.func private @q4nx_chunk_accum_slice_i32_fast(memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "q4nx_fill_perf_inputs",
        "q4nx_clear_accum_fast",
        "q4nx_chunk_accum_slice_i32_fast",
        f"%chunks = arith.constant {FULL_LAYER_TOTAL_WEIGHT_CHUNKS} : index",
        f"memref<{DONE_DWORDS}xi32>",
        MAIN16_KERNEL_OBJECT,
    )
    errors = [f"missing main16 q4nx compute perf marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count(MAIN16_KERNEL_OBJECT) != 3:
        errors.append(f"main16 q4nx perf expected 3 declarations linked with {MAIN16_KERNEL_OBJECT}")
    errors.extend(require_count(CASE_NAME, "main16 compute tiles", mlir.count("_wt = aie.buffer"), MAIN16_TILES))
    errors.extend(require_count(CASE_NAME, "main-to-row1 plus row1-to-shim flows", mlir.count("aie.flow("), MAIN16_TILES + len(MAIN_COLUMNS)))
    errors.extend(require_count(CASE_NAME, "done output queues", mlir.count("S2MM : "), len(MAIN_COLUMNS)))
    errors.extend(require_count(CASE_NAME, "q4nx fast kernel call sites", mlir.count("func.call @q4nx_chunk_accum_slice_i32_fast"), MAIN16_TILES))
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
