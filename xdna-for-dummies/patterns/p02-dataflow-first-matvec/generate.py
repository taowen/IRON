"""Generate MLIR-AIE for P2: dataflow-first matvec.

Demonstrates block-major scheduling with weight ping-pong:
- Weights stream through via ping-pong BD ring (DMA/compute overlap)
- Activation replayed from host (block-major: each output block sees all K chunks)
- Weight stream order = compute loop order (chunk-major pack)
- Each tile emits a record when done

This is block-major scheduling (like Q/K/V in qwen3-layer).
Chunk-major (like O/down) would have activation come from on-chip packet
and multiple output blocks accumulate from the same activation chunk.
"""

from __future__ import annotations

from pathlib import Path

from reference import (
    ACT_DWORDS,
    CHUNK_COLS,
    K_DIM,
    NUM_CHUNKS,
    NUM_TILES,
    OUTPUT_DWORDS,
    RECORD_DWORDS,
    ROWS_PER_TILE,
    TOTAL_WEIGHT_DWORDS,
    WEIGHT_CHUNK_DWORDS,
    WEIGHT_DWORDS_PER_TILE,
)

EXPERIMENT_DIR = Path(__file__).parent.resolve()


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 64 : i32, column = {column} : i32, "
        f"d0_size = 0 : i32, d0_stride = 0 : i32, "
        f"d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = 0 : i32, d1_stride = 0 : i32, "
        f"d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = 0 : i32, d2_stride = 0 : i32, "
        f"d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def npu_push_queue(column: int, direction: str, channel: int, bd_id: int, repeat: int = 0) -> str:
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = true, repeat_count = {repeat} : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _tile_mlir(col: int, tile_idx: int) -> str:
    tile = f"tile{tile_idx}"
    return f"""
    // --- {tile}: block-major scheduling with weight ping-pong ---
    // Weight ping-pong: DMA fills one buffer while core computes on the other
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{WEIGHT_CHUNK_DWORDS}xi32>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{WEIGHT_CHUNK_DWORDS}xi32>
    %{tile}_act = aie.buffer(%{tile}) {{sym_name = "{tile}_act"}} : memref<{CHUNK_COLS}xi32>
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    // Weight ping-pong locks
    %{tile}_wt_empty = aie.lock(%{tile}, 0) {{init = 2 : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}
    // Activation lock
    %{tile}_act_empty = aie.lock(%{tile}, 2) {{init = 1 : i32, sym_name = "{tile}_act_empty"}}
    %{tile}_act_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_act_full"}}
    // Output lock
    %{tile}_out_empty = aie.lock(%{tile}, 4) {{init = 1 : i32, sym_name = "{tile}_out_empty"}}
    %{tile}_out_full = aie.lock(%{tile}, 5) {{init = 0 : i32, sym_name = "{tile}_out_full"}}

    %{tile}_core = aie.core(%{tile}) {{
      %rows = arith.constant {ROWS_PER_TILE} : i32
      %cols = arith.constant {CHUNK_COLS} : i32
      %chunks = arith.constant {NUM_CHUNKS} : index
      %tile_id = arith.constant {tile_idx} : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      func.call @matvec_clear(%rows) : (i32) -> ()

      // Block-major loop: for each chunk, consume weight+activation pair
      scf.for %chunk = %c0 to %chunks step %c1 {{
        %rem = arith.remui %chunk, %c2 : index
        %is_pong = arith.cmpi eq, %rem, %c1 : index
        aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @matvec_mac_chunk(%{tile}_wt_pong, %{tile}_act, %rows, %cols)
            : (memref<{WEIGHT_CHUNK_DWORDS}xi32>, memref<{CHUNK_COLS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @matvec_mac_chunk(%{tile}_wt_ping, %{tile}_act, %rows, %cols)
            : (memref<{WEIGHT_CHUNK_DWORDS}xi32>, memref<{CHUNK_COLS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_wt_empty, Release, 1)
        aie.use_lock(%{tile}_act_empty, Release, 1)
      }}

      aie.use_lock(%{tile}_out_empty, AcquireGreaterEqual, 1)
      func.call @matvec_emit_record(%{tile}_record, %tile_id, %rows)
        : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_out_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      // Weight S2MM: ping-pong BD ring (DMA fills next while core uses current)
      %0 = aie.dma_start(S2MM, 0, ^wt_ping, ^act_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{WEIGHT_CHUNK_DWORDS}xi32>, 0, {WEIGHT_CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{WEIGHT_CHUNK_DWORDS}xi32>, 0, {WEIGHT_CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^act_start:
      %1 = aie.dma_start(S2MM, 1, ^act_recv, ^out_start)
    ^act_recv:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act : memref<{CHUNK_COLS}xi32>, 0, {CHUNK_COLS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_recv

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_send, ^end)
    ^out_send:
      aie.use_lock(%{tile}_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%{tile}_out_empty, Release, 1)
      aie.next_bd ^out_send
    ^end:
      aie.end
    }}
"""


def generate_mlir() -> str:
    tiles_mlir = _tile_mlir(2, 0) + _tile_mlir(3, 1)

    rt_lines = []
    # Interleaved scheduling: for each chunk, send weight+activation to both tiles
    for chunk in range(NUM_CHUNKS):
        for tile_idx, col in enumerate((2, 3)):
            wt_offset = (tile_idx * WEIGHT_DWORDS_PER_TILE + chunk * WEIGHT_CHUNK_DWORDS) * 4
            bd_id = 4 + tile_idx * NUM_CHUNKS + chunk
            rt_lines.extend([
                npu_writebd(col, bd_id, WEIGHT_CHUNK_DWORDS, wt_offset),
                npu_address_patch(col, bd_id, 0, wt_offset),
                npu_push_queue(col, "MM2S", 0, bd_id),
            ])
        act_offset = chunk * CHUNK_COLS * 4
        for col in (2, 3):
            act_bd = 12 + chunk
            rt_lines.extend([
                npu_writebd(col, act_bd, CHUNK_COLS, act_offset),
                npu_address_patch(col, act_bd, 1, act_offset),
                npu_push_queue(col, "MM2S", 1, act_bd),
            ])

    # Collect records
    rt_lines.extend([
        npu_writebd(2, 0, RECORD_DWORDS, 0),
        npu_address_patch(2, 0, 2, 0),
        npu_push_queue(2, "S2MM", 0, 0),
        npu_writebd(3, 0, RECORD_DWORDS, RECORD_DWORDS * 4),
        npu_address_patch(3, 0, 2, RECORD_DWORDS * 4),
        npu_push_queue(3, "S2MM", 0, 0),
        npu_sync(2, 0, direction=0),
        npu_sync(3, 0, direction=0),
    ])

    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(2, 0)
    %shim1 = aie.tile(3, 0)
    %tile0 = aie.tile(2, 2)
    %tile1 = aie.tile(3, 2)

    // Weight streams: each tile gets its own weight chunk stream
    aie.flow(%shim0, DMA : 0, %tile0, DMA : 0)
    aie.flow(%shim1, DMA : 0, %tile1, DMA : 0)
    // Activation: replayed from host to both tiles (block-major: cheap to replay)
    aie.flow(%shim0, DMA : 1, %tile0, DMA : 1)
    aie.flow(%shim1, DMA : 1, %tile1, DMA : 1)
    // Record output
    aie.flow(%tile0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%tile1, DMA : 0, %shim1, DMA : 0)

    func.func private @matvec_clear(i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
    func.func private @matvec_mac_chunk(memref<{WEIGHT_CHUNK_DWORDS}xi32>, memref<{CHUNK_COLS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
    func.func private @matvec_emit_record(memref<{RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
{tiles_mlir}
    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_DWORDS}xi32>, %activation: memref<{ACT_DWORDS}xi32>, %output: memref<{OUTPUT_DWORDS}xi32>) {{
{chr(10).join(rt_lines)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "wt_ping",
        "wt_pong",
        "next_bd_id = 1",
        "next_bd_id = 0",
        "@matvec_mac_chunk",
        "@matvec_emit_record",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
