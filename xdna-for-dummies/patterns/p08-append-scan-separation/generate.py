"""Generate MLIR-AIE for P08: append/scan separation.

Two-phase runtime sequence with npu.sync boundary:
  Phase 1: writer tile produces data → shim writes to cache BO at APPEND_POSITION
  npu.sync (ensures append is visible in host memory)
  Phase 2: shim reads entire cache BO → scanner tile computes sum

This mirrors currentkv in qwen3-layer:
  Phase 1: c1r3 → packet8/9 → shim S2MM → KV cache BO at token offset
  npu.sync
  Phase 2: shim MM2S → KV scan → Shape tiles
"""

from __future__ import annotations

from pathlib import Path

from reference import APPEND_DWORDS, APPEND_POSITION, CACHE_DWORDS, OUTPUT_DWORDS

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


def generate_mlir() -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(2, 0)
    %writer = aie.tile(2, 2)
    %scanner = aie.tile(2, 3)

    // Phase 1 path: writer → shim (append to cache BO)
    aie.flow(%writer, DMA : 0, %shim, DMA : 0)
    // Phase 2 path: shim → scanner (read entire cache BO)
    aie.flow(%shim, DMA : 0, %scanner, DMA : 0)
    // Output: scanner → shim
    aie.flow(%scanner, DMA : 0, %shim, DMA : 1)

    func.func private @writer_produce(memref<{APPEND_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
    func.func private @scanner_sum(memref<{CACHE_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- Writer tile: produces append data (Phase 1) ---
    %wr_buf = aie.buffer(%writer) {{sym_name = "wr_buf"}} : memref<{APPEND_DWORDS}xi32>
    %wr_empty = aie.lock(%writer, 0) {{init = 1 : i32, sym_name = "wr_empty"}}
    %wr_full = aie.lock(%writer, 1) {{init = 0 : i32, sym_name = "wr_full"}}

    %writer_core = aie.core(%writer) {{
      %pos = arith.constant {APPEND_POSITION} : i32
      %len = arith.constant {APPEND_DWORDS} : i32
      aie.use_lock(%wr_empty, AcquireGreaterEqual, 1)
      func.call @writer_produce(%wr_buf, %pos, %len) : (memref<{APPEND_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%wr_full, Release, 1)
      aie.end
    }}

    %writer_mem = aie.mem(%writer) {{
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%wr_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%wr_buf : memref<{APPEND_DWORDS}xi32>, 0, {APPEND_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%wr_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Scanner tile: reads entire cache (Phase 2, after sync) ---
    %sc_cache = aie.buffer(%scanner) {{sym_name = "sc_cache"}} : memref<{CACHE_DWORDS}xi32>
    %sc_result = aie.buffer(%scanner) {{sym_name = "sc_result"}} : memref<{OUTPUT_DWORDS}xi32>
    %sc_in_empty = aie.lock(%scanner, 0) {{init = 1 : i32, sym_name = "sc_in_empty"}}
    %sc_in_full = aie.lock(%scanner, 1) {{init = 0 : i32, sym_name = "sc_in_full"}}
    %sc_out_empty = aie.lock(%scanner, 2) {{init = 1 : i32, sym_name = "sc_out_empty"}}
    %sc_out_full = aie.lock(%scanner, 3) {{init = 0 : i32, sym_name = "sc_out_full"}}

    %scanner_core = aie.core(%scanner) {{
      %len = arith.constant {CACHE_DWORDS} : i32
      aie.use_lock(%sc_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%sc_out_empty, AcquireGreaterEqual, 1)
      func.call @scanner_sum(%sc_cache, %sc_result, %len) : (memref<{CACHE_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%sc_in_empty, Release, 1)
      aie.use_lock(%sc_out_full, Release, 1)
      aie.end
    }}

    %scanner_mem = aie.mem(%scanner) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^out_start)
    ^recv:
      aie.use_lock(%sc_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc_cache : memref<{CACHE_DWORDS}xi32>, 0, {CACHE_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%sc_in_full, Release, 1)
      aie.next_bd ^recv
    ^out_start:
      %1 = aie.dma_start(MM2S, 0, ^out_send, ^end)
    ^out_send:
      aie.use_lock(%sc_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc_result : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%sc_out_empty, Release, 1)
      aie.next_bd ^out_send
    ^end:
      aie.end
    }}

    // --- Runtime sequence: TWO PHASES separated by npu.sync ---
    aie.runtime_sequence(%cache: memref<{CACHE_DWORDS}xi32>, %output: memref<{OUTPUT_DWORDS}xi32>) {{
      // === PHASE 1: APPEND ===
      // Shim S2MM receives writer's output INTO cache BO at APPEND_POSITION offset
      // This is the descriptor-level "append" — buffer_offset is position-dependent!
{npu_writebd(2, 0, APPEND_DWORDS, APPEND_POSITION * 4)}
{npu_address_patch(2, 0, 0, APPEND_POSITION * 4)}
{npu_push_queue(2, "S2MM", 0, 0)}
      // Wait for append to complete (data visible in cache BO)
{npu_sync(2, 0, direction=0)}

      // === PHASE 2: SCAN (only after sync guarantees append is visible) ===
      // Shim MM2S reads ENTIRE cache BO (including just-appended data) → scanner
{npu_writebd(2, 2, CACHE_DWORDS, 0)}
{npu_address_patch(2, 2, 0, 0)}
{npu_push_queue(2, "MM2S", 0, 2)}
      // Collect scanner output
{npu_writebd(2, 3, OUTPUT_DWORDS, 0)}
{npu_address_patch(2, 3, 1, 0)}
{npu_push_queue(2, "S2MM", 1, 3)}
{npu_sync(2, 1, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "aie.flow(%writer, DMA : 0, %shim, DMA : 0)",
        "aie.flow(%shim, DMA : 0, %scanner, DMA : 0)",
        "@writer_produce",
        "@scanner_sum",
        "npu.sync",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    # Verify the two-phase structure: append BD has position-dependent offset
    if f"buffer_offset = {APPEND_POSITION * 4}" not in generate_mlir():
        errors.append("append BD missing position-dependent buffer_offset")
    return errors
