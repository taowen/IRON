"""Generate MLIR-AIE for P6: layout is design.

Two tiles need even-indexed and odd-indexed elements from a 32-element array.
In natural (interleaved) layout [0,1,2,3,...], these are non-contiguous.
Solution: host pre-packs as [all_evens | all_odds], so each tile gets
a contiguous 16-dword block with a simple 1D BD.

This mirrors qwen3-layer's approach:
- Q4NX weights pre-packed into chunk-major (not model tensor layout)
- KV cache in block-major (not token-major)
- Current K/V written even-first-then-odd for linked BD scatter
"""

from __future__ import annotations

from pathlib import Path

from reference import ELEMENTS_PER_TILE, OUTPUT_DWORDS, TOTAL_ELEMENTS

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
    %shim0 = aie.tile(2, 0)
    %shim1 = aie.tile(3, 0)
    %tile0 = aie.tile(2, 2)
    %tile1 = aie.tile(3, 2)

    // Input: packed layout [evens | odds] — each tile gets contiguous block
    aie.flow(%shim0, DMA : 0, %tile0, DMA : 0)
    aie.flow(%shim1, DMA : 0, %tile1, DMA : 0)
    // Output
    aie.flow(%tile0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%tile1, DMA : 0, %shim1, DMA : 0)

    func.func private @sum_array(memref<{ELEMENTS_PER_TILE}xi32>, memref<1xi32>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- tile0: sums even-indexed elements ---
    %t0_in = aie.buffer(%tile0) {{sym_name = "t0_in"}} : memref<{ELEMENTS_PER_TILE}xi32>
    %t0_out = aie.buffer(%tile0) {{sym_name = "t0_out"}} : memref<1xi32>
    %t0_in_empty = aie.lock(%tile0, 0) {{init = 1 : i32, sym_name = "t0_in_empty"}}
    %t0_in_full = aie.lock(%tile0, 1) {{init = 0 : i32, sym_name = "t0_in_full"}}
    %t0_out_empty = aie.lock(%tile0, 2) {{init = 1 : i32, sym_name = "t0_out_empty"}}
    %t0_out_full = aie.lock(%tile0, 3) {{init = 0 : i32, sym_name = "t0_out_full"}}

    %t0_core = aie.core(%tile0) {{
      %len = arith.constant {ELEMENTS_PER_TILE} : i32
      aie.use_lock(%t0_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%t0_out_empty, AcquireGreaterEqual, 1)
      func.call @sum_array(%t0_in, %t0_out, %len) : (memref<{ELEMENTS_PER_TILE}xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%t0_in_empty, Release, 1)
      aie.use_lock(%t0_out_full, Release, 1)
      aie.end
    }}

    %t0_mem = aie.mem(%tile0) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%t0_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%t0_in : memref<{ELEMENTS_PER_TILE}xi32>, 0, {ELEMENTS_PER_TILE}) {{bd_id = 0 : i32}}
      aie.use_lock(%t0_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%t0_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%t0_out : memref<1xi32>, 0, 1) {{bd_id = 1 : i32}}
      aie.use_lock(%t0_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- tile1: sums odd-indexed elements ---
    %t1_in = aie.buffer(%tile1) {{sym_name = "t1_in"}} : memref<{ELEMENTS_PER_TILE}xi32>
    %t1_out = aie.buffer(%tile1) {{sym_name = "t1_out"}} : memref<1xi32>
    %t1_in_empty = aie.lock(%tile1, 0) {{init = 1 : i32, sym_name = "t1_in_empty"}}
    %t1_in_full = aie.lock(%tile1, 1) {{init = 0 : i32, sym_name = "t1_in_full"}}
    %t1_out_empty = aie.lock(%tile1, 2) {{init = 1 : i32, sym_name = "t1_out_empty"}}
    %t1_out_full = aie.lock(%tile1, 3) {{init = 0 : i32, sym_name = "t1_out_full"}}

    %t1_core = aie.core(%tile1) {{
      %len = arith.constant {ELEMENTS_PER_TILE} : i32
      aie.use_lock(%t1_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%t1_out_empty, AcquireGreaterEqual, 1)
      func.call @sum_array(%t1_in, %t1_out, %len) : (memref<{ELEMENTS_PER_TILE}xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%t1_in_empty, Release, 1)
      aie.use_lock(%t1_out_full, Release, 1)
      aie.end
    }}

    %t1_mem = aie.mem(%tile1) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%t1_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%t1_in : memref<{ELEMENTS_PER_TILE}xi32>, 0, {ELEMENTS_PER_TILE}) {{bd_id = 0 : i32}}
      aie.use_lock(%t1_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%t1_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%t1_out : memref<1xi32>, 0, 1) {{bd_id = 1 : i32}}
      aie.use_lock(%t1_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // Runtime: send packed [evens|odds], each tile gets contiguous 16 dwords
    // With pre-pack, only 1 BD per tile needed (no 2D stride!)
    aie.runtime_sequence(%input: memref<{TOTAL_ELEMENTS}xi32>, %output: memref<{OUTPUT_DWORDS}xi32>) {{
      // tile0 gets evens (first half of packed buffer)
{npu_writebd(2, 0, ELEMENTS_PER_TILE, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "MM2S", 0, 0)}
      // tile1 gets odds (second half of packed buffer)
{npu_writebd(3, 0, ELEMENTS_PER_TILE, ELEMENTS_PER_TILE * 4)}
{npu_address_patch(3, 0, 0, ELEMENTS_PER_TILE * 4)}
{npu_push_queue(3, "MM2S", 0, 0)}
      // Collect outputs
{npu_writebd(2, 1, 1, 0)}
{npu_address_patch(2, 1, 1, 0)}
{npu_push_queue(2, "S2MM", 0, 1)}
{npu_writebd(3, 1, 1, 4)}
{npu_address_patch(3, 1, 1, 4)}
{npu_push_queue(3, "S2MM", 0, 1)}
{npu_sync(2, 0, direction=0)}
{npu_sync(3, 0, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = ["@sum_array", "kernel.o", "t0_in", "t1_in"]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
