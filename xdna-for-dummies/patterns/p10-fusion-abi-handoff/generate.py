"""Generate MLIR-AIE for P10: fusion ABI handoff.

Op A outputs a fixed-format record [header | payload] on-chip.
Op B receives it, processes payload, outputs its own record.
The intermediate record never touches host memory.

ABI contract = fixed record size + header encoding + payload layout.
This mirrors qwen3-layer's packet/record handoffs between phases.
"""

from __future__ import annotations

from pathlib import Path

from reference import DATA_DWORDS, RECORD_DWORDS

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
    %tile_a = aie.tile(2, 2)
    %tile_b = aie.tile(2, 3)

    // host -> tile A (raw input)
    aie.flow(%shim, DMA : 0, %tile_a, DMA : 0)
    // tile A -> tile B (ON-CHIP record handoff — the ABI boundary!)
    aie.flow(%tile_a, DMA : 0, %tile_b, DMA : 0)
    // tile B -> host (final record)
    aie.flow(%tile_b, DMA : 0, %shim, DMA : 0)

    func.func private @op_a_emit_record(memref<{DATA_DWORDS}xi32>, memref<{RECORD_DWORDS}xi32>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
    func.func private @op_b_consume_record(memref<{RECORD_DWORDS}xi32>, memref<{RECORD_DWORDS}xi32>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- Tile A: produces a record with ABI header ---
    %a_in = aie.buffer(%tile_a) {{sym_name = "a_in"}} : memref<{DATA_DWORDS}xi32>
    %a_record = aie.buffer(%tile_a) {{sym_name = "a_record"}} : memref<{RECORD_DWORDS}xi32>
    %a_in_empty = aie.lock(%tile_a, 0) {{init = 1 : i32, sym_name = "a_in_empty"}}
    %a_in_full = aie.lock(%tile_a, 1) {{init = 0 : i32, sym_name = "a_in_full"}}
    %a_out_empty = aie.lock(%tile_a, 2) {{init = 1 : i32, sym_name = "a_out_empty"}}
    %a_out_full = aie.lock(%tile_a, 3) {{init = 0 : i32, sym_name = "a_out_full"}}

    %a_core = aie.core(%tile_a) {{
      %plen = arith.constant {DATA_DWORDS} : i32
      aie.use_lock(%a_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%a_out_empty, AcquireGreaterEqual, 1)
      func.call @op_a_emit_record(%a_in, %a_record, %plen) : (memref<{DATA_DWORDS}xi32>, memref<{RECORD_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%a_in_empty, Release, 1)
      aie.use_lock(%a_out_full, Release, 1)
      aie.end
    }}

    %a_mem = aie.mem(%tile_a) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%a_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_in : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%a_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      // Send the FULL RECORD (header + payload) to tile B
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%a_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%a_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Tile B: receives record, processes based on ABI contract ---
    %b_in = aie.buffer(%tile_b) {{sym_name = "b_in"}} : memref<{RECORD_DWORDS}xi32>
    %b_out = aie.buffer(%tile_b) {{sym_name = "b_out"}} : memref<{RECORD_DWORDS}xi32>
    %b_in_empty = aie.lock(%tile_b, 0) {{init = 1 : i32, sym_name = "b_in_empty"}}
    %b_in_full = aie.lock(%tile_b, 1) {{init = 0 : i32, sym_name = "b_in_full"}}
    %b_out_empty = aie.lock(%tile_b, 2) {{init = 1 : i32, sym_name = "b_out_empty"}}
    %b_out_full = aie.lock(%tile_b, 3) {{init = 0 : i32, sym_name = "b_out_full"}}

    %b_core = aie.core(%tile_b) {{
      %plen = arith.constant {DATA_DWORDS} : i32
      aie.use_lock(%b_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%b_out_empty, AcquireGreaterEqual, 1)
      func.call @op_b_consume_record(%b_in, %b_out, %plen) : (memref<{RECORD_DWORDS}xi32>, memref<{RECORD_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%b_in_empty, Release, 1)
      aie.use_lock(%b_out_full, Release, 1)
      aie.end
    }}

    %b_mem = aie.mem(%tile_b) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%b_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_in : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%b_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%b_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_out : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%b_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%input: memref<{DATA_DWORDS}xi32>, %output: memref<{RECORD_DWORDS}xi32>) {{
{npu_writebd(2, 0, DATA_DWORDS, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "MM2S", 0, 0)}
{npu_writebd(2, 1, RECORD_DWORDS, 0)}
{npu_address_patch(2, 1, 1, 0)}
{npu_push_queue(2, "S2MM", 0, 1)}
{npu_sync(2, 0, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "aie.flow(%tile_a, DMA : 0, %tile_b, DMA : 0)",
        "@op_a_emit_record",
        "@op_b_consume_record",
        "a_record",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
