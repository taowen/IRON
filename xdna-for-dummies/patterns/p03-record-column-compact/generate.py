"""Generate MLIR-AIE for P03: on-chip record column compact.

2 producer tiles emit records → memtile collects via separate S2MM channels
→ counting lock drain → compact output (row0 full record + row1 payload only).

Follows the proven pattern from recipes/hierarchical-record-compaction:
- Per-row S2MM channel (row0→ch0, row1→ch1)
- Per-row empty lock + shared full lock
- Drain: AcquireGreaterEqual(NUM_PRODUCERS) on full lock
- Header stripping: row0 keeps full record, row1 BD starts at offset 1 (skips header)
"""

from __future__ import annotations

from pathlib import Path

from reference import COMPACT_DWORDS, NUM_PRODUCERS, PAYLOAD_DWORDS, RECORD_DWORDS

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
    # Row1 dest offsets for compact:
    # Row0: full record at offset 0, length = RECORD_DWORDS
    # Row1: payload only at offset RECORD_DWORDS, length = PAYLOAD_DWORDS
    row0_dest_offset = 0
    row0_length = RECORD_DWORDS
    row1_dest_offset = RECORD_DWORDS
    row1_length = PAYLOAD_DWORDS

    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(2, 0)
    %mt = aie.tile(2, 1)
    %prod0 = aie.tile(2, 2)
    %prod1 = aie.tile(2, 3)

    // Producers → memtile (separate S2MM channels per row)
    aie.flow(%prod0, DMA : 0, %mt, DMA : 0)
    aie.flow(%prod1, DMA : 0, %mt, DMA : 1)
    // Memtile compact output → shim
    aie.flow(%mt, DMA : 5, %shim, DMA : 0)

    func.func private @emit_record(memref<{RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- Producer 0 (row0) ---
    %p0_rec = aie.buffer(%prod0) {{sym_name = "p0_rec"}} : memref<{RECORD_DWORDS}xi32>
    %p0_empty = aie.lock(%prod0, 0) {{init = 1 : i32, sym_name = "p0_empty"}}
    %p0_full = aie.lock(%prod0, 1) {{init = 0 : i32, sym_name = "p0_full"}}

    %p0_core = aie.core(%prod0) {{
      %id = arith.constant 0 : i32
      %plen = arith.constant {PAYLOAD_DWORDS} : i32
      aie.use_lock(%p0_empty, AcquireGreaterEqual, 1)
      func.call @emit_record(%p0_rec, %id, %plen) : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%p0_full, Release, 1)
      aie.end
    }}

    %p0_mem = aie.mem(%prod0) {{
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%p0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%p0_rec : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%p0_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Producer 1 (row1) ---
    %p1_rec = aie.buffer(%prod1) {{sym_name = "p1_rec"}} : memref<{RECORD_DWORDS}xi32>
    %p1_empty = aie.lock(%prod1, 0) {{init = 1 : i32, sym_name = "p1_empty"}}
    %p1_full = aie.lock(%prod1, 1) {{init = 0 : i32, sym_name = "p1_full"}}

    %p1_core = aie.core(%prod1) {{
      %id = arith.constant 1 : i32
      %plen = arith.constant {PAYLOAD_DWORDS} : i32
      aie.use_lock(%p1_empty, AcquireGreaterEqual, 1)
      func.call @emit_record(%p1_rec, %id, %plen) : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%p1_full, Release, 1)
      aie.end
    }}

    %p1_mem = aie.mem(%prod1) {{
      // Send ONLY payload (skip header) — header stripping at source!
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%p1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%p1_rec : memref<{RECORD_DWORDS}xi32>, 1, {PAYLOAD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%p1_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Memtile: on-chip compact ---
    // Compact buffer holds: [row0 full record | row1 payload only]
    %mt_compact = aie.buffer(%mt) {{sym_name = "mt_compact"}} : memref<{COMPACT_DWORDS}xi32>
    // Per-row empty locks (each row has its own S2MM channel)
    // init=1: one slot available per row
    %mt_row0_empty = aie.lock(%mt, 0) {{init = 1 : i32, sym_name = "mt_row0_empty"}}
    %mt_row1_empty = aie.lock(%mt, 1) {{init = 1 : i32, sym_name = "mt_row1_empty"}}
    // Shared full lock: drain waits for BOTH rows (counting lock)
    %mt_full = aie.lock(%mt, 2) {{init = 0 : i32, sym_name = "mt_full"}}
    // Drain token: released by drain, signals back to allow next round
    %mt_drain_token = aie.lock(%mt, 3) {{init = 0 : i32, sym_name = "mt_drain_token"}}

    %mt_dma = aie.memtile_dma(%mt) {{
      // S2MM ch0: receives row0 full record into compact[0..RECORD_DWORDS]
      %0 = aie.dma_start(S2MM, 0, ^row0_recv, ^row1_start)
    ^row0_recv:
      aie.use_lock(%mt_row0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_compact : memref<{COMPACT_DWORDS}xi32>, {row0_dest_offset}, {row0_length}) {{bd_id = 0 : i32}}
      aie.use_lock(%mt_full, Release, 1)
      aie.next_bd ^row0_recv

      // S2MM ch1: receives row1 record — writes at row1 offset in compact buffer
      // Note: we receive full RECORD_DWORDS but only the first PAYLOAD_DWORDS
      // after the header lands at the correct offset. For simplicity here we
      // receive only PAYLOAD_DWORDS into the payload slot.
    ^row1_start:
      %1 = aie.dma_start(S2MM, 1, ^row1_recv, ^drain_start)
    ^row1_recv:
      aie.use_lock(%mt_row1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_compact : memref<{COMPACT_DWORDS}xi32>, {row1_dest_offset}, {row1_length}) {{bd_id = 24 : i32}}
      aie.use_lock(%mt_full, Release, 1)
      aie.next_bd ^row1_recv

      // MM2S ch5: drain compact buffer when BOTH rows have written
      // BD block can only have ONE release — use drain_token
    ^drain_start:
      %2 = aie.dma_start(MM2S, 5, ^drain, ^end)
    ^drain:
      aie.use_lock(%mt_full, AcquireGreaterEqual, {NUM_PRODUCERS})
      aie.dma_bd(%mt_compact : memref<{COMPACT_DWORDS}xi32>, 0, {COMPACT_DWORDS}) {{bd_id = 34 : i32}}
      aie.use_lock(%mt_drain_token, Release, 1)
      aie.next_bd ^drain
    ^end:
      aie.end
    }}

    // --- Runtime sequence ---
    aie.runtime_sequence(%output: memref<{COMPACT_DWORDS}xi32>) {{
{npu_writebd(2, 0, COMPACT_DWORDS, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "S2MM", 0, 0)}
{npu_sync(2, 0, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "aie.flow(%prod0, DMA : 0, %mt, DMA : 0)",
        "aie.flow(%prod1, DMA : 0, %mt, DMA : 1)",
        "aie.flow(%mt, DMA : 5, %shim, DMA : 0)",
        "mt_compact",
        "mt_row0_empty",
        "mt_row1_empty",
        "mt_full",
        f"AcquireGreaterEqual, {NUM_PRODUCERS}",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
