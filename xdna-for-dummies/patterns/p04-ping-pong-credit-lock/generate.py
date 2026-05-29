"""Generate MLIR-AIE for P4: ping-pong credit lock demo."""

from __future__ import annotations

from pathlib import Path

from reference import CHUNK_DWORDS, CONSTANT, NUM_BATCHES, OUTPUT_DWORDS

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
    %producer = aie.tile(2, 2)
    %consumer = aie.tile(2, 3)

    // producer -> consumer stream
    aie.flow(%producer, DMA : 1, %consumer, DMA : 0)
    // consumer -> shim stream (output to host)
    aie.flow(%consumer, DMA : 1, %shim, DMA : 0)

    // --- kernel declarations ---
    func.func private @producer_fill(memref<{CHUNK_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}
    func.func private @consumer_add(memref<{CHUNK_DWORDS}xi32>, memref<{CHUNK_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- producer tile ---
    %prod_ping = aie.buffer(%producer) {{sym_name = "prod_ping"}} : memref<{CHUNK_DWORDS}xi32>
    %prod_pong = aie.buffer(%producer) {{sym_name = "prod_pong"}} : memref<{CHUNK_DWORDS}xi32>
    %prod_empty = aie.lock(%producer, 0) {{init = 2 : i32, sym_name = "prod_empty"}}
    %prod_full = aie.lock(%producer, 1) {{init = 0 : i32, sym_name = "prod_full"}}

    %producer_core = aie.core(%producer) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %len = arith.constant {CHUNK_DWORDS} : i32
      %batches = arith.constant {NUM_BATCHES} : index
      scf.for %batch = %c0 to %batches step %c1 {{
        %batch_i32 = arith.index_cast %batch : index to i32
        %rem = arith.remui %batch, %c2 : index
        %is_pong = arith.cmpi eq, %rem, %c1 : index
        aie.use_lock(%prod_empty, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @producer_fill(%prod_pong, %batch_i32, %len) : (memref<{CHUNK_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @producer_fill(%prod_ping, %batch_i32, %len) : (memref<{CHUNK_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%prod_full, Release, 1)
      }}
      aie.end
    }}

    %producer_mem = aie.mem(%producer) {{
      %0 = aie.dma_start(MM2S, 1, ^send_ping, ^end)
    ^send_ping:
      aie.use_lock(%prod_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%prod_ping : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%prod_empty, Release, 1)
      aie.next_bd ^send_pong
    ^send_pong:
      aie.use_lock(%prod_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%prod_pong : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%prod_empty, Release, 1)
      aie.next_bd ^send_ping
    ^end:
      aie.end
    }}

    // --- consumer tile ---
    %cons_in_ping = aie.buffer(%consumer) {{sym_name = "cons_in_ping"}} : memref<{CHUNK_DWORDS}xi32>
    %cons_in_pong = aie.buffer(%consumer) {{sym_name = "cons_in_pong"}} : memref<{CHUNK_DWORDS}xi32>
    %cons_in_empty = aie.lock(%consumer, 0) {{init = 2 : i32, sym_name = "cons_in_empty"}}
    %cons_in_full = aie.lock(%consumer, 1) {{init = 0 : i32, sym_name = "cons_in_full"}}

    %cons_out_ping = aie.buffer(%consumer) {{sym_name = "cons_out_ping"}} : memref<{CHUNK_DWORDS}xi32>
    %cons_out_pong = aie.buffer(%consumer) {{sym_name = "cons_out_pong"}} : memref<{CHUNK_DWORDS}xi32>
    %cons_out_empty = aie.lock(%consumer, 2) {{init = 2 : i32, sym_name = "cons_out_empty"}}
    %cons_out_full = aie.lock(%consumer, 3) {{init = 0 : i32, sym_name = "cons_out_full"}}

    %consumer_core = aie.core(%consumer) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %len = arith.constant {CHUNK_DWORDS} : i32
      %add_val = arith.constant {CONSTANT} : i32
      %batches = arith.constant {NUM_BATCHES} : index
      scf.for %batch = %c0 to %batches step %c1 {{
        %rem = arith.remui %batch, %c2 : index
        %is_pong = arith.cmpi eq, %rem, %c1 : index
        aie.use_lock(%cons_in_full, AcquireGreaterEqual, 1)
        aie.use_lock(%cons_out_empty, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @consumer_add(%cons_in_pong, %cons_out_pong, %add_val, %len) : (memref<{CHUNK_DWORDS}xi32>, memref<{CHUNK_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @consumer_add(%cons_in_ping, %cons_out_ping, %add_val, %len) : (memref<{CHUNK_DWORDS}xi32>, memref<{CHUNK_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%cons_in_empty, Release, 1)
        aie.use_lock(%cons_out_full, Release, 1)
      }}
      aie.end
    }}

    %consumer_mem = aie.mem(%consumer) {{
      %0 = aie.dma_start(S2MM, 0, ^recv_ping, ^send_start)
    ^recv_ping:
      aie.use_lock(%cons_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%cons_in_ping : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%cons_in_full, Release, 1)
      aie.next_bd ^recv_pong
    ^recv_pong:
      aie.use_lock(%cons_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%cons_in_pong : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%cons_in_full, Release, 1)
      aie.next_bd ^recv_ping

    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^out_ping, ^end)
    ^out_ping:
      aie.use_lock(%cons_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%cons_out_ping : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%cons_out_empty, Release, 1)
      aie.next_bd ^out_pong
    ^out_pong:
      aie.use_lock(%cons_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%cons_out_pong : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%cons_out_empty, Release, 1)
      aie.next_bd ^out_ping
    ^end:
      aie.end
    }}

    // --- runtime sequence ---
    aie.runtime_sequence(%output: memref<{OUTPUT_DWORDS}xi32>) {{
{npu_writebd(2, 0, CHUNK_DWORDS, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "S2MM", 0, 0, repeat=0)}
{npu_writebd(2, 1, CHUNK_DWORDS, CHUNK_DWORDS * 4)}
{npu_address_patch(2, 1, 0, CHUNK_DWORDS * 4)}
{npu_push_queue(2, "S2MM", 0, 1, repeat=0)}
{npu_sync(2, 0, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "aie.flow(%producer, DMA : 1, %consumer, DMA : 0)",
        "aie.flow(%consumer, DMA : 1, %shim, DMA : 0)",
        "prod_ping",
        "prod_pong",
        "cons_in_ping",
        "cons_in_pong",
        "cons_out_ping",
        "cons_out_pong",
        "prod_empty",
        "prod_full",
        "cons_in_empty",
        "cons_in_full",
        "cons_out_empty",
        "cons_out_full",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
