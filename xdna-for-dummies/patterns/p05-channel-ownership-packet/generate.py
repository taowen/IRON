"""Generate MLIR-AIE for P5: channel ownership + packet ID.

Demonstrates:
- A producer tile OWNS MM2S ch0 exclusively
- It sends TWO different payloads via the SAME physical channel
- Packet ID 0 goes to worker0, packet ID 1 goes to worker1
- This is packet_flow: one channel owner, multiple logical streams

This mirrors qwen3-layer's c1r1 shared bridge: one physical MM2S channel
carries both packet2 (attention) and packet1 (FFN down) to different consumers.
"""

from __future__ import annotations

from pathlib import Path

from reference import DATA_DWORDS, OUTPUT_DWORDS, PACKET_ID_0, PACKET_ID_1, WORKER0_CONSTANT, WORKER1_CONSTANT

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
    %worker0 = aie.tile(2, 3)
    %worker1 = aie.tile(2, 4)

    // Input: host → producer (circuit flow, single owner)
    aie.flow(%shim, DMA : 0, %producer, DMA : 0)

    // KEY: producer MM2S ch0 carries BOTH packet flows
    // Same physical channel, different logical streams via packet ID
    aie.packet_flow({PACKET_ID_0}) {{
      aie.packet_source<%producer, DMA : 0>
      aie.packet_dest<%worker0, DMA : 0>
    }}
    aie.packet_flow({PACKET_ID_1}) {{
      aie.packet_source<%producer, DMA : 0>
      aie.packet_dest<%worker1, DMA : 0>
    }}

    // Output: each worker → host via separate channels
    aie.flow(%worker0, DMA : 0, %shim, DMA : 0)
    aie.flow(%worker1, DMA : 0, %shim, DMA : 1)

    func.func private @add_constant(memref<{DATA_DWORDS}xi32>, memref<{DATA_DWORDS}xi32>, i32, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    // --- Producer: sends two payloads on same MM2S ch0 with different packet IDs ---
    %prod_in = aie.buffer(%producer) {{sym_name = "prod_in"}} : memref<{OUTPUT_DWORDS}xi32>
    %prod_out0 = aie.buffer(%producer) {{sym_name = "prod_out0"}} : memref<{DATA_DWORDS}xi32>
    %prod_out1 = aie.buffer(%producer) {{sym_name = "prod_out1"}} : memref<{DATA_DWORDS}xi32>
    %prod_in_empty = aie.lock(%producer, 0) {{init = 1 : i32, sym_name = "prod_in_empty"}}
    %prod_in_full = aie.lock(%producer, 1) {{init = 0 : i32, sym_name = "prod_in_full"}}
    %prod_out0_empty = aie.lock(%producer, 2) {{init = 1 : i32, sym_name = "prod_out0_empty"}}
    %prod_out0_full = aie.lock(%producer, 3) {{init = 0 : i32, sym_name = "prod_out0_full"}}
    %prod_out1_empty = aie.lock(%producer, 4) {{init = 1 : i32, sym_name = "prod_out1_empty"}}
    %prod_out1_full = aie.lock(%producer, 5) {{init = 0 : i32, sym_name = "prod_out1_full"}}

    %prod_core = aie.core(%producer) {{
      %c0 = arith.constant 0 : index
      %half = arith.constant {DATA_DWORDS} : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%prod_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%prod_out0_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%prod_out1_empty, AcquireGreaterEqual, 1)
      // Split input into two halves for the two packet streams
      scf.for %i = %c0 to %half step %c1 {{
        %v0 = memref.load %prod_in[%i] : memref<{OUTPUT_DWORDS}xi32>
        memref.store %v0, %prod_out0[%i] : memref<{DATA_DWORDS}xi32>
      }}
      scf.for %i = %c0 to %half step %c1 {{
        %idx = arith.addi %i, %half : index
        %v1 = memref.load %prod_in[%idx] : memref<{OUTPUT_DWORDS}xi32>
        memref.store %v1, %prod_out1[%i] : memref<{DATA_DWORDS}xi32>
      }}
      aie.use_lock(%prod_in_empty, Release, 1)
      aie.use_lock(%prod_out0_full, Release, 1)
      aie.use_lock(%prod_out1_full, Release, 1)
      aie.end
    }}

    %prod_mem = aie.mem(%producer) {{
      %0 = aie.dma_start(S2MM, 0, ^in_recv, ^out_start)
    ^in_recv:
      aie.use_lock(%prod_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%prod_in : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%prod_in_full, Release, 1)
      aie.next_bd ^in_recv

    ^out_start:
      // MM2S ch0: sends packet0 then packet1 (same channel, different IDs!)
      %1 = aie.dma_start(MM2S, 0, ^send_pkt0, ^end)
    ^send_pkt0:
      aie.use_lock(%prod_out0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%prod_out0 : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_0}>}}
      aie.use_lock(%prod_out0_empty, Release, 1)
      aie.next_bd ^send_pkt1
    ^send_pkt1:
      aie.use_lock(%prod_out1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%prod_out1 : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_1}>}}
      aie.use_lock(%prod_out1_empty, Release, 1)
      aie.next_bd ^send_pkt0
    ^end:
      aie.end
    }}

    // --- Worker 0: receives packet 0 only, adds constant ---
    %w0_in = aie.buffer(%worker0) {{sym_name = "w0_in"}} : memref<{DATA_DWORDS}xi32>
    %w0_out = aie.buffer(%worker0) {{sym_name = "w0_out"}} : memref<{DATA_DWORDS}xi32>
    %w0_in_empty = aie.lock(%worker0, 0) {{init = 1 : i32, sym_name = "w0_in_empty"}}
    %w0_in_full = aie.lock(%worker0, 1) {{init = 0 : i32, sym_name = "w0_in_full"}}
    %w0_out_empty = aie.lock(%worker0, 2) {{init = 1 : i32, sym_name = "w0_out_empty"}}
    %w0_out_full = aie.lock(%worker0, 3) {{init = 0 : i32, sym_name = "w0_out_full"}}

    %w0_core = aie.core(%worker0) {{
      %c = arith.constant {WORKER0_CONSTANT} : i32
      %len = arith.constant {DATA_DWORDS} : i32
      aie.use_lock(%w0_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%w0_out_empty, AcquireGreaterEqual, 1)
      func.call @add_constant(%w0_in, %w0_out, %c, %len) : (memref<{DATA_DWORDS}xi32>, memref<{DATA_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%w0_in_empty, Release, 1)
      aie.use_lock(%w0_out_full, Release, 1)
      aie.end
    }}

    %w0_mem = aie.mem(%worker0) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%w0_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_in : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%w0_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%w0_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_out : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%w0_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Worker 1: receives packet 1 only, adds different constant ---
    %w1_in = aie.buffer(%worker1) {{sym_name = "w1_in"}} : memref<{DATA_DWORDS}xi32>
    %w1_out = aie.buffer(%worker1) {{sym_name = "w1_out"}} : memref<{DATA_DWORDS}xi32>
    %w1_in_empty = aie.lock(%worker1, 0) {{init = 1 : i32, sym_name = "w1_in_empty"}}
    %w1_in_full = aie.lock(%worker1, 1) {{init = 0 : i32, sym_name = "w1_in_full"}}
    %w1_out_empty = aie.lock(%worker1, 2) {{init = 1 : i32, sym_name = "w1_out_empty"}}
    %w1_out_full = aie.lock(%worker1, 3) {{init = 0 : i32, sym_name = "w1_out_full"}}

    %w1_core = aie.core(%worker1) {{
      %c = arith.constant {WORKER1_CONSTANT} : i32
      %len = arith.constant {DATA_DWORDS} : i32
      aie.use_lock(%w1_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%w1_out_empty, AcquireGreaterEqual, 1)
      func.call @add_constant(%w1_in, %w1_out, %c, %len) : (memref<{DATA_DWORDS}xi32>, memref<{DATA_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%w1_in_empty, Release, 1)
      aie.use_lock(%w1_out_full, Release, 1)
      aie.end
    }}

    %w1_mem = aie.mem(%worker1) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%w1_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_in : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%w1_in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.use_lock(%w1_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_out : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%w1_out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    // --- Runtime sequence ---
    aie.runtime_sequence(%input: memref<{OUTPUT_DWORDS}xi32>, %output: memref<{OUTPUT_DWORDS}xi32>) {{
      // Send input to producer
{npu_writebd(2, 0, OUTPUT_DWORDS, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "MM2S", 0, 0)}
      // Collect worker0 output
{npu_writebd(2, 1, DATA_DWORDS, 0)}
{npu_address_patch(2, 1, 1, 0)}
{npu_push_queue(2, "S2MM", 0, 1)}
      // Collect worker1 output
{npu_writebd(2, 2, DATA_DWORDS, DATA_DWORDS * 4)}
{npu_address_patch(2, 2, 1, DATA_DWORDS * 4)}
{npu_push_queue(2, "S2MM", 1, 2)}
      // Wait
{npu_sync(2, 0, direction=0)}
{npu_sync(2, 1, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        f"aie.packet_flow({PACKET_ID_0})",
        f"aie.packet_flow({PACKET_ID_1})",
        "aie.packet_source<%producer, DMA : 0>",
        "aie.packet_dest<%worker0, DMA : 0>",
        "aie.packet_dest<%worker1, DMA : 0>",
        f"pkt_id = {PACKET_ID_0}",
        f"pkt_id = {PACKET_ID_1}",
        "kernel.o",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
