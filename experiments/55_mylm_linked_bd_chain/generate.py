"""Generate raw MLIR-AIE for exp55 MyLM-style linked BD chain."""

from pathlib import Path

NUM_CHUNKS = 8
CHUNK_DWORDS = 64
WORDS_PER_RECORD = 4
TOTAL_INPUT_DWORDS = NUM_CHUNKS * CHUNK_DWORDS
TOTAL_OUTPUT_DWORDS = NUM_CHUNKS * WORDS_PER_RECORD


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(
    column: int,
    bd_id: int,
    buffer_length: int,
    buffer_offset: int,
    next_bd: int = 0,
    use_next_bd: bool = False,
) -> str:
    use_next = 1 if use_next_bd else 0
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
        f"next_bd = {next_bd} : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = {use_next} : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def _npu_push_queue(column: int, direction: str, channel: int, bd_id: int, issue_token: bool = False) -> str:
    token = "true" if issue_token else "false"
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"
    )


def _npu_sync(column: int, channel: int, direction: int) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _lock_pair(prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{prefix}_empty = aie.lock(%core, {base}) {{init = {init_empty} : i32, sym_name = "{prefix}_empty"}}
    %{prefix}_full = aie.lock(%core, {base + 1}) {{init = 0 : i32, sym_name = "{prefix}_full"}}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%input: memref<{TOTAL_INPUT_DWORDS}xi32>, "
        f"%output: memref<{TOTAL_OUTPUT_DWORDS}xi32>) {{",
        _npu_writebd(0, 13, TOTAL_OUTPUT_DWORDS, 0),
        _npu_address_patch(0, 13, 1, 0),
        _npu_push_queue(0, "S2MM", 0, 13, issue_token=True),
    ]

    for chunk in range(NUM_CHUNKS):
        next_bd = chunk + 1 if chunk + 1 < NUM_CHUNKS else 0
        offset = chunk * CHUNK_DWORDS * 4
        lines += [
            _npu_writebd(0, chunk, CHUNK_DWORDS, offset, next_bd=next_bd, use_next_bd=chunk + 1 < NUM_CHUNKS),
            _npu_address_patch(0, chunk, 0, offset),
        ]

    lines += [
        _npu_push_queue(0, "MM2S", 0, 0, issue_token=True),
        _npu_sync(0, 0, direction=1),
        _npu_sync(0, 0, direction=0),
        "    }",
    ]
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.flow(%shim0, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 0, %shim0, DMA : 0)

    func.func private @record_descriptor_chunk(memref<{CHUNK_DWORDS}xi32>, memref<{TOTAL_OUTPUT_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/linked_bd_chain.o"}}

    %in_ping = aie.buffer(%core) {{sym_name = "in_ping"}} : memref<{CHUNK_DWORDS}xi32>
    %in_pong = aie.buffer(%core) {{sym_name = "in_pong"}} : memref<{CHUNK_DWORDS}xi32>
    %out = aie.buffer(%core) {{sym_name = "out"}} : memref<{TOTAL_OUTPUT_DWORDS}xi32>
{_lock_pair("input", 0, init_empty=2)}
{_lock_pair("output", 2)}

    %core_program = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %chunks = arith.constant {NUM_CHUNKS} : index
      %chunk_len = arith.constant {CHUNK_DWORDS} : i32

      aie.use_lock(%output_empty, AcquireGreaterEqual, 1)
      scf.for %chunk = %c0 to %chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%input_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @record_descriptor_chunk(%in_pong, %out, %chunk_i32, %chunk_len)
            : (memref<{CHUNK_DWORDS}xi32>, memref<{TOTAL_OUTPUT_DWORDS}xi32>, i32, i32) -> ()
        }} else {{
          func.call @record_descriptor_chunk(%in_ping, %out, %chunk_i32, %chunk_len)
            : (memref<{CHUNK_DWORDS}xi32>, memref<{TOTAL_OUTPUT_DWORDS}xi32>, i32, i32) -> ()
        }}
        aie.use_lock(%input_empty, Release, 1)
      }}
      aie.use_lock(%output_full, Release, 1)
      aie.end
    }}

    %core_dma = aie.mem(%core) {{
      %0 = aie.dma_start(S2MM, 0, ^in_ping, ^out_start)
    ^in_ping:
      aie.use_lock(%input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_ping : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%input_full, Release, 1)
      aie.next_bd ^in_pong
    ^in_pong:
      aie.use_lock(%input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_pong : memref<{CHUNK_DWORDS}xi32>, 0, {CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%input_full, Release, 1)
      aie.next_bd ^in_ping

    ^out_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%out : memref<{TOTAL_OUTPUT_DWORDS}xi32>, 0, {TOTAL_OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%output_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}

{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
