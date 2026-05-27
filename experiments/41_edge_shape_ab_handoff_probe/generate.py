"""Generate raw MLIR-AIE for exp41 edge shape-A -> shape-B handoff probe."""

from pathlib import Path

CURRENT_DWORDS = 512
HISTORY_DWORDS = 2048
STATE_DWORDS = 17
OUTPUT_DWORDS = 512


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
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


def _npu_sync(column: int, channel: int) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _lock_pair(tile: str, prefix: str, base: int) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = 1 : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _shape_a() -> str:
    return f"""
    %shape_a_current = aie.buffer(%shape_a) {{sym_name = "shape_a_current"}} : memref<{CURRENT_DWORDS}xi32>
    %shape_a_history = aie.buffer(%shape_a) {{sym_name = "shape_a_history"}} : memref<{HISTORY_DWORDS}xi32>
    %shape_a_state = aie.buffer(%shape_a) {{sym_name = "shape_a_state"}} : memref<{STATE_DWORDS}xi32>
{_lock_pair("shape_a", "current", 0)}
{_lock_pair("shape_a", "history", 2)}
{_lock_pair("shape_a", "state", 4)}

    %shape_a_core = aie.core(%shape_a) {{
      aie.use_lock(%shape_a_current_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_a_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_a_state_empty, AcquireGreaterEqual, 1)
      func.call @shape_a_make_state(%shape_a_current, %shape_a_history, %shape_a_state)
        : (memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{STATE_DWORDS}xi32>) -> ()
      aie.use_lock(%shape_a_current_empty, Release, 1)
      aie.use_lock(%shape_a_history_empty, Release, 1)
      aie.use_lock(%shape_a_state_full, Release, 1)
      aie.end
    }}

    %shape_a_mem = aie.mem(%shape_a) {{
      %0 = aie.dma_start(S2MM, 0, ^current_bd, ^history_start)
    ^current_bd:
      aie.use_lock(%shape_a_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_current : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%shape_a_current_full, Release, 1)
      aie.next_bd ^current_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^state_start)
    ^history_bd:
      aie.use_lock(%shape_a_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_history : memref<{HISTORY_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%shape_a_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^state_start:
      %2 = aie.dma_start(MM2S, 0, ^state_bd, ^end)
    ^state_bd:
      aie.use_lock(%shape_a_state_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_a_state : memref<{STATE_DWORDS}xi32>, 0, {STATE_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%shape_a_state_empty, Release, 1)
      aie.next_bd ^state_bd
    ^end:
      aie.end
    }}
"""


def _shape_b() -> str:
    return f"""
    %shape_b_state = aie.buffer(%shape_b) {{sym_name = "shape_b_state"}} : memref<{STATE_DWORDS}xi32>
    %shape_b_history = aie.buffer(%shape_b) {{sym_name = "shape_b_history"}} : memref<{HISTORY_DWORDS}xi32>
    %shape_b_output = aie.buffer(%shape_b) {{sym_name = "shape_b_output"}} : memref<{OUTPUT_DWORDS}xi32>
{_lock_pair("shape_b", "state", 0)}
{_lock_pair("shape_b", "history", 2)}
{_lock_pair("shape_b", "output", 4)}

    %shape_b_core = aie.core(%shape_b) {{
      aie.use_lock(%shape_b_state_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_b_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%shape_b_output_empty, AcquireGreaterEqual, 1)
      func.call @shape_b_consume_state(%shape_b_state, %shape_b_history, %shape_b_output)
        : (memref<{STATE_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) -> ()
      aie.use_lock(%shape_b_state_empty, Release, 1)
      aie.use_lock(%shape_b_history_empty, Release, 1)
      aie.use_lock(%shape_b_output_full, Release, 1)
      aie.end
    }}

    %shape_b_mem = aie.mem(%shape_b) {{
      %0 = aie.dma_start(S2MM, 0, ^state_bd, ^history_start)
    ^state_bd:
      aie.use_lock(%shape_b_state_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_state : memref<{STATE_DWORDS}xi32>, 0, {STATE_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%shape_b_state_full, Release, 1)
      aie.next_bd ^state_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^output_start)
    ^history_bd:
      aie.use_lock(%shape_b_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_history : memref<{HISTORY_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%shape_b_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^output_start:
      %2 = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:
      aie.use_lock(%shape_b_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%shape_b_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%shape_b_output_empty, Release, 1)
      aie.next_bd ^output_bd
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        [
            f"    aie.runtime_sequence(%current: memref<{CURRENT_DWORDS}xi32>, "
            f"%history_a: memref<{HISTORY_DWORDS}xi32>, "
            f"%history_b: memref<{HISTORY_DWORDS}xi32>, "
            f"%output: memref<{OUTPUT_DWORDS}xi32>) {{",
            _npu_writebd(1, 2, OUTPUT_DWORDS, 0),
            _npu_address_patch(1, 2, 3, 0),
            _npu_push_queue(1, "S2MM", 0, 2, issue_token=True),
            _npu_writebd(0, 0, CURRENT_DWORDS, 0),
            _npu_address_patch(0, 0, 0, 0),
            _npu_push_queue(0, "MM2S", 0, 0),
            _npu_writebd(0, 1, HISTORY_DWORDS, 0),
            _npu_address_patch(0, 1, 1, 0),
            _npu_push_queue(0, "MM2S", 1, 1),
            _npu_writebd(1, 0, HISTORY_DWORDS, 0),
            _npu_address_patch(1, 0, 2, 0),
            _npu_push_queue(1, "MM2S", 0, 0),
            _npu_sync(1, 0),
            "    }",
        ]
    )


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shape_a = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %shape_b = aie.tile(1, 2)

    aie.flow(%shim0, DMA : 0, %shape_a, DMA : 0)
    aie.flow(%shim0, DMA : 1, %shape_a, DMA : 1)
    aie.flow(%shape_a, DMA : 0, %shape_b, DMA : 0)
    aie.flow(%shim1, DMA : 0, %shape_b, DMA : 1)
    aie.flow(%shape_b, DMA : 0, %shim1, DMA : 0)

    func.func private @shape_a_make_state(memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{STATE_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/edge_handoff.o"}}
    func.func private @shape_b_consume_state(memref<{STATE_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/edge_handoff.o"}}

{_shape_a()}
{_shape_b()}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
