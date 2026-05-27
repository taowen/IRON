"""Generate raw MLIR-AIE for exp42 attention-output -> O-projection handoff."""

from pathlib import Path

CURRENT_DWORDS = 512
HISTORY_DWORDS = 2048
ATTENTION_DWORDS = 512
ATTENTION_HALF_DWORDS = 256
O_WEIGHT_DWORDS = 2048
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


def _edge_attention() -> str:
    return f"""
    %edge_current = aie.buffer(%edge_attn) {{sym_name = "edge_current"}} : memref<{CURRENT_DWORDS}xi32>
    %edge_history = aie.buffer(%edge_attn) {{sym_name = "edge_history"}} : memref<{HISTORY_DWORDS}xi32>
    %edge_attention_lo = aie.buffer(%edge_attn) {{sym_name = "edge_attention_lo"}} : memref<{ATTENTION_HALF_DWORDS}xi32>
    %edge_attention_hi = aie.buffer(%edge_attn) {{sym_name = "edge_attention_hi"}} : memref<{ATTENTION_HALF_DWORDS}xi32>
{_lock_pair("edge_attn", "current", 0)}
{_lock_pair("edge_attn", "history", 2)}
{_lock_pair("edge_attn", "attention_lo", 4)}
{_lock_pair("edge_attn", "attention_hi", 6)}

    %edge_attn_core = aie.core(%edge_attn) {{
      aie.use_lock(%edge_attn_current_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_attn_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_attn_attention_lo_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%edge_attn_attention_hi_empty, AcquireGreaterEqual, 1)
      func.call @edge_make_attention(%edge_current, %edge_history, %edge_attention_lo, %edge_attention_hi)
        : (memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>) -> ()
      aie.use_lock(%edge_attn_current_empty, Release, 1)
      aie.use_lock(%edge_attn_history_empty, Release, 1)
      aie.use_lock(%edge_attn_attention_lo_full, Release, 1)
      aie.use_lock(%edge_attn_attention_hi_full, Release, 1)
      aie.end
    }}

    %edge_attn_mem = aie.mem(%edge_attn) {{
      %0 = aie.dma_start(S2MM, 0, ^current_bd, ^history_start)
    ^current_bd:
      aie.use_lock(%edge_attn_current_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_current : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%edge_attn_current_full, Release, 1)
      aie.next_bd ^current_bd

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^history_bd, ^attention_start)
    ^history_bd:
      aie.use_lock(%edge_attn_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_history : memref<{HISTORY_DWORDS}xi32>, 0, {HISTORY_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%edge_attn_history_full, Release, 1)
      aie.next_bd ^history_bd

    ^attention_start:
      %2 = aie.dma_start(MM2S, 0, ^attention_lo, ^end)
    ^attention_lo:
      aie.use_lock(%edge_attn_attention_lo_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_attention_lo : memref<{ATTENTION_HALF_DWORDS}xi32>, 0, {ATTENTION_HALF_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%edge_attn_attention_lo_empty, Release, 1)
      aie.next_bd ^attention_hi
    ^attention_hi:
      aie.use_lock(%edge_attn_attention_hi_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%edge_attention_hi : memref<{ATTENTION_HALF_DWORDS}xi32>, 0, {ATTENTION_HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%edge_attn_attention_hi_empty, Release, 1)
      aie.next_bd ^attention_lo
    ^end:
      aie.end
    }}
"""


def _o_projection() -> str:
    return f"""
    %o_attention_lo = aie.buffer(%o_proj) {{sym_name = "o_attention_lo"}} : memref<{ATTENTION_HALF_DWORDS}xi32>
    %o_attention_hi = aie.buffer(%o_proj) {{sym_name = "o_attention_hi"}} : memref<{ATTENTION_HALF_DWORDS}xi32>
    %o_weight = aie.buffer(%o_proj) {{sym_name = "o_weight"}} : memref<{O_WEIGHT_DWORDS}xi32>
    %o_output = aie.buffer(%o_proj) {{sym_name = "o_output"}} : memref<{OUTPUT_DWORDS}xi32>
{_lock_pair("o_proj", "attention_lo", 0)}
{_lock_pair("o_proj", "attention_hi", 2)}
{_lock_pair("o_proj", "weight", 4)}
{_lock_pair("o_proj", "output", 6)}

    %o_proj_core = aie.core(%o_proj) {{
      aie.use_lock(%o_proj_attention_lo_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_attention_hi_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_weight_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_output_empty, AcquireGreaterEqual, 1)
      func.call @o_project_attention(%o_attention_lo, %o_attention_hi, %o_weight, %o_output)
        : (memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) -> ()
      aie.use_lock(%o_proj_attention_lo_empty, Release, 1)
      aie.use_lock(%o_proj_attention_hi_empty, Release, 1)
      aie.use_lock(%o_proj_weight_empty, Release, 1)
      aie.use_lock(%o_proj_output_full, Release, 1)
      aie.end
    }}

    %o_proj_mem = aie.mem(%o_proj) {{
      %0 = aie.dma_start(S2MM, 0, ^attention_lo, ^weight_start)
    ^attention_lo:
      aie.use_lock(%o_proj_attention_lo_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_attention_lo : memref<{ATTENTION_HALF_DWORDS}xi32>, 0, {ATTENTION_HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%o_proj_attention_lo_full, Release, 1)
      aie.next_bd ^attention_hi
    ^attention_hi:
      aie.use_lock(%o_proj_attention_hi_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_attention_hi : memref<{ATTENTION_HALF_DWORDS}xi32>, 0, {ATTENTION_HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%o_proj_attention_hi_full, Release, 1)
      aie.next_bd ^attention_lo

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^weight_bd, ^output_start)
    ^weight_bd:
      aie.use_lock(%o_proj_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_weight : memref<{O_WEIGHT_DWORDS}xi32>, 0, {O_WEIGHT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%o_proj_weight_full, Release, 1)
      aie.next_bd ^weight_bd

    ^output_start:
      %2 = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:
      aie.use_lock(%o_proj_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%o_proj_output_empty, Release, 1)
      aie.next_bd ^output_bd
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        [
            f"    aie.runtime_sequence(%current: memref<{CURRENT_DWORDS}xi32>, "
            f"%history: memref<{HISTORY_DWORDS}xi32>, "
            f"%o_weight_bo: memref<{O_WEIGHT_DWORDS}xi32>, "
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
            _npu_writebd(1, 0, O_WEIGHT_DWORDS, 0),
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
    %edge_attn = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %o_proj = aie.tile(1, 2)

    aie.flow(%shim0, DMA : 0, %edge_attn, DMA : 0)
    aie.flow(%shim0, DMA : 1, %edge_attn, DMA : 1)
    aie.flow(%edge_attn, DMA : 0, %o_proj, DMA : 0)
    aie.flow(%shim1, DMA : 0, %o_proj, DMA : 1)
    aie.flow(%o_proj, DMA : 0, %shim1, DMA : 0)

    func.func private @edge_make_attention(memref<{CURRENT_DWORDS}xi32>, memref<{HISTORY_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/attention_to_o.o"}}
    func.func private @o_project_attention(memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{ATTENTION_HALF_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/attention_to_o.o"}}

{_edge_attention()}
{_o_projection()}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
