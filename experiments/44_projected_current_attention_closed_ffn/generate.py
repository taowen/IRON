"""Generate raw MLIR-AIE for exp44 projected current-write attention closed FFN."""

from pathlib import Path

CONTEXT_LEN = 31
HEAD_DIM = 128
HIDDEN_DWORDS = 512
HALF_DWORDS = 256
QUERY_DWORDS = 512
PLANE_DWORDS = 32 * HEAD_DIM
KV_CACHE_DWORDS = 2 * PLANE_DWORDS
O_WEIGHT_DWORDS = 2048
FFN_WEIGHT_DWORDS = 4096
OUTPUT_DWORDS = 512
CURRENT_TOKEN = CONTEXT_LEN - 1
CURRENT_K_OFFSET = CURRENT_TOKEN * HEAD_DIM
CURRENT_V_OFFSET = PLANE_DWORDS + CURRENT_K_OFFSET


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


def _projection() -> str:
    return f"""
    %proj_hidden = aie.buffer(%proj) {{sym_name = "proj_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %proj_query_lo = aie.buffer(%proj) {{sym_name = "proj_query_lo"}} : memref<{HALF_DWORDS}xi32>
    %proj_query_hi = aie.buffer(%proj) {{sym_name = "proj_query_hi"}} : memref<{HALF_DWORDS}xi32>
    %proj_current_k = aie.buffer(%proj) {{sym_name = "proj_current_k"}} : memref<{HEAD_DIM}xi32>
    %proj_current_v = aie.buffer(%proj) {{sym_name = "proj_current_v"}} : memref<{HEAD_DIM}xi32>
{_lock_pair("proj", "hidden", 0)}
{_lock_pair("proj", "query_lo", 2)}
{_lock_pair("proj", "query_hi", 4)}
{_lock_pair("proj", "current_k", 6)}
{_lock_pair("proj", "current_v", 8)}

    %proj_core = aie.core(%proj) {{
      aie.use_lock(%proj_hidden_full, AcquireGreaterEqual, 1)
      aie.use_lock(%proj_query_lo_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%proj_query_hi_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%proj_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%proj_current_v_empty, AcquireGreaterEqual, 1)
      func.call @project_query_current(%proj_hidden, %proj_query_lo, %proj_query_hi, %proj_current_k, %proj_current_v)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HEAD_DIM}xi32>, memref<{HEAD_DIM}xi32>) -> ()
      aie.use_lock(%proj_hidden_empty, Release, 1)
      aie.use_lock(%proj_query_lo_full, Release, 1)
      aie.use_lock(%proj_query_hi_full, Release, 1)
      aie.use_lock(%proj_current_k_full, Release, 1)
      aie.use_lock(%proj_current_v_full, Release, 1)
      aie.end
    }}

    %proj_mem = aie.mem(%proj) {{
      %0 = aie.dma_start(S2MM, 0, ^hidden_bd, ^query_start)
    ^hidden_bd:
      aie.use_lock(%proj_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%proj_hidden_full, Release, 1)
      aie.next_bd ^hidden_bd

    ^query_start:
      %1 = aie.dma_start(MM2S, 0, ^query_lo, ^current_start)
    ^query_lo:
      aie.use_lock(%proj_query_lo_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_query_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%proj_query_lo_empty, Release, 1)
      aie.next_bd ^query_hi
    ^query_hi:
      aie.use_lock(%proj_query_hi_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_query_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%proj_query_hi_empty, Release, 1)
      aie.next_bd ^query_lo

    ^current_start:
      %2 = aie.dma_start(MM2S, 1, ^current_k, ^end)
    ^current_k:
      aie.use_lock(%proj_current_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_current_k : memref<{HEAD_DIM}xi32>, 0, {HEAD_DIM}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%proj_current_k_empty, Release, 1)
      aie.next_bd ^current_v
    ^current_v:
      aie.use_lock(%proj_current_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%proj_current_v : memref<{HEAD_DIM}xi32>, 0, {HEAD_DIM}) {{bd_id = 5 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%proj_current_v_empty, Release, 1)
      aie.next_bd ^current_k
    ^end:
      aie.end
    }}
"""


def _attention() -> str:
    return f"""
    %attn_query_lo = aie.buffer(%attn) {{sym_name = "attn_query_lo"}} : memref<{HALF_DWORDS}xi32>
    %attn_query_hi = aie.buffer(%attn) {{sym_name = "attn_query_hi"}} : memref<{HALF_DWORDS}xi32>
    %attn_k_history = aie.buffer(%attn) {{sym_name = "attn_k_history"}} : memref<{PLANE_DWORDS}xi32>
    %attn_v_history = aie.buffer(%attn) {{sym_name = "attn_v_history"}} : memref<{PLANE_DWORDS}xi32>
    %attn_out_lo = aie.buffer(%attn) {{sym_name = "attn_out_lo"}} : memref<{HALF_DWORDS}xi32>
    %attn_out_hi = aie.buffer(%attn) {{sym_name = "attn_out_hi"}} : memref<{HALF_DWORDS}xi32>
{_lock_pair("attn", "query_lo", 0)}
{_lock_pair("attn", "query_hi", 2)}
{_lock_pair("attn", "k_history", 4)}
{_lock_pair("attn", "v_history", 6)}
{_lock_pair("attn", "out_lo", 8)}
{_lock_pair("attn", "out_hi", 10)}

    %attn_core = aie.core(%attn) {{
      aie.use_lock(%attn_query_lo_full, AcquireGreaterEqual, 1)
      aie.use_lock(%attn_query_hi_full, AcquireGreaterEqual, 1)
      aie.use_lock(%attn_k_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%attn_v_history_full, AcquireGreaterEqual, 1)
      aie.use_lock(%attn_out_lo_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%attn_out_hi_empty, AcquireGreaterEqual, 1)
      func.call @attention_from_cache(%attn_query_lo, %attn_query_hi, %attn_k_history, %attn_v_history, %attn_out_lo, %attn_out_hi)
        : (memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{PLANE_DWORDS}xi32>, memref<{PLANE_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>) -> ()
      aie.use_lock(%attn_query_lo_empty, Release, 1)
      aie.use_lock(%attn_query_hi_empty, Release, 1)
      aie.use_lock(%attn_k_history_empty, Release, 1)
      aie.use_lock(%attn_v_history_empty, Release, 1)
      aie.use_lock(%attn_out_lo_full, Release, 1)
      aie.use_lock(%attn_out_hi_full, Release, 1)
      aie.end
    }}

    %attn_mem = aie.mem(%attn) {{
      %0 = aie.dma_start(S2MM, 0, ^query_lo, ^history_start)
    ^query_lo:
      aie.use_lock(%attn_query_lo_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_query_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%attn_query_lo_full, Release, 1)
      aie.next_bd ^query_hi
    ^query_hi:
      aie.use_lock(%attn_query_hi_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_query_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%attn_query_hi_full, Release, 1)
      aie.next_bd ^query_lo

    ^history_start:
      %1 = aie.dma_start(S2MM, 1, ^k_history, ^out_start)
    ^k_history:
      aie.use_lock(%attn_k_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_k_history : memref<{PLANE_DWORDS}xi32>, 0, {PLANE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%attn_k_history_full, Release, 1)
      aie.next_bd ^v_history
    ^v_history:
      aie.use_lock(%attn_v_history_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_v_history : memref<{PLANE_DWORDS}xi32>, 0, {PLANE_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%attn_v_history_full, Release, 1)
      aie.next_bd ^k_history

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_lo, ^end)
    ^out_lo:
      aie.use_lock(%attn_out_lo_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_out_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%attn_out_lo_empty, Release, 1)
      aie.next_bd ^out_hi
    ^out_hi:
      aie.use_lock(%attn_out_hi_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%attn_out_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%attn_out_hi_empty, Release, 1)
      aie.next_bd ^out_lo
    ^end:
      aie.end
    }}
"""


def _o_projection() -> str:
    return f"""
    %o_attention_lo = aie.buffer(%o_proj) {{sym_name = "o_attention_lo"}} : memref<{HALF_DWORDS}xi32>
    %o_attention_hi = aie.buffer(%o_proj) {{sym_name = "o_attention_hi"}} : memref<{HALF_DWORDS}xi32>
    %o_weight = aie.buffer(%o_proj) {{sym_name = "o_weight"}} : memref<{O_WEIGHT_DWORDS}xi32>
    %o_out_lo = aie.buffer(%o_proj) {{sym_name = "o_out_lo"}} : memref<{HALF_DWORDS}xi32>
    %o_out_hi = aie.buffer(%o_proj) {{sym_name = "o_out_hi"}} : memref<{HALF_DWORDS}xi32>
{_lock_pair("o_proj", "attention_lo", 0)}
{_lock_pair("o_proj", "attention_hi", 2)}
{_lock_pair("o_proj", "weight", 4)}
{_lock_pair("o_proj", "out_lo", 6)}
{_lock_pair("o_proj", "out_hi", 8)}

    %o_proj_core = aie.core(%o_proj) {{
      aie.use_lock(%o_proj_attention_lo_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_attention_hi_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_weight_full, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_out_lo_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%o_proj_out_hi_empty, AcquireGreaterEqual, 1)
      func.call @o_project_attention(%o_attention_lo, %o_attention_hi, %o_weight, %o_out_lo, %o_out_hi)
        : (memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>) -> ()
      aie.use_lock(%o_proj_attention_lo_empty, Release, 1)
      aie.use_lock(%o_proj_attention_hi_empty, Release, 1)
      aie.use_lock(%o_proj_weight_empty, Release, 1)
      aie.use_lock(%o_proj_out_lo_full, Release, 1)
      aie.use_lock(%o_proj_out_hi_full, Release, 1)
      aie.end
    }}

    %o_proj_mem = aie.mem(%o_proj) {{
      %0 = aie.dma_start(S2MM, 0, ^attention_lo, ^weight_start)
    ^attention_lo:
      aie.use_lock(%o_proj_attention_lo_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_attention_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%o_proj_attention_lo_full, Release, 1)
      aie.next_bd ^attention_hi
    ^attention_hi:
      aie.use_lock(%o_proj_attention_hi_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_attention_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%o_proj_attention_hi_full, Release, 1)
      aie.next_bd ^attention_lo

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^weight_bd, ^out_start)
    ^weight_bd:
      aie.use_lock(%o_proj_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_weight : memref<{O_WEIGHT_DWORDS}xi32>, 0, {O_WEIGHT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%o_proj_weight_full, Release, 1)
      aie.next_bd ^weight_bd

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_lo, ^end)
    ^out_lo:
      aie.use_lock(%o_proj_out_lo_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%o_proj_out_lo_empty, Release, 1)
      aie.next_bd ^out_hi
    ^out_hi:
      aie.use_lock(%o_proj_out_hi_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%o_proj_out_hi_empty, Release, 1)
      aie.next_bd ^out_lo
    ^end:
      aie.end
    }}
"""


def _ffn_tail() -> str:
    return f"""
    %ffn_o_lo = aie.buffer(%ffn_tail) {{sym_name = "ffn_o_lo"}} : memref<{HALF_DWORDS}xi32>
    %ffn_o_hi = aie.buffer(%ffn_tail) {{sym_name = "ffn_o_hi"}} : memref<{HALF_DWORDS}xi32>
    %ffn_weight = aie.buffer(%ffn_tail) {{sym_name = "ffn_weight"}} : memref<{FFN_WEIGHT_DWORDS}xi32>
    %ffn_gate = aie.buffer(%ffn_tail) {{sym_name = "ffn_gate"}} : memref<{OUTPUT_DWORDS}xi32>
    %ffn_up = aie.buffer(%ffn_tail) {{sym_name = "ffn_up"}} : memref<{OUTPUT_DWORDS}xi32>
    %ffn_swiglu = aie.buffer(%ffn_tail) {{sym_name = "ffn_swiglu"}} : memref<{OUTPUT_DWORDS}xi32>
    %ffn_output = aie.buffer(%ffn_tail) {{sym_name = "ffn_output"}} : memref<{OUTPUT_DWORDS}xi32>
{_lock_pair("ffn_tail", "o_lo", 0)}
{_lock_pair("ffn_tail", "o_hi", 2)}
{_lock_pair("ffn_tail", "weight", 4)}
{_lock_pair("ffn_tail", "output", 6)}

    %ffn_tail_core = aie.core(%ffn_tail) {{
      aie.use_lock(%ffn_tail_o_lo_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ffn_tail_o_hi_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ffn_tail_weight_full, AcquireGreaterEqual, 1)
      aie.use_lock(%ffn_tail_output_empty, AcquireGreaterEqual, 1)
      func.call @ffn_tail(%ffn_o_lo, %ffn_o_hi, %ffn_weight, %ffn_gate, %ffn_up, %ffn_swiglu, %ffn_output)
        : (memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{FFN_WEIGHT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) -> ()
      aie.use_lock(%ffn_tail_o_lo_empty, Release, 1)
      aie.use_lock(%ffn_tail_o_hi_empty, Release, 1)
      aie.use_lock(%ffn_tail_weight_empty, Release, 1)
      aie.use_lock(%ffn_tail_output_full, Release, 1)
      aie.end
    }}

    %ffn_tail_mem = aie.mem(%ffn_tail) {{
      %0 = aie.dma_start(S2MM, 0, ^o_lo, ^weight_start)
    ^o_lo:
      aie.use_lock(%ffn_tail_o_lo_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_o_lo : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%ffn_tail_o_lo_full, Release, 1)
      aie.next_bd ^o_hi
    ^o_hi:
      aie.use_lock(%ffn_tail_o_hi_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_o_hi : memref<{HALF_DWORDS}xi32>, 0, {HALF_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%ffn_tail_o_hi_full, Release, 1)
      aie.next_bd ^o_lo

    ^weight_start:
      %1 = aie.dma_start(S2MM, 1, ^weight_bd, ^output_start)
    ^weight_bd:
      aie.use_lock(%ffn_tail_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_weight : memref<{FFN_WEIGHT_DWORDS}xi32>, 0, {FFN_WEIGHT_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%ffn_tail_weight_full, Release, 1)
      aie.next_bd ^weight_bd

    ^output_start:
      %2 = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:
      aie.use_lock(%ffn_tail_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%ffn_tail_output_empty, Release, 1)
      aie.next_bd ^output_bd
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        [
            f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_DWORDS}xi32>, "
            f"%kv_cache: memref<{KV_CACHE_DWORDS}xi32>, "
            f"%o_weight_bo: memref<{O_WEIGHT_DWORDS}xi32>, "
            f"%ffn_weight_bo: memref<{FFN_WEIGHT_DWORDS}xi32>, "
            f"%output: memref<{OUTPUT_DWORDS}xi32>) {{",
            _npu_writebd(3, 2, OUTPUT_DWORDS, 0),
            _npu_address_patch(3, 2, 4, 0),
            _npu_push_queue(3, "S2MM", 0, 2, issue_token=True),
            _npu_writebd(0, 2, HEAD_DIM, CURRENT_K_OFFSET * 4),
            _npu_address_patch(0, 2, 1, CURRENT_K_OFFSET * 4),
            _npu_push_queue(0, "S2MM", 0, 2),
            _npu_writebd(0, 3, HEAD_DIM, CURRENT_V_OFFSET * 4),
            _npu_address_patch(0, 3, 1, CURRENT_V_OFFSET * 4),
            _npu_push_queue(0, "S2MM", 0, 3, issue_token=True),
            _npu_writebd(0, 0, HIDDEN_DWORDS, 0),
            _npu_address_patch(0, 0, 0, 0),
            _npu_push_queue(0, "MM2S", 0, 0),
            _npu_sync(0, 0),
            _npu_writebd(1, 0, PLANE_DWORDS, 0),
            _npu_address_patch(1, 0, 1, 0),
            _npu_push_queue(1, "MM2S", 0, 0),
            _npu_writebd(1, 1, PLANE_DWORDS, PLANE_DWORDS * 4),
            _npu_address_patch(1, 1, 1, PLANE_DWORDS * 4),
            _npu_push_queue(1, "MM2S", 0, 1),
            _npu_writebd(2, 0, O_WEIGHT_DWORDS, 0),
            _npu_address_patch(2, 0, 2, 0),
            _npu_push_queue(2, "MM2S", 0, 0),
            _npu_writebd(3, 0, FFN_WEIGHT_DWORDS, 0),
            _npu_address_patch(3, 0, 3, 0),
            _npu_push_queue(3, "MM2S", 0, 0),
            _npu_sync(3, 0),
            "    }",
        ]
    )


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %proj = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %attn = aie.tile(1, 2)
    %shim2 = aie.tile(2, 0)
    %o_proj = aie.tile(2, 2)
    %shim3 = aie.tile(3, 0)
    %ffn_tail = aie.tile(3, 2)

    aie.flow(%shim0, DMA : 0, %proj, DMA : 0)
    aie.flow(%proj, DMA : 0, %attn, DMA : 0)
    aie.flow(%proj, DMA : 1, %shim0, DMA : 0)
    aie.flow(%shim1, DMA : 0, %attn, DMA : 1)
    aie.flow(%attn, DMA : 0, %o_proj, DMA : 0)
    aie.flow(%shim2, DMA : 0, %o_proj, DMA : 1)
    aie.flow(%o_proj, DMA : 0, %ffn_tail, DMA : 0)
    aie.flow(%shim3, DMA : 0, %ffn_tail, DMA : 1)
    aie.flow(%ffn_tail, DMA : 0, %shim3, DMA : 0)

    func.func private @project_query_current(memref<{HIDDEN_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HEAD_DIM}xi32>, memref<{HEAD_DIM}xi32>) attributes {{link_with = "{experiment_dir}/projected_closed_tail.o"}}
    func.func private @attention_from_cache(memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{PLANE_DWORDS}xi32>, memref<{PLANE_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/projected_closed_tail.o"}}
    func.func private @o_project_attention(memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/projected_closed_tail.o"}}
    func.func private @ffn_tail(memref<{HALF_DWORDS}xi32>, memref<{HALF_DWORDS}xi32>, memref<{FFN_WEIGHT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/projected_closed_tail.o"}}

{_projection()}
{_attention()}
{_o_projection()}
{_ffn_tail()}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
