"""Shared K/V cache writeback and rounded KV scan dataflow."""

from __future__ import annotations

from attention_dataflow import KV_OUT_BDS
from cases.decode_cache_reference import (
    CACHE_BLOCK_DWORDS,
    CURRENT_DWORDS,
    DecodeSchedule,
    HEAD_DWORDS,
    K_WINDOW_DWORDS,
    KV_HEADS,
    V_WINDOW_DWORDS,
    WINDOW_HEAD_DWORDS,
)
from mlir_utils import npu_address_patch, npu_push_queue, npu_writebd

K_SCAN_BD = 0
V_SCAN_BD = 1
KV_SCAN_BDS = (K_SCAN_BD, V_SCAN_BD)
CURRENT_WRITE_BDS = (2, 3)
CURRENT_WRITE_CHANNEL = 1
KV_SPLIT_K_IN_BDS = (0, 1)
KV_SPLIT_V_IN_BDS = (28, 29)
K_SIDE_SCAN_DWORDS = K_WINDOW_DWORDS * 2
V_SIDE_SCAN_DWORDS = V_WINDOW_DWORDS * 2


def shape_blocks_name(tile: str) -> str:
    return f"{tile}_blocks"


def shape_tail_tokens_name(tile: str) -> str:
    return f"{tile}_tail_tokens"


def shape_runtime_start_name(tile: str) -> str:
    return f"{tile}_runtime_start"


def kv_split_scan_memtile(side: int) -> str:
    tile = "kv_left" if side == 0 else "kv_right"
    logical_slots = (
        ("k", 0, 0, K_WINDOW_DWORDS),
        ("v", 1, 0, V_WINDOW_DWORDS),
        ("k", 2, K_WINDOW_DWORDS, K_WINDOW_DWORDS),
        ("v", 3, V_WINDOW_DWORDS, V_WINDOW_DWORDS),
    )
    locks = []
    for _, slot, _, _ in logical_slots:
        locks.append(
            f"""    %{tile}_slot{slot}_empty = aie.lock(%{tile}, {slot * 2}) {{init = 1 : i32, sym_name = "{tile}_slot{slot}_empty"}}
    %{tile}_slot{slot}_full = aie.lock(%{tile}, {slot * 2 + 1}) {{init = 0 : i32, sym_name = "{tile}_slot{slot}_full"}}"""
        )

    outputs = []
    for slot, bd_id in enumerate(KV_OUT_BDS):
        next_start = f"^out{slot + 1}_start" if slot + 1 < len(KV_OUT_BDS) else "^end"
        plane, _, offset, length = logical_slots[slot]
        buffer = f"%{tile}_{plane}_payload"
        payload_dwords = K_SIDE_SCAN_DWORDS if plane == "k" else V_SIDE_SCAN_DWORDS
        outputs.append(f"""    ^out{slot}_start:
      %out{slot}_dma = aie.dma_start(MM2S, {slot}, ^out{slot}, {next_start})
    ^out{slot}:
      aie.use_lock(%{tile}_slot{slot}_full, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer} : memref<{payload_dwords}xi32>, {offset}, {length}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%{tile}_slot{slot}_empty, Release, 1)
      aie.next_bd ^out{slot}""")

    return f"""
    %{tile}_k_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_k_payload"}} : memref<{K_SIDE_SCAN_DWORDS}xi32>
    %{tile}_v_payload = aie.buffer(%{tile}) {{sym_name = "{tile}_v_payload"}} : memref<{V_SIDE_SCAN_DWORDS}xi32>
{chr(10).join(locks)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %k_dma = aie.dma_start(S2MM, 0, ^k0_in, ^v_start)
    ^k0_in:
      aie.use_lock(%{tile}_slot0_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k_payload : memref<{K_SIDE_SCAN_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_K_IN_BDS[0]} : i32, next_bd_id = {KV_SPLIT_K_IN_BDS[1]} : i32}}
      aie.use_lock(%{tile}_slot0_full, Release, 1)
      aie.next_bd ^k1_in
    ^k1_in:
      aie.use_lock(%{tile}_slot2_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k_payload : memref<{K_SIDE_SCAN_DWORDS}xi32>, {K_WINDOW_DWORDS}, {K_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_K_IN_BDS[1]} : i32, next_bd_id = {KV_SPLIT_K_IN_BDS[0]} : i32}}
      aie.use_lock(%{tile}_slot2_full, Release, 1)
      aie.next_bd ^k0_in

    ^v_start:
      %v_dma = aie.dma_start(S2MM, 1, ^v0_in, ^out0_start)
    ^v0_in:
      aie.use_lock(%{tile}_slot1_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v_payload : memref<{V_SIDE_SCAN_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_V_IN_BDS[0]} : i32, next_bd_id = {KV_SPLIT_V_IN_BDS[1]} : i32}}
      aie.use_lock(%{tile}_slot1_full, Release, 1)
      aie.next_bd ^v1_in
    ^v1_in:
      aie.use_lock(%{tile}_slot3_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v_payload : memref<{V_SIDE_SCAN_DWORDS}xi32>, {V_WINDOW_DWORDS}, {V_WINDOW_DWORDS}) {{bd_id = {KV_SPLIT_V_IN_BDS[1]} : i32, next_bd_id = {KV_SPLIT_V_IN_BDS[0]} : i32}}
      aie.use_lock(%{tile}_slot3_full, Release, 1)
      aie.next_bd ^v0_in

{chr(10).join(outputs)}
    ^end:
      aie.end
    }}
"""


def push_current_cache_write(column: int, cache_arg: int, schedule: DecodeSchedule) -> list[str]:
    lines: list[str] = []
    base_offset = schedule.current_write_byte_offset
    half_current_dwords = CURRENT_DWORDS // 2
    half_head_dwords = HEAD_DWORDS // 2
    for parity, bd_id in enumerate(CURRENT_WRITE_BDS):
        has_next = parity + 1 < len(CURRENT_WRITE_BDS)
        next_bd = CURRENT_WRITE_BDS[parity + 1] if has_next else 0
        offset = base_offset + parity * 4
        lines.extend(
            (
                npu_writebd(
                    column,
                    bd_id,
                    half_current_dwords,
                    offset,
                    next_bd=next_bd,
                    use_next_bd=has_next,
                    d0_size=half_head_dwords,
                    d0_stride=1,
                    d1_size=KV_HEADS,
                    d1_stride=WINDOW_HEAD_DWORDS - 1,
                ),
                npu_address_patch(column, bd_id, cache_arg, offset),
            )
        )
    lines.append(npu_push_queue(column, "S2MM", CURRENT_WRITE_CHANNEL, CURRENT_WRITE_BDS[0]))
    return lines


def push_kv_scan_from_cache(
    column: int,
    k_arg: int,
    v_arg: int,
    head_base: int,
    schedule: DecodeSchedule,
) -> list[str]:
    k_offset = head_base * WINDOW_HEAD_DWORDS * 4
    v_offset = head_base * WINDOW_HEAD_DWORDS * 4
    return [
        npu_writebd(
            column,
            K_SCAN_BD,
            K_SIDE_SCAN_DWORDS,
            k_offset,
            iteration_size=schedule.kv_blocks,
            iteration_stride=CACHE_BLOCK_DWORDS - 1,
        ),
        npu_address_patch(column, K_SCAN_BD, k_arg, k_offset),
        npu_push_queue(column, "MM2S", 0, K_SCAN_BD, repeat_count=schedule.kv_blocks - 1),
        npu_writebd(
            column,
            V_SCAN_BD,
            V_SIDE_SCAN_DWORDS,
            v_offset,
            iteration_size=schedule.kv_blocks,
            iteration_stride=CACHE_BLOCK_DWORDS - 1,
        ),
        npu_address_patch(column, V_SCAN_BD, v_arg, v_offset),
        npu_push_queue(column, "MM2S", 1, V_SCAN_BD, repeat_count=schedule.kv_blocks - 1),
    ]
