"""Small MLIR-AIE text helpers shared by qwen3-layer cases."""

from __future__ import annotations

import re


def shim_bd_address(column: int, bd_id: int) -> int:
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
        f"      aiex.npu.address_patch {{addr = {shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def npu_push_queue(
    column: int,
    direction: str,
    channel: int,
    bd_id: int,
    issue_token: bool = True,
) -> str:
    token = "true" if issue_token else "false"
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def flow(src_tile: str, src_dma: int, dst_tile: str, dst_dma: int) -> str:
    return f"    aie.flow(%{src_tile}, DMA : {src_dma}, %{dst_tile}, DMA : {dst_dma})"


def packet_flow(packet_id: int, src_tile: str, src_dma: int, dst_tile: str, dst_dma: int) -> str:
    return "\n".join(
        (
            f"    aie.packet_flow({packet_id}) {{",
            f"      aie.packet_source<%{src_tile}, DMA : {src_dma}>",
            f"      aie.packet_dest<%{dst_tile}, DMA : {dst_dma}>",
            "    }",
        )
    )


def require_unique_bd_ids(scope: str, bd_ids: tuple[int, ...]) -> list[str]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for bd_id in bd_ids:
        if bd_id in seen and bd_id not in duplicates:
            duplicates.append(bd_id)
        seen.add(bd_id)
    return [f"{scope}: duplicate BD id {bd_id}" for bd_id in duplicates]


def require_disjoint_bd_ids(scope: str, left: tuple[int, ...], right: tuple[int, ...]) -> list[str]:
    overlap = tuple(sorted(set(left).intersection(right)))
    return [f"{scope}: overlapping BD id {bd_id}" for bd_id in overlap]


def require_c1r1_s2mm3_high_bds(scope: str, bd_ids: tuple[int, ...]) -> list[str]:
    return [
        f"{scope}: c1r1 S2MM3 BD {bd_id} is in the illegal low bank"
        for bd_id in bd_ids
        if bd_id < 41
    ]


def require_compact_payload_slice(
    scope: str,
    source_offset: int,
    source_length: int,
    payload_offset: int,
    payload_length: int,
) -> list[str]:
    if source_offset == payload_offset and source_length == payload_length:
        return []
    return [
        f"{scope}: expected compact payload slice offset={payload_offset} "
        f"length={payload_length}, got offset={source_offset} length={source_length}"
    ]


def require_count(scope: str, name: str, actual: int, expected: int) -> list[str]:
    if actual == expected:
        return []
    return [f"{scope}: {name} count {actual} != {expected}"]


def require_kv16_attention_shapes(
    scope: str,
    k_window_dwords: int,
    v_window_dwords: int,
    kv_side_dwords: int,
    k_cache_side_dwords: int,
    v_cache_side_dwords: int,
    carrier_dwords: int,
    weight_dwords: int,
    scalar_dwords: int,
    output_dwords: int,
) -> list[str]:
    errors: list[str] = []
    if k_window_dwords != 2048:
        errors.append(f"{scope}: K window must be 2048 dwords, got {k_window_dwords}")
    if v_window_dwords != 2048:
        errors.append(f"{scope}: V window must be 2048 dwords, got {v_window_dwords}")
    if kv_side_dwords != 8192:
        errors.append(f"{scope}: KV side payload must be 8192 dwords, got {kv_side_dwords}")
    if k_cache_side_dwords != k_window_dwords * 2:
        errors.append(
            f"{scope}: K cache side must contain two windows, got {k_cache_side_dwords}"
        )
    if v_cache_side_dwords != v_window_dwords * 2:
        errors.append(
            f"{scope}: V cache side must contain two windows, got {v_cache_side_dwords}"
        )
    if carrier_dwords != 80:
        errors.append(f"{scope}: carrier must be 80 dwords, got {carrier_dwords}")
    if weight_dwords + scalar_dwords != carrier_dwords:
        errors.append(
            f"{scope}: carrier split {weight_dwords}+{scalar_dwords} != {carrier_dwords}"
        )
    if output_dwords != 512:
        errors.append(f"{scope}: attention return window must be 512 dwords, got {output_dwords}")
    return errors


def require_max_address_patch_arg(scope: str, mlir: str, max_arg_idx: int) -> list[str]:
    arg_indices = [int(match) for match in re.findall(r"arg_idx = ([0-9]+) : i32", mlir)]
    return [
        f"{scope}: address_patch arg_idx {arg_idx} exceeds supported max {max_arg_idx}"
        for arg_idx in sorted(set(arg_indices))
        if arg_idx > max_arg_idx
    ]
