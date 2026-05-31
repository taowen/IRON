"""Small MLIR-AIE text helpers shared by qwen3-layer cases."""

from __future__ import annotations

import re

MAIN16_LAYER_SCHEDULER_NAME = "q4nx_main16_layer_scheduler"
MAIN16_CONTROL_ADDR = 0x3D00
MAIN16_CONTROL_BYTES = 24
MAIN16_RECORD_CONTROL_GUARD_BYTES = 0xE4


def shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def npu_writebd(
    column: int,
    bd_id: int,
    buffer_length: int,
    buffer_offset: int,
    next_bd: int = 0,
    use_next_bd: bool = False,
    d0_size: int = 0,
    d0_stride: int = 0,
    d1_size: int = 0,
    d1_stride: int = 0,
    d2_size: int = 0,
    d2_stride: int = 0,
    iteration_size: int = 0,
    iteration_stride: int = 0,
    enable_packet: bool = False,
    packet_id: int = 0,
    packet_type: int = 0,
) -> str:
    """Emit raw AIEX NPU BD setup. Stride fields use the encoded step value."""
    use_next = 1 if use_next_bd else 0
    packet_enable = 1 if enable_packet else 0
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 64 : i32, column = {column} : i32, "
        f"d0_size = {d0_size} : i32, d0_stride = {d0_stride} : i32, "
        f"d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = {d1_size} : i32, d1_stride = {d1_stride} : i32, "
        f"d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = {d2_size} : i32, d2_stride = {d2_stride} : i32, "
        f"d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = {packet_enable} : i32, iteration_current = 0 : i32, "
        f"iteration_size = {iteration_size} : i32, iteration_stride = {iteration_stride} : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = {next_bd} : i32, out_of_order_id = 0 : i32, "
        f"packet_id = {packet_id} : i32, packet_type = {packet_type} : i32, "
        f"row = 0 : i32, use_next_bd = {use_next} : i32, valid_bd = 1 : i32}}"
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
    repeat_count: int = 0,
) -> str:
    token = "true" if issue_token else "false"
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = {repeat_count} : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def npu_rtp_write(buffer_name: str, index: int, value: int) -> str:
    return f"      aiex.npu.rtp_write(@{buffer_name}, {index}, {value})"


def npu_set_lock(lock_name: str, value: int) -> str:
    return f"      aiex.set_lock(%{lock_name}, {value})"


def lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def flow(src_tile: str, src_dma: int, dst_tile: str, dst_dma: int) -> str:
    return f"    aie.flow(%{src_tile}, DMA : {src_dma}, %{dst_tile}, DMA : {dst_dma})"


def packet_flow(
    packet_id: int,
    src_tile: str,
    src_dma: int,
    dst_tile: str,
    dst_dma: int,
    *,
    keep_pkt_header: bool = False,
) -> str:
    attrs = " {keep_pkt_header = true}" if keep_pkt_header else ""
    return "\n".join(
        (
            f"    aie.packet_flow({packet_id}) {{",
            f"      aie.packet_source<%{src_tile}, DMA : {src_dma}>",
            f"      aie.packet_dest<%{dst_tile}, DMA : {dst_dma}>",
            f"    }}{attrs}",
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


def require_absent_markers(scope: str, mlir: str, markers: tuple[str, ...]) -> list[str]:
    return [f"{scope}: forbidden marker present: {marker}" for marker in markers if marker in mlir]


def require_marker_order(scope: str, mlir: str, markers: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    previous = -1
    for marker in markers:
        position = mlir.find(marker, previous + 1)
        if position == -1:
            errors.append(f"{scope}: ordered marker missing: {marker}")
            continue
        previous = position
    return errors


def require_dma_bd_lock_balance(
    scope: str,
    mlir: str,
    allowed_unbalanced_blocks: tuple[str, ...] = (),
) -> list[str]:
    errors: list[str] = []
    block_name = "entry"
    block_lines: list[str] = []

    def flush_block() -> None:
        if not block_lines:
            return
        block = "\n".join(block_lines)
        if "aie.dma_bd(" not in block:
            return
        has_acquire = re.search(r"aie\.use_lock\([^)]*,\s*Acquire", block) is not None
        has_release = re.search(r"aie\.use_lock\([^)]*,\s*Release", block) is not None
        if has_acquire != has_release and block_name not in allowed_unbalanced_blocks:
            errors.append(f"{scope}: {block_name} has unbalanced DMA BD locks")

    for line in mlir.splitlines():
        stripped = line.strip()
        if stripped.startswith("^"):
            flush_block()
            block_name = stripped.split(":", 1)[0]
            block_lines = []
        else:
            block_lines.append(line)
    flush_block()
    return errors


def require_source_side_packet_replay(
    scope: str,
    mlir: str,
    label_prefix: str,
    packet_id: int,
    replay_count: int,
    first_acquire: str,
    last_release: str,
) -> list[str]:
    errors: list[str] = []
    labels = tuple(f"^{label_prefix}{idx}" for idx in range(replay_count))
    for label in labels:
        if f"    {label}:" not in mlir:
            errors.append(f"{scope}: source-side replay label missing: {label}")
    if errors:
        return errors
    first = mlir.find(f"    {labels[0]}:")
    last = mlir.find(f"    {labels[-1]}:")
    if first > last:
        errors.append(f"{scope}: source-side replay labels are out of order for {label_prefix}")
    last_block_end = mlir.find("    ^", last + 1)
    if last_block_end == -1:
        last_block_end = len(mlir)
    replay_region = mlir[first:last_block_end]
    packet_marker = f"pkt_id = {packet_id}>"
    count = replay_region.count(packet_marker)
    if count != replay_count:
        errors.append(f"{scope}: source-side replay packet {packet_id} count {count} != {replay_count}")
    first_block_end = mlir.find(f"    {labels[1]}:", first) if replay_count > 1 else last
    first_block = mlir[first:first_block_end]
    if first_acquire not in first_block:
        errors.append(f"{scope}: source-side replay first block must acquire {first_acquire}")
    last_block = mlir[last:last_block_end]
    if last_release not in last_block:
        errors.append(f"{scope}: source-side replay last block must release {last_release}")
    return errors


def _dma_regions(mlir: str) -> list[tuple[str, str]]:
    regions: list[tuple[str, str]] = []
    region_name = ""
    region_lines: list[str] = []
    in_region = False
    depth = 0
    for line in mlir.splitlines():
        if not in_region:
            match = re.search(r"%([A-Za-z0-9_]+)\s*=\s*aie\.(mem|memtile_dma)\(", line)
            if match:
                in_region = True
                region_name = match.group(1)
                region_lines = [line]
                depth = line.count("{") - line.count("}")
            continue
        region_lines.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            regions.append((region_name, "\n".join(region_lines)))
            in_region = False
            region_name = ""
            region_lines = []
    return regions


def require_dma_next_bd_labels(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    for region_name, region in _dma_regions(mlir):
        labels = set(re.findall(r"^\s*\^([A-Za-z0-9_]+):", region, flags=re.MULTILINE))
        for label in re.findall(r"aie\.dma_start\([^,]+,\s*[0-9]+,\s*\^([A-Za-z0-9_]+)", region):
            if label not in labels:
                errors.append(f"{scope}: {region_name} dma_start target ^{label} is missing")
        for label in re.findall(r"aie\.next_bd\s+\^([A-Za-z0-9_]+)", region):
            if label not in labels:
                errors.append(f"{scope}: {region_name} next_bd target ^{label} is missing")
    return errors


def require_dma_bd_next_ids(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    for region_name, region in _dma_regions(mlir):
        bd_ids = {int(item) for item in re.findall(r"(?<!next_)bd_id = ([0-9]+) : i32", region)}
        for next_id in (int(item) for item in re.findall(r"next_bd_id = ([0-9]+) : i32", region)):
            if next_id not in bd_ids:
                errors.append(f"{scope}: {region_name} next_bd_id {next_id} has no matching bd_id")
    return errors


def require_unique_packet_flows(scope: str, mlir: str) -> list[str]:
    packet_ids = [int(item) for item in re.findall(r"aie\.packet_flow\(([0-9]+)\)", mlir)]
    seen: set[int] = set()
    duplicates: list[int] = []
    for packet_id in packet_ids:
        if packet_id in seen and packet_id not in duplicates:
            duplicates.append(packet_id)
        seen.add(packet_id)
    return [f"{scope}: duplicate packet_flow id {packet_id}" for packet_id in duplicates]


def require_main_record_phase_barrier(scope: str, mlir: str, first_acquire: int) -> list[str]:
    errors: list[str] = []
    pattern = (
        r"%(m[0-9]+_[0-9]+)_records_empty = aie\.lock\([^)]*\) "
        r"\{init = ([0-9]+) : i32"
    )
    for tile, init_text in re.findall(pattern, mlir):
        lock_name = f"{tile}_records_empty"
        init = int(init_text)
        acquires = re.findall(
            rf"aie\.use_lock\(%{lock_name},\s*AcquireGreaterEqual,\s*([0-9]+)\)",
            mlir,
        )
        if not acquires:
            errors.append(f"{scope}: {lock_name} is never acquired by the main core")
            continue
        first = int(acquires[0])
        if init != first_acquire:
            errors.append(f"{scope}: {lock_name} init {init} != first phase acquire {first_acquire}")
        if first != first_acquire:
            errors.append(f"{scope}: {lock_name} first acquire {first} != expected {first_acquire}")
        if init > first:
            errors.append(f"{scope}: {lock_name} can advance into a later phase before record DMA drains")
    return errors


def require_main_record_pingpong(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    linked_main16_core_kernel = MAIN16_LAYER_SCHEDULER_NAME in mlir
    pattern = (
        r"%(m[0-9]+_[0-9]+)_records_empty = aie\.lock\([^)]*\) "
        r"\{init = ([0-9]+) : i32"
    )
    for tile, init_text in re.findall(pattern, mlir):
        row = int(tile.rsplit("_", 1)[1])
        source_offset = 0 if row == 0 else 1
        source_length = 17 if row == 0 else 16
        if int(init_text) != 2:
            errors.append(f"{scope}: {tile}_records_empty init {init_text} != ping/pong depth 2")

        empty_acquires = re.findall(
            rf"aie\.use_lock\(%{tile}_records_empty,\s*AcquireGreaterEqual,\s*([0-9]+)\)",
            mlir,
        )
        full_releases = re.findall(
            rf"aie\.use_lock\(%{tile}_records_full,\s*Release,\s*([0-9]+)\)",
            mlir,
        )
        if linked_main16_core_kernel:
            control_buffer = re.search(
                rf"%{tile}_[A-Za-z0-9_]*control[A-Za-z0-9_]* = "
                rf"aie\.buffer\(%{tile}\) "
                rf"\{{address = {MAIN16_CONTROL_ADDR} : i32",
                mlir,
            )
            if control_buffer is not None:
                errors.append(
                    f"{scope}: {tile} declares a standalone control buffer at "
                    f"0x{MAIN16_CONTROL_ADDR:04x}; generated main16 control words "
                    "must live inside record_ping ownership"
                )
        if not empty_acquires and not linked_main16_core_kernel:
            errors.append(f"{scope}: {tile}_records_empty is never acquired by the main core")
        for value in empty_acquires:
            if int(value) != 1:
                errors.append(f"{scope}: {tile}_records_empty acquire {value} != per-record acquire 1")
        for value in full_releases:
            if int(value) != 1:
                errors.append(f"{scope}: {tile}_records_full release {value} != per-record release 1")

        for suffix, bd_id, next_bd in (("ping", 4, 5), ("pong", 5, 4)):
            buffer_name = f"{tile}_record_{suffix}"
            buffer_marker = f"%{buffer_name} = aie.buffer(%{tile}) "
            buffer_decl = re.search(
                rf"%{buffer_name} = aie\.buffer\(%{tile}\) "
                rf"\{{address = ([0-9]+) : i32, sym_name = \"{buffer_name}\"\}} "
                rf": memref<([0-9]+)xi32>",
                mlir,
            )
            if buffer_marker not in mlir:
                errors.append(f"{scope}: missing {buffer_name} buffer")
            if linked_main16_core_kernel and suffix == "ping":
                if buffer_decl is None:
                    errors.append(f"{scope}: missing parseable {buffer_name} declaration")
                else:
                    address = int(buffer_decl.group(1))
                    size_bytes = int(buffer_decl.group(2)) * 4
                    if not (
                        address <= MAIN16_CONTROL_ADDR
                        and MAIN16_CONTROL_ADDR + MAIN16_CONTROL_BYTES <= address + size_bytes
                    ):
                        errors.append(
                            f"{scope}: {buffer_name} range "
                            f"0x{address:04x}..0x{address + size_bytes:04x} does not own "
                            f"generated main16 control words "
                            f"0x{MAIN16_CONTROL_ADDR:04x}.."
                            f"0x{MAIN16_CONTROL_ADDR + MAIN16_CONTROL_BYTES:04x}"
                        )
                    if MAIN16_CONTROL_ADDR < address + MAIN16_RECORD_CONTROL_GUARD_BYTES:
                        errors.append(
                            f"{scope}: {buffer_name} control words start at "
                            f"0x{MAIN16_CONTROL_ADDR:04x}, inside the guarded record "
                            f"flush area ending at 0x{address + MAIN16_RECORD_CONTROL_GUARD_BYTES:04x}"
                        )
            bd_pattern = (
                rf"aie\.dma_bd\(%{buffer_name} : memref<[0-9]+xi32>, "
                rf"{source_offset}, {source_length}\) "
                rf"\{{(?=[^}}]*bd_id = {bd_id} : i32)"
                rf"(?=[^}}]*next_bd_id = {next_bd} : i32)[^}}]*\}}"
            )
            if re.search(bd_pattern, mlir) is None:
                errors.append(
                    f"{scope}: missing {buffer_name} record BD{bd_id}->{next_bd} "
                    f"offset={source_offset} length={source_length}"
                )
    return errors


def require_compact_record_packet_granularity(scope: str, mlir: str) -> list[str]:
    main_record_source = (
        re.search(r"record_ping : memref<[0-9]+xi32>, 0, 17", mlir) is not None
        and re.search(r"record_pong : memref<[0-9]+xi32>, 0, 17", mlir) is not None
    )
    phase_sized_column_receive = (
        "memref<520xi32>, 0, 136" in mlir
        or "memref<520xi32>, 17, 128" in mlir
    )
    record_granular_column_receive = (
        "memref<65xi32>, 0, 17" in mlir
        and "memref<65xi32>, 17, 16" in mlir
        and "memref<65xi32>, 33, 16" in mlir
        and "memref<65xi32>, 49, 16" in mlir
    )
    record_granular_column_output = (
        "memref<65xi32>, 0, 65" in mlir
        and "memref<65xi32>, 1, 64" in mlir
    )
    phase_sized_bridge_receive = (
        "memref<2056xi32>, 0, 520" in mlir
        or "memref<2056xi32>, 65, 512" in mlir
    )
    record_granular_bridge_receive = (
        "memref<257xi32>, 0, 65" in mlir
        and "memref<257xi32>, 65, 64" in mlir
        and "memref<257xi32>, 129, 64" in mlir
        and "memref<257xi32>, 193, 64" in mlir
    )
    record_granular_bridge_output = "memref<257xi32>, 0, 257" in mlir
    if main_record_source and phase_sized_column_receive:
        return [
            (
                f"{scope}: main16 emits one 17/16-dword packet per compact record, "
                "but row1 compact receive BDs still expect phase-sized Q/K/V bursts. "
                "Convert row1 to the MyLM-style 17+16+16+16 -> 65-dword record packer "
                "before running on hardware."
            )
        ]
    if main_record_source and not (record_granular_column_receive and record_granular_column_output):
        return [
            (
                f"{scope}: main16 record output is enabled, but row1 does not expose the "
                "MyLM-style 65-dword compact packer shape: S2MM lengths must be "
                "17,16,16,16; column0 MM2S must emit 65 dwords, and column1..3 "
                "must emit only the 64-dword payload."
            )
        ]
    if main_record_source and record_granular_column_receive and phase_sized_bridge_receive:
        return [
            (
                f"{scope}: row1 now emits one 65/64-dword column compact per record, "
                "but c1r1 still receives Q/K/V as phase-sized 520/512-dword bursts. "
                "This mixed state compiles but times out on hardware. Convert c1r1 "
                "to the MyLM-style 65+64+64+64 -> 257-dword record packer before "
                "running this topology."
            )
        ]
    if main_record_source and record_granular_column_receive and not (
        record_granular_bridge_receive and record_granular_bridge_output
    ):
        return [
            (
                f"{scope}: row1 emits record-granular 65-dword column compacts, "
                "but c1r1 does not expose the MyLM-style 257-dword global record "
                "packer shape: S2MM lengths must be 65,64,64,64 and MM2S must "
                "emit the 257-dword record with the manual packet header."
            )
        ]
    return []


def require_memtile_dma_bd_bank(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    region_name = ""
    region_lines: list[str] = []
    in_region = False

    def flush_region() -> None:
        if not region_lines:
            return
        channels_by_label: dict[str, tuple[str, int]] = {}
        blocks_by_label: dict[str, str] = {}
        current_label = ""
        current_lines: list[str] = []

        def flush_block() -> None:
            if current_label:
                blocks_by_label[current_label] = "\n".join(current_lines)

        for region_line in region_lines:
            start = re.search(r"aie\.dma_start\((S2MM|MM2S),\s*([0-9]+),\s*\^([A-Za-z0-9_]+)", region_line)
            if start:
                channels_by_label[start.group(3)] = (start.group(1), int(start.group(2)))
            stripped = region_line.strip()
            if stripped.startswith("^"):
                flush_block()
                current_label = stripped.split(":", 1)[0][1:]
                current_lines = []
            else:
                current_lines.append(region_line)
        flush_block()

        changed = True
        while changed:
            changed = False
            for label, block in blocks_by_label.items():
                channel = channels_by_label.get(label)
                if channel is None:
                    continue
                for next_label in re.findall(r"aie\.next_bd\s+\^([A-Za-z0-9_]+)", block):
                    if next_label in blocks_by_label and next_label not in channels_by_label:
                        channels_by_label[next_label] = channel
                        changed = True

        for label, block in blocks_by_label.items():
            channel = channels_by_label.get(label)
            if channel is None:
                continue
            direction, channel_id = channel
            for bd_id_text in re.findall(r"aie\.dma_bd\([^\n]*\{bd_id = ([0-9]+) : i32", block):
                bd_id = int(bd_id_text)
                high_bank = bd_id > 23
                odd_channel = channel_id % 2 == 1
                if high_bank != odd_channel:
                    errors.append(
                        f"{scope}: {region_name} {direction}{channel_id} block ^{label} "
                        f"uses memtile BD {bd_id} from the wrong bank"
                    )

    for line in mlir.splitlines():
        if not in_region:
            match = re.search(r"%([A-Za-z0-9_]+)\s*=\s*aie\.memtile_dma", line)
            if match:
                in_region = True
                region_name = match.group(1)
                region_lines = []
            continue
        if line.strip() == "}":
            flush_region()
            in_region = False
            region_name = ""
            region_lines = []
        else:
            region_lines.append(line)
    return errors


def require_no_compute_kv_materialization(
    scope: str,
    mlir: str,
    kv_side_dwords: int,
    kv_all_dwords: int,
) -> list[str]:
    errors: list[str] = []
    for line in mlir.splitlines():
        has_buffer = "aie.buffer(%" in line
        has_large_kv_side = f"memref<{kv_side_dwords}xi32>" in line
        if has_buffer and has_large_kv_side and "%kv_left" not in line and "%kv_right" not in line:
            errors.append(f"{scope}: large KV side buffer outside row1 KV scan: {line.strip()}")
        if f"memref<{kv_all_dwords}xi32>" in line:
            errors.append(f"{scope}: full KV materialization marker present: {line.strip()}")
    return errors


def require_attention_block_shapes(
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


def require_npu_writebd_id_limit(scope: str, mlir: str, max_bd_id: int) -> list[str]:
    bd_ids = [int(match) for match in re.findall(r"aiex\.npu\.writebd \{bd_id = ([0-9]+) : i32", mlir)]
    return [
        f"{scope}: npu.writebd BD id {bd_id} exceeds supported max {max_bd_id}"
        for bd_id in sorted(set(bd_ids))
        if bd_id > max_bd_id
    ]


def require_npu_writebd_field_ranges(scope: str, mlir: str) -> list[str]:
    ranges = {
        "d0_size": 1023,
        "d0_stride": (1 << 20) - 1,
        "d1_size": 1023,
        "d1_stride": (1 << 20) - 1,
        "d2_stride": (1 << 20) - 1,
        "iteration_size": 63,
        "iteration_stride": (1 << 20) - 1,
    }
    errors: list[str] = []
    for bd_match in re.finditer(r"aiex\.npu\.writebd \{([^}]*)\}", mlir):
        attrs = bd_match.group(1)
        bd_id_match = re.search(r"bd_id = ([0-9]+) : i32", attrs)
        bd_id = bd_id_match.group(1) if bd_id_match else "?"
        for field, max_value in ranges.items():
            value_match = re.search(rf"{field} = ([0-9]+) : i32", attrs)
            if value_match and int(value_match.group(1)) > max_value:
                errors.append(
                    f"{scope}: npu.writebd BD {bd_id} {field}={value_match.group(1)} "
                    f"exceeds max {max_value}"
                )
    return errors


def require_npu_push_queue_repeat_range(scope: str, mlir: str) -> list[str]:
    repeat_counts = [int(match) for match in re.findall(r"repeat_count = ([0-9]+) : i32", mlir)]
    return [
        f"{scope}: npu.push_queue repeat_count {repeat_count} exceeds supported max 255"
        for repeat_count in sorted(set(repeat_counts))
        if repeat_count > 255
    ]
