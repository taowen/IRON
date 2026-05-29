"""Shared compact-record and row1 weight-stream dataflow for qwen3-layer."""

from __future__ import annotations

from dataclasses import dataclass

from attention_dataflow import HUB_Q_OUT_BDS, HUB_RETURN_IN_BDS
from contract import (
    C1R2_PACKET_DWORDS,
    C1R2_UPGATE_REPLAYS,
    C6R2_HALF_DWORDS,
    COMPACT_PACKET_DWORDS,
    DOWN_PACKET_DWORDS,
    MAIN_COLUMNS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)
from mlir_utils import lock_pair
from weight_stream import (
    WeightStreamConfig,
    weight_stream_buffers,
    weight_stream_input_rings,
    weight_stream_lock_defs,
    weight_stream_row_streams,
)
from qkv_compact_reference import (
    COLUMN_COMPACT_DWORDS,
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    V_GLOBAL_PACKET_ID,
    WINDOW_DWORDS,
    column_packet,
    main_packet,
)

FULL_LAYER_RECORD_STAGES = ("q", "k", "v", "o", "up", "gate", "down")
MAIN_RECORD_DWORDS = RECORD_DWORDS * len(FULL_LAYER_RECORD_STAGES)
FFN_GLOBAL_PACKET_ID = 14
DOWN_GLOBAL_PACKET_ID = 15
FULL_REPLAY_PACKET_ID = 0
DOWN_ACT_PACKET_ID = 1
DOWN_CHUNKS = DOWN_PACKET_DWORDS // MAIN_CHUNK_DWORDS
DOWN_PHASE = 6
MAIN_CHUNKS_PER_REPLAY = (C1R2_PACKET_DWORDS - 1) // MAIN_CHUNK_DWORDS
TOTAL_MAIN_CHUNKS = C1R2_UPGATE_REPLAYS * MAIN_CHUNKS_PER_REPLAY

BODY_PHASES = ("q", "k", "v", "o", "upgate", "down")
BODY_RECORD_SLOTS = (0, 1, 2, 3, -1, 6)
UPGATE_BODY_RECORDS = C1R2_UPGATE_REPLAYS
UPGATE_MAIN_RECORD_DWORDS = UPGATE_BODY_RECORDS * RECORD_DWORDS

COLUMN_RECEIVE_BDS = (
    (0, 1, 2, 3, 4, 5),
    (24, 25, 26, 27, 28, 29),
    (7, 8, 9, 10, 11, 12),
    (31, 32, 33, 41, 42, 43),
)
BRIDGE_RECEIVE_BDS = (
    (0, 1, 2, 3, 4, 5),
    (24, 25, 26, 27, 30, 31),
    (9, 10, 11, 12, 13, 14),
    (41, 42, 43, 44, 45, 46),
)
COMPACT_OUT_BDS = (34, 35, 36, 37, 38, 39)
BRIDGE_PACKET_IN_BDS = (6, 7)
BRIDGE_PACKET_OUT_BDS = (28, 29)
MAIN_RECORD_BDS = (2, 3, 4, 5, 6, 7)
WEIGHT_PATCH_INPUT_BDS = ((14, 15), (30, 40))
WEIGHT_ROW_BDS = ((16, 17), (44, 45), (18, 19), (46, 47))

HUB_FFN_IN_BD = 36
HUB_ATTENTION_OUT_BD = 34
HUB_FFN_OUT_BD = 35


@dataclass(frozen=True)
class CompactPhase:
    label: str
    logical_phase: str
    record_slot: int
    packet_id: int
    output_offset: int
    output_length: int
    body_records: int


def _phase_packet_id(logical_phase: str) -> int:
    if logical_phase == "q":
        return Q_GLOBAL_PACKET_ID
    if logical_phase == "k":
        return K_GLOBAL_PACKET_ID
    if logical_phase == "v":
        return V_GLOBAL_PACKET_ID
    if logical_phase == "o":
        return O_GLOBAL_PACKET_ID
    if logical_phase == "upgate":
        return FFN_GLOBAL_PACKET_ID
    if logical_phase == "down":
        return DOWN_GLOBAL_PACKET_ID
    raise ValueError(f"unknown compact phase: {logical_phase}")


def _phase_output_slice(logical_phase: str, body_records: int) -> tuple[int, int]:
    if logical_phase == "upgate":
        return 1, body_records * C6R2_HALF_DWORDS
    return 0, body_records * COMPACT_PACKET_DWORDS


def _compact_phase(
    label: str,
    logical_phase: str,
    record_slot: int,
    body_records: int | None = None,
) -> CompactPhase:
    records = UPGATE_BODY_RECORDS if logical_phase == "upgate" else 1
    if body_records is not None:
        records = body_records
    output_offset, output_length = _phase_output_slice(logical_phase, records)
    return CompactPhase(
        label=label,
        logical_phase=logical_phase,
        record_slot=record_slot,
        packet_id=_phase_packet_id(logical_phase),
        output_offset=output_offset,
        output_length=output_length,
        body_records=records,
    )


COMPACT_PHASE_TRACE = tuple(
    _compact_phase(label=stage, logical_phase=stage, record_slot=BODY_RECORD_SLOTS[stage_idx])
    for stage_idx, stage in enumerate(BODY_PHASES)
)
WEIGHT_LOCK_BASE = len(COMPACT_PHASE_TRACE) + ROWS_PER_COLUMN + 1


def _phase_trace_marker(phase_trace: tuple[CompactPhase, ...]) -> str:
    return ",".join(phase.label for phase in phase_trace)


def _phase_trace_logical_marker(phase_trace: tuple[CompactPhase, ...]) -> str:
    return ",".join(phase.logical_phase for phase in phase_trace)


def _next_phase(phase_trace: tuple[CompactPhase, ...], stage_idx: int) -> CompactPhase:
    return phase_trace[(stage_idx + 1) % len(phase_trace)]


def _phase_trace_errors(phase_trace: tuple[CompactPhase, ...]) -> list[str]:
    errors: list[str] = []
    labels = tuple(phase.label for phase in phase_trace)
    logical_phases = tuple(phase.logical_phase for phase in phase_trace)
    record_slots = tuple(phase.record_slot for phase in phase_trace)
    if len(set(labels)) != len(labels):
        errors.append(f"compact phase trace labels are not unique: {_phase_trace_marker(phase_trace)}")
    if logical_phases != BODY_PHASES:
        errors.append(
            "compact phase trace logical phases do not match current body schedule: "
            f"{_phase_trace_logical_marker(phase_trace)}"
        )
    if record_slots != BODY_RECORD_SLOTS:
        errors.append(f"compact phase trace record slots do not match current body schedule: {record_slots}")
    if len(phase_trace) != len(COMPACT_OUT_BDS):
        errors.append("compact phase trace/bridge output BD count mismatch")
    if len(phase_trace) != len(MAIN_RECORD_BDS):
        errors.append("compact phase trace/main record BD count mismatch")
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        if len(phase_trace) != len(bd_ids):
            errors.append(f"compact phase trace/column row {row} BD count mismatch")
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        if len(phase_trace) != len(bd_ids):
            errors.append(f"compact phase trace/bridge group {group} BD count mismatch")
    return errors


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def down_record_header(group: int, row: int) -> int:
    return (DOWN_PHASE << 24) | (group << 16) | (row << 8) | 0xD0


def _segment(row: int) -> tuple[int, int]:
    if row == 0:
        return 0, RECORD_DWORDS
    return RECORD_DWORDS + (row - 1) * RECORD_PAYLOAD_DWORDS, RECORD_PAYLOAD_DWORDS


def _source_segment(stage: int, row: int) -> tuple[int, int]:
    base = stage * RECORD_DWORDS
    if row == 0:
        return base, RECORD_DWORDS
    return base + 1, RECORD_PAYLOAD_DWORDS


def _bd_dimensions(dimensions: tuple[tuple[int, int], ...]) -> str:
    if not dimensions:
        return ""
    pairs = ", ".join(f"<size = {size}, stride = {stride}>" for size, stride in dimensions)
    return f", [{pairs}]"


def _phase_buffer(tile: str, phase: CompactPhase) -> tuple[str, str]:
    if phase.body_records > 1:
        return f"%{tile}_{phase.label}", f"memref<{phase.body_records * COLUMN_COMPACT_DWORDS}xi32>"
    return f"%{tile}_{phase.label}", f"memref<{COLUMN_COMPACT_DWORDS}xi32>"


def _bridge_phase_buffer(phase: CompactPhase) -> tuple[str, str]:
    if phase.body_records > 1:
        return f"%bridge_{phase.label}", f"memref<{phase.body_records * COMPACT_PACKET_DWORDS}xi32>"
    return f"%bridge_{phase.label}", f"memref<{COMPACT_PACKET_DWORDS}xi32>"


def _main_record_transfer(phase: CompactPhase, row: int) -> tuple[str, int, int, tuple[tuple[int, int], ...]]:
    if phase.body_records > 1:
        buffer_name = f"{phase.label}_records"
        if row == 0:
            return buffer_name, 0, phase.body_records * RECORD_DWORDS, (
                (phase.body_records, RECORD_DWORDS),
                (RECORD_DWORDS, 1),
            )
        return buffer_name, 1, phase.body_records * RECORD_PAYLOAD_DWORDS, (
            (phase.body_records, RECORD_DWORDS),
            (RECORD_PAYLOAD_DWORDS, 1),
        )
    offset, length = _source_segment(phase.record_slot, row)
    return "records", offset, length, ()


def _column_receive_transfer(phase: CompactPhase, row: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    dest_offset, length = _segment(row)
    if phase.body_records > 1:
        return dest_offset, phase.body_records * length, (
            (phase.body_records, COLUMN_COMPACT_DWORDS),
            (length, 1),
        )
    return dest_offset, length, ()


def _column_output_transfer(group: int, phase: CompactPhase) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    source_offset = 0 if group == 0 else 1
    source_length = COLUMN_COMPACT_DWORDS if group == 0 else COLUMN_COMPACT_DWORDS - 1
    if phase.body_records > 1:
        if group == 0:
            return 0, phase.body_records * COLUMN_COMPACT_DWORDS, ()
        return 1, phase.body_records * (COLUMN_COMPACT_DWORDS - 1), (
            (phase.body_records, COLUMN_COMPACT_DWORDS),
            (COLUMN_COMPACT_DWORDS - 1, 1),
        )
    return source_offset, source_length, ()


def _bridge_receive_transfer(group: int, phase: CompactPhase) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if group == 0:
        dest_offset = 0
        length = COLUMN_COMPACT_DWORDS
    else:
        length = COLUMN_COMPACT_DWORDS - 1
        dest_offset = COLUMN_COMPACT_DWORDS + (group - 1) * (COLUMN_COMPACT_DWORDS - 1)
    if phase.body_records > 1:
        return dest_offset, phase.body_records * length, (
            (phase.body_records, COMPACT_PACKET_DWORDS),
            (length, 1),
        )
    return dest_offset, length, ()


def _bridge_output_transfer(phase: CompactPhase) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if phase.logical_phase == "upgate":
        return 1, phase.body_records * C6R2_HALF_DWORDS, (
            (phase.body_records, COMPACT_PACKET_DWORDS),
            (C6R2_HALF_DWORDS, 1),
        )
    if phase.logical_phase in ("q", "k", "v") and phase.body_records > 1:
        return 1, phase.body_records * C6R2_HALF_DWORDS, (
            (phase.body_records, COMPACT_PACKET_DWORDS),
            (C6R2_HALF_DWORDS, 1),
        )
    return phase.output_offset, phase.output_length, ()


def _column_lock_defs(tile: str, phase_trace: tuple[CompactPhase, ...]) -> str:
    lines: list[str] = []
    for stage_idx, phase in enumerate(phase_trace):
        lines.append(
            f'    %{tile}_{phase.label}_full = aie.lock(%{tile}, {stage_idx}) '
            f'{{init = 0 : i32, sym_name = "{tile}_{phase.label}_full"}}\n'
        )
    for row in range(ROWS_PER_COLUMN):
        lines.append(
            f'    %{tile}_row{row}_empty = aie.lock(%{tile}, {len(phase_trace) + row}) '
            f'{{init = {len(phase_trace)} : i32, sym_name = "{tile}_row{row}_empty"}}\n'
        )
    lines.append(
        f'    %{tile}_drain_token = aie.lock(%{tile}, {len(phase_trace) + ROWS_PER_COLUMN}) '
        f'{{init = 0 : i32, sym_name = "{tile}_drain_token"}}\n'
    )
    return "".join(lines)


def _bridge_lock_defs(phase_trace: tuple[CompactPhase, ...]) -> str:
    lines: list[str] = []
    for stage_idx, phase in enumerate(phase_trace):
        lines.append(
            f'    %bridge_{phase.label}_full = aie.lock(%bridge, {stage_idx}) '
            f'{{init = 0 : i32, sym_name = "bridge_{phase.label}_full"}}\n'
        )
    lines.append(lock_pair("bridge", "packet", len(phase_trace), init_empty=2))
    group_lock_base = len(phase_trace) + 2
    for group in range(len(MAIN_COLUMNS)):
        lines.append(
            f'    %bridge_g{group}_empty = aie.lock(%bridge, {group_lock_base + group}) '
            f'{{init = {len(phase_trace)} : i32, sym_name = "bridge_g{group}_empty"}}\n'
        )
    lines.append(
        f'    %bridge_drain_token = aie.lock(%bridge, {group_lock_base + len(MAIN_COLUMNS)}) '
        '{init = 0 : i32, sym_name = "bridge_drain_token"}\n'
    )
    return "".join(lines)


def compact_column_memtile(group: int, phase_trace: tuple[CompactPhase, ...]) -> str:
    tile = f"mt{group}"
    packet = column_packet(group)
    receive_starts: list[str] = []
    for row in range(ROWS_PER_COLUMN):
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^out_start"
        bds = COLUMN_RECEIVE_BDS[row]
        stage_blocks: list[str] = []
        for stage_idx, phase in enumerate(phase_trace):
            next_phase = _next_phase(phase_trace, stage_idx)
            dest_offset, length, dimensions = _column_receive_transfer(phase, row)
            buffer_name, buffer_type = _phase_buffer(tile, phase)
            stage_blocks.append(f"""    ^row{row}_{phase.label}:
      aie.use_lock(%{tile}_row{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer_name} : {buffer_type}, {dest_offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = {bds[stage_idx]} : i32, next_bd_id = {bds[(stage_idx + 1) % len(phase_trace)]} : i32}}
      aie.use_lock(%{tile}_{phase.label}_full, Release, 1)
      aie.next_bd ^row{row}_{next_phase.label}""")
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_q, {next_start})
{chr(10).join(stage_blocks)}"""
        )

    out_blocks: list[str] = []
    for stage_idx, phase in enumerate(phase_trace):
        next_phase = _next_phase(phase_trace, stage_idx)
        source_offset, source_length, dimensions = _column_output_transfer(group, phase)
        buffer_name, buffer_type = _phase_buffer(tile, phase)
        out_blocks.append(f"""    ^{phase.label}_out:
      aie.use_lock(%{tile}_{phase.label}_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd({buffer_name} : {buffer_type}, {source_offset}, {source_length}{_bd_dimensions(dimensions)}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(phase_trace)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^{next_phase.label}_out""")

    buffers = "\n".join(
        f'    %{tile}_{phase.label} = aie.buffer(%{tile}) {{sym_name = "{tile}_{phase.label}"}} : memref<{phase.body_records * COLUMN_COMPACT_DWORDS}xi32>'
        for phase in phase_trace
    )
    return f"""
{buffers}
{_column_lock_defs(tile, phase_trace)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^q_out, ^end)
{chr(10).join(out_blocks)}
    ^end:
      aie.end
    }}
"""


def q4nx_weight_column_memtile(group: int, phase_trace: tuple[CompactPhase, ...]) -> str:
    tile = f"mt{group}"
    packet = column_packet(group)
    weight_config = WeightStreamConfig(
        group=group,
        input_channels=(4, 5),
        patch_input_bds=WEIGHT_PATCH_INPUT_BDS,
        row_bds=WEIGHT_ROW_BDS,
        row_block_prefix="wt_row",
        row_terminal_label="^out_start",
        input_block_prefix="q4nx_",
    )
    receive_starts: list[str] = []
    for row in range(ROWS_PER_COLUMN):
        start_label = "" if row == 0 else f"    ^row{row}_start:\n"
        next_start = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^patch0_start"
        bds = COLUMN_RECEIVE_BDS[row]
        stage_blocks: list[str] = []
        for stage_idx, phase in enumerate(phase_trace):
            next_phase = _next_phase(phase_trace, stage_idx)
            dest_offset, length, dimensions = _column_receive_transfer(phase, row)
            buffer_name, buffer_type = _phase_buffer(tile, phase)
            stage_blocks.append(f"""    ^row{row}_{phase.label}:
      aie.use_lock(%{tile}_row{row}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer_name} : {buffer_type}, {dest_offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = {bds[stage_idx]} : i32, next_bd_id = {bds[(stage_idx + 1) % len(phase_trace)]} : i32}}
      aie.use_lock(%{tile}_{phase.label}_full, Release, 1)
      aie.next_bd ^row{row}_{next_phase.label}""")
        receive_starts.append(
            f"""{start_label}      %row{row}_dma = aie.dma_start(S2MM, {row}, ^row{row}_q, {next_start})
{chr(10).join(stage_blocks)}"""
        )

    out_blocks: list[str] = []
    for stage_idx, phase in enumerate(phase_trace):
        next_phase = _next_phase(phase_trace, stage_idx)
        source_offset, source_length, dimensions = _column_output_transfer(group, phase)
        buffer_name, buffer_type = _phase_buffer(tile, phase)
        out_blocks.append(f"""    ^{phase.label}_out:
      aie.use_lock(%{tile}_{phase.label}_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd({buffer_name} : {buffer_type}, {source_offset}, {source_length}{_bd_dimensions(dimensions)}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(phase_trace)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {packet}>}}
      aie.use_lock(%{tile}_drain_token, Release, 1)
      aie.next_bd ^{next_phase.label}_out""")

    buffers = "\n".join(
        f'    %{tile}_{phase.label} = aie.buffer(%{tile}) {{sym_name = "{tile}_{phase.label}"}} : memref<{phase.body_records * COLUMN_COMPACT_DWORDS}xi32>'
        for phase in phase_trace
    )
    return f"""
{buffers}
{weight_stream_buffers(tile)}
{_column_lock_defs(tile, phase_trace)}
{weight_stream_lock_defs(tile, WEIGHT_LOCK_BASE)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{chr(10).join(receive_starts)}

{weight_stream_input_rings(weight_config)}

{weight_stream_row_streams(weight_config)}

    ^out_start:
      %out_dma = aie.dma_start(MM2S, 5, ^q_out, ^end)
{chr(10).join(out_blocks)}
    ^end:
      aie.end
    }}
"""


def _bridge_receive_starts(phase_trace: tuple[CompactPhase, ...]) -> str:
    starts: list[str] = []
    for group in range(len(MAIN_COLUMNS)):
        start_label = "" if group == 0 else f"    ^g{group}_start:\n"
        next_start = f"^g{group + 1}_start" if group + 1 < len(MAIN_COLUMNS) else "^compact_out_start"
        bds = BRIDGE_RECEIVE_BDS[group]
        stage_blocks: list[str] = []
        for stage_idx, phase in enumerate(phase_trace):
            next_phase = _next_phase(phase_trace, stage_idx)
            dest_offset, length, dimensions = _bridge_receive_transfer(group, phase)
            buffer_name, buffer_type = _bridge_phase_buffer(phase)
            stage_blocks.append(f"""    ^g{group}_{phase.label}:
      aie.use_lock(%bridge_g{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd({buffer_name} : {buffer_type}, {dest_offset}, {length}{_bd_dimensions(dimensions)}) {{bd_id = {bds[stage_idx]} : i32, next_bd_id = {bds[(stage_idx + 1) % len(phase_trace)]} : i32}}
      aie.use_lock(%bridge_{phase.label}_full, Release, 1)
      aie.next_bd ^g{group}_{next_phase.label}""")
        starts.append(
            f"""{start_label}      %g{group}_dma = aie.dma_start(S2MM, {group}, ^g{group}_q, {next_start})
{chr(10).join(stage_blocks)}"""
        )
    return "\n".join(starts)


def _bridge(phase_trace: tuple[CompactPhase, ...]) -> str:
    compact_out_blocks: list[str] = []
    for stage_idx, phase in enumerate(phase_trace):
        next_phase = _next_phase(phase_trace, stage_idx)
        output_offset, output_length, dimensions = _bridge_output_transfer(phase)
        buffer_name, buffer_type = _bridge_phase_buffer(phase)
        compact_out_blocks.append(f"""    ^{phase.label}_out:
      aie.use_lock(%bridge_{phase.label}_full, AcquireGreaterEqual, {len(MAIN_COLUMNS)})
      aie.dma_bd({buffer_name} : {buffer_type}, {output_offset}, {output_length}{_bd_dimensions(dimensions)}) {{bd_id = {COMPACT_OUT_BDS[stage_idx]} : i32, next_bd_id = {COMPACT_OUT_BDS[(stage_idx + 1) % len(phase_trace)]} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {phase.packet_id}>}}
      aie.use_lock(%bridge_drain_token, Release, 1)
      aie.next_bd ^{next_phase.label}_out""")

    buffers = "\n".join(
        f'    %bridge_{phase.label} = aie.buffer(%bridge) {{sym_name = "bridge_{phase.label}"}} : memref<{phase.body_records * COMPACT_PACKET_DWORDS}xi32>'
        for phase in phase_trace
    )
    return f"""
    // compact phase trace {_phase_trace_marker(phase_trace)}
{buffers}
    %bridge_packet_ping = aie.buffer(%bridge) {{sym_name = "bridge_packet_ping"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %bridge_packet_pong = aie.buffer(%bridge) {{sym_name = "bridge_packet_pong"}} : memref<{C6R2_HALF_DWORDS}xi32>
{_bridge_lock_defs(phase_trace)}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{_bridge_receive_starts(phase_trace)}

    ^compact_out_start:
      %compact_out_dma = aie.dma_start(MM2S, 5, ^q_out, ^packet_in_start)
{chr(10).join(compact_out_blocks)}

    ^packet_in_start:
      %packet_in_dma = aie.dma_start(S2MM, 4, ^packet_in_ping, ^packet_out_start)
    ^packet_in_ping:
      aie.use_lock(%bridge_packet_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_ping : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_pong
    ^packet_in_pong:
      aie.use_lock(%bridge_packet_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_IN_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_IN_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_full, Release, 1)
      aie.next_bd ^packet_in_ping

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 1, ^packet_out_ping, ^end)
    ^packet_out_ping:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_ping : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_pong
    ^packet_out_pong:
      aie.use_lock(%bridge_packet_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_packet_pong : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {BRIDGE_PACKET_OUT_BDS[1]} : i32, next_bd_id = {BRIDGE_PACKET_OUT_BDS[0]} : i32}}
      aie.use_lock(%bridge_packet_empty, Release, 1)
      aie.next_bd ^packet_out_ping
    ^end:
      aie.end
    }}
"""


def _hub() -> str:
    q_outs: list[str] = []
    for window, bd_id in enumerate(HUB_Q_OUT_BDS):
        next_start = f"^q{window + 1}_start" if window + 1 < 4 else "^return0_start"
        q_outs.append(f"""    ^q{window}_start:
      %q{window}_dma = aie.dma_start(MM2S, {window}, ^q{window}_out, {next_start})
    ^q{window}_out:
      aie.use_lock(%hub_q_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_q_empty, Release, 1)
      aie.next_bd ^q{window}_out""")

    return_ins: list[str] = []
    for window, bd_id in enumerate(HUB_RETURN_IN_BDS):
        next_start = f"^return{window + 1}_start" if window + 1 < 4 else "^ffn_in_start"
        return_ins.append(f"""    ^return{window}_start:
      %return{window}_dma = aie.dma_start(S2MM, {window + 1}, ^return{window}_in, {next_start})
    ^return{window}_in:
      aie.use_lock(%hub_return_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_return_full, Release, 1)
      aie.next_bd ^return{window}_in""")

    return f"""
    %hub_q = aie.buffer(%hub) {{sym_name = "hub_q"}} : memref<{Q_DWORDS}xi32>
    %hub_return = aie.buffer(%hub) {{sym_name = "hub_return"}} : memref<{Q_DWORDS}xi32>
    %hub_ffn = aie.buffer(%hub) {{sym_name = "hub_ffn"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %hub_q_empty = aie.lock(%hub, 0) {{init = 4 : i32, sym_name = "hub_q_empty"}}
    %hub_q_full = aie.lock(%hub, 1) {{init = 0 : i32, sym_name = "hub_q_full"}}
    %hub_return_empty = aie.lock(%hub, 2) {{init = 4 : i32, sym_name = "hub_return_empty"}}
    %hub_return_full = aie.lock(%hub, 3) {{init = 0 : i32, sym_name = "hub_return_full"}}
{lock_pair("hub", "ffn", 4)}

    %hub_dma = aie.memtile_dma(%hub) {{
      %q_in_dma = aie.dma_start(S2MM, 0, ^q_in, ^q0_start)
    ^q_in:
      aie.use_lock(%hub_q_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%hub_q_full, Release, 4)
      aie.next_bd ^q_in

{chr(10).join(q_outs)}

{chr(10).join(return_ins)}

    ^ffn_in_start:
      %ffn_in_dma = aie.dma_start(S2MM, 5, ^ffn_in, ^packet_out_start)
    ^ffn_in:
      aie.use_lock(%hub_ffn_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_ffn : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {HUB_FFN_IN_BD} : i32}}
      aie.use_lock(%hub_ffn_full, Release, 1)
      aie.next_bd ^ffn_in

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 5, ^attention_out, ^end)
    ^attention_out:
      aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = {HUB_ATTENTION_OUT_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_ATTENTION}>}}
      aie.use_lock(%hub_return_empty, Release, 4)
      aie.next_bd ^ffn_out
    ^ffn_out:
      aie.use_lock(%hub_ffn_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_ffn : memref<{C6R2_HALF_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = {HUB_FFN_OUT_BD} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {DOWN_ACT_PACKET_ID}>}}
      aie.use_lock(%hub_ffn_empty, Release, 1)
      aie.next_bd ^ffn_out
    ^end:
      aie.end
    }}
"""
