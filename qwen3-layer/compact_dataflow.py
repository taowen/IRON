"""Shared compact-record and row1 weight-stream dataflow for qwen3-layer."""

from __future__ import annotations

from dataclasses import dataclass

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
from projection_schedule import DOWN_BODY_RECORDS, KV_BODY_RECORDS, O_BODY_RECORDS, Q_BODY_RECORDS
from qkv_compact_reference import (
    COLUMN_COMPACT_DWORDS,
    DOWN_GLOBAL_PACKET_ID,
    FFN_GLOBAL_PACKET_ID,
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    V_GLOBAL_PACKET_ID,
    WINDOW_DWORDS,
    body_record_header,
)

FULL_LAYER_RECORD_STAGES = ("q", "k", "v", "o", "up", "gate", "down")
MAIN_RECORD_DWORDS = RECORD_DWORDS * len(FULL_LAYER_RECORD_STAGES)
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
BODY_RECORDS_BY_PHASE = {
    "q": Q_BODY_RECORDS,
    "k": KV_BODY_RECORDS,
    "v": KV_BODY_RECORDS,
    "o": O_BODY_RECORDS,
    "upgate": UPGATE_BODY_RECORDS,
    "down": DOWN_BODY_RECORDS,
}

COLUMN_RECEIVE_BDS = (
    (0, 1),
    (24, 25),
    (2, 3),
    (26, 27),
)
BRIDGE_RECEIVE_BDS = (
    (0, 1),
    (24, 25),
    (2, 3),
    (26, 27),
)
COLUMN_OUT_BDS = (10, 11)
COMPACT_OUT_BDS = (4, 5)
BRIDGE_PACKET_IN_BDS = (6, 7)
BRIDGE_PACKET_OUT_BDS = (28, 29)
MAIN_RECORD_BDS = (4, 5, 6, 7, 8, 9)
WEIGHT_PATCH_INPUT_BDS = ((4, 5), (32, 33))
WEIGHT_ROW_BDS = ((6, 7), (28, 29), (8, 9), (30, 31))
COLUMN_OUT_CHANNEL = 4
BRIDGE_COMPACT_OUT_CHANNEL = 0

HUB_Q_IN_CHANNEL = 1
HUB_Q_IN_BD = 24
HUB_Q_OUT_CHANNELS = (1, 2, 3, 4)
HUB_Q_OUT_BDS = (25, 2, 26, 3)
HUB_RETURN_IN_CHANNELS = (2, 3, 4, 5)
HUB_RETURN_IN_BDS = (4, 28, 6, 30)
HUB_FFN_IN_CHANNEL = 0
HUB_FFN_IN_BD = 0
HUB_ATTENTION_OUT_BDS = tuple(range(34, 34 + O_BODY_RECORDS))
HUB_DOWN_OUT_BDS = (27, 29, 31, 32, 33, 42, 43, 44)


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


def compact_phase_trace(labels: tuple[str, ...]) -> tuple[CompactPhase, ...]:
    phases: list[CompactPhase] = []
    for label in labels:
        if label not in BODY_PHASES:
            raise ValueError(f"unknown compact phase label: {label}")
        stage_idx = BODY_PHASES.index(label)
        phases.append(
            _compact_phase(
                label=label,
                logical_phase=label,
                record_slot=BODY_RECORD_SLOTS[stage_idx],
                body_records=BODY_RECORDS_BY_PHASE[label],
            )
        )
    return tuple(phases)


COMPACT_PHASE_TRACE = compact_phase_trace(BODY_PHASES)
COMPACT_PIPELINE_LOCKS = ROWS_PER_COLUMN + 1
WEIGHT_LOCK_BASE = COMPACT_PIPELINE_LOCKS


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
    if len(COMPACT_OUT_BDS) != 2:
        errors.append("compact bridge output must be a 2-BD ping/pong ring")
    if len(phase_trace) != len(MAIN_RECORD_BDS):
        errors.append("compact phase trace/main record BD count mismatch")
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        if len(bd_ids) != 2:
            errors.append(f"compact bridge group {group} must use a 2-BD ping/pong ring")
    return errors


def _main_symbol(group: int, row: int) -> str:
    return f"m{group}_{row}"


def down_record_header(group: int, row: int) -> int:
    return body_record_header(DOWN_PHASE, 0, group, row)


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


def _bridge_receive_transfer(group: int) -> tuple[int, int]:
    if group == 0:
        dest_offset = 0
        length = COLUMN_COMPACT_DWORDS
    else:
        length = COLUMN_COMPACT_DWORDS - 1
        dest_offset = COLUMN_COMPACT_DWORDS + (group - 1) * (COLUMN_COMPACT_DWORDS - 1)
    return dest_offset, length


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


def _column_record_lock_defs(tile: str) -> str:
    lines = [
        f'    %{tile}_compact_stage0 = aie.lock(%{tile}, 0) '
        f'{{init = 2 : i32, sym_name = "{tile}_compact_stage0"}}\n'
    ]
    for stage in range(1, COMPACT_PIPELINE_LOCKS):
        lines.append(
            f'    %{tile}_compact_stage{stage} = aie.lock(%{tile}, {stage}) '
            f'{{init = 0 : i32, sym_name = "{tile}_compact_stage{stage}"}}\n'
        )
    return "".join(lines)


def _column_record_buffers(tile: str) -> str:
    return "\n".join(
        (
            f'    %{tile}_compact_ping = aie.buffer(%{tile}) '
            f'{{sym_name = "{tile}_compact_ping"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>',
            f'    %{tile}_compact_pong = aie.buffer(%{tile}) '
            f'{{sym_name = "{tile}_compact_pong"}} : memref<{COLUMN_COMPACT_DWORDS}xi32>',
        )
    )


def _column_record_receive_starts(tile: str, terminal_label: str) -> str:
    receive_starts: list[str] = []
    for row in range(ROWS_PER_COLUMN):
        start_label = "" if row == 0 else f"    ^compact_row{row}_start:\n"
        next_start = f"^compact_row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else terminal_label
        ping_bd, pong_bd = COLUMN_RECEIVE_BDS[row]
        dest_offset, length = _segment(row)
        receive_starts.append(
            f"""{start_label}      %compact_row{row}_dma = aie.dma_start(S2MM, {row}, ^compact_row{row}_ping, {next_start})
    ^compact_row{row}_ping:
      aie.use_lock(%{tile}_compact_stage{row}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_compact_ping : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock(%{tile}_compact_stage{row + 1}, Release, 1)
      aie.next_bd ^compact_row{row}_pong
    ^compact_row{row}_pong:
      aie.use_lock(%{tile}_compact_stage{row}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_compact_pong : memref<{COLUMN_COMPACT_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock(%{tile}_compact_stage{row + 1}, Release, 1)
      aie.next_bd ^compact_row{row}_ping"""
        )
    return "\n".join(receive_starts)


def _column_record_output_blocks(tile: str, group: int) -> str:
    ping_bd, pong_bd = COLUMN_OUT_BDS
    source_offset = 0 if group == 0 else 1
    source_length = COLUMN_COMPACT_DWORDS if group == 0 else COLUMN_COMPACT_DWORDS - 1
    return f"""    ^out_start:
      %out_dma = aie.dma_start(MM2S, {COLUMN_OUT_CHANNEL}, ^compact_ping_out, ^end)
    ^compact_ping_out:
      aie.use_lock(%{tile}_compact_stage{ROWS_PER_COLUMN}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_compact_ping : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock(%{tile}_compact_stage0, Release, 1)
      aie.next_bd ^compact_pong_out
    ^compact_pong_out:
      aie.use_lock(%{tile}_compact_stage{ROWS_PER_COLUMN}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_compact_pong : memref<{COLUMN_COMPACT_DWORDS}xi32>, {source_offset}, {source_length}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock(%{tile}_compact_stage0, Release, 1)
      aie.next_bd ^compact_ping_out"""


def _bridge_lock_defs() -> str:
    lines = [
        '    %bridge_compact_stage0 = aie.lock(%bridge, 0) '
        '{init = 2 : i32, sym_name = "bridge_compact_stage0"}\n'
    ]
    for stage in range(1, COMPACT_PIPELINE_LOCKS):
        lines.append(
            f'    %bridge_compact_stage{stage} = aie.lock(%bridge, {stage}) '
            f'{{init = 0 : i32, sym_name = "bridge_compact_stage{stage}"}}\n'
        )
    lines.append(lock_pair("bridge", "packet", COMPACT_PIPELINE_LOCKS, init_empty=2))
    return "".join(lines)


def compact_column_memtile(group: int, phase_trace: tuple[CompactPhase, ...]) -> str:
    tile = f"mt{group}"
    return f"""
    // compact phase trace {_phase_trace_marker(phase_trace)}
{_column_record_buffers(tile)}
{_column_record_lock_defs(tile)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{_column_record_receive_starts(tile, "^out_start")}

{_column_record_output_blocks(tile, group)}
    ^end:
      aie.end
    }}
"""


def q4nx_weight_column_memtile(group: int, phase_trace: tuple[CompactPhase, ...]) -> str:
    tile = f"mt{group}"
    weight_config = WeightStreamConfig(
        group=group,
        input_channels=(4, 5),
        patch_input_bds=WEIGHT_PATCH_INPUT_BDS,
        row_bds=WEIGHT_ROW_BDS,
        row_block_prefix="wt_row",
        row_terminal_label="^out_start",
        input_block_prefix="q4nx_",
    )
    return f"""
    // compact phase trace {_phase_trace_marker(phase_trace)}
{_column_record_buffers(tile)}
{weight_stream_buffers(tile)}
{_column_record_lock_defs(tile)}
{weight_stream_lock_defs(tile, WEIGHT_LOCK_BASE)}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{_column_record_receive_starts(tile, "^patch0_start")}

{weight_stream_input_rings(weight_config)}

{weight_stream_row_streams(weight_config)}

{_column_record_output_blocks(tile, group)}
    ^end:
      aie.end
    }}
"""


def _bridge_receive_starts() -> str:
    starts: list[str] = []
    for group in range(len(MAIN_COLUMNS)):
        start_label = "" if group == 0 else f"    ^g{group}_start:\n"
        next_start = f"^g{group + 1}_start" if group + 1 < len(MAIN_COLUMNS) else "^compact_out_start"
        ping_bd, pong_bd = BRIDGE_RECEIVE_BDS[group]
        dest_offset, length = _bridge_receive_transfer(group)
        starts.append(
            f"""{start_label}      %g{group}_dma = aie.dma_start(S2MM, {group}, ^g{group}_q, {next_start})
    ^g{group}_q:
      aie.use_lock(%bridge_compact_stage{group}, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_compact_ping : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock(%bridge_compact_stage{group + 1}, Release, 1)
      aie.next_bd ^g{group}_pong
    ^g{group}_pong:
      aie.use_lock(%bridge_compact_stage{group}, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_compact_pong : memref<{COMPACT_PACKET_DWORDS}xi32>, {dest_offset}, {length}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock(%bridge_compact_stage{group + 1}, Release, 1)
      aie.next_bd ^g{group}_q"""
        )
    return "\n".join(starts)


def _bridge_output_blocks() -> str:
    ping_bd, pong_bd = COMPACT_OUT_BDS
    return f"""    ^compact_ping_out:
      aie.use_lock(%bridge_compact_stage{ROWS_PER_COLUMN}, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_compact_ping : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock(%bridge_compact_stage0, Release, 1)
      aie.next_bd ^compact_pong_out
    ^compact_pong_out:
      aie.use_lock(%bridge_compact_stage{ROWS_PER_COLUMN}, AcquireGreaterEqual, 1)
      aie.dma_bd(%bridge_compact_pong : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock(%bridge_compact_stage0, Release, 1)
      aie.next_bd ^compact_ping_out"""


def _bridge(phase_trace: tuple[CompactPhase, ...]) -> str:
    return f"""
    // compact phase trace {_phase_trace_marker(phase_trace)}
    %bridge_compact_ping = aie.buffer(%bridge) {{sym_name = "bridge_compact_ping"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_compact_pong = aie.buffer(%bridge) {{sym_name = "bridge_compact_pong"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %bridge_packet_ping = aie.buffer(%bridge) {{sym_name = "bridge_packet_ping"}} : memref<{C6R2_HALF_DWORDS}xi32>
    %bridge_packet_pong = aie.buffer(%bridge) {{sym_name = "bridge_packet_pong"}} : memref<{C6R2_HALF_DWORDS}xi32>
{_bridge_lock_defs()}

    %bridge_dma = aie.memtile_dma(%bridge) {{
{_bridge_receive_starts()}

    ^compact_out_start:
      %compact_out_dma = aie.dma_start(MM2S, {BRIDGE_COMPACT_OUT_CHANNEL}, ^compact_ping_out, ^packet_in_start)
{_bridge_output_blocks()}

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
    for window, (channel, bd_id) in enumerate(zip(HUB_Q_OUT_CHANNELS, HUB_Q_OUT_BDS, strict=True)):
        next_start = f"^q{window + 1}_start" if window + 1 < 4 else "^return0_start"
        q_outs.append(f"""    ^q{window}_start:
      %q{window}_dma = aie.dma_start(MM2S, {channel}, ^q{window}_out, {next_start})
    ^q{window}_out:
      aie.use_lock(%hub_q_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_q_empty, Release, 1)
      aie.next_bd ^q{window}_out""")

    return_ins: list[str] = []
    for window, (channel, bd_id) in enumerate(zip(HUB_RETURN_IN_CHANNELS, HUB_RETURN_IN_BDS, strict=True)):
        next_start = f"^return{window + 1}_start" if window + 1 < 4 else "^ffn_in_start"
        return_ins.append(f"""    ^return{window}_start:
      %return{window}_dma = aie.dma_start(S2MM, {channel}, ^return{window}_in, {next_start})
    ^return{window}_in:
      aie.use_lock(%hub_return_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, {window * WINDOW_DWORDS}, {WINDOW_DWORDS}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%hub_return_full, Release, 1)
      aie.next_bd ^return{window}_in""")

    attention_outs: list[str] = []
    for replay, bd_id in enumerate(HUB_ATTENTION_OUT_BDS):
        next_label = f"^attention_out{replay + 1}" if replay + 1 < O_BODY_RECORDS else "^down_out0"
        if replay == 0:
            acquire = "      aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)\n"
            release = "      aie.use_lock(%hub_attention_replay, Release, 1)\n"
        elif replay + 1 == O_BODY_RECORDS:
            acquire = "      aie.use_lock(%hub_attention_replay, AcquireGreaterEqual, 1)\n"
            release = "      aie.use_lock(%hub_return_empty, Release, 4)\n"
        else:
            acquire = "      aie.use_lock(%hub_attention_replay, AcquireGreaterEqual, 1)\n"
            release = "      aie.use_lock(%hub_attention_replay, Release, 1)\n"
        attention_outs.append(f"""    ^attention_out{replay}:
{acquire}      aie.dma_bd(%hub_return : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = {bd_id} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {PACKET_ID_ATTENTION}>}}
{release}      aie.next_bd {next_label}""")

    down_outs: list[str] = []
    for replay, bd_id in enumerate(HUB_DOWN_OUT_BDS):
        next_label = f"^down_out{replay + 1}" if replay + 1 < DOWN_BODY_RECORDS else "^down_out0"
        if replay == 0:
            acquire = "      aie.use_lock(%hub_ffn_full, AcquireGreaterEqual, 1)\n"
            release = "      aie.use_lock(%hub_down_replay, Release, 1)\n"
        elif replay + 1 == DOWN_BODY_RECORDS:
            acquire = "      aie.use_lock(%hub_down_replay, AcquireGreaterEqual, 1)\n"
            release = "      aie.use_lock(%hub_ffn_empty, Release, 1)\n"
        else:
            acquire = "      aie.use_lock(%hub_down_replay, AcquireGreaterEqual, 1)\n"
            release = "      aie.use_lock(%hub_down_replay, Release, 1)\n"
        down_outs.append(f"""    ^down_out{replay}:
{acquire}      aie.dma_bd(%hub_ffn : memref<{DOWN_PACKET_DWORDS}xi32>, 0, {DOWN_PACKET_DWORDS}) {{bd_id = {bd_id} : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {DOWN_ACT_PACKET_ID}>}}
{release}      aie.next_bd {next_label}""")

    return f"""
    %hub_q = aie.buffer(%hub) {{address = 147456 : i32, sym_name = "hub_q"}} : memref<{Q_DWORDS}xi32>
    %hub_return = aie.buffer(%hub) {{address = 163840 : i32, sym_name = "hub_return"}} : memref<{Q_DWORDS}xi32>
    %hub_ffn = aie.buffer(%hub) {{address = 180224 : i32, sym_name = "hub_ffn"}} : memref<{DOWN_PACKET_DWORDS}xi32>
    %hub_q_empty = aie.lock(%hub, 0) {{init = 4 : i32, sym_name = "hub_q_empty"}}
    %hub_q_full = aie.lock(%hub, 1) {{init = 0 : i32, sym_name = "hub_q_full"}}
    %hub_return_empty = aie.lock(%hub, 2) {{init = 4 : i32, sym_name = "hub_return_empty"}}
    %hub_return_full = aie.lock(%hub, 3) {{init = 0 : i32, sym_name = "hub_return_full"}}
{lock_pair("hub", "ffn", 4)}
    %hub_attention_replay = aie.lock(%hub, 6) {{init = 0 : i32, sym_name = "hub_attention_replay"}}
    %hub_down_replay = aie.lock(%hub, 7) {{init = 0 : i32, sym_name = "hub_down_replay"}}

    %hub_dma = aie.memtile_dma(%hub) {{
      %q_in_dma = aie.dma_start(S2MM, {HUB_Q_IN_CHANNEL}, ^q_in, ^q0_start)
    ^q_in:
      aie.use_lock(%hub_q_empty, AcquireGreaterEqual, 4)
      aie.dma_bd(%hub_q : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = {HUB_Q_IN_BD} : i32}}
      aie.use_lock(%hub_q_full, Release, 4)
      aie.next_bd ^q_in

{chr(10).join(q_outs)}

{chr(10).join(return_ins)}

    ^ffn_in_start:
      %ffn_in_dma = aie.dma_start(S2MM, {HUB_FFN_IN_CHANNEL}, ^ffn_in, ^packet_out_start)
    ^ffn_in:
      aie.use_lock(%hub_ffn_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%hub_ffn : memref<{DOWN_PACKET_DWORDS}xi32>, 0, {DOWN_PACKET_DWORDS}) {{bd_id = {HUB_FFN_IN_BD} : i32}}
      aie.use_lock(%hub_ffn_full, Release, 1)
      aie.next_bd ^ffn_in

    ^packet_out_start:
      %packet_out_dma = aie.dma_start(MM2S, 5, ^attention_out0, ^end)
{chr(10).join(attention_outs)}
{chr(10).join(down_outs)}
    ^end:
      aie.end
    }}
"""
