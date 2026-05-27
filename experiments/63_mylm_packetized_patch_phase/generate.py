"""Generate raw MLIR-AIE for exp63 packetized patch-phase projection."""

import os
from pathlib import Path

VARIANT = os.environ.get("EXP63_VARIANT", "packet_full")
if VARIANT == "packet_onecol":
    MAIN_COLUMNS = (2,)
    EDGE_COLUMNS = (0,)
elif VARIANT == "packet_full":
    MAIN_COLUMNS = (2, 3, 4, 5)
    EDGE_COLUMNS = (0, 1, 6, 7)
else:
    raise ValueError(f"unsupported EXP63_VARIANT={VARIANT!r}")

ROWS_PER_COLUMN = 4
M_PER_TILE = 32
K = 4096
K_CHUNK = 256
K_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
ACT_SLICE_BF16 = K_CHUNK
CHUNK_BF16 = (M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2) // 2
PATCH_BF16 = 2 * K_CHUNKS * CHUNK_BF16
PATCHES_PER_COLUMN = 2
COLUMN_WEIGHT_BF16 = PATCHES_PER_COLUMN * PATCH_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
OUT_RECORD_BF16 = M_PER_TILE + 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_RECORD_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2

ROW_BDS = (
    (2, 3),
    (24, 25),
    (4, 5),
    (26, 27),
)
OUTPUT_COLLECT_BD_BASE = 7
OUTPUT_DRAIN_BD = 34
PATCH_PACKET_BASE = 16


def _patch_packet_id(group: int, patch: int) -> int:
    return PATCH_PACKET_BASE + group * PATCHES_PER_COLUMN + patch


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(
    column: int,
    bd_id: int,
    buffer_length: int,
    buffer_offset: int,
    next_bd: int = 0,
    use_next_bd: bool = False,
    packet_id: int | None = None,
) -> str:
    use_next = 1 if use_next_bd else 0
    enable_packet = 1 if packet_id is not None else 0
    packet_id_value = 0 if packet_id is None else packet_id
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
        f"enable_packet = {enable_packet} : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = {next_bd} : i32, out_of_order_id = 0 : i32, "
        f"packet_id = {packet_id_value} : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = {use_next} : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(
    column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int
) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def _npu_push_queue(
    column: int, direction: str, channel: int, bd_id: int, issue_token: bool = False
) -> str:
    token = "true" if issue_token else "false"
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"
    )


def _npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _packet_id(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _edge_tile(group: int, row: int) -> str:
    tile = f"edge{group}_{row}"
    return f"""
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
{_lock_pair(tile, "act", 0, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %num_chunks = arith.constant {K_CHUNKS} : index
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @edge_make_activation_slice(%{tile}_act_pong, %group_i32, %row_i32, %chunk_i32)
            : (memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) -> ()
        }} else {{
          func.call @edge_make_activation_slice(%{tile}_act_ping, %group_i32, %row_i32, %chunk_i32)
            : (memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) -> ()
        }}
        aie.use_lock(%{tile}_act_full, Release, 1)
      }}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(MM2S, 0, ^act_ping, ^end)
    ^act_ping:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 0 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_ping
    ^end:
      aie.end
    }}
"""


def _main_tile(group: int, row: int) -> str:
    tile = f"m{group}_{row}"
    return f"""
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{OUT_RECORD_BF16}xbf16>
{_lock_pair(tile, "act", 0, init_empty=2)}
{_lock_pair(tile, "wt", 2, init_empty=2)}
{_lock_pair(tile, "output", 4)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %num_chunks = arith.constant {K_CHUNKS} : index
      %m_i32 = arith.constant {M_PER_TILE} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %rem = arith.remsi %chunk_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
        aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
        scf.if %is_pong {{
          func.call @q4nx_chunk_accum_slice(%{tile}_wt_pong, %{tile}_act_pong, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
        }} else {{
          func.call @q4nx_chunk_accum_slice(%{tile}_wt_ping, %{tile}_act_ping, %m_i32)
            : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
        }}
        aie.use_lock(%{tile}_wt_empty, Release, 1)
        aie.use_lock(%{tile}_act_empty, Release, 1)
      }}
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @flush_projection_output(%{tile}_output, %group_i32, %row_i32, %m_i32)
        : (memref<{OUT_RECORD_BF16}xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^act_ping, ^wt_start)
    ^act_ping:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_ping

    ^wt_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^out_start:
      %2 = aie.dma_start(MM2S, 1, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{OUT_RECORD_BF16}xbf16>, 0, {OUT_RECORD_BF16}) {{bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_packet_id(group, row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}
"""


def _slot_lock(prefix: str, patch: int, slot: str, lock_suffix: str) -> str:
    return f"%{prefix}_patch{patch}_{slot}_{lock_suffix}"


def _row_stream(
    group: int, row: int, patch: int, row_in_patch: int, ping_bd: int, pong_bd: int
) -> str:
    prefix = f"mt{group}"
    offset = row_in_patch * CHUNK_BF16
    next_label = (
        f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^output_start"
    )
    return f"""    ^row{row}_start:
      %row{row}_dma = aie.dma_start(MM2S, {row}, ^row{row}_ping, {next_label})
    ^row{row}_ping:
      aie.use_lock({_slot_lock(prefix, patch, "ping", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{patch}_ping : memref<{2 * CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, patch, "ping", "empty")}, Release, 1)
      aie.next_bd ^row{row}_pong
    ^row{row}_pong:
      aie.use_lock({_slot_lock(prefix, patch, "pong", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{patch}_pong : memref<{2 * CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, patch, "pong", "empty")}, Release, 1)
      aie.next_bd ^row{row}_ping"""


def _split_input_ring(
    group: int, patch: int, channel: int, ping_bd: int, pong_bd: int
) -> str:
    prefix = f"mt{group}"
    start_label = f"^patch{patch}_split_ping"
    next_label = (
        "^row0_start" if patch + 1 == PATCHES_PER_COLUMN else f"^patch{patch + 1}_start"
    )
    return f"""    ^patch{patch}_start:
      %patch{patch}_dma = aie.dma_start(S2MM, {channel}, {start_label}, {next_label})
    ^patch{patch}_split_ping:
      aie.use_lock({_slot_lock(prefix, patch, "ping", "empty")}, AcquireGreaterEqual, 2)
      aie.dma_bd(%{prefix}_patch{patch}_ping : memref<{2 * CHUNK_BF16}xbf16>, 0, {2 * CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, patch, "ping", "full")}, Release, 2)
      aie.next_bd ^patch{patch}_split_pong
    ^patch{patch}_split_pong:
      aie.use_lock({_slot_lock(prefix, patch, "pong", "empty")}, AcquireGreaterEqual, 2)
      aie.dma_bd(%{prefix}_patch{patch}_pong : memref<{2 * CHUNK_BF16}xbf16>, 0, {2 * CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, patch, "pong", "full")}, Release, 2)
      aie.next_bd ^patch{patch}_split_ping"""


def _input_dma_start(group: int) -> str:
    return "\n".join(
        (
            _split_input_ring(group, 0, 0, 0, 1),
            _split_input_ring(group, 1, 1, 28, 29),
        )
    )


def _memtile_column(group: int) -> str:
    prefix = f"mt{group}"
    output_bds = []
    for row in range(ROWS_PER_COLUMN):
        next_row = (row + 1) % ROWS_PER_COLUMN
        bd_id = OUTPUT_COLLECT_BD_BASE + row
        next_bd_id = OUTPUT_COLLECT_BD_BASE + next_row
        output_bds.append(f"""    ^out{row}:
      aie.use_lock(%{prefix}_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, {row * OUT_RECORD_BF16}, {OUT_RECORD_BF16}) {{bd_id = {bd_id} : i32, next_bd_id = {next_bd_id} : i32}}
      aie.use_lock(%{prefix}_output_full, Release, 1)
      aie.next_bd ^out{next_row}""")

    row_bds = [
        _row_stream(group, row, row // 2, row % 2, ROW_BDS[row][0], ROW_BDS[row][1])
        for row in range(ROWS_PER_COLUMN)
    ]

    lock_defs = []
    for patch in range(PATCHES_PER_COLUMN):
        base = patch * 4
        lock_defs += [
            f"""    %{prefix}_patch{patch}_ping_empty = aie.lock(%mt{group}, {base}) {{init = 2 : i32, sym_name = "mt{group}_patch{patch}_ping_empty"}}""",
            f"""    %{prefix}_patch{patch}_ping_full = aie.lock(%mt{group}, {base + 1}) {{init = 0 : i32, sym_name = "mt{group}_patch{patch}_ping_full"}}""",
            f"""    %{prefix}_patch{patch}_pong_empty = aie.lock(%mt{group}, {base + 2}) {{init = 2 : i32, sym_name = "mt{group}_patch{patch}_pong_empty"}}""",
            f"""    %{prefix}_patch{patch}_pong_full = aie.lock(%mt{group}, {base + 3}) {{init = 0 : i32, sym_name = "mt{group}_patch{patch}_pong_full"}}""",
        ]

    return f"""
    %{prefix}_patch0_ping = aie.buffer(%mt{group}) {{sym_name = "mt{group}_patch0_ping"}} : memref<{2 * CHUNK_BF16}xbf16>
    %{prefix}_patch0_pong = aie.buffer(%mt{group}) {{sym_name = "mt{group}_patch0_pong"}} : memref<{2 * CHUNK_BF16}xbf16>
    %{prefix}_patch1_ping = aie.buffer(%mt{group}) {{sym_name = "mt{group}_patch1_ping"}} : memref<{2 * CHUNK_BF16}xbf16>
    %{prefix}_patch1_pong = aie.buffer(%mt{group}) {{sym_name = "mt{group}_patch1_pong"}} : memref<{2 * CHUNK_BF16}xbf16>
    %{prefix}_output = aie.buffer(%mt{group}) {{sym_name = "mt{group}_output"}} : memref<{COLUMN_OUTPUT_BF16}xbf16>
{chr(10).join(lock_defs)}
    %{prefix}_output_empty = aie.lock(%mt{group}, 8) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "mt{group}_output_empty"}}
    %{prefix}_output_full = aie.lock(%mt{group}, 9) {{init = 0 : i32, sym_name = "mt{group}_output_full"}}

    %mt{group}_dma = aie.memtile_dma(%mt{group}) {{
{_input_dma_start(group)}

{chr(10).join(row_bds)}

    ^output_start:
      %output_collect = aie.dma_start(S2MM, 2, ^out0, ^drain_start)
{chr(10).join(output_bds)}

    ^drain_start:
      %output_drain = aie.dma_start(MM2S, 5, ^drain, ^end)
    ^drain:
      aie.use_lock(%{prefix}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{prefix}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, 0, {COLUMN_OUTPUT_BF16}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%{prefix}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%output: memref<{OUT_TOTAL_I32}xi32>) {{"
    ]
    output_channel = 0
    for group, column in enumerate(MAIN_COLUMNS):
        output_offset = group * COLUMN_OUTPUT_BF16 * 2
        lines += [
            _npu_writebd(column, 13, COLUMN_OUTPUT_BF16 // 2, output_offset),
            _npu_address_patch(column, 13, 1, output_offset),
            _npu_push_queue(column, "S2MM", output_channel, 13, issue_token=True),
        ]

    for group, column in enumerate(MAIN_COLUMNS):
        column_base = group * COLUMN_WEIGHT_BF16 * 2
        patch_bytes = PATCH_BF16 * 2
        lines += [
            _npu_writebd(
                column,
                0,
                PATCH_BF16 // 2,
                column_base,
                next_bd=1,
                use_next_bd=True,
                packet_id=_patch_packet_id(group, 0),
            ),
            _npu_address_patch(column, 0, 0, column_base),
            _npu_writebd(
                column,
                1,
                PATCH_BF16 // 2,
                column_base + patch_bytes,
                packet_id=_patch_packet_id(group, 1),
            ),
            _npu_address_patch(column, 1, 0, column_base + patch_bytes),
            _npu_push_queue(column, "MM2S", 0, 0, issue_token=True),
            _npu_sync(column, 0, direction=1),
        ]

    for column in MAIN_COLUMNS:
        lines.append(_npu_sync(column, output_channel))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    tile_defs = []
    flows = []
    blocks = []

    for group, main_col in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({main_col}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({main_col}, 1)")
        for patch in range(PATCHES_PER_COLUMN):
            flows.append(f"    aie.packet_flow({_patch_packet_id(group, patch)}) {{")
            flows.append(f"      aie.packet_source<%shim{group}, DMA : 0>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : {patch}>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mt{group}, DMA : 5, %shim{group}, DMA : 0)")
        blocks.append(_memtile_column(group))
        for row in range(ROWS_PER_COLUMN):
            tile_defs.append(
                f"    %edge{group}_{row} = aie.tile({EDGE_COLUMNS[group]}, {row + 2})"
            )
            tile_defs.append(f"    %m{group}_{row} = aie.tile({main_col}, {row + 2})")
            flows.append(
                f"    aie.flow(%edge{group}_{row}, DMA : 0, %m{group}_{row}, DMA : 0)"
            )
            flows.append(
                f"    aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : 1)"
            )
            flows.append(f"    aie.packet_flow({_packet_id(group, row)}) {{")
            flows.append(f"      aie.packet_source<%m{group}_{row}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : 2>")
            flows.append("    }")
            blocks.append(_edge_tile(group, row))
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @edge_make_activation_slice(memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/projection_nblock.o"}}
    func.func private @q4nx_chunk_accum_slice(memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/projection_nblock.o"}}
    func.func private @flush_projection_output(memref<{OUT_RECORD_BF16}xbf16>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/projection_nblock.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
