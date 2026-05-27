"""Generate raw MLIR-AIE for exp67 MyLM global QKV/O layout contract."""

from pathlib import Path

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
OUTPUT_BLOCK_ROWS = 512
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 12288
ACT_SLICE_BF16 = 256
K_CHUNK = 256
GROUP_SIZE = 32
RECORD_DWORDS = 17
PHASE_NAMES = ("Q", "K", "V", "O", "UP", "GATE", "DOWN")
PHASE_INPUT_DIMS = (4096, 4096, 4096, 4096, 4096, 4096, 12288)
PHASE_OUTPUT_DIMS = (4096, 1024, 1024, 4096, 12288, 12288, 4096)
PHASE_BLOCKS = tuple(
    output_dim // OUTPUT_BLOCK_ROWS for output_dim in PHASE_OUTPUT_DIMS
)
PHASE_CHUNKS = tuple(input_dim // K_CHUNK for input_dim in PHASE_INPUT_DIMS)
NUM_PHASES = len(PHASE_NAMES)
TOTAL_LOGICAL_BLOCKS = sum(PHASE_BLOCKS)
PATCHES_PER_COLUMN = 2
ROWS_PER_PATCH = 2
O_PHASE = PHASE_NAMES.index("O")
CONTEXT_LEN = 31

CHUNK_BF16 = 2560
OUT_RECORD_BF16 = M_PER_TILE + 2
PATCH_BF16_BY_PHASE = tuple(
    ROWS_PER_PATCH * PHASE_CHUNKS[phase] * CHUNK_BF16 for phase in range(NUM_PHASES)
)
PHASE_PATCH_COUNTS = tuple(
    PHASE_BLOCKS[phase] * len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    for phase in range(NUM_PHASES)
)
PHASE_WEIGHT_BF16 = tuple(
    PHASE_PATCH_COUNTS[phase] * PATCH_BF16_BY_PHASE[phase]
    for phase in range(NUM_PHASES)
)
TOTAL_PATCHES = sum(PHASE_PATCH_COUNTS)
PATCH_DESCRIPTORS_PER_COLUMN = TOTAL_LOGICAL_BLOCKS * PATCHES_PER_COLUMN
TOTAL_WEIGHT_BF16 = sum(PHASE_WEIGHT_BF16)
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_RECORD_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2
WEIGHT_BD_IDS = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15)
WEIGHT_BD_BATCH = len(WEIGHT_BD_IDS)
ROW_BDS = (
    (2, 3),
    (24, 25),
    (4, 5),
    (26, 27),
)
OUTPUT_COLLECT_BD_BASE = 7
OUTPUT_DRAIN_BD = 34
PATCH_PACKET_BASE = 16


def _patch_packet_id(group: int, pair: int) -> int:
    return PATCH_PACKET_BASE + group * PATCHES_PER_COLUMN + pair


def _phase_base_bf16(phase: int) -> int:
    return sum(PHASE_WEIGHT_BF16[:phase])


def _patch_offset_bf16(phase: int, block: int, group: int, pair: int) -> int:
    patches_per_block = len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    patch_bf16 = PATCH_BF16_BY_PHASE[phase]
    patch_in_phase = block * patches_per_block + group * PATCHES_PER_COLUMN + pair
    return _phase_base_bf16(phase) + patch_in_phase * patch_bf16


def _column_patch_descriptors(group: int) -> list[tuple[int, int, int, int, int]]:
    descriptors: list[tuple[int, int, int, int, int]] = []
    for phase in range(NUM_PHASES):
        patch_bf16 = PATCH_BF16_BY_PHASE[phase]
        for block in range(PHASE_BLOCKS[phase]):
            for pair in range(PATCHES_PER_COLUMN):
                descriptors.append(
                    (
                        phase,
                        block,
                        pair,
                        _patch_offset_bf16(phase, block, group, pair),
                        patch_bf16,
                    )
                )
    return descriptors


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


def _output_packet(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _main_chunk_loop(tile: str, phase: int, chunks: int) -> str:
    return f"""
        %p{phase}_chunks = arith.constant {chunks} : index
        scf.for %p{phase}_chunk = %c0 to %p{phase}_chunks step %c1 {{
          %chunk_i32 = arith.index_cast %p{phase}_chunk : index to i32
          %rem = arith.remsi %chunk_i32, %c2_i32 : i32
          %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
          aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
          aie.use_lock(%{tile}_wt_full, AcquireGreaterEqual, 1)
          scf.if %is_pong {{
            func.call @q4nx_chunk_accum_slice(%{tile}_wt_pong, %{tile}_act_pong, %m_i32)
              : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
          }} else {{
            func.call @q4nx_chunk_accum_slice(%{tile}_wt_ping, %{tile}_act_ping, %m_i32)
              : (memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) -> ()
          }}
          aie.use_lock(%{tile}_act_empty, Release, 1)
          aie.use_lock(%{tile}_wt_empty, Release, 1)
        }}
"""


def _main_phase_loop(
    tile: str, phase: int, blocks: int, chunks: int, is_last_phase: bool
) -> str:
    final_branch = ""
    if is_last_phase:
        final_branch = f"""
        %last_block = arith.constant {blocks - 1} : i32
        %is_last_block = arith.cmpi eq, %block_i32, %last_block : i32
        scf.if %is_last_block {{
          func.call @accumulate_block_summary(%{tile}_phase, %{tile}_summary, %p{phase}_i32, %block_i32, %m_i32)
            : (memref<{M_PER_TILE}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
          func.call @flush_schedule_output_with_header(%{tile}_output, %{tile}_summary, %group_i32, %row_i32, %m_i32)
            : (memref<{OUT_RECORD_BF16}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{tile}_output_full, Release, 1)
        }} else {{
          aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
          func.call @emit_next_block_sideband(%{tile}_record, %{tile}_phase, %{tile}_summary, %p{phase}_i32, %block_i32, %group_i32, %row_i32, %m_i32)
            : (memref<{RECORD_DWORDS}xi32>, memref<{M_PER_TILE}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32, i32, i32) -> ()
          aie.use_lock(%{tile}_record_full, Release, 1)
        }}
"""
    else:
        final_branch = f"""
        aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
        func.call @emit_next_block_sideband(%{tile}_record, %{tile}_phase, %{tile}_summary, %p{phase}_i32, %block_i32, %group_i32, %row_i32, %m_i32)
          : (memref<{RECORD_DWORDS}xi32>, memref<{M_PER_TILE}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%{tile}_record_full, Release, 1)
"""

    return f"""
      %p{phase}_blocks = arith.constant {blocks} : index
      %p{phase}_i32 = arith.constant {phase} : i32
      scf.for %p{phase}_block = %c0 to %p{phase}_blocks step %c1 {{
        %block_i32 = arith.index_cast %p{phase}_block : index to i32
{_main_chunk_loop(tile, phase, chunks)}
        func.call @q4nx_flush_output(%{tile}_phase, %m_i32)
          : (memref<{M_PER_TILE}xbf16>, i32) -> ()
{final_branch}
      }}
"""


def _main_tile(group: int, row: int) -> str:
    tile = f"m{group}_{row}"
    phase_loops = []
    for phase, blocks in enumerate(PHASE_BLOCKS):
        phase_loops.append(
            _main_phase_loop(
                tile, phase, blocks, PHASE_CHUNKS[phase], phase == NUM_PHASES - 1
            )
        )

    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_wt_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_ping"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_wt_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_wt_pong"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_phase = aie.buffer(%{tile}) {{sym_name = "{tile}_phase"}} : memref<{M_PER_TILE}xbf16>
    %{tile}_summary = aie.buffer(%{tile}) {{sym_name = "{tile}_summary"}} : memref<{M_PER_TILE}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{OUT_RECORD_BF16}xbf16>
{_lock_pair(tile, "record", 0)}
{_lock_pair(tile, "act", 2, init_empty=2)}
{_lock_pair(tile, "wt", 4, init_empty=2)}
{_lock_pair(tile, "output", 6)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %m_i32 = arith.constant {M_PER_TILE} : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

      func.call @clear_summary(%{tile}_summary, %m_i32)
        : (memref<{M_PER_TILE}xbf16>, i32) -> ()
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      func.call @emit_seed_sideband(%{tile}_record, %group_i32, %row_i32)
        : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_record_full, Release, 1)

{chr(10).join(phase_loops)}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^wt_ping, ^act_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 0 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^act_start:
      %1 = aie.dma_start(S2MM, 1, ^act_ping, ^record_start)
    ^act_ping:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 1 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 5 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_act_full, Release, 1)
      aie.next_bd ^act_ping

    ^record_start:
      %2 = aie.dma_start(MM2S, 0, ^record_out, ^output_start)
    ^record_out:
      aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_record_empty, Release, 1)
      aie.next_bd ^record_out

    ^output_start:
      %3 = aie.dma_start(MM2S, 1, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{OUT_RECORD_BF16}xbf16>, 0, {OUT_RECORD_BF16}) {{bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_output_packet(group, row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _edge_phase_loop(tile: str, phase: int, blocks: int, chunks: int) -> str:
    return f"""
      %p{phase}_blocks = arith.constant {blocks} : index
      %p{phase}_chunks = arith.constant {chunks} : index
      %p{phase}_i32 = arith.constant {phase} : i32
      scf.for %p{phase}_block = %c0 to %p{phase}_blocks step %c1 {{
        %block_i32 = arith.index_cast %p{phase}_block : index to i32
        aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
        scf.for %p{phase}_chunk = %c0 to %p{phase}_chunks step %c1 {{
          %chunk_i32 = arith.index_cast %p{phase}_chunk : index to i32
          %rem = arith.remsi %chunk_i32, %c2_i32 : i32
          %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32
          aie.use_lock(%{tile}_act_empty, AcquireGreaterEqual, 1)
          scf.if %is_pong {{
            func.call @edge_make_block_slice(%{tile}_record, %{tile}_act_pong, %p{phase}_i32, %block_i32, %chunk_i32, %group_i32, %row_i32)
              : (memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32, i32, i32) -> ()
          }} else {{
            func.call @edge_make_block_slice(%{tile}_record, %{tile}_act_ping, %p{phase}_i32, %block_i32, %chunk_i32, %group_i32, %row_i32)
              : (memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32, i32, i32) -> ()
          }}
          aie.use_lock(%{tile}_act_full, Release, 1)
        }}
        aie.use_lock(%{tile}_record_empty, Release, 1)
      }}
"""


def _edge_tile(group: int, row: int) -> str:
    tile = f"edge{group}_{row}"
    phase_loops = []
    for phase, blocks in enumerate(PHASE_BLOCKS):
        phase_loops.append(_edge_phase_loop(tile, phase, blocks, PHASE_CHUNKS[phase]))

    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_act_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_act_ping"}} : memref<{ACT_SLICE_BF16}xbf16>
    %{tile}_act_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_act_pong"}} : memref<{ACT_SLICE_BF16}xbf16>
{_lock_pair(tile, "record", 0)}
{_lock_pair(tile, "act", 2, init_empty=2)}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32

{chr(10).join(phase_loops)}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^record_in, ^act_start)
    ^record_in:
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_record_full, Release, 1)
      aie.next_bd ^record_in

    ^act_start:
      %1 = aie.dma_start(MM2S, 0, ^act_ping, ^end)
    ^act_ping:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_ping : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 1 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_pong
    ^act_pong:
      aie.use_lock(%{tile}_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_act_pong : memref<{ACT_SLICE_BF16}xbf16>, 0, {ACT_SLICE_BF16}) {{bd_id = 5 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_act_empty, Release, 1)
      aie.next_bd ^act_ping
    ^end:
      aie.end
    }}
"""


def _slot_lock(prefix: str, pair: int, slot: str, lock_suffix: str) -> str:
    return f"%{prefix}_patch{pair}_{slot}_{lock_suffix}"


def _row_stream(
    group: int, row: int, pair: int, row_in_pair: int, ping_bd: int, pong_bd: int
) -> str:
    prefix = f"mt{group}"
    offset = row_in_pair * CHUNK_BF16
    next_label = (
        f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^output_collect_start"
    )
    return f"""    ^row{row}_start:
      %row{row}_dma = aie.dma_start(MM2S, {row}, ^row{row}_ping, {next_label})
    ^row{row}_ping:
      aie.use_lock({_slot_lock(prefix, pair, "ping", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{pair}_ping : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, pair, "ping", "empty")}, Release, 1)
      aie.next_bd ^row{row}_pong
    ^row{row}_pong:
      aie.use_lock({_slot_lock(prefix, pair, "pong", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{pair}_pong : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, pair, "pong", "empty")}, Release, 1)
      aie.next_bd ^row{row}_ping"""


def _split_input_ring(
    group: int, pair: int, channel: int, ping_bd: int, pong_bd: int
) -> str:
    prefix = f"mt{group}"
    start_label = f"^patch{pair}_split_ping"
    next_label = (
        "^row0_start" if pair + 1 == PATCHES_PER_COLUMN else f"^patch{pair + 1}_start"
    )
    return f"""    ^patch{pair}_start:
      %patch{pair}_dma = aie.dma_start(S2MM, {channel}, {start_label}, {next_label})
    ^patch{pair}_split_ping:
      aie.use_lock({_slot_lock(prefix, pair, "ping", "empty")}, AcquireGreaterEqual, {ROWS_PER_PATCH})
      aie.dma_bd(%{prefix}_patch{pair}_ping : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>, 0, {ROWS_PER_PATCH * CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, pair, "ping", "full")}, Release, {ROWS_PER_PATCH})
      aie.next_bd ^patch{pair}_split_pong
    ^patch{pair}_split_pong:
      aie.use_lock({_slot_lock(prefix, pair, "pong", "empty")}, AcquireGreaterEqual, {ROWS_PER_PATCH})
      aie.dma_bd(%{prefix}_patch{pair}_pong : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>, 0, {ROWS_PER_PATCH * CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({_slot_lock(prefix, pair, "pong", "full")}, Release, {ROWS_PER_PATCH})
      aie.next_bd ^patch{pair}_split_ping"""


def _input_dma_start(group: int) -> str:
    return "\n".join(
        (
            _split_input_ring(group, 0, 0, 0, 1),
            _split_input_ring(group, 1, 1, 28, 29),
        )
    )


def _memtile_column(group: int) -> str:
    tile = f"mt{group}"
    prefix = tile
    output_bds = []
    for row in range(ROWS_PER_COLUMN):
        next_row = (row + 1) % ROWS_PER_COLUMN
        bd_id = OUTPUT_COLLECT_BD_BASE + row
        next_bd_id = OUTPUT_COLLECT_BD_BASE + next_row
        output_bds.append(f"""    ^out{row}:
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, {row * OUT_RECORD_BF16}, {OUT_RECORD_BF16}) {{bd_id = {bd_id} : i32, next_bd_id = {next_bd_id} : i32}}
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.next_bd ^out{next_row}""")

    row_streams = [
        _row_stream(
            group,
            row,
            row // ROWS_PER_PATCH,
            row % ROWS_PER_PATCH,
            ROW_BDS[row][0],
            ROW_BDS[row][1],
        )
        for row in range(ROWS_PER_COLUMN)
    ]

    lock_defs = []
    for pair in range(PATCHES_PER_COLUMN):
        base = pair * 4
        lock_defs += [
            f"""    %{prefix}_patch{pair}_ping_empty = aie.lock(%mt{group}, {base}) {{init = {ROWS_PER_PATCH} : i32, sym_name = "mt{group}_patch{pair}_ping_empty"}}""",
            f"""    %{prefix}_patch{pair}_ping_full = aie.lock(%mt{group}, {base + 1}) {{init = 0 : i32, sym_name = "mt{group}_patch{pair}_ping_full"}}""",
            f"""    %{prefix}_patch{pair}_pong_empty = aie.lock(%mt{group}, {base + 2}) {{init = {ROWS_PER_PATCH} : i32, sym_name = "mt{group}_patch{pair}_pong_empty"}}""",
            f"""    %{prefix}_patch{pair}_pong_full = aie.lock(%mt{group}, {base + 3}) {{init = 0 : i32, sym_name = "mt{group}_patch{pair}_pong_full"}}""",
        ]

    return f"""
    %{tile}_patch0_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_patch0_ping"}} : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>
    %{tile}_patch0_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_patch0_pong"}} : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>
    %{tile}_patch1_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_patch1_ping"}} : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>
    %{tile}_patch1_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_patch1_pong"}} : memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{COLUMN_OUTPUT_BF16}xbf16>
{chr(10).join(lock_defs)}
    %{tile}_output_empty = aie.lock(%{tile}, 8) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_output_empty"}}
    %{tile}_output_full = aie.lock(%{tile}, 9) {{init = 0 : i32, sym_name = "{tile}_output_full"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
{_input_dma_start(group)}

{chr(10).join(row_streams)}

    ^output_collect_start:
      %output_collect_dma = aie.dma_start(S2MM, 2, ^out0, ^output_drain_start)
{chr(10).join(output_bds)}

    ^output_drain_start:
      %output_drain_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_BF16}xbf16>, 0, {COLUMN_OUTPUT_BF16}) {{bd_id = {OUTPUT_DRAIN_BD} : i32}}
      aie.use_lock(%{tile}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_I32}xi32>, "
        f"%output: memref<{OUT_TOTAL_I32}xi32>) {{"
    ]
    output_channel = 1
    for group, column in enumerate(MAIN_COLUMNS):
        output_offset = group * COLUMN_OUTPUT_BF16 * 2
        lines += [
            _npu_writebd(column, 13, COLUMN_OUTPUT_BF16 // 2, output_offset),
            _npu_address_patch(column, 13, 1, output_offset),
            _npu_push_queue(column, "S2MM", output_channel, 13, issue_token=True),
        ]

    per_column_descriptors = [
        _column_patch_descriptors(group) for group in range(len(MAIN_COLUMNS))
    ]
    for descriptors in per_column_descriptors:
        if len(descriptors) != PATCH_DESCRIPTORS_PER_COLUMN:
            raise RuntimeError(
                f"bad descriptor count: {len(descriptors)} != {PATCH_DESCRIPTORS_PER_COLUMN}"
            )

    for batch_start in range(0, PATCH_DESCRIPTORS_PER_COLUMN, WEIGHT_BD_BATCH):
        for group, column in enumerate(MAIN_COLUMNS):
            batch = per_column_descriptors[group][
                batch_start : batch_start + WEIGHT_BD_BATCH
            ]
            for bd_slot, (
                _phase,
                _block,
                pair,
                offset_bf16,
                patch_bf16,
            ) in enumerate(batch):
                bd_id = WEIGHT_BD_IDS[bd_slot]
                next_bd = WEIGHT_BD_IDS[bd_slot + 1] if bd_slot + 1 < len(batch) else 0
                weight_offset = offset_bf16 * 2
                lines += [
                    _npu_writebd(
                        column,
                        bd_id,
                        patch_bf16 // 2,
                        weight_offset,
                        next_bd=next_bd,
                        use_next_bd=bd_slot + 1 < len(batch),
                        packet_id=_patch_packet_id(group, pair),
                    ),
                    _npu_address_patch(column, bd_id, 0, weight_offset),
                ]
        head_bd = WEIGHT_BD_IDS[0]
        for column in MAIN_COLUMNS:
            lines += [
                _npu_push_queue(column, "MM2S", 0, head_bd, issue_token=True),
                _npu_sync(column, 0, direction=1),
            ]

    for column in MAIN_COLUMNS:
        lines.append(_npu_sync(column, output_channel))
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    tile_defs = []
    for group, main_col in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({main_col}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({main_col}, 1)")
        for row in range(ROWS_PER_COLUMN):
            tile_defs.append(
                f"    %edge{group}_{row} = aie.tile({EDGE_COLUMNS[group]}, {row + 2})"
            )
            tile_defs.append(f"    %m{group}_{row} = aie.tile({main_col}, {row + 2})")

    flows = []
    for group in range(len(MAIN_COLUMNS)):
        for pair in range(PATCHES_PER_COLUMN):
            flows.append(f"    aie.packet_flow({_patch_packet_id(group, pair)}) {{")
            flows.append(f"      aie.packet_source<%shim{group}, DMA : 0>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : {pair}>")
            flows.append("    }")
        for row in range(ROWS_PER_COLUMN):
            flows.append(
                f"    aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : 0)"
            )
            flows.append(
                f"    aie.flow(%m{group}_{row}, DMA : 0, %edge{group}_{row}, DMA : 0)"
            )
            flows.append(
                f"    aie.flow(%edge{group}_{row}, DMA : 0, %m{group}_{row}, DMA : 1)"
            )
            flows.append(f"    aie.packet_flow({_output_packet(group, row)}) {{")
            flows.append(f"      aie.packet_source<%m{group}_{row}, DMA : 1>")
            flows.append(f"      aie.packet_dest<%mt{group}, DMA : 2>")
            flows.append("    }")
        flows.append(f"    aie.flow(%mt{group}, DMA : 5, %shim{group}, DMA : 1)")

    blocks = []
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(_memtile_column(group))
        for row in range(ROWS_PER_COLUMN):
            blocks.append(_edge_tile(group, row))
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @clear_summary(memref<{M_PER_TILE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @emit_seed_sideband(memref<{RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @emit_next_block_sideband(memref<{RECORD_DWORDS}xi32>, memref<{M_PER_TILE}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @accumulate_block_summary(memref<{M_PER_TILE}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @edge_make_block_slice(memref<{RECORD_DWORDS}xi32>, memref<{ACT_SLICE_BF16}xbf16>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @q4nx_chunk_accum_slice(memref<{CHUNK_BF16}xbf16>, memref<{ACT_SLICE_BF16}xbf16>, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @q4nx_flush_output(memref<{M_PER_TILE}xbf16>, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}
    func.func private @flush_schedule_output_with_header(memref<{OUT_RECORD_BF16}xbf16>, memref<{M_PER_TILE}xbf16>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/patch_schedule.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
