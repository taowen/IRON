"""Generate raw MLIR-AIE for exp58 MyLM patch-pair row1 split."""

from pathlib import Path

MAIN_COLUMN = 2
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
K_CHUNK = 256
GROUP_SIZE = 32
GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BF16 = (M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2) // 2
PATCH_PAIR_BF16 = CHUNK_BF16 * 2
NUM_PATCH_PAIRS = 2
TOTAL_INPUT_BF16 = NUM_PATCH_PAIRS * PATCH_PAIR_BF16
TOTAL_INPUT_I32 = TOTAL_INPUT_BF16 // 2
WORDS_PER_RECORD = 4
TOTAL_OUTPUT_DWORDS = ROWS_PER_COLUMN * WORDS_PER_RECORD


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(
    column: int,
    bd_id: int,
    buffer_length: int,
    buffer_offset: int,
    next_bd: int = 0,
    use_next_bd: bool = False,
) -> str:
    use_next = 1 if use_next_bd else 0
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
        f"next_bd = {next_bd} : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = {use_next} : i32, valid_bd = 1 : i32}}"
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


def _npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def _packet_id(row: int) -> int:
    return row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _main_tile(row: int) -> str:
    tile = f"m{row}"
    return f"""
    %{tile}_chunk = aie.buffer(%{tile}) {{sym_name = "{tile}_chunk"}} : memref<{CHUNK_BF16}xbf16>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{WORDS_PER_RECORD}xi32>
{_lock_pair(tile, "chunk", 0)}
{_lock_pair(tile, "output", 2)}

    %{tile}_core = aie.core(%{tile}) {{
      %row_i32 = arith.constant {row} : i32
      %chunk_i32 = arith.constant {CHUNK_BF16} : i32
      aie.use_lock(%{tile}_chunk_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @record_patch_chunk(%{tile}_chunk, %{tile}_output, %row_i32, %chunk_i32)
        : (memref<{CHUNK_BF16}xbf16>, memref<{WORDS_PER_RECORD}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_chunk_empty, Release, 1)
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^chunk_in, ^out_start)
    ^chunk_in:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_in

    ^out_start:
      %1 = aie.dma_start(MM2S, 1, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{WORDS_PER_RECORD}xi32>, 0, {WORDS_PER_RECORD}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_packet_id(row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}
"""


def _memtile() -> str:
    output_bds = []
    for row in range(ROWS_PER_COLUMN):
        next_row = (row + 1) % ROWS_PER_COLUMN
        output_bds.append(
            f"""    ^out{row}:
      aie.use_lock(%mt_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_output : memref<{TOTAL_OUTPUT_DWORDS}xi32>, {row * WORDS_PER_RECORD}, {WORDS_PER_RECORD}) {{bd_id = {7 + row} : i32, next_bd_id = {7 + next_row} : i32}}
      aie.use_lock(%mt_output_full, Release, 1)
      aie.next_bd ^out{next_row}"""
        )

    row_streams = (
        (0, "pair0", 0, 2),
        (1, "pair0", CHUNK_BF16, 24),
        (2, "pair1", 0, 4),
        (3, "pair1", CHUNK_BF16, 26),
    )
    row_bds = []
    for row, pair, offset, bd_id in row_streams:
        next_label = f"^row{row + 1}_start" if row + 1 < ROWS_PER_COLUMN else "^output_start"
        row_bds.append(
            f"""    ^row{row}_start:
      %row{row}_dma = aie.dma_start(MM2S, {row}, ^row{row}_bd, {next_label})
    ^row{row}_bd:
      aie.use_lock(%mt_{pair}_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_{pair} : memref<{PATCH_PAIR_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {bd_id} : i32}}
      aie.use_lock(%mt_{pair}_empty, Release, 1)
      aie.next_bd ^row{row}_bd"""
        )

    return f"""
    %mt_pair0 = aie.buffer(%mt) {{sym_name = "mt_pair0"}} : memref<{PATCH_PAIR_BF16}xbf16>
    %mt_pair1 = aie.buffer(%mt) {{sym_name = "mt_pair1"}} : memref<{PATCH_PAIR_BF16}xbf16>
    %mt_output = aie.buffer(%mt) {{sym_name = "mt_output"}} : memref<{TOTAL_OUTPUT_DWORDS}xi32>
    %mt_pair0_empty = aie.lock(%mt, 0) {{init = 2 : i32, sym_name = "mt_pair0_empty"}}
    %mt_pair0_full = aie.lock(%mt, 1) {{init = 0 : i32, sym_name = "mt_pair0_full"}}
    %mt_pair1_empty = aie.lock(%mt, 2) {{init = 2 : i32, sym_name = "mt_pair1_empty"}}
    %mt_pair1_full = aie.lock(%mt, 3) {{init = 0 : i32, sym_name = "mt_pair1_full"}}
    %mt_output_empty = aie.lock(%mt, 4) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "mt_output_empty"}}
    %mt_output_full = aie.lock(%mt, 5) {{init = 0 : i32, sym_name = "mt_output_full"}}

    %mt_dma = aie.memtile_dma(%mt) {{
      %0 = aie.dma_start(S2MM, 0, ^pair0_in, ^row0_start)
    ^pair0_in:
      aie.use_lock(%mt_pair0_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt_pair0 : memref<{PATCH_PAIR_BF16}xbf16>, 0, {PATCH_PAIR_BF16}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt_pair0_full, Release, 2)
      aie.next_bd ^pair1_in
    ^pair1_in:
      aie.use_lock(%mt_pair1_empty, AcquireGreaterEqual, 2)
      aie.dma_bd(%mt_pair1 : memref<{PATCH_PAIR_BF16}xbf16>, 0, {PATCH_PAIR_BF16}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt_pair1_full, Release, 2)
      aie.next_bd ^pair0_in

{chr(10).join(row_bds)}

    ^output_start:
      %output_collect = aie.dma_start(S2MM, 2, ^out0, ^drain_start)
{chr(10).join(output_bds)}

    ^drain_start:
      %output_drain = aie.dma_start(MM2S, 5, ^drain, ^end)
    ^drain:
      aie.use_lock(%mt_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%mt_output : memref<{TOTAL_OUTPUT_DWORDS}xi32>, 0, {TOTAL_OUTPUT_DWORDS}) {{bd_id = 34 : i32}}
      aie.use_lock(%mt_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    pair_bytes = PATCH_PAIR_BF16 * 2
    return f"""
    aie.runtime_sequence(%input: memref<{TOTAL_INPUT_I32}xi32>, %output: memref<{TOTAL_OUTPUT_DWORDS}xi32>) {{
{_npu_writebd(MAIN_COLUMN, 13, TOTAL_OUTPUT_DWORDS, 0)}
{_npu_address_patch(MAIN_COLUMN, 13, 1, 0)}
{_npu_push_queue(MAIN_COLUMN, "S2MM", 1, 13, issue_token=True)}
{_npu_writebd(MAIN_COLUMN, 0, PATCH_PAIR_BF16 // 2, 0, next_bd=1, use_next_bd=True)}
{_npu_address_patch(MAIN_COLUMN, 0, 0, 0)}
{_npu_writebd(MAIN_COLUMN, 1, PATCH_PAIR_BF16 // 2, pair_bytes)}
{_npu_address_patch(MAIN_COLUMN, 1, 0, pair_bytes)}
{_npu_push_queue(MAIN_COLUMN, "MM2S", 0, 0, issue_token=True)}
{_npu_sync(MAIN_COLUMN, 0, direction=1)}
{_npu_sync(MAIN_COLUMN, 1, direction=0)}
    }}
"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    tile_defs = [
        f"    %shim = aie.tile({MAIN_COLUMN}, 0)",
        f"    %mt = aie.tile({MAIN_COLUMN}, 1)",
    ]
    for row in range(ROWS_PER_COLUMN):
        tile_defs.append(f"    %m{row} = aie.tile({MAIN_COLUMN}, {row + 2})")

    flows = [
        "    aie.flow(%shim, DMA : 0, %mt, DMA : 0)",
        "    aie.flow(%mt, DMA : 5, %shim, DMA : 1)",
    ]
    for row in range(ROWS_PER_COLUMN):
        flows.append(f"    aie.flow(%mt, DMA : {row}, %m{row}, DMA : 0)")
        flows += [
            f"    aie.packet_flow({_packet_id(row)}) {{",
            f"      aie.packet_source<%m{row}, DMA : 1>",
            "      aie.packet_dest<%mt, DMA : 2>",
            "    }",
        ]

    main_tiles = "\n".join(_main_tile(row) for row in range(ROWS_PER_COLUMN))
    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @record_patch_chunk(memref<{CHUNK_BF16}xbf16>, memref<{WORDS_PER_RECORD}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/patch_pair_split.o"}}

{_memtile()}
{main_tiles}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
