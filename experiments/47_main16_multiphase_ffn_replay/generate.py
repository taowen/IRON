"""Generate raw MLIR-AIE for exp47 main16 multi-phase FFN replay."""

from pathlib import Path

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
SHARD_DWORDS = 32
RECORD_DWORDS = 17
O_WEIGHT_DWORDS = 32
GATE_WEIGHT_DWORDS = 32
UP_WEIGHT_DWORDS = 32
DOWN_WEIGHT_DWORDS = 32
TILE_WEIGHT_DWORDS = O_WEIGHT_DWORDS + GATE_WEIGHT_DWORDS + UP_WEIGHT_DWORDS + DOWN_WEIGHT_DWORDS
COLUMN_OUTPUT_DWORDS = ROWS_PER_COLUMN * SHARD_DWORDS
COLUMN_WEIGHT_DWORDS = ROWS_PER_COLUMN * TILE_WEIGHT_DWORDS
TOTAL_OUTPUT_DWORDS = len(MAIN_COLUMNS) * COLUMN_OUTPUT_DWORDS
TOTAL_WEIGHT_DWORDS = len(MAIN_COLUMNS) * COLUMN_WEIGHT_DWORDS


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


def _output_packet(group: int, row: int) -> int:
    return group * ROWS_PER_COLUMN + row


def _lock_pair(tile: str, prefix: str, base: int, init_empty: int = 1) -> str:
    return f"""
    %{tile}_{prefix}_empty = aie.lock(%{tile}, {base}) {{init = {init_empty} : i32, sym_name = "{tile}_{prefix}_empty"}}
    %{tile}_{prefix}_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_{prefix}_full"}}
"""


def _main_tile(group: int, row: int) -> str:
    tile = f"m{group}_{row}"
    return f"""
    %{tile}_o_weight = aie.buffer(%{tile}) {{sym_name = "{tile}_o_weight"}} : memref<{O_WEIGHT_DWORDS}xi32>
    %{tile}_gate_weight = aie.buffer(%{tile}) {{sym_name = "{tile}_gate_weight"}} : memref<{GATE_WEIGHT_DWORDS}xi32>
    %{tile}_up_weight = aie.buffer(%{tile}) {{sym_name = "{tile}_up_weight"}} : memref<{UP_WEIGHT_DWORDS}xi32>
    %{tile}_down_weight = aie.buffer(%{tile}) {{sym_name = "{tile}_down_weight"}} : memref<{DOWN_WEIGHT_DWORDS}xi32>
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_attention = aie.buffer(%{tile}) {{sym_name = "{tile}_attention"}} : memref<{SHARD_DWORDS}xi32>
    %{tile}_o_output = aie.buffer(%{tile}) {{sym_name = "{tile}_o_output"}} : memref<{SHARD_DWORDS}xi32>
    %{tile}_gate = aie.buffer(%{tile}) {{sym_name = "{tile}_gate"}} : memref<{SHARD_DWORDS}xi32>
    %{tile}_up = aie.buffer(%{tile}) {{sym_name = "{tile}_up"}} : memref<{SHARD_DWORDS}xi32>
    %{tile}_swiglu = aie.buffer(%{tile}) {{sym_name = "{tile}_swiglu"}} : memref<{SHARD_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{SHARD_DWORDS}xi32>
{_lock_pair(tile, "o_weight", 0)}
{_lock_pair(tile, "gate_weight", 2, init_empty=0)}
{_lock_pair(tile, "up_weight", 4, init_empty=0)}
{_lock_pair(tile, "down_weight", 6, init_empty=0)}
{_lock_pair(tile, "attention", 8)}
{_lock_pair(tile, "record", 10)}
{_lock_pair(tile, "output", 12)}

    %{tile}_core = aie.core(%{tile}) {{
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      func.call @emit_main_record(%{tile}_record, %group_i32, %row_i32)
        : (memref<{RECORD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_record_full, Release, 1)

      aie.use_lock(%{tile}_o_weight_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_attention_full, AcquireGreaterEqual, 1)
      func.call @main_o_phase(%{tile}_attention, %{tile}_o_weight, %{tile}_o_output, %group_i32, %row_i32)
        : (memref<{SHARD_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_o_weight_empty, Release, 1)
      aie.use_lock(%{tile}_gate_weight_empty, Release, 1)

      aie.use_lock(%{tile}_gate_weight_full, AcquireGreaterEqual, 1)
      func.call @main_gate_phase(%{tile}_o_output, %{tile}_gate_weight, %{tile}_gate, %group_i32, %row_i32)
        : (memref<{SHARD_DWORDS}xi32>, memref<{GATE_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_gate_weight_empty, Release, 1)
      aie.use_lock(%{tile}_up_weight_empty, Release, 1)

      aie.use_lock(%{tile}_up_weight_full, AcquireGreaterEqual, 1)
      func.call @main_up_phase(%{tile}_o_output, %{tile}_up_weight, %{tile}_up, %group_i32, %row_i32)
        : (memref<{SHARD_DWORDS}xi32>, memref<{UP_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_up_weight_empty, Release, 1)
      func.call @main_swiglu_phase(%{tile}_gate, %{tile}_up, %{tile}_swiglu)
        : (memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>) -> ()
      aie.use_lock(%{tile}_down_weight_empty, Release, 1)

      aie.use_lock(%{tile}_down_weight_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @main_down_phase(%{tile}_o_output, %{tile}_swiglu, %{tile}_down_weight, %{tile}_output, %group_i32, %row_i32)
        : (memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, memref<{DOWN_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_down_weight_empty, Release, 1)
      aie.use_lock(%{tile}_attention_empty, Release, 1)
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^weight_in, ^attention_start)
    ^weight_in:
      aie.use_lock(%{tile}_o_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_o_weight : memref<{O_WEIGHT_DWORDS}xi32>, 0, {O_WEIGHT_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%{tile}_o_weight_full, Release, 1)
      aie.next_bd ^gate_weight_in
    ^gate_weight_in:
      aie.use_lock(%{tile}_gate_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_gate_weight : memref<{GATE_WEIGHT_DWORDS}xi32>, 0, {GATE_WEIGHT_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%{tile}_gate_weight_full, Release, 1)
      aie.next_bd ^up_weight_in
    ^up_weight_in:
      aie.use_lock(%{tile}_up_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_up_weight : memref<{UP_WEIGHT_DWORDS}xi32>, 0, {UP_WEIGHT_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 6 : i32}}
      aie.use_lock(%{tile}_up_weight_full, Release, 1)
      aie.next_bd ^down_weight_in
    ^down_weight_in:
      aie.use_lock(%{tile}_down_weight_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_down_weight : memref<{DOWN_WEIGHT_DWORDS}xi32>, 0, {DOWN_WEIGHT_DWORDS}) {{bd_id = 6 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_down_weight_full, Release, 1)
      aie.next_bd ^weight_in

    ^attention_start:
      %1 = aie.dma_start(S2MM, 1, ^attention_in, ^record_start)
    ^attention_in:
      aie.use_lock(%{tile}_attention_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_attention : memref<{SHARD_DWORDS}xi32>, 0, {SHARD_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_attention_full, Release, 1)
      aie.next_bd ^attention_in

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
      aie.dma_bd(%{tile}_output : memref<{SHARD_DWORDS}xi32>, 0, {SHARD_DWORDS}) {{bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {_output_packet(group, row)}>}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _edge_tile(group: int, row: int) -> str:
    tile = f"edge{group}_{row}"
    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
    %{tile}_attention = aie.buffer(%{tile}) {{sym_name = "{tile}_attention"}} : memref<{SHARD_DWORDS}xi32>
{_lock_pair(tile, "record", 0)}
{_lock_pair(tile, "attention", 2)}

    %{tile}_core = aie.core(%{tile}) {{
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
      aie.use_lock(%{tile}_attention_empty, AcquireGreaterEqual, 1)
      func.call @edge_make_attention_row(%{tile}_record, %{tile}_attention, %group_i32, %row_i32)
        : (memref<{RECORD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) -> ()
      aie.use_lock(%{tile}_record_empty, Release, 1)
      aie.use_lock(%{tile}_attention_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^record_in, ^attention_start)
    ^record_in:
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_record_full, Release, 1)
      aie.next_bd ^record_in

    ^attention_start:
      %1 = aie.dma_start(MM2S, 0, ^attention_out, ^end)
    ^attention_out:
      aie.use_lock(%{tile}_attention_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_attention : memref<{SHARD_DWORDS}xi32>, 0, {SHARD_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_attention_empty, Release, 1)
      aie.next_bd ^attention_out
    ^end:
      aie.end
    }}
"""


def _memtile_column(group: int) -> str:
    tile = f"mt{group}"

    output_bds = []
    output_order = (0, 1, 3, 2) if group in (1, 2) else (0, 1, 2, 3)
    for bd_index, row in enumerate(output_order):
        next_bd_index = (bd_index + 1) % ROWS_PER_COLUMN
        output_bds.append(
            f"""    ^out_collect{bd_index}:
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_DWORDS}xi32>, {row * SHARD_DWORDS}, {SHARD_DWORDS}) {{bd_id = {7 + bd_index} : i32, next_bd_id = {7 + next_bd_index} : i32}}
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.next_bd ^out_collect{next_bd_index}"""
        )

    weight_channels = []
    bds = (1, 30, 3, 32)
    for row in range(ROWS_PER_COLUMN):
        ch = row
        weight_channels.append(
            f"""    ^out{ch}_start:
      %out{ch}_dma = aie.dma_start(MM2S, {ch}, ^out{ch}_weight, ^out{ch + 1}_start)
    ^out{ch}_weight:
      aie.use_lock(%{tile}_weight_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_weight : memref<{COLUMN_WEIGHT_DWORDS}xi32>, {row * TILE_WEIGHT_DWORDS}, {TILE_WEIGHT_DWORDS}) {{bd_id = {bds[row]} : i32}}
      aie.use_lock(%{tile}_weight_empty, Release, 1)
      aie.next_bd ^out{ch}_weight"""
        )

    return f"""
    %{tile}_weight = aie.buffer(%{tile}) {{sym_name = "{tile}_weight"}} : memref<{COLUMN_WEIGHT_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{COLUMN_OUTPUT_DWORDS}xi32>
    %{tile}_weight_empty = aie.lock(%{tile}, 0) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_weight_empty"}}
    %{tile}_weight_full = aie.lock(%{tile}, 1) {{init = 0 : i32, sym_name = "{tile}_weight_full"}}
    %{tile}_output_empty = aie.lock(%{tile}, 2) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "{tile}_output_empty"}}
    %{tile}_output_full = aie.lock(%{tile}, 3) {{init = 0 : i32, sym_name = "{tile}_output_full"}}

    %{tile}_dma = aie.memtile_dma(%{tile}) {{
      %0 = aie.dma_start(S2MM, 0, ^weight_in, ^output_collect0_start)
    ^weight_in:
      aie.use_lock(%{tile}_weight_empty, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_weight : memref<{COLUMN_WEIGHT_DWORDS}xi32>, 0, {COLUMN_WEIGHT_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_weight_full, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^weight_in

    ^output_collect0_start:
      %output_collect_dma = aie.dma_start(S2MM, 2, ^out_collect0, ^out0_start)
{chr(10).join(output_bds)}

{chr(10).join(weight_channels)}
    ^out4_start:
      %output_drain_dma = aie.dma_start(MM2S, 5, ^output_drain, ^end)
    ^output_drain:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%{tile}_output : memref<{COLUMN_OUTPUT_DWORDS}xi32>, 0, {COLUMN_OUTPUT_DWORDS}) {{bd_id = 34 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^output_drain
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    lines = [
        f"    aie.runtime_sequence(%weights: memref<{TOTAL_WEIGHT_DWORDS}xi32>, "
        f"%output: memref<{TOTAL_OUTPUT_DWORDS}xi32>) {{"
    ]
    for group, main_col in enumerate(MAIN_COLUMNS):
        weight_offset = group * COLUMN_WEIGHT_DWORDS * 4
        output_offset = group * COLUMN_OUTPUT_DWORDS * 4
        lines += [
            _npu_writebd(main_col, 13, COLUMN_OUTPUT_DWORDS, output_offset),
            _npu_address_patch(main_col, 13, 1, output_offset),
            _npu_push_queue(main_col, "S2MM", 1, 13, issue_token=True),
            _npu_writebd(main_col, 0, COLUMN_WEIGHT_DWORDS, weight_offset),
            _npu_address_patch(main_col, 0, 0, weight_offset),
            _npu_push_queue(main_col, "MM2S", 0, 0),
            _npu_sync(main_col, 1),
        ]
    lines.append("    }")
    return "\n".join(lines)


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()

    tile_defs = []
    for group, main_col in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({main_col}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({main_col}, 1)")
        for row in range(ROWS_PER_COLUMN):
            tile_defs.append(f"    %edge{group}_{row} = aie.tile({EDGE_COLUMNS[group]}, {row + 2})")
            tile_defs.append(f"    %m{group}_{row} = aie.tile({main_col}, {row + 2})")

    flows = []
    for group, _main_col in enumerate(MAIN_COLUMNS):
        flows.append(f"    aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : 0)")
        for row in range(ROWS_PER_COLUMN):
            flows.append(f"    aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : 0)")
            flows.append(f"    aie.flow(%m{group}_{row}, DMA : 0, %edge{group}_{row}, DMA : 0)")
            flows.append(f"    aie.flow(%edge{group}_{row}, DMA : 0, %m{group}_{row}, DMA : 1)")
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

    func.func private @emit_main_record(memref<{RECORD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @edge_make_attention_row(memref<{RECORD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @main_o_phase(memref<{SHARD_DWORDS}xi32>, memref<{O_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @main_gate_phase(memref<{SHARD_DWORDS}xi32>, memref<{GATE_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @main_up_phase(memref<{SHARD_DWORDS}xi32>, memref<{UP_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @main_swiglu_phase(memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}
    func.func private @main_down_phase(memref<{SHARD_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, memref<{DOWN_WEIGHT_DWORDS}xi32>, memref<{SHARD_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/multiphase_ffn_replay.o"}}

{chr(10).join(blocks)}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
