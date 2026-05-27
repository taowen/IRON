"""Generate raw MLIR-AIE for exp40 projection-record handoff ABI."""

from pathlib import Path

RECORD_DWORDS = 17
ROWS_PER_COLUMN = 4
RECORDS_PER_BLOCK = 16
FFN_BLOCKS = 24
ALL_RECORD_DWORDS = FFN_BLOCKS * RECORDS_PER_BLOCK * RECORD_DWORDS

RAW_COLUMN_DWORDS = ROWS_PER_COLUMN * RECORD_DWORDS
COLUMN65_DWORDS = 65
REPLAY64_DWORDS = 64
BLOCK257_DWORDS = 257
HIDDEN2049_DWORDS = 2049
FFN6144_DWORDS = 6144
TOTAL_OUTPUT_DWORDS = (
    RAW_COLUMN_DWORDS
    + COLUMN65_DWORDS
    + REPLAY64_DWORDS
    + BLOCK257_DWORDS
    + HIDDEN2049_DWORDS
    + FFN6144_DWORDS
)
OUTPUT_SPLIT_DWORDS = 4096
OUTPUT_REMAINDER_DWORDS = TOTAL_OUTPUT_DWORDS - OUTPUT_SPLIT_DWORDS


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


def _source_tile(row: int) -> str:
    tile = f"src{row}"
    return f"""
    %{tile}_record = aie.buffer(%{tile}) {{sym_name = "{tile}_record"}} : memref<{RECORD_DWORDS}xi32>
{_lock_pair(tile, "record", 0)}

    %{tile}_core = aie.core(%{tile}) {{
      %phase = arith.constant 1 : i32
      %block = arith.constant 0 : i32
      %col = arith.constant 0 : i32
      %row = arith.constant {row} : i32
      aie.use_lock(%{tile}_record_empty, AcquireGreaterEqual, 1)
      func.call @emit_projection_record(%{tile}_record, %phase, %block, %col, %row)
        : (memref<{RECORD_DWORDS}xi32>, i32, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_record_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %0 = aie.dma_start(MM2S, 0, ^record_bd, ^end)
    ^record_bd:
      aie.use_lock(%{tile}_record_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_record : memref<{RECORD_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {row}>}}
      aie.use_lock(%{tile}_record_empty, Release, 1)
      aie.next_bd ^record_bd
    ^end:
      aie.end
    }}
"""


def _memtile_gather() -> str:
    s2mm_bds = []
    for row in range(ROWS_PER_COLUMN):
        label = f"raw{row}"
        next_label = f"raw{row + 1}" if row + 1 < ROWS_PER_COLUMN else "raw0"
        s2mm_bds.append(
            f"""    ^{label}:
      aie.use_lock(%mt0_raw_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt0_raw : memref<{RAW_COLUMN_DWORDS}xi32>, {row * RECORD_DWORDS}, {RECORD_DWORDS}) {{bd_id = {row} : i32, next_bd_id = {(row + 1) % ROWS_PER_COLUMN} : i32}}
      aie.use_lock(%mt0_raw_full, Release, 1)
      aie.next_bd ^{next_label}"""
        )
    return f"""
    %mt0_raw = aie.buffer(%mt0) {{sym_name = "mt0_raw"}} : memref<{RAW_COLUMN_DWORDS}xi32>
    %mt0_raw_empty = aie.lock(%mt0, 0) {{init = {ROWS_PER_COLUMN} : i32, sym_name = "mt0_raw_empty"}}
    %mt0_raw_full = aie.lock(%mt0, 1) {{init = 0 : i32, sym_name = "mt0_raw_full"}}

    %mt0_dma = aie.memtile_dma(%mt0) {{
      %0 = aie.dma_start(S2MM, 0, ^raw0, ^raw_out_start)
{chr(10).join(s2mm_bds)}

    ^raw_out_start:
      %1 = aie.dma_start(MM2S, 1, ^raw_out, ^end)
    ^raw_out:
      aie.use_lock(%mt0_raw_full, AcquireGreaterEqual, {ROWS_PER_COLUMN})
      aie.dma_bd(%mt0_raw : memref<{RAW_COLUMN_DWORDS}xi32>, 0, {RAW_COLUMN_DWORDS}) {{bd_id = 24 : i32}}
      aie.use_lock(%mt0_raw_empty, Release, {ROWS_PER_COLUMN})
      aie.next_bd ^raw_out
    ^end:
      aie.end
    }}
"""


def _aggregator_tile() -> str:
    return f"""
    %agg_raw = aie.buffer(%agg) {{sym_name = "agg_raw"}} : memref<{RAW_COLUMN_DWORDS}xi32>
    %agg_all = aie.buffer(%agg) {{sym_name = "agg_all"}} : memref<{ALL_RECORD_DWORDS}xi32>
    %agg_out0 = aie.buffer(%agg) {{sym_name = "agg_out0"}} : memref<{OUTPUT_SPLIT_DWORDS}xi32>
    %agg_out1 = aie.buffer(%agg) {{sym_name = "agg_out1"}} : memref<{OUTPUT_REMAINDER_DWORDS}xi32>
{_lock_pair("agg", "raw", 0)}
{_lock_pair("agg", "all", 2)}
{_lock_pair("agg", "out0", 4)}
{_lock_pair("agg", "out1", 6)}

    %agg_core = aie.core(%agg) {{
      aie.use_lock(%agg_raw_full, AcquireGreaterEqual, 1)
      aie.use_lock(%agg_all_full, AcquireGreaterEqual, 1)
      aie.use_lock(%agg_out0_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%agg_out1_empty, AcquireGreaterEqual, 1)
      func.call @aggregate_record_ladder(%agg_raw, %agg_all, %agg_out0, %agg_out1)
        : (memref<{RAW_COLUMN_DWORDS}xi32>, memref<{ALL_RECORD_DWORDS}xi32>, memref<{OUTPUT_SPLIT_DWORDS}xi32>, memref<{OUTPUT_REMAINDER_DWORDS}xi32>) -> ()
      aie.use_lock(%agg_raw_empty, Release, 1)
      aie.use_lock(%agg_all_empty, Release, 1)
      aie.use_lock(%agg_out0_full, Release, 1)
      aie.use_lock(%agg_out1_full, Release, 1)
      aie.end
    }}

    %agg_mem = aie.mem(%agg) {{
      %0 = aie.dma_start(S2MM, 0, ^raw_in, ^all_start)
    ^raw_in:
      aie.use_lock(%agg_raw_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%agg_raw : memref<{RAW_COLUMN_DWORDS}xi32>, 0, {RAW_COLUMN_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%agg_raw_full, Release, 1)
      aie.next_bd ^raw_in

    ^all_start:
      %1 = aie.dma_start(S2MM, 1, ^all_in, ^out_start)
    ^all_in:
      aie.use_lock(%agg_all_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%agg_all : memref<{ALL_RECORD_DWORDS}xi32>, 0, {ALL_RECORD_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%agg_all_full, Release, 1)
      aie.next_bd ^all_in

    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out0, ^end)
    ^out0:
      aie.use_lock(%agg_out0_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%agg_out0 : memref<{OUTPUT_SPLIT_DWORDS}xi32>, 0, {OUTPUT_SPLIT_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%agg_out0_empty, Release, 1)
      aie.next_bd ^out1
    ^out1:
      aie.use_lock(%agg_out1_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%agg_out1 : memref<{OUTPUT_REMAINDER_DWORDS}xi32>, 0, {OUTPUT_REMAINDER_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%agg_out1_empty, Release, 1)
      aie.next_bd ^out0
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    split_bytes = OUTPUT_SPLIT_DWORDS * 4
    return f"""
    aie.runtime_sequence(%all_records: memref<{ALL_RECORD_DWORDS}xi32>, %output: memref<{TOTAL_OUTPUT_DWORDS}xi32>) {{
{_npu_writebd(1, 2, OUTPUT_SPLIT_DWORDS, 0)}
{_npu_address_patch(1, 2, 1, 0)}
{_npu_push_queue(1, "S2MM", 0, 2)}
{_npu_writebd(1, 3, OUTPUT_REMAINDER_DWORDS, split_bytes)}
{_npu_address_patch(1, 3, 1, split_bytes)}
{_npu_push_queue(1, "S2MM", 0, 3, issue_token=True)}
{_npu_writebd(1, 0, ALL_RECORD_DWORDS, 0)}
{_npu_address_patch(1, 0, 0, 0)}
{_npu_push_queue(1, "MM2S", 0, 0)}
{_npu_sync(1, 0)}
    }}
"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.resolve()
    sources = "\n".join(_source_tile(row) for row in range(ROWS_PER_COLUMN))
    packet_flows = []
    for row in range(ROWS_PER_COLUMN):
        packet_flows += [
            f"    aie.packet_flow({row}) {{",
            f"      aie.packet_source<%src{row}, DMA : 0>",
            "      aie.packet_dest<%mt0, DMA : 0>",
            "    }",
        ]
    packet_flow_defs = "\n".join(packet_flows)

    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %mt0 = aie.tile(0, 1)
    %src0 = aie.tile(0, 2)
    %src1 = aie.tile(0, 3)
    %src2 = aie.tile(0, 4)
    %src3 = aie.tile(0, 5)
    %shim1 = aie.tile(1, 0)
    %agg = aie.tile(1, 2)

{packet_flow_defs}
    aie.flow(%mt0, DMA : 1, %agg, DMA : 0)
    aie.flow(%shim1, DMA : 0, %agg, DMA : 1)
    aie.flow(%agg, DMA : 0, %shim1, DMA : 0)

    func.func private @emit_projection_record(memref<{RECORD_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/record_handoff.o"}}
    func.func private @aggregate_record_ladder(memref<{RAW_COLUMN_DWORDS}xi32>, memref<{ALL_RECORD_DWORDS}xi32>, memref<{OUTPUT_SPLIT_DWORDS}xi32>, memref<{OUTPUT_REMAINDER_DWORDS}xi32>) attributes {{link_with = "{experiment_dir}/record_handoff.o"}}

{sources}
{_memtile_gather()}
{_aggregator_tile()}
{_runtime_sequence()}
  }}
}}
"""


if __name__ == "__main__":
    print(generate_mlir())
