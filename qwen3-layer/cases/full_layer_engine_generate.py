"""Generate MLIR-AIE for the qwen3 full-layer fused engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from attention_dataflow import (
    KV_OUT_BDS,
    SHAPE_A_TILES,
    SHAPE_B_TILES,
    shape_a_symbol,
    shape_b_symbol,
)
from contract import (
    CHUNK_BF16,
    C1R2_QKV_REPLAYS,
    C1R2_PACKET_DWORDS,
    C1R2_UPGATE_REPLAYS,
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_PAYLOAD_DWORDS,
    RECORD_DWORDS,
    ROWS_PER_PATCH,
    SHAPE_CARRIER_DWORDS,
)
from mlir_utils import (
    flow,
    lock_pair,
    npu_address_patch,
    npu_push_queue,
    npu_rtp_write,
    npu_set_lock,
    npu_sync,
    npu_writebd,
    packet_flow,
    require_absent_markers,
    require_compact_record_packet_granularity,
    require_count,
    require_disjoint_bd_ids,
    require_dma_bd_next_ids,
    require_dma_bd_lock_balance,
    require_dma_next_bd_labels,
    require_attention_block_shapes,
    require_main_record_pingpong,
    require_marker_order,
    require_max_address_patch_arg,
    require_memtile_dma_bd_bank,
    require_no_compute_kv_materialization,
    require_npu_push_queue_repeat_range,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
    require_source_side_packet_replay,
    require_unique_packet_flows,
    require_unique_bd_ids,
)
from compact_dataflow import (
    BODY_RECORD_SLOTS,
    BRIDGE_COMPACT_OUT_CHANNEL,
    BRIDGE_PACKET_IN_BDS,
    BRIDGE_PACKET_OUT_BDS,
    BRIDGE_RECEIVE_BDS,
    COLUMN_OUT_CHANNEL,
    COLUMN_OUT_BDS,
    COLUMN_RECEIVE_BDS,
    COMPACT_OUT_BDS,
    CompactPhase,
    DOWN_ACT_PACKET_ID,
    DOWN_GLOBAL_PACKET_ID,
    FFN_GLOBAL_PACKET_ID,
    FULL_REPLAY_PACKET_ID,
    HUB_DOWN_OUT_BDS,
    HUB_FFN_IN_CHANNEL,
    HUB_Q_IN_CHANNEL,
    HUB_Q_OUT_BDS,
    HUB_Q_OUT_CHANNELS,
    HUB_RETURN_IN_BDS,
    HUB_RETURN_IN_CHANNELS,
    K_GLOBAL_PACKET_ID,
    MAIN_CHUNK_DWORDS,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    TOTAL_MAIN_CHUNKS,
    V_GLOBAL_PACKET_ID,
    WEIGHT_PATCH_INPUT_BDS,
    WEIGHT_ROW_BDS,
    _bd_dimensions,
    _bridge,
    _hub,
    _main_symbol,
    _phase_trace_errors,
    _phase_trace_marker,
    compact_phase_trace,
    q4nx_weight_column_memtile,
)
from physical_contract import validate_q4nx_down_full_layer_ownership
from projection_schedule import (
    DOWN_CHUNKS,
    DOWN_BODY_RECORDS,
    DOWN_WEIGHT_CHUNKS,
    FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE,
    FULL_LAYER_O_WEIGHT_CHUNK_BASE,
    FULL_LAYER_TOTAL_WEIGHT_CHUNKS,
    FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE,
    K_CHUNKS_PER_RECORD,
    K_WEIGHT_CHUNK_BASE,
    KV_BODY_RECORDS,
    O_BODY_RECORDS,
    O_CHUNKS_PER_RECORD,
    O_WEIGHT_CHUNKS,
    Q_BODY_RECORDS,
    Q_CHUNKS_PER_RECORD,
    Q_WEIGHT_CHUNK_BASE,
    QKV_BODY_WEIGHT_CHUNKS,
    UPGATE_CHUNKS_PER_REPLAY,
    UPGATE_WEIGHT_CHUNKS,
    V_CHUNKS_PER_RECORD,
    V_WEIGHT_CHUNK_BASE,
)
from resource_manifest import (
    BDResource,
    BufferResource,
    LockResource,
    ResourceManifest,
    TileResource,
    validate_manifest_matches_mlir,
    validate_resource_manifest,
)
from cases.full_layer_engine_reference import (
    CASE_NAME,
    COLUMN_WEIGHT_BF16,
    DEFAULT_SCHEDULE,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    PATCH_WEIGHT_BF16,
    QK_ROPE_DWORDS,
    RMS_NORM_DWORDS,
    AUX_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
)
from cases.kv_cache_dataflow import (
    CURRENT_WRITE_BDS,
    CURRENT_WRITE_CHANNEL,
    K_SCAN_BD,
    K_SIDE_SCAN_DWORDS,
    KV_SCAN_BDS,
    KV_SPLIT_K_IN_BDS,
    KV_SPLIT_V_IN_BDS,
    V_SCAN_BD,
    V_SIDE_SCAN_DWORDS,
    kv_split_scan_memtile,
    push_current_cache_write,
    push_kv_scan_from_cache,
    shape_blocks_name,
    shape_runtime_start_name,
    shape_tail_tokens_name,
)
from cases.decode_cache_reference import (
    ACCUM_LANES,
    CACHE_BLOCK_DWORDS,
    CURRENT_DWORDS,
    CURRENT_PACKET_K,
    CURRENT_PACKET_V,
    DecodeSchedule,
    K_CACHE_SIDE_DWORDS,
    K_WINDOW_DWORDS,
    KV_SIDE_DWORDS,
    SCALAR_DWORDS,
    V_CACHE_SIDE_DWORDS,
    V_WINDOW_DWORDS,
)
from cases.kv_scan_reference import OUTPUT_DWORDS as ATTENTION_OUTPUT_DWORDS
from cases.kv_scan_reference import WEIGHT_DWORDS, WINDOW_DWORDS

QKV_PREFIX_PHASE_TRACE = compact_phase_trace(("q", "k", "v"))
QKVO_PHASE_TRACE = compact_phase_trace(("q", "k", "v", "o"))
QKV_BODY_PHASE_TRACE = compact_phase_trace(("q", "k", "v", "o", "upgate", "down"))
MAIN16_KERNEL_OBJECT = "main_projection_q4nx_fast.o"
MAIN16_LAYER_SCHEDULER = "q4nx_main16_layer_scheduler"
MAIN16_PHASE_LIMIT_QKV = 3
MAIN16_PHASE_LIMIT_QKVO = 4
MAIN16_PHASE_LIMIT_FULL = 7
QKV_BODY_DWORDS = Q_DWORDS + CURRENT_DWORDS * 2
MAIN_RECORD_PINGPONG_DWORDS = 128
MAIN16_ALL_PHASES = ("Q", "K", "V", "O", "UPGATE", "DOWN")
MAIN_ACTIVATION_EMPTY_LOCK = 0
MAIN_ACTIVATION_FULL_LOCK = 1
MAIN_WEIGHT_EMPTY_LOCK = 2
MAIN_WEIGHT_FULL_LOCK = 3
MAIN_RECORD_EMPTY_LOCK = 4
MAIN_RECORD_FULL_LOCK = 5
MAIN_ACTIVATION_BDS = (0, 1)
MAIN_WEIGHT_BDS = (2, 3)
MAIN_RECORD_BDS = (4, 5)
MAIN_ACCUM_ADDR = 0x2000
MAIN_ACCUM_DWORDS = RECORD_PAYLOAD_DWORDS * 2
MAIN_RECORD_PING_ADDR = 0x3C1C
MAIN_CONTROL_ADDR = 0x3D00
MAIN_RECORD_PONG_ADDR = 0x541C


@dataclass(frozen=True)
class Main16Buffer:
    name: str
    address: int
    size_bytes: int
    phases: tuple[str, ...]
    semantic: str
    memref: str


MAIN16_BUFFERS = (
    Main16Buffer(
        "accum",
        MAIN_ACCUM_ADDR,
        MAIN_ACCUM_DWORDS * 4,
        MAIN16_ALL_PHASES,
        "generated main16 scheduler FP32 accumulator scratch",
        f"memref<{MAIN_ACCUM_DWORDS}xi32>",
    ),
    Main16Buffer(
        "wt_ping",
        0x2800,
        CHUNK_BF16 * 2,
        MAIN16_ALL_PHASES,
        "DMA1 weight ping",
        f"memref<{CHUNK_BF16}xbf16>",
    ),
    Main16Buffer(
        "record_ping",
        MAIN_RECORD_PING_ADDR,
        MAIN_RECORD_PINGPONG_DWORDS * 4,
        MAIN16_ALL_PHASES,
        "compact record ping plus generated scheduler control scratch",
        f"memref<{MAIN_RECORD_PINGPONG_DWORDS}xi32>",
    ),
    Main16Buffer(
        "wt_pong",
        0x4000,
        CHUNK_BF16 * 2,
        MAIN16_ALL_PHASES,
        "DMA1 weight pong",
        f"memref<{CHUNK_BF16}xbf16>",
    ),
    Main16Buffer(
        "record_pong",
        MAIN_RECORD_PONG_ADDR,
        MAIN_RECORD_PINGPONG_DWORDS * 4,
        MAIN16_ALL_PHASES,
        "compact record pong",
        f"memref<{MAIN_RECORD_PINGPONG_DWORDS}xi32>",
    ),
    Main16Buffer(
        "chunk_ping",
        0x8000,
        MAIN_CHUNK_DWORDS * 4,
        MAIN16_ALL_PHASES,
        "DMA0 activation ping",
        f"memref<{MAIN_CHUNK_DWORDS}xi32>",
    ),
    Main16Buffer(
        "chunk_pong",
        0xC000,
        MAIN_CHUNK_DWORDS * 4,
        MAIN16_ALL_PHASES,
        "DMA0 activation pong",
        f"memref<{MAIN_CHUNK_DWORDS}xi32>",
    ),
)
MAIN16_BUFFER_BY_NAME = {buffer.name: buffer for buffer in MAIN16_BUFFERS}
WEIGHT_PATCH_BD_IDS = (
    (0, 2, 4, 6, 8, 10, 12, 14),
    (1, 3, 5, 7, 9, 11, 13, 15),
)
WEIGHT_SPAN_CHUNKS = QKV_BODY_WEIGHT_CHUNKS


def _split_weight_span(chunk_base: int, chunks: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for offset in range(0, chunks, WEIGHT_SPAN_CHUNKS):
        spans.append((chunk_base + offset, min(WEIGHT_SPAN_CHUNKS, chunks - offset)))
    return tuple(spans)


def _full_weight_spans() -> tuple[tuple[int, int], ...]:
    spans = (
        (Q_WEIGHT_CHUNK_BASE, QKV_BODY_WEIGHT_CHUNKS),
        (FULL_LAYER_O_WEIGHT_CHUNK_BASE, O_WEIGHT_CHUNKS),
        *_split_weight_span(FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE, UPGATE_WEIGHT_CHUNKS),
        *_split_weight_span(FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE, DOWN_WEIGHT_CHUNKS),
    )
    if len(spans) != len(WEIGHT_PATCH_BD_IDS[0]):
        raise RuntimeError(f"bad full-layer weight span count: {len(spans)}")
    return spans


def _runtime_sequence(schedule: DecodeSchedule) -> str:
    lines = [
        f"    aie.runtime_sequence(%k_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%v_cache: memref<{schedule.kv_cache_dwords}xi32>, "
        f"%weights: memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>, "
        f"%output: memref<{OUTPUT_DWORDS}xi32>, "
        f"%hidden: memref<{HIDDEN_DWORDS}xi32>) {{"
    ]
    lines.append(npu_rtp_write("post_current_token", 0, schedule.current_token))
    for window in range(4):
        lines.append(npu_rtp_write(shape_blocks_name(shape_a_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(shape_blocks_name(shape_b_symbol(window)), 0, schedule.kv_blocks))
        lines.append(npu_rtp_write(shape_tail_tokens_name(shape_a_symbol(window)), 0, schedule.tail_tokens))
    lines.extend(
        (
            npu_writebd(
                1,
                10,
                QK_ROPE_DWORDS,
                RMS_NORM_DWORDS * 4,
            ),
            npu_address_patch(1, 10, 2, RMS_NORM_DWORDS * 4),
            npu_push_queue(1, "MM2S", 1, 10),
        )
    )
    lines.extend(push_current_cache_write(0, 0, schedule))
    lines.extend(push_current_cache_write(7, 1, schedule))
    lines.extend(
        (
            npu_writebd(1, 13, OUTPUT_DWORDS, 0),
            npu_address_patch(1, 13, 3, 0),
            npu_push_queue(1, "S2MM", 1, 13),
            npu_writebd(1, 12, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 12, 4, 0),
            npu_push_queue(1, "MM2S", 0, 12),
            npu_writebd(1, 14, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 14, 2, 0),
            npu_push_queue(1, "MM2S", 0, 14),
            npu_writebd(1, 15, HIDDEN_DWORDS, HIDDEN_DWORDS * 4),
            npu_address_patch(1, 15, 2, HIDDEN_DWORDS * 4),
            npu_push_queue(1, "MM2S", 0, 15),
        )
    )
    weight_spans = _full_weight_spans()
    chunk_pair_bytes = ROWS_PER_PATCH * CHUNK_BF16 * 2
    for group, column in enumerate(MAIN_COLUMNS):
        column_base = AUX_DWORDS * 4 + group * COLUMN_WEIGHT_BF16 * 2
        for patch, bd_ids in enumerate(WEIGHT_PATCH_BD_IDS):
            patch_base = column_base + patch * PATCH_WEIGHT_BF16 * 2
            for span_idx, (chunk_base, chunk_count) in enumerate(weight_spans):
                bd_id = bd_ids[span_idx]
                has_next = span_idx + 1 < len(weight_spans)
                next_bd = bd_ids[span_idx + 1] if has_next else 0
                byte_offset = patch_base + chunk_base * chunk_pair_bytes
                dwords = chunk_count * ROWS_PER_PATCH * CHUNK_BF16 // 2
                lines.extend(
                    (
                        npu_writebd(
                            column,
                            bd_id,
                            dwords,
                            0,
                            next_bd=next_bd,
                            use_next_bd=has_next,
                        ),
                        npu_address_patch(column, bd_id, 2, byte_offset),
                    )
                )
            lines.append(npu_push_queue(column, "MM2S", patch, bd_ids[0]))
    lines.append(npu_set_lock("post_runtime_start", 1))
    lines.extend((npu_sync(0, CURRENT_WRITE_CHANNEL), npu_sync(7, CURRENT_WRITE_CHANNEL)))
    lines.extend(push_kv_scan_from_cache(0, 0, 1, 0, schedule))
    lines.extend(push_kv_scan_from_cache(7, 0, 1, 4, schedule))
    for window in range(4):
        lines.append(npu_set_lock(shape_runtime_start_name(shape_a_symbol(window)), 1))
        lines.append(npu_set_lock(shape_runtime_start_name(shape_b_symbol(window)), 1))
    lines.extend(
        (
            npu_sync(0, 0, direction=1),
            npu_sync(0, 1, direction=1),
            npu_sync(7, 0, direction=1),
            npu_sync(7, 1, direction=1),
        )
    )
    for column in MAIN_COLUMNS:
        lines.extend((npu_sync(column, 0, direction=1), npu_sync(column, 1, direction=1)))
    lines.append(npu_sync(1, 1))
    lines.append(npu_sync(1, 0, direction=1))
    lines.append(npu_sync(1, 1, direction=1))
    lines.append("    }")
    return "\n".join(lines)


def _shape_a_multiblock_bf16(window: int) -> str:
    tile = shape_a_symbol(window)
    blocks_name = shape_blocks_name(tile)
    tail_tokens_name = shape_tail_tokens_name(tile)
    runtime_start = shape_runtime_start_name(tile)
    return f"""
    %{tile}_q = aie.buffer(%{tile}) {{sym_name = "{tile}_q"}} : memref<{WINDOW_DWORDS}xi32>
    %{tile}_k = aie.buffer(%{tile}) {{sym_name = "{tile}_k"}} : memref<{K_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{blocks_name} = aie.buffer(%{tile}) {{sym_name = "{blocks_name}"}} : memref<1xi32>
    %{tail_tokens_name} = aie.buffer(%{tile}) {{sym_name = "{tail_tokens_name}"}} : memref<1xi32>
{lock_pair(tile, "q", 0)}
{lock_pair(tile, "k", 2)}
{lock_pair(tile, "carrier", 4)}
    %{runtime_start} = aie.lock(%{tile}, 6) {{init = 0 : i32, sym_name = "{runtime_start}"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%{runtime_start}, Acquire, 1)
      %blocks_i32 = memref.load %{blocks_name}[%c0] : memref<1xi32>
      %tail_tokens_i32 = memref.load %{tail_tokens_name}[%c0] : memref<1xi32>
      %blocks = arith.index_cast %blocks_i32 : i32 to index
      %window_i32 = arith.constant {window} : i32
      %q_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %k_dwords_i32 = arith.constant {K_WINDOW_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      aie.use_lock(%{tile}_q_full, AcquireGreaterEqual, 1)
      scf.for %block = %c0 to %blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%{tile}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
        func.call @qwen3_attention_bf16_make_carrier_masked(%{tile}_q, %{tile}_k, %{tile}_carrier, %window_i32, %block_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
          : (memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%{tile}_k_empty, Release, 1)
        aie.use_lock(%{tile}_carrier_full, Release, 1)
      }}
      aie.use_lock(%{tile}_q_empty, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %q_dma = aie.dma_start(S2MM, 0, ^q_in, ^k_start)
    ^q_in:
      aie.use_lock(%{tile}_q_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_q : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_q_full, Release, 1)
      aie.next_bd ^q_in

    ^k_start:
      %k_dma = aie.dma_start(S2MM, 1, ^k_in, ^carrier_start)
    ^k_in:
      aie.use_lock(%{tile}_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_k : memref<{K_WINDOW_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_k_full, Release, 1)
      aie.next_bd ^k_in

    ^carrier_start:
      %carrier_dma = aie.dma_start(MM2S, 0, ^carrier_out, ^end)
    ^carrier_out:
      aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_carrier : memref<{SHAPE_CARRIER_DWORDS}xi32>, 0, {SHAPE_CARRIER_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_carrier_empty, Release, 1)
      aie.next_bd ^carrier_out
    ^end:
      aie.end
    }}
"""


def _shape_b_multiblock_bf16(window: int) -> str:
    tile = shape_b_symbol(window)
    blocks_name = shape_blocks_name(tile)
    runtime_start = shape_runtime_start_name(tile)
    return f"""
    %{tile}_v = aie.buffer(%{tile}) {{sym_name = "{tile}_v"}} : memref<{V_WINDOW_DWORDS}xi32>
    %{tile}_carrier = aie.buffer(%{tile}) {{sym_name = "{tile}_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %{tile}_accum = aie.buffer(%{tile}) {{sym_name = "{tile}_accum"}} : memref<{ACCUM_LANES}xi32>
    %{tile}_state = aie.buffer(%{tile}) {{sym_name = "{tile}_state"}} : memref<{SCALAR_DWORDS}xi32>
    %{tile}_output = aie.buffer(%{tile}) {{sym_name = "{tile}_output"}} : memref<{ATTENTION_OUTPUT_DWORDS}xi32>
    %{blocks_name} = aie.buffer(%{tile}) {{sym_name = "{blocks_name}"}} : memref<1xi32>
{lock_pair(tile, "v", 0)}
{lock_pair(tile, "carrier", 2)}
{lock_pair(tile, "output", 4)}
    %{runtime_start} = aie.lock(%{tile}, 6) {{init = 0 : i32, sym_name = "{runtime_start}"}}

    %{tile}_core = aie.core(%{tile}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%{runtime_start}, Acquire, 1)
      %blocks_i32 = memref.load %{blocks_name}[%c0] : memref<1xi32>
      %blocks = arith.index_cast %blocks_i32 : i32 to index
      %v_dwords_i32 = arith.constant {V_WINDOW_DWORDS} : i32
      %out_dwords_i32 = arith.constant {ATTENTION_OUTPUT_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      %accum_lanes_i32 = arith.constant {ACCUM_LANES} : i32
      %state_dwords_i32 = arith.constant {SCALAR_DWORDS} : i32
      func.call @qwen3_attention_bf16_init_accum(%{tile}_accum, %{tile}_state, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) -> ()
      scf.for %block = %c0 to %blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%{tile}_v_full, AcquireGreaterEqual, 1)
        aie.use_lock(%{tile}_carrier_full, AcquireGreaterEqual, 1)
        func.call @qwen3_attention_bf16_accum_block(%{tile}_v, %{tile}_carrier, %{tile}_accum, %{tile}_state, %block_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
          : (memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%{tile}_v_empty, Release, 1)
        aie.use_lock(%{tile}_carrier_empty, Release, 1)
      }}
      aie.use_lock(%{tile}_output_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_attention_bf16_finish_accum(%{tile}_accum, %{tile}_state, %{tile}_output, %out_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{ATTENTION_OUTPUT_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%{tile}_output_full, Release, 1)
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %v_dma = aie.dma_start(S2MM, 0, ^v_in, ^carrier_start)
    ^v_in:
      aie.use_lock(%{tile}_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_v : memref<{V_WINDOW_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%{tile}_v_full, Release, 1)
      aie.next_bd ^v_in

    ^carrier_start:
      %carrier_dma = aie.dma_start(S2MM, 1, ^carrier_in, ^output_start)
    ^carrier_in:
      aie.use_lock(%{tile}_carrier_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_carrier : memref<{SHAPE_CARRIER_DWORDS}xi32>, 0, {SHAPE_CARRIER_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%{tile}_carrier_full, Release, 1)
      aie.next_bd ^carrier_in

    ^output_start:
      %output_dma = aie.dma_start(MM2S, 0, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%{tile}_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_output : memref<{ATTENTION_OUTPUT_DWORDS}xi32>, 0, {ATTENTION_OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%{tile}_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _postprocess_qkv_body() -> str:
    return f"""
    %post_qkv_payload = aie.buffer(%post) {{sym_name = "post_qkv_payload"}} : memref<{COMPACT_PACKET_DWORDS - 1}xi32>
    %post_q_body = aie.buffer(%post) {{sym_name = "post_q_body"}} : memref<{Q_DWORDS}xi32>
    %post_k_body = aie.buffer(%post) {{sym_name = "post_k_body"}} : memref<{CURRENT_DWORDS}xi32>
    %post_v_body = aie.buffer(%post) {{sym_name = "post_v_body"}} : memref<{CURRENT_DWORDS}xi32>
    %post_qk_rope_side = aie.buffer(%post) {{sym_name = "post_qk_rope_side"}} : memref<{QK_ROPE_DWORDS}xi32>
    %post_q_payload = aie.buffer(%post) {{sym_name = "post_q_payload"}} : memref<{Q_DWORDS}xi32>
    %post_current_k = aie.buffer(%post) {{sym_name = "post_current_k"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_v = aie.buffer(%post) {{sym_name = "post_current_v"}} : memref<{CURRENT_DWORDS}xi32>
    %post_current_token = aie.buffer(%post) {{sym_name = "post_current_token"}} : memref<1xi32>
{lock_pair("post", "qkv_payload", 0)}
{lock_pair("post", "q_payload", 6)}
{lock_pair("post", "current_k", 8)}
{lock_pair("post", "current_v", 10)}
{lock_pair("post", "qk_rope_side", 13)}
    %post_runtime_start = aie.lock(%post, 12) {{init = 0 : i32, sym_name = "post_runtime_start"}}

    %post_core = aie.core(%post) {{
      aie.use_lock(%post_runtime_start, Acquire, 1)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %qkv_records = arith.constant {Q_BODY_RECORDS + KV_BODY_RECORDS * 2} : index
      %q_dwords_i32 = arith.constant {Q_DWORDS} : i32
      %current_dwords_i32 = arith.constant {CURRENT_DWORDS} : i32
      scf.for %record = %c0 to %qkv_records step %c1 {{
        %record_i32 = arith.index_cast %record : index to i32
        aie.use_lock(%post_qkv_payload_full, AcquireGreaterEqual, 1)
        func.call @qwen3_postprocess_absorb_qkv_payload_record(%post_qkv_payload, %post_q_body, %post_k_body, %post_v_body, %record_i32)
          : (memref<{COMPACT_PACKET_DWORDS - 1}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, i32) -> ()
        aie.use_lock(%post_qkv_payload_empty, Release, 1)
      }}
      aie.use_lock(%post_qk_rope_side_full, AcquireGreaterEqual, 1)
      aie.use_lock(%post_q_payload_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_k_empty, AcquireGreaterEqual, 1)
      aie.use_lock(%post_current_v_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_postprocess_q4nx_body_payload(%post_q_body, %post_k_body, %post_v_body, %post_qk_rope_side, %post_q_payload, %post_current_k, %post_current_v, %post_current_token, %q_dwords_i32, %current_dwords_i32)
        : (memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) -> ()
      aie.use_lock(%post_qk_rope_side_empty, Release, 1)
      aie.use_lock(%post_q_payload_full, Release, 1)
      aie.use_lock(%post_current_k_full, Release, 1)
      aie.use_lock(%post_current_v_full, Release, 1)
      aie.end
    }}

    %post_mem = aie.mem(%post) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^qkv_in, ^side_start)
    ^qkv_in:
      aie.use_lock(%post_qkv_payload_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_qkv_payload : memref<{COMPACT_PACKET_DWORDS - 1}xi32>, 0, {COMPACT_PACKET_DWORDS - 1}) {{bd_id = 0 : i32}}
      aie.use_lock(%post_qkv_payload_full, Release, 1)
      aie.next_bd ^qkv_in

    ^side_start:
      %side_dma = aie.dma_start(S2MM, 1, ^side_in, ^q_out_start)
    ^side_in:
      aie.use_lock(%post_qk_rope_side_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_qk_rope_side : memref<{QK_ROPE_DWORDS}xi32>, 0, {QK_ROPE_DWORDS}) {{bd_id = 6 : i32}}
      aie.use_lock(%post_qk_rope_side_full, Release, 1)
      aie.next_bd ^side_in

    ^q_out_start:
      %q_dma = aie.dma_start(MM2S, 0, ^q_out, ^current_out_start)
    ^q_out:
      aie.use_lock(%post_q_payload_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_q_payload : memref<{Q_DWORDS}xi32>, 0, {Q_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%post_q_payload_empty, Release, 1)
      aie.next_bd ^q_out

    ^current_out_start:
      %current_dma = aie.dma_start(MM2S, 1, ^current_k_out, ^end)
    ^current_k_out:
      aie.use_lock(%post_current_k_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_k : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 5 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {CURRENT_PACKET_K}>}}
      aie.use_lock(%post_current_k_empty, Release, 1)
      aie.next_bd ^current_v_out
    ^current_v_out:
      aie.use_lock(%post_current_v_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%post_current_v : memref<{CURRENT_DWORDS}xi32>, 0, {CURRENT_DWORDS}) {{bd_id = 5 : i32, next_bd_id = 4 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {CURRENT_PACKET_V}>}}
      aie.use_lock(%post_current_v_empty, Release, 1)
      aie.next_bd ^current_k_out
    ^end:
      aie.end
    }}
"""


def _main_buffer_name(tile: str, buffer_name: str) -> str:
    return f"%{tile}_{buffer_name}"


def _record_ping_name(tile: str) -> str:
    return _main_buffer_name(tile, "record_ping")


def _record_pong_name(tile: str) -> str:
    return _main_buffer_name(tile, "record_pong")


def _record_pingpong_type() -> str:
    return f"memref<{MAIN_RECORD_PINGPONG_DWORDS}xi32>"


def _main_buffer_attr(tile: str, address: int) -> str:
    return f'{{address = {address} : i32, sym_name = "{tile}"}}'


def _main_lock_decls(tile: str, record_empty_init: int) -> str:
    return f"""
    %{tile}_chunk_empty = aie.lock(%{tile}, {MAIN_ACTIVATION_EMPTY_LOCK}) {{init = 2 : i32, sym_name = "{tile}_chunk_empty"}}
    %{tile}_chunk_full = aie.lock(%{tile}, {MAIN_ACTIVATION_FULL_LOCK}) {{init = 0 : i32, sym_name = "{tile}_chunk_full"}}
    %{tile}_wt_empty = aie.lock(%{tile}, {MAIN_WEIGHT_EMPTY_LOCK}) {{init = 2 : i32, sym_name = "{tile}_wt_empty"}}
    %{tile}_wt_full = aie.lock(%{tile}, {MAIN_WEIGHT_FULL_LOCK}) {{init = 0 : i32, sym_name = "{tile}_wt_full"}}
    %{tile}_records_empty = aie.lock(%{tile}, {MAIN_RECORD_EMPTY_LOCK}) {{init = {record_empty_init} : i32, sym_name = "{tile}_records_empty"}}
    %{tile}_records_full = aie.lock(%{tile}, {MAIN_RECORD_FULL_LOCK}) {{init = 0 : i32, sym_name = "{tile}_records_full"}}"""


def validate_main_buffer_residency(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            tile = _main_symbol(group, row)
            for buffer in MAIN16_BUFFERS:
                marker = (
                    f"%{tile}_{buffer.name} = aie.buffer(%{tile}) "
                    f'{{address = {buffer.address} : i32, sym_name = "{tile}_{buffer.name}"}}'
                )
                if marker not in mlir:
                    errors.append(
                        f"{scope}: main16 {tile}_{buffer.name} must stay pinned at 0x{buffer.address:04x}"
                    )
    return errors


def resource_manifest_for_case(
    case_name: str,
    phase_trace: tuple[CompactPhase, ...] = QKV_BODY_PHASE_TRACE,
) -> ResourceManifest:
    return ResourceManifest(
        schema="qwen3-layer-resource-manifest-v1",
        case=case_name,
        tiles=tuple(
            _main16_tile_resources(group, row, phase_trace)
            for group in range(len(MAIN_COLUMNS))
            for row in range(len(MAIN_ROWS))
        ),
    )


def resource_manifest(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> ResourceManifest:
    return resource_manifest_for_case(CASE_NAME)


def _main16_tile_resources(group: int, row: int, phase_trace: tuple[CompactPhase, ...]) -> TileResource:
    tile = _main_symbol(group, row)
    return TileResource(
        name=tile,
        coord=(MAIN_COLUMNS[group], MAIN_ROWS[row]),
        role="main16",
        buffers=tuple(_main_buffer_resource(tile, buffer) for buffer in MAIN16_BUFFERS),
        locks=(
            _main_lock_resource(
                tile,
                "chunk_empty",
                MAIN_ACTIVATION_EMPTY_LOCK,
                2,
                "main16 DMA0 activation ping/pong reusable",
            ),
            _main_lock_resource(
                tile,
                "chunk_full",
                MAIN_ACTIVATION_FULL_LOCK,
                0,
                "main16 DMA0 activation ping/pong ready",
            ),
            _main_lock_resource(
                tile,
                "wt_empty",
                MAIN_WEIGHT_EMPTY_LOCK,
                2,
                "main16 DMA1 Q4NX weight ping/pong reusable",
            ),
            _main_lock_resource(
                tile,
                "wt_full",
                MAIN_WEIGHT_FULL_LOCK,
                0,
                "main16 DMA1 Q4NX weight ping/pong ready",
            ),
            _main_lock_resource(
                tile,
                "records_empty",
                MAIN_RECORD_EMPTY_LOCK,
                2,
                "main16 compact record ping/pong reusable",
            ),
            _main_lock_resource(
                tile,
                "records_full",
                MAIN_RECORD_FULL_LOCK,
                0,
                "main16 compact record ping/pong ready",
            ),
        ),
        bds=(
            BDResource(
                name="activation_ping",
                bd_id=MAIN_ACTIVATION_BDS[0],
                channel=0,
                direction="S2MM",
                length=MAIN_CHUNK_DWORDS,
                buffer=f"{tile}_chunk_ping",
                phases=_all_phases(),
                semantic="main16 DMA0 activation ping",
                next_bd_id=MAIN_ACTIVATION_BDS[1],
            ),
            BDResource(
                name="activation_pong",
                bd_id=MAIN_ACTIVATION_BDS[1],
                channel=0,
                direction="S2MM",
                length=MAIN_CHUNK_DWORDS,
                buffer=f"{tile}_chunk_pong",
                phases=_all_phases(),
                semantic="main16 DMA0 activation pong",
                next_bd_id=MAIN_ACTIVATION_BDS[0],
            ),
            *_main_record_bds(tile, row, phase_trace),
            BDResource(
                name="weight_ping",
                bd_id=MAIN_WEIGHT_BDS[0],
                channel=1,
                direction="S2MM",
                length=CHUNK_BF16,
                buffer=f"{tile}_wt_ping",
                phases=_all_phases(),
                semantic="main16 DMA1 Q4NX weight ping",
                next_bd_id=MAIN_WEIGHT_BDS[1],
            ),
            BDResource(
                name="weight_pong",
                bd_id=MAIN_WEIGHT_BDS[1],
                channel=1,
                direction="S2MM",
                length=CHUNK_BF16,
                buffer=f"{tile}_wt_pong",
                phases=_all_phases(),
                semantic="main16 DMA1 Q4NX weight pong",
                next_bd_id=MAIN_WEIGHT_BDS[0],
            ),
        ),
    )


def _main_buffer_resource(
    tile: str,
    buffer: Main16Buffer,
) -> BufferResource:
    return BufferResource(
        name=buffer.name,
        mlir_name=f"{tile}_{buffer.name}",
        address=buffer.address,
        size_bytes=buffer.size_bytes,
        phases=buffer.phases,
        semantic=f"main16 {buffer.semantic}",
    )


def _main_lock_resource(tile: str, suffix: str, lock_id: int, init: int, semantic: str) -> LockResource:
    return LockResource(
        name=suffix,
        mlir_name=f"{tile}_{suffix}",
        lock_id=lock_id,
        init=init,
        phases=_all_phases(),
        semantic=semantic,
    )


def _main_record_bds(tile: str, row: int, phase_trace: tuple[CompactPhase, ...]) -> tuple[BDResource, ...]:
    row_length = RECORD_DWORDS if row == 0 else RECORD_PAYLOAD_DWORDS
    return (
        BDResource(
            name="record_ping",
            bd_id=MAIN_RECORD_BDS[0],
            channel=1,
            direction="MM2S",
            length=row_length,
            buffer=f"{tile}_record_ping",
            phases=_all_phases(),
            semantic="main16 compact record ping",
            next_bd_id=MAIN_RECORD_BDS[1],
        ),
        BDResource(
            name="record_pong",
            bd_id=MAIN_RECORD_BDS[1],
            channel=1,
            direction="MM2S",
            length=row_length,
            buffer=f"{tile}_record_pong",
            phases=_all_phases(),
            semantic="main16 compact record pong",
            next_bd_id=MAIN_RECORD_BDS[0],
        ),
    )


def _phases(phase: str) -> tuple[str, ...]:
    if phase == "ALL":
        return _all_phases()
    return (phase,)


def _all_phases() -> tuple[str, ...]:
    return MAIN16_ALL_PHASES


def _main_record_dma_blocks(tile: str, row: int) -> str:
    source_offset = 0 if row == 0 else 1
    source_length = RECORD_DWORDS if row == 0 else RECORD_PAYLOAD_DWORDS
    record_type = _record_pingpong_type()
    return f"""    ^record_ping:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd({_record_ping_name(tile)} : {record_type}, {source_offset}, {source_length}) {{bd_id = {MAIN_RECORD_BDS[0]} : i32, next_bd_id = {MAIN_RECORD_BDS[1]} : i32}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^record_pong
    ^record_pong:
      aie.use_lock(%{tile}_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd({_record_pong_name(tile)} : {record_type}, {source_offset}, {source_length}) {{bd_id = {MAIN_RECORD_BDS[1]} : i32, next_bd_id = {MAIN_RECORD_BDS[0]} : i32}}
      aie.use_lock(%{tile}_records_empty, Release, 1)
      aie.next_bd ^record_ping"""


def _main16_layer_scheduler_call(tile: str, group: int, row: int, phase_limit: int) -> str:
    record_type = _record_pingpong_type()
    return f"""
      %m_i32 = arith.constant 32 : i32
      %group_i32 = arith.constant {group} : i32
      %row_i32 = arith.constant {row} : i32
      %main16_phase_limit_i32 = arith.constant {phase_limit} : i32
      func.call @{MAIN16_LAYER_SCHEDULER}(%{tile}_wt_ping, %{tile}_wt_pong, %{tile}_chunk_ping, %{tile}_chunk_pong, {_record_ping_name(tile)}, {_record_pong_name(tile)}, %group_i32, %row_i32, %m_i32, %main16_phase_limit_i32)
        : (memref<{CHUNK_BF16}xbf16>, memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_CHUNK_DWORDS}xi32>, {record_type}, {record_type}, i32, i32, i32, i32) -> ()"""


def main16_qkv_prefix_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    buffer_decls = "\n".join(
        f"    %{tile}_{buffer.name} = aie.buffer(%{tile}) "
        f'{_main_buffer_attr(f"{tile}_{buffer.name}", buffer.address)} : {buffer.memref}'
        for buffer in MAIN16_BUFFERS
    )
    return f"""
{buffer_decls}
{_main_lock_decls(tile, 2)}

    %{tile}_core = aie.core(%{tile}) {{
{_main16_layer_scheduler_call(tile, group, row, MAIN16_PHASE_LIMIT_QKV)}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^wt_start)
    ^chunk_ping:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_ping : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_pong
    ^chunk_pong:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_pong : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_ping

    ^wt_start:
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[0]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[1]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[1]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[0]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^record_ping, ^end)
{_main_record_dma_blocks(tile, row)}
    ^end:
      aie.end
    }}
"""


def main16_qkvo_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    buffer_decls = "\n".join(
        f"    %{tile}_{buffer.name} = aie.buffer(%{tile}) "
        f'{_main_buffer_attr(f"{tile}_{buffer.name}", buffer.address)} : {buffer.memref}'
        for buffer in MAIN16_BUFFERS
    )
    return f"""
{buffer_decls}
{_main_lock_decls(tile, 2)}

    %{tile}_core = aie.core(%{tile}) {{
{_main16_layer_scheduler_call(tile, group, row, MAIN16_PHASE_LIMIT_QKVO)}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^wt_start)
    ^chunk_ping:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_ping : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_pong
    ^chunk_pong:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_pong : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_ping

    ^wt_start:
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[0]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[1]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[1]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[0]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^record_ping, ^end)
{_main_record_dma_blocks(tile, row)}
    ^end:
      aie.end
    }}
"""


def _main_tile(group: int, row: int) -> str:
    tile = _main_symbol(group, row)
    buffer_decls = "\n".join(
        f"    %{tile}_{buffer.name} = aie.buffer(%{tile}) "
        f'{_main_buffer_attr(f"{tile}_{buffer.name}", buffer.address)} : {buffer.memref}'
        for buffer in MAIN16_BUFFERS
    )
    return f"""
{buffer_decls}
{_main_lock_decls(tile, 2)}

    %{tile}_core = aie.core(%{tile}) {{
{_main16_layer_scheduler_call(tile, group, row, MAIN16_PHASE_LIMIT_FULL)}
      aie.end
    }}

    %{tile}_mem = aie.mem(%{tile}) {{
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^wt_start)
    ^chunk_ping:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_ping : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_pong
    ^chunk_pong:
      aie.use_lock(%{tile}_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_chunk_pong : memref<{MAIN_CHUNK_DWORDS}xi32>, 0, {MAIN_CHUNK_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%{tile}_chunk_full, Release, 1)
      aie.next_bd ^chunk_ping

    ^wt_start:
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)
    ^wt_ping:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_ping : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[0]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[1]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%{tile}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{tile}_wt_pong : memref<{CHUNK_BF16}xbf16>, 0, {CHUNK_BF16}) {{bd_id = {MAIN_WEIGHT_BDS[1]} : i32, next_bd_id = {MAIN_WEIGHT_BDS[0]} : i32}}
      aie.use_lock(%{tile}_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^record_ping, ^end)
{_main_record_dma_blocks(tile, row)}
    ^end:
      aie.end
    }}
"""


def _full_vector_q4nx_output() -> str:
    replay_payload_dwords = C1R2_PACKET_DWORDS - 1
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_vector = aie.buffer(%full) {{sym_name = "full_vector"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_compact = aie.buffer(%full) {{sym_name = "full_compact"}} : memref<{COMPACT_PACKET_DWORDS}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{replay_payload_dwords}xi32>
    %full_output = aie.buffer(%full) {{sym_name = "full_output"}} : memref<{OUTPUT_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}
    %full_vector_empty = aie.lock(%full, 2) {{init = 1 : i32, sym_name = "full_vector_empty"}}
    %full_vector_full = aie.lock(%full, 3) {{init = 0 : i32, sym_name = "full_vector_full"}}
    %full_compact_empty = aie.lock(%full, 4) {{init = 1 : i32, sym_name = "full_compact_empty"}}
    %full_compact_full = aie.lock(%full, 5) {{init = 0 : i32, sym_name = "full_compact_full"}}
    %full_replay_empty = aie.lock(%full, 6) {{init = 1 : i32, sym_name = "full_replay_empty"}}
    %full_replay_full = aie.lock(%full, 7) {{init = 0 : i32, sym_name = "full_replay_full"}}
    %full_output_empty = aie.lock(%full, 8) {{init = 1 : i32, sym_name = "full_output_empty"}}
    %full_output_full = aie.lock(%full, 9) {{init = 0 : i32, sym_name = "full_output_full"}}

    %full_core = aie.core(%full) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %o_blocks = arith.constant {O_BODY_RECORDS} : index
      %down_blocks = arith.constant {DOWN_BODY_RECORDS} : index
      %payload_i32 = arith.constant {replay_payload_dwords} : i32

      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_vector_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_make_input_norm_payload(%full_hidden, %full_vector, %full_replay, %payload_i32)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{replay_payload_dwords}xi32>, i32) -> ()
      aie.use_lock(%full_replay_full, Release, {C1R2_QKV_REPLAYS})
      aie.use_lock(%full_vector_empty, Release, 1)

      scf.for %block = %c0 to %o_blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
        func.call @full_c1r2_add_o_compact_to_residual(%full_hidden, %full_compact, %block_i32)
          : (memref<{HIDDEN_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, i32) -> ()
        aie.use_lock(%full_compact_empty, Release, 1)
      }}

      aie.use_lock(%full_vector_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_replay_empty, AcquireGreaterEqual, {C1R2_QKV_REPLAYS})
      func.call @full_c1r2_make_post_norm_payload(%full_hidden, %full_vector, %full_replay, %payload_i32)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{replay_payload_dwords}xi32>, i32) -> ()
      aie.use_lock(%full_replay_full, Release, {C1R2_UPGATE_REPLAYS})
      aie.use_lock(%full_vector_empty, Release, 1)

      aie.use_lock(%full_output_empty, AcquireGreaterEqual, 1)
      scf.for %block = %c0 to %down_blocks step %c1 {{
        %block_i32 = arith.index_cast %block : index to i32
        aie.use_lock(%full_compact_full, AcquireGreaterEqual, 1)
        func.call @full_c1r2_write_down_block(%full_hidden, %full_compact, %full_output, %block_i32)
          : (memref<{HIDDEN_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) -> ()
        aie.use_lock(%full_compact_empty, Release, 1)
      }}
      aie.use_lock(%full_output_full, Release, 1)
      aie.use_lock(%full_hidden_empty, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %compact_dma = aie.dma_start(S2MM, 0, ^compact_in, ^hidden_in_start)
    ^compact_in:
      aie.use_lock(%full_compact_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_compact : memref<{COMPACT_PACKET_DWORDS}xi32>, 0, {COMPACT_PACKET_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%full_compact_full, Release, 1)
      aie.next_bd ^compact_in

    ^hidden_in_start:
      %hidden_dma = aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 3 : i32}}
      aie.use_lock(%full_hidden_full, Release, 1)
      aie.next_bd ^input_norm_in
    ^input_norm_in:
      aie.use_lock(%full_vector_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_vector : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 4 : i32}}
      aie.use_lock(%full_vector_full, Release, 1)
      aie.next_bd ^post_norm_in
    ^post_norm_in:
      aie.use_lock(%full_vector_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_vector : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 5 : i32}}
      aie.use_lock(%full_vector_full, Release, 1)
      aie.next_bd ^input_end
    ^input_end:
      aie.end

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 1, ^replay_out, ^output_out_start)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{replay_payload_dwords}xi32>, 0, {replay_payload_dwords}) {{bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {FULL_REPLAY_PACKET_ID}>}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^replay_out

    ^output_out_start:
      %output_dma = aie.dma_start(MM2S, 0, ^output_out, ^end)
    ^output_out:
      aie.use_lock(%full_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%full_output_empty, Release, 1)
      aie.next_bd ^output_out
    ^end:
      aie.end
    }}
"""


def _swiglu_bf16() -> str:
    return f"""
    %swiglu_input = aie.buffer(%swiglu) {{sym_name = "swiglu_input"}} : memref<{C6R2_INPUT_DWORDS}xi32>
    %swiglu_output = aie.buffer(%swiglu) {{sym_name = "swiglu_output"}} : memref<{C6R2_HALF_DWORDS * 2}xbf16>
    %swiglu_input_empty = aie.lock(%swiglu, 0) {{init = 2 : i32, sym_name = "swiglu_input_empty"}}
    %swiglu_input_full = aie.lock(%swiglu, 1) {{init = 0 : i32, sym_name = "swiglu_input_full"}}
    %swiglu_output_empty = aie.lock(%swiglu, 2) {{init = 1 : i32, sym_name = "swiglu_output_empty"}}
    %swiglu_output_full = aie.lock(%swiglu, 3) {{init = 0 : i32, sym_name = "swiglu_output_full"}}

    %swiglu_core = aie.core(%swiglu) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %repeats = arith.constant {C1R2_UPGATE_REPLAYS // 2} : index
      %dwords_i32 = arith.constant {C6R2_INPUT_DWORDS} : i32
      scf.for %repeat = %c0 to %repeats step %c1 {{
        %slice_i32 = arith.index_cast %repeat : index to i32
        aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)
        aie.use_lock(%swiglu_output_empty, AcquireGreaterEqual, 1)
        func.call @ffn_swiglu_slice_bf16_inputs(%swiglu_input, %swiglu_output, %dwords_i32, %slice_i32)
          : (memref<{C6R2_INPUT_DWORDS}xi32>, memref<{C6R2_HALF_DWORDS * 2}xbf16>, i32, i32) -> ()
        aie.use_lock(%swiglu_output_full, Release, 1)
        aie.use_lock(%swiglu_input_empty, Release, 2)
      }}
      aie.end
    }}

    %swiglu_mem = aie.mem(%swiglu) {{
      %input_dma = aie.dma_start(S2MM, 0, ^up_in, ^out_start)
    ^up_in:
      aie.use_lock(%swiglu_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_input : memref<{C6R2_INPUT_DWORDS}xi32>, 0, {C6R2_HALF_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%swiglu_input_full, Release, 1)
      aie.next_bd ^gate_in
    ^gate_in:
      aie.use_lock(%swiglu_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_input : memref<{C6R2_INPUT_DWORDS}xi32>, {C6R2_HALF_DWORDS}, {C6R2_HALF_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%swiglu_input_full, Release, 1)
      aie.next_bd ^up_in

    ^out_start:
      %output_dma = aie.dma_start(MM2S, 1, ^out, ^end)
    ^out:
      aie.use_lock(%swiglu_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%swiglu_output : memref<{C6R2_HALF_DWORDS * 2}xbf16>, 0, {C6R2_HALF_DWORDS * 2}) {{bd_id = 2 : i32}}
      aie.use_lock(%swiglu_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}
"""


def generate_mlir(schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    tile_defs = [
        "    %shim_left = aie.tile(0, 0)",
        "    %kv_left = aie.tile(0, 1)",
        "    %shim_out = aie.tile(1, 0)",
        "    %bridge = aie.tile(1, 1)",
        "    %full = aie.tile(1, 2)",
        "    %post = aie.tile(1, 3)",
        "    %hub = aie.tile(6, 1)",
        "    %swiglu = aie.tile(6, 2)",
        "    %shim_right = aie.tile(7, 0)",
        "    %kv_right = aie.tile(7, 1)",
    ]
    for group, column in enumerate(MAIN_COLUMNS):
        tile_defs.append(f"    %shim{group} = aie.tile({column}, 0)")
        tile_defs.append(f"    %mt{group} = aie.tile({column}, 1)")
        for row_idx, row in enumerate(MAIN_ROWS):
            tile_defs.append(f"    %{_main_symbol(group, row_idx)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_A_TILES):
        tile_defs.append(f"    %{shape_a_symbol(window)} = aie.tile({column}, {row})")
    for window, (column, row) in enumerate(SHAPE_B_TILES):
        tile_defs.append(f"    %{shape_b_symbol(window)} = aie.tile({column}, {row})")

    flows = [f"    // case marker {CASE_NAME}"]
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            flows.append(flow(_main_symbol(group, row), 1, f"mt{group}", row))
            flows.append(flow("bridge", 1, _main_symbol(group, row), 0))
            flows.append(flow(f"mt{group}", row, _main_symbol(group, row), 1))
        flows.append(flow(f"mt{group}", COLUMN_OUT_CHANNEL, "bridge", group))
        flows.append(flow(f"shim{group}", 0, f"mt{group}", 4))
        flows.append(flow(f"shim{group}", 1, f"mt{group}", 5))
    for packet in (Q_GLOBAL_PACKET_ID, K_GLOBAL_PACKET_ID, V_GLOBAL_PACKET_ID):
        flows.append(packet_flow(packet, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "post", 0))
    flows.extend(
        (
            packet_flow(CURRENT_PACKET_K, "post", 1, "shim_left", 1),
            packet_flow(CURRENT_PACKET_V, "post", 1, "shim_right", 1),
            packet_flow(O_GLOBAL_PACKET_ID, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "full", 0, keep_pkt_header=True),
            flow("post", 0, "hub", HUB_Q_IN_CHANNEL),
            flow("shim_left", 0, "kv_left", 0),
            flow("shim_left", 1, "kv_left", 1),
            flow("shim_right", 0, "kv_right", 0),
            flow("shim_right", 1, "kv_right", 1),
        )
    )
    for window in range(4):
        kv_tile = "kv_left" if window < 2 else "kv_right"
        kv_k_channel = 0 if window in (0, 2) else 2
        kv_v_channel = 1 if window in (0, 2) else 3
        flows.append(flow("hub", HUB_Q_OUT_CHANNELS[window], shape_a_symbol(window), 0))
        flows.append(flow(kv_tile, kv_k_channel, shape_a_symbol(window), 1))
        flows.append(flow(kv_tile, kv_v_channel, shape_b_symbol(window), 0))
        flows.append(flow(shape_a_symbol(window), 0, shape_b_symbol(window), 1))
        flows.append(flow(shape_b_symbol(window), 0, "hub", HUB_RETURN_IN_CHANNELS[window]))
    flows.extend(
        (
            packet_flow(PACKET_ID_ATTENTION, "hub", 5, "bridge", 4),
            packet_flow(FULL_REPLAY_PACKET_ID, "full", 1, "bridge", 4),
            packet_flow(FFN_GLOBAL_PACKET_ID, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "swiglu", 0),
            flow("swiglu", 1, "hub", HUB_FFN_IN_CHANNEL),
            packet_flow(DOWN_ACT_PACKET_ID, "hub", 5, "bridge", 4),
            packet_flow(DOWN_GLOBAL_PACKET_ID, "bridge", BRIDGE_COMPACT_OUT_CHANNEL, "full", 0, keep_pkt_header=True),
            flow("shim_out", 0, "full", 1),
            flow("shim_out", 1, "post", 1),
            flow("full", 0, "shim_out", 1),
        )
    )

    blocks = [
        _bridge(QKV_BODY_PHASE_TRACE),
        _postprocess_qkv_body(),
        _hub(),
        kv_split_scan_memtile(0),
        kv_split_scan_memtile(1),
        _full_vector_q4nx_output(),
        _swiglu_bf16(),
    ]
    for window in range(4):
        blocks.append(_shape_a_multiblock_bf16(window))
        blocks.append(_shape_b_multiblock_bf16(window))
    for group in range(len(MAIN_COLUMNS)):
        blocks.append(q4nx_weight_column_memtile(group, QKV_BODY_PHASE_TRACE))
        for row in range(len(MAIN_ROWS)):
            blocks.append(_main_tile(group, row))

    return f"""module {{
  aie.device(npu2) {{
{chr(10).join(tile_defs)}

{chr(10).join(flows)}

    func.func private @qwen3_postprocess_absorb_qkv_payload_record(memref<{COMPACT_PACKET_DWORDS - 1}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @qwen3_postprocess_q4nx_body_payload(memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{QK_ROPE_DWORDS}xi32>, memref<{Q_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<{CURRENT_DWORDS}xi32>, memref<1xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/postprocess_qkv.o"}}
    func.func private @full_c1r2_make_input_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_add_o_compact_to_residual(memref<{HIDDEN_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_make_post_norm_payload(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @full_c1r2_write_down_block(memref<{HIDDEN_DWORDS}xi32>, memref<{COMPACT_PACKET_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}
    func.func private @ffn_swiglu_slice_bf16_inputs(memref<{C6R2_INPUT_DWORDS}xi32>, memref<{C6R2_HALF_DWORDS * 2}xbf16>, i32, i32) attributes {{link_with = "{experiment_dir}/swiglu.o"}}
    func.func private @qwen3_attention_bf16_make_carrier_masked(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_init_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_accum_block(memref<{V_WINDOW_DWORDS}xi32>, memref<{SCALAR_DWORDS + WEIGHT_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @qwen3_attention_bf16_finish_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{ATTENTION_OUTPUT_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/edge_attention.o"}}
    func.func private @{MAIN16_LAYER_SCHEDULER}(memref<{CHUNK_BF16}xbf16>, memref<{CHUNK_BF16}xbf16>, memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_CHUNK_DWORDS}xi32>, memref<{MAIN_RECORD_PINGPONG_DWORDS}xi32>, memref<{MAIN_RECORD_PINGPONG_DWORDS}xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{MAIN16_KERNEL_OBJECT}"}}

{chr(10).join(blocks)}
{_runtime_sequence(schedule)}
  }}
}}
"""


def validate_generated_mlir(mlir: str, schedule: DecodeSchedule = DEFAULT_SCHEDULE) -> list[str]:
    ownership = resource_manifest(schedule)
    required = (
        f"case marker {CASE_NAME}",
        f"compact phase trace {_phase_trace_marker(QKV_BODY_PHASE_TRACE)}",
        "qwen3_postprocess_absorb_qkv_payload_record",
        "qwen3_postprocess_q4nx_body_payload",
        f"aie.dma_bd(%post_qkv_payload : memref<{COMPACT_PACKET_DWORDS - 1}xi32>, 0, {COMPACT_PACKET_DWORDS - 1})",
        f"aie.dma_bd(%post_qk_rope_side : memref<{QK_ROPE_DWORDS}xi32>, 0, {QK_ROPE_DWORDS})",
        "qwen3_attention_bf16_make_carrier_masked",
        "qwen3_attention_bf16_finish_accum",
        "full_c1r2_make_input_norm_payload",
        "full_c1r2_add_o_compact_to_residual",
        "full_c1r2_make_post_norm_payload",
        MAIN16_LAYER_SCHEDULER,
        "ffn_swiglu_slice_bf16_inputs",
        "full_c1r2_write_down_block",
        MAIN16_KERNEL_OBJECT,
        f"aie.packet_flow({CURRENT_PACKET_K})",
        f"aie.packet_flow({CURRENT_PACKET_V})",
        f"aie.packet_flow({FFN_GLOBAL_PACKET_ID})",
        f"aie.packet_flow({DOWN_GLOBAL_PACKET_ID})",
        f"pkt_id = {CURRENT_PACKET_K}",
        f"pkt_id = {CURRENT_PACKET_V}",
        f"memref<{schedule.kv_cache_dwords}xi32>",
        f"memref<{TOTAL_WEIGHT_AND_AUX_I32}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
        f"aiex.npu.rtp_write(@shape_a0_blocks, 0, {schedule.kv_blocks})",
        f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
        "postprocess_qkv.o",
        "full_vector_station.o",
        "swiglu.o",
        "edge_attention.o",
        "aiex.set_lock(%post_runtime_start, 1)",
        "aiex.set_lock(%shape_a0_runtime_start, 1)",
        f"iteration_size = {schedule.kv_blocks} : i32",
        f"iteration_stride = {CACHE_BLOCK_DWORDS - 1} : i32",
        f"repeat_count = {schedule.kv_blocks - 1} : i32",
        "aie.dma_start(S2MM, 4, ^patch0_q4nx_ping, ^patch1_start)",
        "aie.dma_start(S2MM, 5, ^patch1_q4nx_ping, ^wt_row0_start)",
        "aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)",
        "aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)",
        "arg_idx = 3 : i32",
        "arg_idx = 2 : i32",
        "arg_idx = 4 : i32",
    )
    errors = [f"missing full-layer engine marker: {marker}" for marker in required if marker not in mlir]
    if mlir.count(MAIN16_KERNEL_OBJECT) != 1:
        errors.append(f"full-layer engine expected 1 declaration linked with {MAIN16_KERNEL_OBJECT}")
    errors.extend(validate_resource_manifest(CASE_NAME, ownership))
    errors.extend(validate_manifest_matches_mlir(CASE_NAME, ownership, mlir))
    if "qwen3_layer.o" in mlir:
        errors.append("full-layer engine must not link the old mixed qwen3_layer object")
    if "qwen3_bridge.o" in mlir:
        errors.append("full-layer engine must not link the old mixed qwen3_bridge object")
    if "debug_contract.o" in mlir:
        errors.append("full-layer engine must not link debug_contract.o")
    errors.extend(_phase_trace_errors(QKV_BODY_PHASE_TRACE))
    errors.extend(
        require_marker_order(
            CASE_NAME,
            mlir,
            (
                f"aiex.npu.rtp_write(@post_current_token, 0, {schedule.current_token})",
                f"aiex.npu.rtp_write(@shape_a0_tail_tokens, 0, {schedule.tail_tokens})",
                f"aiex.npu.push_queue(0, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                f"aiex.npu.push_queue(7, 0, S2MM : {CURRENT_WRITE_CHANNEL}) {{bd_id = {CURRENT_WRITE_BDS[0]} : i32",
                "aiex.npu.push_queue(1, 0, S2MM : 1) {bd_id = 13 : i32",
                "aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 15 : i32",
                "aiex.npu.push_queue(5, 0, MM2S : 1) {bd_id = 1 : i32",
                "aiex.set_lock(%post_runtime_start, 1)",
                f"aiex.npu.sync {{channel = {CURRENT_WRITE_CHANNEL} : i32, column = 0 : i32",
                f"aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = {K_SCAN_BD} : i32",
                f"aiex.npu.push_queue(7, 0, MM2S : 1) {{bd_id = {V_SCAN_BD} : i32",
                "aiex.set_lock(%shape_a0_runtime_start, 1)",
            ),
        )
    )
    expected_packets = 11
    errors.extend(require_count(CASE_NAME, "packet flow", mlir.count("aie.packet_flow("), expected_packets))
    main_tile_count = len(MAIN_COLUMNS) * len(MAIN_ROWS)
    errors.extend(require_count(CASE_NAME, "q4nx main16 layer scheduler calls", mlir.count(f"func.call @{MAIN16_LAYER_SCHEDULER}"), main_tile_count))
    errors.extend(require_count(CASE_NAME, "main16 full phase limit constants", mlir.count(f"%main16_phase_limit_i32 = arith.constant {MAIN16_PHASE_LIMIT_FULL} : i32"), main_tile_count))
    errors.extend(require_count(CASE_NAME, "q4nx old q emit calls in MLIR", mlir.count("func.call @q4nx_emit_q_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "q4nx old k emit calls in MLIR", mlir.count("func.call @q4nx_emit_k_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "q4nx old v emit calls in MLIR", mlir.count("func.call @q4nx_emit_v_accum_body_record"), 0))
    errors.extend(require_count(CASE_NAME, "qkv q-loop index constants", mlir.count("%c8 = arith.constant 8 : index"), 0))
    errors.extend(require_count(CASE_NAME, "qkv chunk-loop index constants", mlir.count("%c16 = arith.constant 16 : index"), 0))
    errors.extend(require_count(CASE_NAME, "qwen3 compact record absorb", mlir.count("qwen3_postprocess_absorb_qkv_payload_record"), 2))
    errors.extend(require_count(CASE_NAME, "qwen3 q4nx body postprocess", mlir.count("qwen3_postprocess_q4nx_body_payload"), 2))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_make_carrier_masked", mlir.count("qwen3_attention_bf16_make_carrier_masked"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_init_accum", mlir.count("qwen3_attention_bf16_init_accum"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_accum_block", mlir.count("qwen3_attention_bf16_accum_block"), 5))
    errors.extend(require_count(CASE_NAME, "qwen3_attention_bf16_finish_accum", mlir.count("qwen3_attention_bf16_finish_accum"), 5))
    errors.extend(require_count(CASE_NAME, "q4nx old o emit calls in MLIR", mlir.count("func.call @q4nx_emit_o_block_accum_record"), 0))
    errors.extend(require_count(CASE_NAME, "q4nx old upgate emit calls in MLIR", mlir.count("func.call @q4nx_emit_upgate_accum_record"), 0))
    errors.extend(require_count(CASE_NAME, "q4nx fast single-accum chunk call sites in MLIR", mlir.count("func.call @q4nx_chunk_accum_slice_i32_fast("), 0))
    errors.extend(require_count(CASE_NAME, "q4nx fast block-accum chunk call sites in MLIR", mlir.count("func.call @q4nx_chunk_accum_block_slice_i32_fast("), 0))
    errors.extend(require_count(CASE_NAME, "q4nx old down emit calls in MLIR", mlir.count("func.call @q4nx_emit_down_block_accum_record"), 0))
    expected_arg2_patches = 3 + len(MAIN_COLUMNS) * len(WEIGHT_PATCH_BD_IDS) * len(_full_weight_spans())
    errors.extend(
        require_count(
            CASE_NAME,
            "aux-prefixed weight arg2 address patches",
            mlir.count("arg_idx = 2 : i32"),
            expected_arg2_patches,
        )
    )
    errors.extend(require_count(CASE_NAME, "output arg3 address patches", mlir.count("arg_idx = 3 : i32"), 1))
    errors.extend(require_count(CASE_NAME, "hidden arg4 address patches", mlir.count("arg_idx = 4 : i32"), 1))
    errors.extend(require_unique_packet_flows(CASE_NAME, mlir))
    errors.extend(require_dma_next_bd_labels(CASE_NAME, mlir))
    errors.extend(require_dma_bd_next_ids(CASE_NAME, mlir))
    errors.extend(require_main_record_pingpong(CASE_NAME, mlir))
    errors.extend(require_compact_record_packet_granularity(CASE_NAME, mlir))
    attention_replay_blocks = tuple(f"^attention_out{idx}" for idx in range(O_BODY_RECORDS))
    down_replay_blocks = tuple(f"^down_out{idx}" for idx in range(DOWN_BODY_RECORDS))
    errors.extend(
        require_dma_bd_lock_balance(
            CASE_NAME,
            mlir,
            allowed_unbalanced_blocks=(
                attention_replay_blocks[0],
                attention_replay_blocks[-1],
                down_replay_blocks[0],
                down_replay_blocks[-1],
            ),
        )
    )
    errors.extend(
        require_source_side_packet_replay(
            CASE_NAME,
            mlir,
            "attention_out",
            PACKET_ID_ATTENTION,
            O_BODY_RECORDS,
            "aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)",
            "aie.use_lock(%hub_return_empty, Release, 4)",
        )
    )
    errors.extend(
        require_source_side_packet_replay(
            CASE_NAME,
            mlir,
            "down_out",
            DOWN_ACT_PACKET_ID,
            DOWN_BODY_RECORDS,
            "aie.use_lock(%hub_ffn_full, AcquireGreaterEqual, 1)",
            "aie.use_lock(%hub_ffn_empty, Release, 1)",
        )
    )
    errors.extend(require_memtile_dma_bd_bank(CASE_NAME, mlir))
    errors.extend(validate_main_buffer_residency(CASE_NAME, mlir))
    errors.extend(
        require_attention_block_shapes(
            CASE_NAME,
            K_WINDOW_DWORDS,
            V_WINDOW_DWORDS,
            KV_SIDE_DWORDS,
            K_CACHE_SIDE_DWORDS,
            V_CACHE_SIDE_DWORDS,
            SCALAR_DWORDS + WEIGHT_DWORDS,
            WEIGHT_DWORDS,
            SCALAR_DWORDS,
            WEIGHT_DWORDS * 8,
        )
    )
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 4))
    errors.extend(require_no_compute_kv_materialization(CASE_NAME, mlir, KV_SIDE_DWORDS, schedule.kv_cache_dwords * 2))
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SCAN_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, CURRENT_WRITE_BDS))
    errors.extend(require_unique_bd_ids(CASE_NAME, KV_SPLIT_K_IN_BDS + KV_SPLIT_V_IN_BDS + KV_OUT_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, KV_SCAN_BDS, CURRENT_WRITE_BDS))
    errors.extend(require_disjoint_bd_ids(CASE_NAME, BRIDGE_PACKET_IN_BDS, BRIDGE_PACKET_OUT_BDS))
    for group, bd_ids in enumerate(BRIDGE_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} bridge compact receive group {group}", bd_ids))
    for row, bd_ids in enumerate(COLUMN_RECEIVE_BDS):
        errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact receive row {row}", bd_ids))
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row compact output", COLUMN_OUT_BDS))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    errors.extend(require_npu_push_queue_repeat_range(CASE_NAME, mlir))
    errors.extend(
        require_absent_markers(
            CASE_NAME,
            mlir,
            (
                "qkv_postprocess_payload",
                "qkv_split_kv_payload",
                "currentkv_postprocess_payload",
                "currentkv_postprocess_body_payload",
                "currentkv_postprocess_q4nx_body_payload",
                "attention_kv16_make_carrier_masked",
                "attention_kv16_finish_accum_bf16",
                "qkv_emit_qkv_body_records",
                "qkv_emit_qkv_records",
                "c1r2_main_accum_chunk",
                "func.call @attention_kv16_finish_accum(",
                "qkv_main_init_summary",
                "qkv_main_accum_chunk",
                "qkv_main_emit_o_record",
                "ffn_swiglu_slice_bf16_inputs_contract",
                "full_main_emit_upgate_slice_record",
                "full_main_init_down_accum",
                "full_main_emit_down_record",
                "aie.packet_flow(3)",
            ),
        )
    )
    errors.extend(validate_q4nx_down_full_layer_ownership(CASE_NAME, mlir))
    all_weight_bds = tuple(bd for pair in WEIGHT_PATCH_INPUT_BDS + WEIGHT_ROW_BDS for bd in pair)
    errors.extend(require_unique_bd_ids(f"{CASE_NAME} row1 weight stream", all_weight_bds))
    compact_bds = tuple(bd for row_bds in COLUMN_RECEIVE_BDS for bd in row_bds) + COLUMN_OUT_BDS
    errors.extend(require_disjoint_bd_ids(CASE_NAME, all_weight_bds, compact_bds))
    if f"buffer_length = {PATCH_WEIGHT_BF16 // 2} : i32" in mlir:
        errors.append("full-layer weight ingress must use bounded descriptor spans, not one monolithic patch BD")
    if CURRENT_PACKET_K != 8 or CURRENT_PACKET_V != 9:
        errors.append("current K/V packets must stay at 8/9 to avoid full-layer packet14/15 traffic")
    if DOWN_CHUNKS != 48 or TOTAL_MAIN_CHUNKS != 768:
        errors.append("full-layer replay/down chunk contract mismatch")
    if (
        FULL_LAYER_O_WEIGHT_CHUNK_BASE != QKV_BODY_WEIGHT_CHUNKS
        or FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE != FULL_LAYER_O_WEIGHT_CHUNK_BASE + O_WEIGHT_CHUNKS
        or FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE != FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE + UPGATE_WEIGHT_CHUNKS
        or FULL_LAYER_TOTAL_WEIGHT_CHUNKS != QKV_BODY_WEIGHT_CHUNKS + O_WEIGHT_CHUNKS + UPGATE_WEIGHT_CHUNKS + DOWN_WEIGHT_CHUNKS
    ):
        errors.append("full-layer Q4NX weight chunk schedule mismatch")
    if BODY_RECORD_SLOTS != (0, 1, 2, 3, -1, 6):
        errors.append("full-layer q4nx body record slots changed unexpectedly")
    if (
        HUB_Q_IN_CHANNEL != 1
        or HUB_Q_OUT_CHANNELS != (1, 2, 3, 4)
        or HUB_Q_OUT_BDS != (25, 2, 26, 3)
        or HUB_RETURN_IN_CHANNELS != (2, 3, 4, 5)
        or HUB_RETURN_IN_BDS != (4, 28, 6, 30)
        or HUB_DOWN_OUT_BDS != (27, 29, 31, 32, 33, 42, 43, 44)
    ):
        errors.append("hub BD contract mismatch")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
