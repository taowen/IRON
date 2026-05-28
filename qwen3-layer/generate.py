"""Generate the qwen3-dataflow MLIR-AIE physical skeleton.

This file intentionally does not contain the old exp67 projection backend.  It
emits the tile map, packet routes, and circuit-style flow skeleton required by
experiments/qwen3-dataflow.md so that every implementation check is tied to the
target physical dataflow rather than to a legacy main/edge replay scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract import (
    ATTENTION_PACKET_DWORDS,
    C1R2_PACKET_DWORDS,
    C1R2_QKV_REPLAYS,
    C1R2_UPGATE_REPLAYS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    DOWN_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    O_CHUNKS,
    PHASE_BLOCKS,
    PHASE_CHUNKS,
    PHASE_NAMES,
    RECORD_DWORDS,
    SHAPE_CARRIER_DWORDS,
    SHAPE_WINDOW_DWORDS,
    SWIGLU_SLICES,
    TOTAL_PATCHES,
    TOTAL_WEIGHT_I32,
)
from dataflow import build_dataflow


@dataclass(frozen=True)
class TileDef:
    symbol: str
    column: int
    row: int
    role: str


@dataclass(frozen=True)
class CircuitFlow:
    name: str
    source: str
    source_dma: int
    target: str
    target_dma: int
    payload: str


@dataclass(frozen=True)
class PacketRoute:
    name: str
    packet: int
    source: str
    source_dma: int
    target: str
    target_dma: int
    payload: str
    dwords: int


def main_symbol(column: int, row: int) -> str:
    return f"main_{column}_{row}"


def main_mem_symbol(column: int) -> str:
    return f"main_mem_{column}"


def tile_defs() -> tuple[TileDef, ...]:
    defs = [
        TileDef("shim_k", 0, 0, "current K writeback / KV scan ingress"),
        TileDef("shim_hidden", 1, 0, "hidden-in / hidden-out runtime boundary"),
        TileDef("shim_v", 7, 0, "current V writeback / KV scan ingress"),
        TileDef("kv_left", 0, 1, "left KV history splitter"),
        TileDef("activation_bridge", 1, 1, "c1r1 shared activation bridge"),
        TileDef("hub", 6, 1, "c6r1 Q fanout + attention/FFN gather"),
        TileDef("kv_right", 7, 1, "right KV history splitter"),
        TileDef("full_vector", 1, 2, "c1r2 RMSNorm/residual/full-vector station"),
        TileDef("postprocess", 1, 3, "c1r3 Q/K norm + RoPE + current KV"),
        TileDef("swiglu", 6, 2, "c6r2 up/gate SwiGLU slice station"),
        TileDef("shape_a_0", 0, 2, "shape-A heads 0..7 score/softmax"),
        TileDef("shape_b_0", 0, 3, "shape-B heads 0..7 weighted V"),
        TileDef("shape_a_1", 0, 4, "shape-A heads 8..15 score/softmax"),
        TileDef("shape_b_1", 0, 5, "shape-B heads 8..15 weighted V"),
        TileDef("shape_a_2", 7, 2, "shape-A heads 16..23 score/softmax"),
        TileDef("shape_b_2", 7, 3, "shape-B heads 16..23 weighted V"),
        TileDef("shape_a_3", 7, 4, "shape-A heads 24..31 score/softmax"),
        TileDef("shape_b_3", 7, 5, "shape-B heads 24..31 weighted V"),
    ]
    defs.extend(
        TileDef(main_mem_symbol(column), column, 1, "main16 row1 weight/compact memtile")
        for column in MAIN_COLUMNS
    )
    defs.extend(
        TileDef(main_symbol(column, row), column, row, "main16 Q4NX projection worker")
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    return tuple(defs)


def circuit_flows() -> tuple[CircuitFlow, ...]:
    flows: list[CircuitFlow] = [
        CircuitFlow(
            "hidden_in_to_c1r2",
            "shim_hidden",
            0,
            "full_vector",
            0,
            "2048-dword hidden full-vector input",
        ),
        CircuitFlow(
            "c1r1_to_c1r2_fullvector",
            "activation_bridge",
            0,
            "full_vector",
            1,
            "O/down compact-result full-vector ingress",
        ),
        CircuitFlow(
            "c1r3_q_to_c6r1",
            "postprocess",
            0,
            "hub",
            1,
            "2048-dword Q circuit output",
        ),
        CircuitFlow(
            "c6r2_swiglu_to_c6r1",
            "swiglu",
            0,
            "hub",
            0,
            "24 x 256-dword SwiGLU slices",
        ),
        CircuitFlow(
            "c1r2_hidden_out",
            "full_vector",
            2,
            "shim_hidden",
            1,
            "final 2048-dword hidden output",
        ),
    ]
    flows.extend(
        CircuitFlow(
            f"c1r1_activation_to_{main_symbol(column, row)}",
            "activation_bridge",
            1,
            main_symbol(column, row),
            0,
            "128-dword main16 activation ring",
        )
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    flows.extend(
        CircuitFlow(
            f"weight_{main_mem_symbol(column)}_to_{main_symbol(column, row)}",
            main_mem_symbol(column),
            row - 2,
            main_symbol(column, row),
            1,
            "1280-dword Q4NX weight chunk",
        )
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    flows.extend(
        CircuitFlow(
            f"{main_symbol(column, row)}_qkv_to_c1r3",
            main_symbol(column, row),
            1,
            "postprocess",
            0,
            "17-dword Q/K/V compact record",
        )
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    flows.extend(
        CircuitFlow(
            f"{main_symbol(column, row)}_o_down_to_c1r1",
            main_symbol(column, row),
            2,
            "activation_bridge",
            2,
            "17-dword O/down compact record",
        )
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    flows.extend(
        CircuitFlow(
            f"{main_symbol(column, row)}_upgate_to_{main_mem_symbol(column)}",
            main_symbol(column, row),
            3,
            main_mem_symbol(column),
            2,
            "17-dword up/gate compact record",
        )
        for column in MAIN_COLUMNS
        for row in MAIN_ROWS
    )
    flows.extend(
        CircuitFlow(
            f"{main_mem_symbol(column)}_compact_to_c1r1",
            main_mem_symbol(column),
            5,
            "activation_bridge",
            3,
            "65-dword column compact packet",
        )
        for column in MAIN_COLUMNS
    )
    flows.extend(
        CircuitFlow(
            f"q_window_{idx}",
            "hub",
            idx + 2,
            f"shape_a_{idx}",
            0,
            f"512-dword Q window {idx}",
        )
        for idx in range(4)
    )
    flows.extend(
        (
            CircuitFlow("kv_scan_left", "shim_k", 0, "kv_left", 0, "K/V groups 0..3 scan"),
            CircuitFlow("kv_scan_right", "shim_v", 0, "kv_right", 0, "K/V groups 4..7 scan"),
        )
    )
    flows.extend(
        (
            CircuitFlow("k_left_to_shape_a0", "kv_left", 0, "shape_a_0", 1, "K groups 0..1"),
            CircuitFlow("v_left_to_shape_b0", "kv_left", 1, "shape_b_0", 1, "V groups 0..1"),
            CircuitFlow("k_left_to_shape_a1", "kv_left", 2, "shape_a_1", 1, "K groups 2..3"),
            CircuitFlow("v_left_to_shape_b1", "kv_left", 3, "shape_b_1", 1, "V groups 2..3"),
            CircuitFlow("k_right_to_shape_a2", "kv_right", 0, "shape_a_2", 1, "K groups 4..5"),
            CircuitFlow("v_right_to_shape_b2", "kv_right", 1, "shape_b_2", 1, "V groups 4..5"),
            CircuitFlow("k_right_to_shape_a3", "kv_right", 2, "shape_a_3", 1, "K groups 6..7"),
            CircuitFlow("v_right_to_shape_b3", "kv_right", 3, "shape_b_3", 1, "V groups 6..7"),
        )
    )
    flows.extend(
        CircuitFlow(
            f"shape_a{idx}_carrier_to_shape_b{idx}",
            f"shape_a_{idx}",
            2,
            f"shape_b_{idx}",
            2,
            f"{SHAPE_CARRIER_DWORDS}-dword neighbor carrier",
        )
        for idx in range(4)
    )
    flows.extend(
        CircuitFlow(
            f"shape_b{idx}_return_to_c6r1",
            f"shape_b_{idx}",
            0,
            "hub",
            idx + 6,
            f"{SHAPE_WINDOW_DWORDS}-dword attention return window {idx}",
        )
        for idx in range(4)
    )
    return tuple(flows)


def packet_routes() -> tuple[PacketRoute, ...]:
    return (
        PacketRoute(
            "c1r2_fullvector_packet0",
            0,
            "full_vector",
            3,
            "activation_bridge",
            4,
            "+12/+48/+1 manual-header full-vector replay",
            C1R2_PACKET_DWORDS,
        ),
        PacketRoute(
            "c1r3_current_k_packet14",
            14,
            "postprocess",
            1,
            "shim_k",
            0,
            "current K writeback",
            512,
        ),
        PacketRoute(
            "c1r3_current_v_packet15",
            15,
            "postprocess",
            2,
            "shim_v",
            0,
            "current V writeback",
            512,
        ),
        PacketRoute(
            "c6r1_attention_packet2",
            2,
            "hub",
            11,
            "activation_bridge",
            4,
            "2048-dword attention return",
            ATTENTION_PACKET_DWORDS,
        ),
        PacketRoute(
            "c6r1_down_packet0",
            0,
            "hub",
            12,
            "activation_bridge",
            4,
            "6144-dword FFN intermediate return",
            DOWN_PACKET_DWORDS,
        ),
        PacketRoute(
            "c1r1_global_compact_to_c6r2",
            8,
            "activation_bridge",
            5,
            "swiglu",
            0,
            "257-dword up/gate global compact packet",
            COMPACT_PACKET_DWORDS,
        ),
    )


def _tile_lines() -> list[str]:
    return [
        f"    %{tile.symbol} = aie.tile({tile.column}, {tile.row}) // {tile.role}"
        for tile in tile_defs()
    ]


def _flow_lines() -> list[str]:
    lines: list[str] = []
    for flow in circuit_flows():
        lines.append(f"    // qwen3-flow {flow.name}: {flow.payload}")
        lines.append(
            f"    aie.flow(%{flow.source}, DMA : {flow.source_dma}, "
            f"%{flow.target}, DMA : {flow.target_dma})"
        )
    return lines


def _packet_lines() -> list[str]:
    lines: list[str] = []
    for route in packet_routes():
        lines.append(
            f"    // qwen3-packet {route.name}: packet{route.packet}, "
            f"{route.dwords} dwords, {route.payload}"
        )
        lines.append(f"    aie.packet_flow({route.packet}) {{")
        lines.append(f"      aie.packet_source<%{route.source}, DMA : {route.source_dma}>")
        lines.append(f"      aie.packet_dest<%{route.target}, DMA : {route.target_dma}>")
        lines.append("    }")
    return lines


def _contract_comment_lines() -> list[str]:
    graph = build_dataflow()
    lines = [
        f"    // qwen3-contract phases={','.join(PHASE_NAMES)}",
        f"    // qwen3-contract phase_blocks={PHASE_BLOCKS}",
        f"    // qwen3-contract phase_chunks={PHASE_CHUNKS}",
        f"    // qwen3-contract total_patches={TOTAL_PATCHES}",
        f"    // qwen3-contract total_weight_i32={TOTAL_WEIGHT_I32}",
        f"    // qwen3-contract c1r2_replays={C1R2_QKV_REPLAYS},{C1R2_UPGATE_REPLAYS},1",
        f"    // qwen3-contract o_chunks={O_CHUNKS}",
        f"    // qwen3-contract swiglu_slices={SWIGLU_SLICES}",
        f"    // qwen3-contract c6r2_input_dwords={C6R2_INPUT_DWORDS}",
    ]
    lines.extend(
        f"    // qwen3-edge {edge.name}: {edge.source}->{edge.target}, "
        f"dwords={edge.dwords}, packet={edge.packet}, count={edge.count}, {edge.payload}"
        for edge in graph.edges
    )
    return lines


def generate_mlir() -> str:
    return "\n".join(
        (
            "module {",
            "  aie.device(npu2) {",
            *_tile_lines(),
            "",
            *_contract_comment_lines(),
            "",
            *_flow_lines(),
            "",
            *_packet_lines(),
            "  }",
            "}",
            "",
        )
    )


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        "aie.tile(1, 1) // c1r1 shared activation bridge",
        "aie.tile(1, 2) // c1r2 RMSNorm/residual/full-vector station",
        "aie.tile(1, 3) // c1r3 Q/K norm + RoPE + current KV",
        "aie.tile(6, 1) // c6r1 Q fanout + attention/FFN gather",
        "aie.tile(6, 2) // c6r2 up/gate SwiGLU slice station",
        "qwen3-packet c6r1_attention_packet2: packet2",
        "qwen3-packet c6r1_down_packet0: packet0",
        "qwen3-packet c1r3_current_k_packet14: packet14",
        "qwen3-packet c1r3_current_v_packet15: packet15",
        "qwen3-flow c1r3_q_to_c6r1",
        "qwen3-flow c6r2_swiglu_to_c6r1",
        "qwen3-flow shape_a0_carrier_to_shape_b0",
        "qwen3-flow c1r1_activation_to_main_2_2",
        "qwen3-edge bridge_to_main_o",
        "qwen3-edge bridge_to_main_down",
        f"qwen3-contract total_patches={TOTAL_PATCHES}",
    )
    forbidden = (
        "edge_make_block_slice",
        "q4nx_chunk_accum_slice",
        "m0_0",
        "edge0_0",
        "PATCH_DESCRIPTORS_PER_COLUMN",
        "legacy exp67",
        "generate raw MLIR-AIE for exp67",
    )
    errors = [f"missing generated marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(f"forbidden legacy marker found: {marker}" for marker in forbidden if marker in mlir)
    if mlir.count("aie.packet_flow(2)") != 1:
        errors.append("attention packet2 route must appear exactly once")
    if mlir.count("aie.tile(") != len(tile_defs()):
        errors.append(f"tile count mismatch: {mlir.count('aie.tile(')} != {len(tile_defs())}")
    return errors


if __name__ == "__main__":
    print(generate_mlir())
