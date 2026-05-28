"""Static qwen3-dataflow implementation graph for the fused decode layer."""

from __future__ import annotations

from dataclasses import dataclass

from contract import (
    C1R2_PACKET_DWORDS,
    C1R2_QKV_REPLAYS,
    C1R2_UPGATE_REPLAYS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    HEAD_DIM,
    NUM_Q_HEADS,
    O_CHUNKS,
    SWIGLU_SLICES,
)


@dataclass(frozen=True)
class Node:
    name: str
    tile: str
    role: str


@dataclass(frozen=True)
class Edge:
    name: str
    source: str
    target: str
    payload: str
    dwords: int
    packet: int | None
    count: int


@dataclass(frozen=True)
class LayerDataflow:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def _main16_nodes() -> tuple[Node, ...]:
    return tuple(
        Node(f"main_{column}_{row}", f"c{column}r{row}", "main16_projection")
        for column in range(2, 6)
        for row in range(2, 6)
    )


def _shape_nodes() -> tuple[Node, ...]:
    return (
        Node("shape_a_0", "c0r2", "shape_a_score_softmax"),
        Node("shape_b_0", "c0r3", "shape_b_weighted_v"),
        Node("shape_a_1", "c0r4", "shape_a_score_softmax"),
        Node("shape_b_1", "c0r5", "shape_b_weighted_v"),
        Node("shape_a_2", "c7r2", "shape_a_score_softmax"),
        Node("shape_b_2", "c7r3", "shape_b_weighted_v"),
        Node("shape_a_3", "c7r4", "shape_a_score_softmax"),
        Node("shape_b_3", "c7r5", "shape_b_weighted_v"),
    )


def build_dataflow() -> LayerDataflow:
    nodes = (
        Node("host", "host", "runtime_boundary"),
        Node("shim_k", "c0r0", "current_k_writeback_and_kv_scan"),
        Node("shim_v", "c7r0", "current_v_writeback_and_kv_scan"),
        Node("kv_left", "c0r1", "kv_history_split"),
        Node("kv_right", "c7r1", "kv_history_split"),
        Node("activation_bridge", "c1r1", "packet_to_main16_activation_bridge"),
        Node("full_vector", "c1r2", "rmsnorm_residual_final_output"),
        Node("postprocess", "c1r3", "qk_norm_rope_current_kv"),
        Node("hub", "c6r1", "q_fanout_attention_and_ffn_gather"),
        Node("swiglu", "c6r2", "up_gate_swiglu"),
        Node("row1_column_compact", "c2r1..c5r1", "per_column_compact_65d"),
        Node("row1_global_compact", "c1r1", "global_compact_257d"),
        *_main16_nodes(),
        *_shape_nodes(),
    )

    edges = (
        Edge("hidden_in", "host", "full_vector", "hidden full-vector", 2048, None, 1),
        Edge(
            "c1r2_qkv_replay",
            "full_vector",
            "activation_bridge",
            "packet0 full-vector replay for Q/K/V",
            C1R2_PACKET_DWORDS,
            0,
            C1R2_QKV_REPLAYS,
        ),
        Edge(
            "bridge_to_main_qkv",
            "activation_bridge",
            "main16",
            "256-bf16 activation chunks from 12 full-vector replays",
            128,
            None,
            C1R2_QKV_REPLAYS * 16,
        ),
        Edge("main_qkv_records", "main16", "postprocess", "Q/K/V compact records", 17, None, 12),
        Edge("current_k_writeback", "postprocess", "shim_k", "current K", 512, 14, 1),
        Edge("current_v_writeback", "postprocess", "shim_v", "current V", 512, 15, 1),
        Edge("q_to_hub", "postprocess", "hub", "Q[32][128]", 2048, None, 1),
        Edge("q_window_0", "hub", "shape_a_0", "Q heads 0..7", 512, None, 1),
        Edge("q_window_1", "hub", "shape_a_1", "Q heads 8..15", 512, None, 1),
        Edge("q_window_2", "hub", "shape_a_2", "Q heads 16..23", 512, None, 1),
        Edge("q_window_3", "hub", "shape_a_3", "Q heads 24..31", 512, None, 1),
        Edge("kv_scan_left", "shim_k", "kv_left", "K/V groups 0..3", 2048, None, 2),
        Edge("kv_scan_right", "shim_v", "kv_right", "K/V groups 4..7", 2048, None, 2),
        Edge("k_left_to_shape_a0", "kv_left", "shape_a_0", "K history groups 0..1", 2048, None, 1),
        Edge("v_left_to_shape_b0", "kv_left", "shape_b_0", "V history groups 0..1", 2048, None, 1),
        Edge("k_left_to_shape_a1", "kv_left", "shape_a_1", "K history groups 2..3", 2048, None, 1),
        Edge("v_left_to_shape_b1", "kv_left", "shape_b_1", "V history groups 2..3", 2048, None, 1),
        Edge("k_right_to_shape_a2", "kv_right", "shape_a_2", "K history groups 4..5", 2048, None, 1),
        Edge("v_right_to_shape_b2", "kv_right", "shape_b_2", "V history groups 4..5", 2048, None, 1),
        Edge("k_right_to_shape_a3", "kv_right", "shape_a_3", "K history groups 6..7", 2048, None, 1),
        Edge("v_right_to_shape_b3", "kv_right", "shape_b_3", "V history groups 6..7", 2048, None, 1),
        Edge("carrier_0", "shape_a_0", "shape_b_0", "base[0x100]+scalar[0x40]", 80, None, 1),
        Edge("carrier_1", "shape_a_1", "shape_b_1", "base[0x100]+scalar[0x40]", 80, None, 1),
        Edge("carrier_2", "shape_a_2", "shape_b_2", "base[0x100]+scalar[0x40]", 80, None, 1),
        Edge("carrier_3", "shape_a_3", "shape_b_3", "base[0x100]+scalar[0x40]", 80, None, 1),
        Edge("shape_b0_return", "shape_b_0", "hub", "attention heads 0..7", 512, None, 1),
        Edge("shape_b1_return", "shape_b_1", "hub", "attention heads 8..15", 512, None, 1),
        Edge("shape_b2_return", "shape_b_2", "hub", "attention heads 16..23", 512, None, 1),
        Edge("shape_b3_return", "shape_b_3", "hub", "attention heads 24..31", 512, None, 1),
        Edge("attention_packet2", "hub", "activation_bridge", "Attn[32][128]", 2048, 2, 1),
        Edge("bridge_to_main_o", "activation_bridge", "main16", "O activation chunks", 128, None, O_CHUNKS),
        Edge("main_o_records", "main16", "full_vector", "O compact records", 17, None, 8),
        Edge(
            "c1r2_upgate_replay",
            "full_vector",
            "activation_bridge",
            "packet0 full-vector replay for up/gate",
            C1R2_PACKET_DWORDS,
            0,
            C1R2_UPGATE_REPLAYS,
        ),
        Edge(
            "bridge_to_main_upgate",
            "activation_bridge",
            "main16",
            "up/gate activations from 48 full-vector replays",
            128,
            None,
            C1R2_UPGATE_REPLAYS * 16,
        ),
        Edge("main_upgate_records", "main16", "row1_column_compact", "up/gate 17-dword records", 17, None, 48),
        Edge("column_compact", "row1_column_compact", "row1_global_compact", "65-dword column compact", 65, None, 4),
        Edge("global_compact", "row1_global_compact", "swiglu", "257-dword compact packet", COMPACT_PACKET_DWORDS, None, SWIGLU_SLICES * 2),
        Edge("swiglu_to_hub", "swiglu", "hub", "SwiGLU slices", 256, None, SWIGLU_SLICES),
        Edge("down_packet0", "hub", "activation_bridge", "FFN intermediate", 6144, 0, 1),
        Edge("bridge_to_main_down", "activation_bridge", "main16", "down activation chunks", 128, None, 48),
        Edge("main_down_records", "main16", "full_vector", "down compact records", 17, None, 8),
        Edge("hidden_out", "full_vector", "host", "final hidden output", 2048, None, 1),
    )
    return LayerDataflow(nodes=nodes, edges=edges)


def _node_names(graph: LayerDataflow) -> set[str]:
    return {node.name for node in graph.nodes}


def _edge_by_name(graph: LayerDataflow) -> dict[str, Edge]:
    return {edge.name: edge for edge in graph.edges}


def validate_dataflow(graph: LayerDataflow | None = None) -> list[str]:
    current = build_dataflow() if graph is None else graph
    errors: list[str] = []
    names = _node_names(current)
    edges = _edge_by_name(current)

    required_nodes = (
        "postprocess",
        "hub",
        "activation_bridge",
        "full_vector",
        "swiglu",
        "row1_column_compact",
        "row1_global_compact",
        "shape_a_0",
        "shape_b_3",
        "shim_k",
        "shim_v",
    )
    for node in required_nodes:
        if node not in names:
            errors.append(f"missing node {node}")

    for edge in current.edges:
        if edge.source not in names and edge.source != "main16":
            errors.append(f"{edge.name}: missing source {edge.source}")
        if edge.target not in names and edge.target != "main16":
            errors.append(f"{edge.name}: missing target {edge.target}")

    expected_packets = {
        "current_k_writeback": 14,
        "current_v_writeback": 15,
        "attention_packet2": 2,
        "down_packet0": 0,
        "c1r2_qkv_replay": 0,
        "c1r2_upgate_replay": 0,
    }
    for name, packet in expected_packets.items():
        edge = edges.get(name)
        if edge is None:
            errors.append(f"missing edge {name}")
        elif edge.packet != packet:
            errors.append(f"{name}: packet {edge.packet}, expected {packet}")

    expected_sizes = {
        "attention_packet2": 2048,
        "down_packet0": 6144,
        "c1r2_qkv_replay": 2049,
        "c1r2_upgate_replay": 2049,
        "global_compact": 257,
        "swiglu_to_hub": 256,
    }
    for name, dwords in expected_sizes.items():
        edge = edges.get(name)
        if edge is None:
            errors.append(f"missing edge {name}")
        elif edge.dwords != dwords:
            errors.append(f"{name}: {edge.dwords} dwords, expected {dwords}")

    if edges["bridge_to_main_o"].count != 16:
        errors.append("O bridge must produce 16 chunks")
    if edges["bridge_to_main_qkv"].count != C1R2_QKV_REPLAYS * 16:
        errors.append("Q/K/V bridge chunk count must be +12 full-vector replays * 16")
    if edges["bridge_to_main_upgate"].count != C1R2_UPGATE_REPLAYS * 16:
        errors.append("up/gate bridge chunk count must be +48 full-vector replays * 16")
    if edges["bridge_to_main_down"].count != 48:
        errors.append("down bridge must produce 48 chunks")
    if edges["global_compact"].count != SWIGLU_SLICES * 2:
        errors.append("up/gate global compact must produce adjacent packet pairs")

    legacy_direct = [
        edge.name
        for edge in current.edges
        if edge.source == "main16" and edge.target.startswith("edge")
    ]
    if legacy_direct:
        errors.append(f"legacy direct main-to-edge edges present: {legacy_direct}")

    if C6R2_INPUT_DWORDS != 512:
        errors.append(f"c6r2 input dwords mismatch: {C6R2_INPUT_DWORDS}")
    if NUM_Q_HEADS * HEAD_DIM != 4096:
        errors.append("attention vector size mismatch")
    return errors


def dataflow_lines(graph: LayerDataflow | None = None) -> list[str]:
    current = build_dataflow() if graph is None else graph
    lines = ["qwen3-dataflow graph:"]
    lines.extend(
        f"  node {node.name}: {node.tile} {node.role}" for node in current.nodes
    )
    lines.append("  edges:")
    lines.extend(
        "    "
        f"{edge.name}: {edge.source} -> {edge.target}, "
        f"{edge.payload}, dwords={edge.dwords}, packet={edge.packet}, count={edge.count}"
        for edge in current.edges
    )
    return lines
