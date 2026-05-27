"""Audit fused-layer resource planning against experiments and MyLM notes.

This script turns the first-principles fused-layer plan into computed facts:
tile ownership, Q4NX projection patch counts, local working-set size, and edge
KV stream shapes.  It then compares those facts against the retained IRON
experiments and the MyLM reverse-engineering notes.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MYLM_ROOT = Path("/var/home/taowen/projects/MyLM")

Q4_ROWS = 32
Q4_K_CHUNK = 256
Q4_CHUNK_BYTES = 5120
OUTPUT_BLOCK_ROWS = 512
PATCH_ROWS = 64
MAIN_COLS = tuple(range(2, 6))
MAIN_ROWS = tuple(range(2, 6))


@dataclass(frozen=True)
class ProjectionPhase:
    name: str
    input_dim: int
    output_dim: int

    @property
    def output_blocks(self) -> int:
        return self.output_dim // OUTPUT_BLOCK_ROWS

    @property
    def patches(self) -> int:
        return self.output_blocks * 8

    @property
    def k_chunks(self) -> int:
        return self.input_dim // Q4_K_CHUNK

    @property
    def patch_bytes(self) -> int:
        return (PATCH_ROWS // Q4_ROWS) * self.k_chunks * Q4_CHUNK_BYTES


@dataclass(frozen=True)
class FusedLayerPlan:
    hidden_dim: int = 4096
    intermediate_dim: int = 12288
    q_dim: int = 4096
    kv_dim: int = 1024
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    tokens_per_kv_tile: int = 16

    @property
    def phases(self) -> tuple[ProjectionPhase, ...]:
        return (
            ProjectionPhase("Q", self.hidden_dim, self.q_dim),
            ProjectionPhase("K", self.hidden_dim, self.kv_dim),
            ProjectionPhase("V", self.hidden_dim, self.kv_dim),
            ProjectionPhase("O", self.hidden_dim, self.q_dim),
            ProjectionPhase("up", self.hidden_dim, self.intermediate_dim),
            ProjectionPhase("gate", self.hidden_dim, self.intermediate_dim),
            ProjectionPhase("down", self.intermediate_dim, self.hidden_dim),
        )

    @property
    def main_tiles(self) -> tuple[str, ...]:
        return tuple(f"c{col}r{row}" for col in MAIN_COLS for row in MAIN_ROWS)

    @property
    def edge_tiles(self) -> tuple[str, ...]:
        return tuple(
            f"c{col}r{row}" for col in (0, 1, 6, 7) for row in MAIN_ROWS
        )

    @property
    def q4_local_working_set_bytes(self) -> int:
        weight_ping_pong = 2 * Q4_CHUNK_BYTES
        activation_chunk = Q4_K_CHUNK * 2
        accumulator = Q4_ROWS * 4
        output = Q4_ROWS * 2
        return weight_ping_pong + activation_chunk + accumulator + output

    @property
    def kv_plane_tile_dwords(self) -> int:
        return 4 * self.tokens_per_kv_tile * self.head_dim * 2 // 4

    @property
    def kv_half_tile_dwords(self) -> int:
        return self.kv_plane_tile_dwords // 2

    @property
    def current_packet_dwords(self) -> int:
        return 4 * self.head_dim

    @property
    def total_weight_patches(self) -> int:
        return sum(phase.patches for phase in self.phases)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def source_contains(path: Path, pattern: str) -> bool:
    return re.search(pattern, read_text(path), re.MULTILINE) is not None


def literal_constant(path: Path, name: str) -> int | None:
    text = read_text(path)
    match = re.search(rf"^{re.escape(name)}\s*=\s*(0x[0-9a-fA-F]+|\d+)\s*$", text, re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1), 0)


def status(pass_condition: bool, partial_condition: bool = False) -> str:
    if pass_condition:
        return "PASS"
    if partial_condition:
        return "PARTIAL"
    return "GAP"


def experiment_checks(plan: FusedLayerPlan) -> tuple[CheckResult, ...]:
    exp23_ref = ROOT / "experiments/23_qkv_current_attention/reference.py"
    exp25_gen = ROOT / "experiments/25_mylm_edge_bd_ring/generate.py"
    exp26_gen = ROOT / "experiments/26_kv_edge_aux_reshape/generate.py"
    exp27_gen = ROOT / "experiments/27_shape_ab_attention_contract/generate.py"
    exp28_gen = ROOT / "experiments/28_multitile_shape_ab_attention/generate.py"

    exp23_hidden = literal_constant(exp23_ref, "HIDDEN_DIM")
    exp25_plane = literal_constant(exp25_gen, "PLANE_TILE_DWORDS")
    exp26_current = literal_constant(exp26_gen, "CURRENT_DWORDS")
    exp26_history_source = literal_constant(exp26_gen, "HISTORY_SOURCE_DWORDS")
    exp26_history = literal_constant(exp26_gen, "HISTORY_DWORDS")
    exp26_sideband = literal_constant(exp26_gen, "SIDEBAND_DWORDS")
    exp27_current = literal_constant(exp27_gen, "CURRENT_DWORDS")
    exp27_history_source = literal_constant(exp27_gen, "HISTORY_SOURCE_DWORDS")
    exp27_history = literal_constant(exp27_gen, "HISTORY_DWORDS")
    exp27_sideband = literal_constant(exp27_gen, "SIDEBAND_DWORDS")
    exp27_output = literal_constant(exp27_gen, "ATTENTION_OUT_DWORDS")
    exp28_current = literal_constant(exp28_gen, "CURRENT_DWORDS")
    exp28_history_source = literal_constant(exp28_gen, "HISTORY_SOURCE_DWORDS")
    exp28_history = literal_constant(exp28_gen, "HISTORY_DWORDS")
    exp28_sideband = literal_constant(exp28_gen, "SIDEBAND_DWORDS")
    exp28_output = literal_constant(exp28_gen, "ATTENTION_OUT_DWORDS")

    exp25_static_ring = (
        exp25_plane == plan.kv_plane_tile_dwords
        and source_contains(exp25_gen, r"HALF_TILE_DWORDS\s*=\s*PLANE_TILE_DWORDS\s*//\s*2")
        and source_contains(exp25_gen, r"bd_id = 0 : i32, next_bd_id = 1")
        and source_contains(exp25_gen, r"bd_id = 24 : i32, next_bd_id = 25")
    )
    exp26_selector = (
        exp26_current == plan.current_packet_dwords
        and exp26_history_source == plan.kv_plane_tile_dwords
        and exp26_history == plan.kv_half_tile_dwords
        and exp26_sideband == 17
        and source_contains(exp26_gen, r'"left14": Variant\(name="left14", packet_id=14, shape_col=0\)')
        and source_contains(exp26_gen, r'"right15": Variant\(name="right15", packet_id=15, shape_col=7\)')
        and source_contains(exp26_gen, r"aie.flow\(%side_sink, DMA : 1, %shape_b, DMA : 0\)")
    )
    exp27_attention = (
        exp27_current == plan.current_packet_dwords
        and exp27_history_source == plan.kv_plane_tile_dwords
        and exp27_history == plan.kv_half_tile_dwords
        and exp27_sideband == 17
        and exp27_output == plan.current_packet_dwords
        and source_contains(exp27_gen, r"aie\.packet_flow\(14\)")
        and source_contains(exp27_gen, r"aie.flow\(%mem0, DMA : 0, %shape_a, DMA : 1\)")
        and source_contains(exp27_gen, r"aie.flow\(%mem7, DMA : 0, %shape_b, DMA : 1\)")
        and source_contains(exp27_gen, r"func.call @shape_a_softmax_sideband")
        and source_contains(exp27_gen, r"func.call @shape_b_weighted_value")
    )
    exp28_attention = (
        exp28_current == plan.current_packet_dwords
        and exp28_history_source == plan.kv_plane_tile_dwords
        and exp28_history == plan.kv_half_tile_dwords
        and exp28_sideband == 17
        and exp28_output == plan.current_packet_dwords
        and source_contains(exp28_gen, r"aie\.packet_flow\(14\)")
        and source_contains(exp28_gen, r"aie.flow\(%mem0, DMA : 0, %shape_a, DMA : 1\)")
        and source_contains(exp28_gen, r"aie.flow\(%mem7, DMA : 0, %shape_b, DMA : 1\)")
        and source_contains(exp28_gen, r"func.call @shape_a_tile_sideband")
        and source_contains(exp28_gen, r"func.call @shape_b_accumulate_tile")
        and source_contains(exp28_gen, r"history_total = num_tiles \* HISTORY_SOURCE_DWORDS")
    )

    return (
        CheckResult(
            "exp23 Q/K/V current attention",
            status(exp23_hidden == plan.hidden_dim, exp23_hidden == 512),
            f"toy hidden_dim={exp23_hidden}; full plan hidden_dim={plan.hidden_dim}",
        ),
        CheckResult(
            "exp25 row1 static KV ring",
            status(exp25_static_ring),
            f"plane={exp25_plane}, expected={plan.kv_plane_tile_dwords}; half={plan.kv_half_tile_dwords}",
        ),
        CheckResult(
            "exp26 selector/history/sideband",
            status(exp26_selector),
            (
                f"current={exp26_current}, history_source={exp26_history_source}, "
                f"history={exp26_history}, sideband={exp26_sideband}"
            ),
        ),
        CheckResult(
            "exp27 shape-A/B attention contract",
            status(exp27_attention),
            (
                f"current={exp27_current}, history_source={exp27_history_source}, "
                f"history={exp27_history}, sideband={exp27_sideband}, output={exp27_output}"
            ),
        ),
        CheckResult(
            "exp28 multi-tile online attention",
            status(exp28_attention),
            (
                f"current={exp28_current}, history_source={exp28_history_source}, "
                f"history={exp28_history}, sideband={exp28_sideband}, output={exp28_output}"
            ),
        ),
    )


def mylm_doc_checks(plan: FusedLayerPlan, mylm_root: Path) -> tuple[CheckResult, ...]:
    q4nx = read_text(mylm_root / "tools/re/Q4NX_LAYOUT.md")
    current = read_text(mylm_root / "tools/re/fused-layer-engine/current-understanding.md")

    q4nx_ok = (
        "block_size = 5120" in q4nx
        and "32 x 256" in q4nx
        and f"Total: {plan.total_weight_patches} patches" in q4nx
    )
    main16_ok = (
        "The main projection fabric is spatially parallel" in current
        and "c2r2/c3r2/c4r2/c5r2" in current
        and "input  ch1 bd2 len=1280" in current
        and "output ch2 bd4 len=17" in current
    )
    edge_ok = (
        "edge shape A" in current
        and "input ch0 bd0 base=0x78000 len=512" in current
        and "input ch1 bd2 base=0x70400 len=2048" in current
        and "edge shape B" in current
        and "output ch2 bd2 base=0x78000 len=512" in current
    )
    packet_ok = (
        "packet14" in current
        and "packet15" in current
        and "Two 256-dword BDs per packet id equal 512 dwords" in current
    )
    sideband_ok = (
        "Route8 has no packet-enabled BD source" in current
        and "len=17" in current
        and "compact sideband" in current
    )

    notes_status = "PASS" if q4nx and current else "MISSING"
    return (
        CheckResult(
            "MyLM notes available",
            notes_status,
            f"root={mylm_root}",
        ),
        CheckResult(
            "MyLM Q4NX patch plan",
            status(q4nx_ok),
            f"expected total patches={plan.total_weight_patches}, chunk={Q4_CHUNK_BYTES}B",
        ),
        CheckResult(
            "MyLM main16 projection shape",
            status(main16_ok),
            "expects 16 reusable main tiles with 1280-dword Q4 input and 17-dword sideband",
        ),
        CheckResult(
            "MyLM edge shape A/B",
            status(edge_ok),
            "shape A: current+history, shape B: history->512-dword output",
        ),
        CheckResult(
            "MyLM packet current",
            status(packet_ok),
            "packet14/15 are two 256-dword BDs each, total 512 dwords per side",
        ),
        CheckResult(
            "MyLM route8 sideband",
            status(sideband_ok),
            "17-dword non-packet compact stream, not full tensor route",
        ),
    )


def parse_field(fields: str, name: str) -> int | None:
    match = re.search(rf"\b{re.escape(name)}=(-?0x[0-9a-fA-F]+|-?\d+)\b", fields)
    if match is None:
        return None
    return int(match.group(1), 0)


def bd_csv_checks(plan: FusedLayerPlan, csv_path: Path) -> tuple[CheckResult, ...]:
    if not csv_path.exists():
        return (CheckResult("BD CSV", "MISSING", f"{csv_path}"),)

    rows: list[tuple[str, int, int, int]] = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            fields = row["fields"]
            length = parse_field(fields, "len")
            packet_en = parse_field(fields, "packet_en")
            packet_id = parse_field(fields, "packet_id")
            if length is None or packet_en is None or packet_id is None:
                continue
            rows.append((row["tile"], length, packet_en, packet_id))

    has_ring_load = any(tile in {"c0r1", "c7r1"} and length == plan.kv_plane_tile_dwords for tile, length, _, _ in rows)
    has_ring_half = any(tile in {"c0r1", "c7r1"} and length == plan.kv_half_tile_dwords for tile, length, _, _ in rows)
    has_packet14 = any(packet_en == 1 and packet_id == 14 for _, _, packet_en, packet_id in rows)
    has_packet15 = any(packet_en == 1 and packet_id == 15 for _, _, packet_en, packet_id in rows)
    has_sideband17 = any(length == 17 and packet_en == 0 for _, length, packet_en, _ in rows)
    has_shape_a = any(tile in {"c0r2", "c7r2"} and length == plan.current_packet_dwords for tile, length, _, _ in rows) and any(
        tile in {"c0r2", "c7r2"} and length == plan.kv_half_tile_dwords for tile, length, _, _ in rows
    )

    return (
        CheckResult("BD CSV row1 ring load", status(has_ring_load), f"needs {plan.kv_plane_tile_dwords}-dword c0r1/c7r1 rows"),
        CheckResult("BD CSV row1 half stream", status(has_ring_half), f"needs {plan.kv_half_tile_dwords}-dword c0r1/c7r1 rows"),
        CheckResult("BD CSV packet14/15", status(has_packet14 and has_packet15, has_packet14 or has_packet15), "packet-enabled current descriptors"),
        CheckResult("BD CSV sideband17", status(has_sideband17), "non-packet 17-dword descriptor"),
        CheckResult("BD CSV shape-A ports", status(has_shape_a), "shape-A current 512 + history 2048"),
    )


def print_phase_plan(plan: FusedLayerPlan) -> None:
    print("First-principles Qwen3 decode layer plan")
    print("=========================================")
    print(f"main tiles: {len(plan.main_tiles)} {', '.join(plan.main_tiles)}")
    print(f"edge/aux candidate tiles: {len(plan.edge_tiles)} {', '.join(plan.edge_tiles)}")
    print(f"Q4NX chunk: {Q4_ROWS}x{Q4_K_CHUNK}, {Q4_CHUNK_BYTES} bytes")
    print(f"per-main-tile Q4 working set: {plan.q4_local_working_set_bytes} bytes")
    print(f"KV plane tile: {plan.kv_plane_tile_dwords} dwords, half stream: {plan.kv_half_tile_dwords} dwords")
    print(f"current packet side: {plan.current_packet_dwords} dwords")
    print()
    print("projection phases:")
    print("  phase  input  output  blocks  patches  patch_bytes")
    for phase in plan.phases:
        print(
            f"  {phase.name:<5} {phase.input_dim:>5} {phase.output_dim:>7} "
            f"{phase.output_blocks:>7} {phase.patches:>8} 0x{phase.patch_bytes:x}"
        )
    print(f"  total patches: {plan.total_weight_patches}")
    print()


def print_checks(title: str, checks: tuple[CheckResult, ...]) -> None:
    print(title)
    print("=" * len(title))
    for check in checks:
        print(f"{check.status:<8} {check.name}: {check.detail}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mylm-root", type=Path, default=DEFAULT_MYLM_ROOT)
    parser.add_argument("--bd-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = FusedLayerPlan()
    print_phase_plan(plan)
    print_checks("Experiment comparison", experiment_checks(plan))
    print_checks("MyLM note comparison", mylm_doc_checks(plan, args.mylm_root))
    if args.bd_csv is not None:
        print_checks("Decoded BD CSV comparison", bd_csv_checks(plan, args.bd_csv))


if __name__ == "__main__":
    main()
