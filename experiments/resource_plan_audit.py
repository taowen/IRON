"""Audit retained fused-layer experiments against the current resource plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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
        chunks_per_patch = PATCH_ROWS // Q4_ROWS
        return chunks_per_patch * self.k_chunks * Q4_CHUNK_BYTES


@dataclass(frozen=True)
class FusedLayerPlan:
    hidden_dim: int = 4096
    intermediate_dim: int = 12288
    q_dim: int = 4096
    kv_dim: int = 1024
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
    def main_tile_count(self) -> int:
        return len(MAIN_COLS) * len(MAIN_ROWS)

    @property
    def edge_tile_count(self) -> int:
        return 32 - self.main_tile_count

    @property
    def q4_local_working_set_bytes(self) -> int:
        weight_ping_pong = 2 * Q4_CHUNK_BYTES
        activation_chunk = Q4_K_CHUNK * 2
        accumulator = Q4_ROWS * 4
        output = Q4_ROWS * 2
        return weight_ping_pong + activation_chunk + accumulator + output

    @property
    def kv_plane_tile_dwords(self) -> int:
        bytes_per_tile = 4 * self.tokens_per_kv_tile * self.head_dim * 2
        return bytes_per_tile // 4

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
class ExperimentExpectation:
    directory: str
    files: tuple[str, ...]
    markers: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def read_text(path: Path) -> str:
    return path.read_text()


def retained_experiments() -> tuple[ExperimentExpectation, ...]:
    return (
        ExperimentExpectation(
            "25_mylm_edge_bd_ring",
            ("README.md", "generate.py"),
            ("L=128", "PASS", "4096-dword", "2048-dword"),
        ),
        ExperimentExpectation(
            "26_kv_edge_aux_reshape",
            ("README.md", "generate.py"),
            ("packet14/15", "SIDEBAND_DWORDS = 17", "HISTORY_DWORDS"),
        ),
        ExperimentExpectation(
            "38_full_attention_fabric",
            ("README.md", "generate.py"),
            ("32Q/8KV", "row1", "PASS"),
        ),
        ExperimentExpectation(
            "39_projected_current_write_full_attention",
            ("README.md", "generate.py"),
            ("current K/V writeback", "full attention fabric", "PASS"),
        ),
        ExperimentExpectation(
            "48_main16_fullk_q4nx_phase_replay",
            ("README.md", "generate.py"),
            ("main16", "full-K Q4NX", "PASS"),
        ),
        ExperimentExpectation(
            "52_fullk_edge_slice_replay_q4nx_o_phase",
            ("README.md", "generate.py"),
            ("full-K O", "edge-replayed", "K=4096"),
        ),
        ExperimentExpectation(
            "53_full_layer_phase_chain_contract",
            ("README.md", "generate.py"),
            ("Q,K,V,O,gate,up,down", "seven", "PASS"),
        ),
        ExperimentExpectation(
            "54_real_qwen_patch_schedule",
            ("README.md", "run_npu.py"),
            ("608", "real Qwen3", "patch schedule"),
        ),
        ExperimentExpectation(
            "55_mylm_linked_bd_chain",
            ("README.md", "generate.py"),
            ("linked", "BD", "pushes only `BD0`"),
        ),
        ExperimentExpectation(
            "56_mylm_linked_qwen_schedule",
            ("README.md", "generate.py"),
            ("608", "linked", "Qwen3"),
        ),
        ExperimentExpectation(
            "57_mylm_exact_patch_manifest",
            ("README.md", "schedule.py"),
            ("608", "0x28000", "0x78000"),
        ),
        ExperimentExpectation(
            "58_mylm_patch_pair_row1_split",
            ("README.md", "generate.py"),
            ("two 64-row", "row1", "PASS"),
        ),
        ExperimentExpectation(
            "59_mylm_exact_nblock_projection",
            ("README.md", "generate.py"),
            ("0x28000", "512-row", "PASS"),
        ),
        ExperimentExpectation(
            "60_mylm_chunk_ring_projection",
            ("README.md", "generate.py"),
            ("chunk-sized", "ERT_CMD_STATE_TIMEOUT", "Current status"),
        ),
    )


def check_plan(plan: FusedLayerPlan) -> tuple[CheckResult, ...]:
    hidden_phase = plan.phases[0]
    down_phase = plan.phases[-1]
    checks = (
        CheckResult(
            "main/edge fabric",
            "PASS" if plan.main_tile_count == 16 and plan.edge_tile_count == 16 else "GAP",
            f"main={plan.main_tile_count}, edge={plan.edge_tile_count}",
        ),
        CheckResult(
            "Q4 local working set",
            "PASS" if plan.q4_local_working_set_bytes < 16 * 1024 else "GAP",
            f"{plan.q4_local_working_set_bytes} bytes per main tile",
        ),
        CheckResult(
            "Qwen patch count",
            "PASS" if plan.total_weight_patches == 608 else "GAP",
            f"{plan.total_weight_patches} patches",
        ),
        CheckResult(
            "hidden patch size",
            "PASS" if hidden_phase.patch_bytes == 0x28000 else "GAP",
            f"0x{hidden_phase.patch_bytes:x}",
        ),
        CheckResult(
            "down patch size",
            "PASS" if down_phase.patch_bytes == 0x78000 else "GAP",
            f"0x{down_phase.patch_bytes:x}",
        ),
        CheckResult(
            "KV history tile",
            "PASS" if plan.kv_plane_tile_dwords == 4096 else "GAP",
            f"{plan.kv_plane_tile_dwords} dwords",
        ),
        CheckResult(
            "KV half tile",
            "PASS" if plan.kv_half_tile_dwords == 2048 else "GAP",
            f"{plan.kv_half_tile_dwords} dwords",
        ),
        CheckResult(
            "current packet",
            "PASS" if plan.current_packet_dwords == 512 else "GAP",
            f"{plan.current_packet_dwords} dwords",
        ),
    )
    return checks


def check_experiment(expectation: ExperimentExpectation) -> CheckResult:
    exp_dir = ROOT / "experiments" / expectation.directory
    missing_files = tuple(
        name for name in expectation.files if not (exp_dir / name).exists()
    )
    if missing_files:
        return CheckResult(
            expectation.directory,
            "GAP",
            f"missing files: {', '.join(missing_files)}",
        )

    text = "\n".join(read_text(exp_dir / name) for name in expectation.files)
    missing_markers = tuple(
        marker for marker in expectation.markers if marker not in text
    )
    if missing_markers:
        return CheckResult(
            expectation.directory,
            "PARTIAL",
            f"missing markers: {', '.join(missing_markers)}",
        )
    return CheckResult(expectation.directory, "PASS", "retained milestone present")


def print_phase_table(plan: FusedLayerPlan) -> None:
    print("Projection schedule")
    print("phase  input  output  blocks  k_chunks  patches  patch_bytes")
    for phase in plan.phases:
        print(
            f"{phase.name:<5} "
            f"{phase.input_dim:>5} "
            f"{phase.output_dim:>6} "
            f"{phase.output_blocks:>6} "
            f"{phase.k_chunks:>8} "
            f"{phase.patches:>7} "
            f"0x{phase.patch_bytes:x}"
        )


def print_checks(title: str, checks: tuple[CheckResult, ...]) -> None:
    print()
    print(title)
    for check in checks:
        print(f"{check.status:<7} {check.name}: {check.detail}")


def run_audit() -> int:
    plan = FusedLayerPlan()
    print_phase_table(plan)
    plan_checks = check_plan(plan)
    experiment_checks = tuple(check_experiment(item) for item in retained_experiments())
    print_checks("First-principles resource checks", plan_checks)
    print_checks("Retained experiment checks", experiment_checks)
    all_checks = plan_checks + experiment_checks
    return 0 if all(check.status == "PASS" for check in all_checks) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully when retained experiments are present but some markers changed.",
    )
    args = parser.parse_args()
    code = run_audit()
    if code == 1 and args.allow_partial:
        raise SystemExit(0)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
