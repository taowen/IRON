#!/usr/bin/env python3
"""Gate modified Q4NX body candidates against the exp120 section contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONTRACT_JSON = REPO_ROOT / "experiments/120_mylm_q4nx_generator_contract/mylm_q4nx_generator_contract.json"
DEFAULT_REPORT = EXPERIMENT_DIR / "q4nx_modified_body_numeric_gate.md"
DEFAULT_JSON = EXPERIMENT_DIR / "q4nx_modified_body_numeric_gate.json"

ROWS = 32
GROUPS = 8
GROUP_SIZE = 32
SEED = 121
SAMPLES = 256
SCENARIOS = (
    ("nominal", 0.06, 0.15, 0.7),
    ("stress", 0.30, 64.0, 32.0),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    static_macs: int
    dynamic_macs: int
    preserves_exp120_mac_shape: bool
    exact_reference_contract: bool
    note: str


@dataclass(frozen=True)
class DiffStats:
    scenario: str
    name: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    mismatches_1e_3: int
    mismatches_1e_2: int
    mismatches_5e_2: int


@dataclass(frozen=True)
class ContractSummary:
    sections: int
    original_slots: int
    total_group_macs: int
    exact_instruction_match: bool
    stable_boundary_pressure: bool


def bf16_round(values: np.ndarray) -> np.ndarray:
    return values.astype(bfloat16).astype(np.float32)


def load_contract(path: Path) -> ContractSummary:
    data = json.loads(path.read_text())
    checks = data["checks"]
    return ContractSummary(
        sections=len(data["sections"]),
        original_slots=int(checks["original_slots"]),
        total_group_macs=int(checks["total_group_macs"]),
        exact_instruction_match=bool(checks["exact_instruction_match"]),
        stable_boundary_pressure=bool(checks["stable_boundary_pressure"]),
    )


def deterministic_inputs(
    samples: int,
    scale_std: float,
    zero_std: float,
    activation_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    q = rng.integers(0, 16, size=(samples, ROWS, GROUPS, GROUP_SIZE), dtype=np.uint8).astype(np.float32)
    scale = bf16_round(rng.normal(loc=0.0, scale=scale_std, size=(samples, ROWS, GROUPS)).astype(np.float32))
    zero = bf16_round(rng.normal(loc=0.0, scale=zero_std, size=(samples, ROWS, GROUPS)).astype(np.float32))
    activation = bf16_round(
        rng.normal(loc=0.0, scale=activation_std, size=(samples, GROUPS, GROUP_SIZE)).astype(np.float32)
    )
    return q, scale, zero, activation


def exact_reference(q: np.ndarray, scale: np.ndarray, zero: np.ndarray, activation: np.ndarray) -> np.ndarray:
    scaled = bf16_round(q * scale[..., None])
    coeff = bf16_round(scaled + zero[..., None])
    return np.sum(coeff * activation[:, None, :, :], axis=(2, 3), dtype=np.float32)


def exact_recompose_coeff(q: np.ndarray, scale: np.ndarray, zero: np.ndarray, activation: np.ndarray) -> np.ndarray:
    scaled = bf16_round(q * scale[..., None])
    coeff = bf16_round(scaled + zero[..., None])
    centered = coeff - zero[..., None]
    recomposed = centered + zero[..., None]
    return np.sum(recomposed * activation[:, None, :, :], axis=(2, 3), dtype=np.float32)


def mylm_group_correction(q: np.ndarray, scale: np.ndarray, zero: np.ndarray, activation: np.ndarray) -> np.ndarray:
    scaled = bf16_round(q * scale[..., None])
    coeff = bf16_round(scaled + zero[..., None])
    centered = coeff - zero[..., None]
    centered_sum = np.sum(centered * activation[:, None, :, :], axis=(2, 3), dtype=np.float32)
    group_sum = np.sum(activation, axis=2, dtype=np.float32)
    zero_sum = np.sum(zero * group_sum[:, None, :], axis=2, dtype=np.float32)
    return centered_sum + zero_sum


def split_zero_per_dim(q: np.ndarray, scale: np.ndarray, zero: np.ndarray, activation: np.ndarray) -> np.ndarray:
    scaled = bf16_round(q * scale[..., None])
    coeff = bf16_round(scaled + zero[..., None])
    centered = coeff - zero[..., None]
    centered_sum = np.sum(centered * activation[:, None, :, :], axis=(2, 3), dtype=np.float32)
    zero_sum = np.sum(zero[..., None] * activation[:, None, :, :], axis=(2, 3), dtype=np.float32)
    return centered_sum + zero_sum


def diff_stats(scenario: str, name: str, got: np.ndarray, expected: np.ndarray) -> DiffStats:
    diff = np.abs(got.astype(np.float32) - expected.astype(np.float32)).reshape(-1)
    return DiffStats(
        scenario=scenario,
        name=name,
        max_abs=float(np.max(diff)),
        mean_abs=float(np.mean(diff)),
        p99_abs=float(np.percentile(diff, 99.0)),
        mismatches_1e_3=int(np.count_nonzero(diff > 1.0e-3)),
        mismatches_1e_2=int(np.count_nonzero(diff > 1.0e-2)),
        mismatches_5e_2=int(np.count_nonzero(diff > 5.0e-2)),
    )


def candidates(contract: ContractSummary) -> tuple[Candidate, ...]:
    return (
        Candidate(
            name="mylm_group_correction",
            static_macs=contract.total_group_macs,
            dynamic_macs=contract.total_group_macs * 2,
            preserves_exp120_mac_shape=True,
            exact_reference_contract=False,
            note="Preserves the MyLM 32+1 MAC/group shape; requires token-quality acceptance.",
        ),
        Candidate(
            name="exact_recompose_coeff",
            static_macs=GROUPS * GROUP_SIZE,
            dynamic_macs=GROUPS * GROUP_SIZE * 2,
            preserves_exp120_mac_shape=False,
            exact_reference_contract=True,
            note="Exact parity route; zero must be recomposed into the coefficient before MAC.",
        ),
        Candidate(
            name="split_zero_per_dim",
            static_macs=GROUPS * GROUP_SIZE * 2,
            dynamic_macs=GROUPS * GROUP_SIZE * 2 * 2,
            preserves_exp120_mac_shape=False,
            exact_reference_contract=False,
            note="Diagnostic only: per-dim split zero changes fp32 accumulation order.",
        ),
    )


def render_diff_table(stats: tuple[DiffStats, ...]) -> list[str]:
    lines = [
        "| Scenario | Candidate | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in stats:
        lines.append(
            f"| `{item.scenario}` | `{item.name}` | {item.max_abs:.9f} | {item.p99_abs:.9f} | "
            f"{item.mean_abs:.9f} | {item.mismatches_1e_3} | {item.mismatches_1e_2} | {item.mismatches_5e_2} |"
        )
    return lines


def render_candidate_table(items: tuple[Candidate, ...]) -> list[str]:
    lines = [
        "| Candidate | Static MACs | Dynamic MACs | Keeps exp120 Shape | Exact Contract | Note |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in items:
        keeps_shape = "yes" if item.preserves_exp120_mac_shape else "no"
        exact = "yes" if item.exact_reference_contract else "no"
        lines.append(
            f"| `{item.name}` | {item.static_macs} | {item.dynamic_macs} | "
            f"{keeps_shape} | {exact} | {item.note} |"
        )
    return lines


def render_report(contract: ContractSummary, stats: tuple[DiffStats, ...], candidate_items: tuple[Candidate, ...]) -> str:
    exact_items = tuple(item for item in stats if item.name == "exact_recompose_coeff")
    mylm_items = tuple(item for item in stats if item.name == "mylm_group_correction")
    if all(item.mismatches_1e_2 == 0 for item in exact_items):
        exact_decision = "exact route is numerically valid for this gate"
    else:
        exact_decision = "exact route failed this gate"
    if all(item.mismatches_1e_2 == 0 for item in mylm_items):
        mylm_decision = "MyLM-like route passed this synthetic tolerance"
    else:
        mylm_decision = "MyLM-like route needs multi-layer/token acceptance, not exact migration"
    lines = [
        "# Q4NX Modified Body Numeric Gate",
        "",
        "This experiment uses the exp120 section contract as the body boundary and",
        "tests the two realistic numerical branches before changing production asm.",
        "",
        "## Exp120 Contract",
        "",
        f"- Sections: `{contract.sections}`",
        f"- Original slots: `{contract.original_slots}`",
        f"- Total static group MACs: `{contract.total_group_macs}`",
        f"- Exact instruction replay: `{contract.exact_instruction_match}`",
        f"- Stable boundary pressure: `{contract.stable_boundary_pressure}`",
        "",
        "## Candidate Body Shapes",
        "",
        *render_candidate_table(candidate_items),
        "",
        "## Synthetic Numeric Gate",
        "",
        f"- Samples: `{SAMPLES}`",
        f"- Rows/sample: `{ROWS}`",
        f"- Groups: `{GROUPS}`",
        f"- Group size: `{GROUP_SIZE}`",
        f"- Scenarios: `{', '.join(scenario[0] for scenario in SCENARIOS)}`",
        "",
        *render_diff_table(stats),
        "",
        "## Decision",
        "",
        f"- `{exact_decision}`.",
        f"- `{mylm_decision}`.",
        "",
        "The first production-safe modified body should therefore target",
        "`exact_recompose_coeff`: preserve exp120 section boundaries and live state,",
        "but accept that the MyLM 33-MAC/group count changes to an exact 32-MAC/group",
        "body unless a later real layer/token gate explicitly accepts the MyLM-like numerical contract.",
    ]
    return "\n".join(lines) + "\n"


def render_json(contract: ContractSummary, stats: tuple[DiffStats, ...], candidate_items: tuple[Candidate, ...]):
    return {
        "contract": {
            "sections": contract.sections,
            "original_slots": contract.original_slots,
            "total_group_macs": contract.total_group_macs,
            "exact_instruction_match": contract.exact_instruction_match,
            "stable_boundary_pressure": contract.stable_boundary_pressure,
        },
        "candidates": [item.__dict__ for item in candidate_items],
        "stats": [item.__dict__ for item in stats],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_contract(CONTRACT_JSON)
    stats_list: list[DiffStats] = []
    for scenario_name, scale_std, zero_std, activation_std in SCENARIOS:
        q, scale, zero, activation = deterministic_inputs(SAMPLES, scale_std, zero_std, activation_std)
        expected = exact_reference(q, scale, zero, activation)
        stats_list.extend(
            (
                diff_stats(
                    scenario_name,
                    "mylm_group_correction",
                    mylm_group_correction(q, scale, zero, activation),
                    expected,
                ),
                diff_stats(
                    scenario_name,
                    "exact_recompose_coeff",
                    exact_recompose_coeff(q, scale, zero, activation),
                    expected,
                ),
                diff_stats(
                    scenario_name,
                    "split_zero_per_dim",
                    split_zero_per_dim(q, scale, zero, activation),
                    expected,
                ),
            )
        )
    stats = tuple(stats_list)
    candidate_items = candidates(contract)
    args.report.write_text(render_report(contract, stats, candidate_items))
    args.json_output.write_text(json.dumps(render_json(contract, stats, candidate_items), indent=2) + "\n")
    print(f"wrote {args.report}")
    print(f"wrote {args.json_output}")
    for item in stats:
        print(f"{item.scenario}/{item.name}: max_abs={item.max_abs:.9f} >1e-2={item.mismatches_1e_2}")
    exact_items = tuple(item for item in stats if item.name == "exact_recompose_coeff")
    return 0 if all(item.mismatches_1e_2 == 0 for item in exact_items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
