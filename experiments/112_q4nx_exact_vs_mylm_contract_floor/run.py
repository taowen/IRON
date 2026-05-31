#!/usr/bin/env python3
"""Compare exact-Q4NX lower-bound costs with the MyLM-like contract."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "experiments/112_q4nx_exact_vs_mylm_contract_floor/q4nx_exact_vs_mylm_contract_floor.md"

ROWS_PER_CHUNK = 32
ROWS_PER_VECTOR = 16
LANE_PASSES = ROWS_PER_CHUNK // ROWS_PER_VECTOR
GROUPS = 8
DIMS_PER_GROUP = 32
VECTOR_DIMS_PER_CHUNK = LANE_PASSES * GROUPS * DIMS_PER_GROUP
PACKED_4D_BLOCKS = LANE_PASSES * GROUPS * (DIMS_PER_GROUP // 4)

OPS = (
    "vmac.f",
    "vextbcst.16",
    "vunpack",
    "vups.4x",
    "vups.2x",
    "vmul.f",
    "vadd.f",
    "vsub.f",
    "vconv.bf16.fp32",
    "vconv.fp32.bf16",
    "crupsmode",
    "crunpacksize",
    "nop",
)


@dataclass(frozen=True)
class ContractCost:
    name: str
    counts: OrderedDict[str, int]
    notes: tuple[str, ...]


def empty_counts() -> OrderedDict[str, int]:
    return OrderedDict((op, 0) for op in OPS)


def active_exact_cost() -> ContractCost:
    counts = empty_counts()
    counts.update(
        {
            "vmac.f": 512,
            "vextbcst.16": 512,
            "vunpack": 512,
            "vups.2x": 256,
            "vmul.f": 512,
            "vadd.f": 512,
            "vsub.f": 256,
            "vconv.bf16.fp32": 1280,
            "vconv.fp32.bf16": 768,
            "crupsmode": 128,
            "crunpacksize": 512,
            "nop": 3088,
        }
    )
    return ContractCost(
        name="active_exact",
        counts=counts,
        notes=(
            "Current q4nx_chunk_accum_asm_zol dynamic counts from exp111.",
            "Numerically exact against the current reference, but not software-pipelined.",
        ),
    )


def exact_lower_bound_cost() -> ContractCost:
    counts = empty_counts()
    counts.update(
        {
            "vmac.f": VECTOR_DIMS_PER_CHUNK,
            "vextbcst.16": VECTOR_DIMS_PER_CHUNK,
            "vunpack": PACKED_4D_BLOCKS,
            "vups.4x": PACKED_4D_BLOCKS,
            "vmul.f": VECTOR_DIMS_PER_CHUNK,
            "vadd.f": VECTOR_DIMS_PER_CHUNK,
            "vconv.bf16.fp32": VECTOR_DIMS_PER_CHUNK * 2,
            "vconv.fp32.bf16": VECTOR_DIMS_PER_CHUNK,
        }
    )
    return ContractCost(
        name="exact_lower_bound",
        counts=counts,
        notes=(
            "Semantic lower bound for bf16(bf16(q*scale)+zero) before MAC.",
            "Assumes a perfect 4-dim unpack/vups schedule and no redundant control setup.",
            "Still needs two bf16 roundings and one bf16->fp32 conversion per vector dim.",
        ),
    )


def mylm_contract_cost() -> ContractCost:
    counts = empty_counts()
    counts.update(
        {
            "vmac.f": 528,
            "vextbcst.16": 512,
            "vunpack": 128,
            "vups.4x": 128,
            "vmul.f": 16,
            "vsub.f": 128,
            "vconv.bf16.fp32": 272,
            "nop": 216,
        }
    )
    return ContractCost(
        name="mylm_group_contract",
        counts=counts,
        notes=(
            "MyLM raw hot-loop dynamic counts from exp111.",
            "Uses 512 main MACs plus 16 zero/group-sum correction MACs.",
            "Not bit-equivalent to current exact reference per exp110.",
        ),
    )


def improvement(before: int, after: int) -> str:
    if before == 0:
        return "n/a"
    return f"{(before - after) / before * 100.0:.1f}%"


def ratio(left: int, right: int) -> str:
    if right == 0:
        return "n/a"
    return f"{left / right:.2f}x"


def render_cost_table(costs: tuple[ContractCost, ...]) -> list[str]:
    by_name = {cost.name: cost.counts for cost in costs}
    active = by_name["active_exact"]
    lower = by_name["exact_lower_bound"]
    mylm = by_name["mylm_group_contract"]
    lines = [
        "| Op | Active Exact | Exact Lower Bound | MyLM-like Contract | Active -> Lower | Lower/MyLM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for op in OPS:
        lines.append(
            f"| `{op}` | {active[op]} | {lower[op]} | {mylm[op]} | "
            f"{improvement(active[op], lower[op])} | {ratio(lower[op], mylm[op])} |"
        )
    return lines


def render_notes(costs: tuple[ContractCost, ...]) -> list[str]:
    lines: list[str] = []
    for cost in costs:
        lines.append(f"### `{cost.name}`")
        lines.append("")
        for note in cost.notes:
            lines.append(f"- {note}")
        lines.append("")
    return lines


def render_report(costs: tuple[ContractCost, ...]) -> str:
    active = costs[0].counts
    lower = costs[1].counts
    mylm = costs[2].counts
    lines = [
        "# Q4NX Exact vs MyLM Contract Floor",
        "",
        "## Chunk Shape",
        "",
        f"- Rows per chunk: `{ROWS_PER_CHUNK}`",
        f"- Rows per vector pass: `{ROWS_PER_VECTOR}`",
        f"- Lane passes: `{LANE_PASSES}`",
        f"- Groups: `{GROUPS}`",
        f"- Dims per group: `{DIMS_PER_GROUP}`",
        f"- Vector dims per chunk: `{VECTOR_DIMS_PER_CHUNK}`",
        f"- 4-dim packed blocks per chunk: `{PACKED_4D_BLOCKS}`",
        "",
        "## Contract Notes",
        "",
        *render_notes(costs),
        "## Dynamic Cost Model",
        "",
        *render_cost_table(costs),
        "",
        "## Decision",
        "",
        "- Exact parity can still improve the active body by removing redundant unpack/control/conversion traffic.",
        f"- The exact lower bound still needs `{lower['vmul.f']}` vector multiplies and "
        f"`{lower['vconv.bf16.fp32'] + lower['vconv.fp32.bf16']}` conversion operations per chunk.",
        f"- MyLM-like group correction needs only `{mylm['vmul.f']}` vector multiplies and "
        f"`{mylm['vconv.bf16.fp32'] + mylm['vconv.fp32.bf16']}` conversion operations per chunk.",
        "- Therefore exact-parity assembly can narrow the gap, but it cannot reach the MyLM raw hot-loop shape unless there is an unknown fused instruction path for the two bf16 roundings.",
        "- The performance route to MyLM-like speed should now branch explicitly: optimize exact as a safe baseline, and separately run multi-layer token gates for the MyLM-like numerical contract.",
        "",
        "## Immediate Target",
        "",
        f"- Safe exact target: reduce active `vunpack` `{active['vunpack']} -> {lower['vunpack']}`, "
        f"`vconv.bf16.fp32` `{active['vconv.bf16.fp32']} -> {lower['vconv.bf16.fp32']}`, and "
        f"`vconv.fp32.bf16` `{active['vconv.fp32.bf16']} -> {lower['vconv.fp32.bf16']}`.",
        "- Risk-bearing fast target: accept the MyLM-like zero/group-sum correction contract, then prove token quality across many layers before migrating production.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    costs = (
        active_exact_cost(),
        exact_lower_bound_cost(),
        mylm_contract_cost(),
    )
    OUTPUT.write_text(render_report(costs))
    print(f"wrote {OUTPUT}")
    for cost in costs:
        conversions = cost.counts["vconv.bf16.fp32"] + cost.counts["vconv.fp32.bf16"]
        print(f"{cost.name}: vmac={cost.counts['vmac.f']} vmul={cost.counts['vmul.f']} conversions={conversions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
