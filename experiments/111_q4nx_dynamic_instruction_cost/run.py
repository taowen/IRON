#!/usr/bin/env python3
"""Expand Q4NX hot-loop instruction costs by hardware-loop trip counts."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments/aie_intrinsics_api_probe"))

from analyze_mylm_main16_kernel import parse_disasm, summarize  # noqa: E402

DEFAULT_IRON_ASM = REPO_ROOT / "qwen3-layer/main_projection_q4nx_asm.s"
DEFAULT_MYLM_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
DEFAULT_OUTPUT = REPO_ROOT / "experiments/111_q4nx_dynamic_instruction_cost/q4nx_dynamic_instruction_cost.md"

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
    "vlda",
    "vldb",
    "vldb.128",
    "vst",
    "crunpacksize",
    "crupsmode",
    "vbcst.16",
    "vmov",
    "nop",
)
MAC_GROUPS = 8
BLOCKS_PER_GROUP = 8
LANE_PASSES = 2
GROUP_BODY_REPEATS = MAC_GROUPS * LANE_PASSES
EXACT4_REPEATS = BLOCKS_PER_GROUP * GROUP_BODY_REPEATS


@dataclass(frozen=True)
class CostModel:
    name: str
    loop_shape: str
    dynamic_ops: Counter[str]
    notes: tuple[str, ...]


def macro_body(source: str, name: str) -> str:
    match = re.search(rf"^\s*\.macro\s+{re.escape(name)}[^\n]*\n(.*?)^\s*\.endm", source, re.S | re.M)
    if match is None:
        raise ValueError(f"macro not found: {name}")
    return match.group(1)


def count_ops(text: str) -> Counter[str]:
    return Counter({op: text.count(op) for op in OPS})


def scaled_counts(counts: Counter[str], scale: int) -> Counter[str]:
    return Counter({op: value * scale for op, value in counts.items()})


def active_iron_cost(asm_path: Path) -> CostModel:
    source = asm_path.read_text()
    exact4 = count_ops(macro_body(source, "Q4_EXACT4_BLOCK"))
    handoff = count_ops(macro_body(source, "Q4_HANDOFF"))
    group_direct = macro_body(source, "Q4_EXACT32_GROUP_DIRECT")
    group_scaffold = group_direct
    for line in group_direct.splitlines():
        if line.strip().startswith("Q4_EXACT4_BLOCK") or line.strip().startswith("Q4_HANDOFF"):
            group_scaffold = group_scaffold.replace(line, "")
    group_scaffold_counts = count_ops(group_scaffold)
    lane_body = macro_body(source, "Q4_EXACT_LANE_ZOL")
    lane_scaffold = lane_body
    for marker in ("Q4_EXACT32_GROUP_DIRECT",):
        lane_scaffold = lane_scaffold.replace(marker, "")
    lane_scaffold_counts = count_ops(lane_scaffold)
    function_text = source.split("q4nx_chunk_accum_asm_zol:", 1)[1]
    function_scaffold = function_text
    for marker in ("Q4_EXACT_LANE_ZOL",):
        function_scaffold = function_scaffold.replace(marker, "")
    function_scaffold_counts = count_ops(function_scaffold)

    dynamic = Counter()
    dynamic.update(scaled_counts(exact4, EXACT4_REPEATS))
    dynamic.update(scaled_counts(handoff, EXACT4_REPEATS))
    dynamic.update(scaled_counts(group_scaffold_counts, GROUP_BODY_REPEATS))
    dynamic.update(scaled_counts(lane_scaffold_counts, LANE_PASSES))
    dynamic.update(function_scaffold_counts)
    return CostModel(
        name="IRON active exact asm",
        loop_shape="2 lane passes * 8 groups * 8 exact4 blocks",
        dynamic_ops=dynamic,
        notes=(
            "Preserves current exact bf16(q*scale+zero) coefficient before MAC.",
            "Counts are source-expanded because the active body is generated from macros.",
        ),
    )


def mylm_cost(disasm_path: Path) -> CostModel:
    summary = summarize(parse_disasm(disasm_path))
    dynamic = Counter({op: summary.static_ops[op] * summary.loop_count for op in OPS})
    return CostModel(
        name="MyLM raw Q4NX hot loop",
        loop_shape=f"static 0x260..0x1850 * lc={summary.loop_count}",
        dynamic_ops=dynamic,
        notes=(
            "Uses 512 main MACs plus 16 offset/group-sum correction MACs per chunk.",
            "This is the fast numerical contract candidate, not current exact parity.",
        ),
    )


def target_cost() -> CostModel:
    dynamic = Counter(
        {
            "vmac.f": 512,
            "vextbcst.16": 512,
        }
    )
    return CostModel(
        name="Exact recomposed-coeff target floor",
        loop_shape="2 lane passes * 8 groups * 32 direct coefficient MACs",
        dynamic_ops=dynamic,
        notes=(
            "From exp110: exact parity requires centered+zero recomposed before MAC.",
            "This is a semantic floor, not an assembled schedule; dequant/rounding ops are still unknown.",
        ),
    )


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2f}x"


def render_cost_table(costs: tuple[CostModel, ...]) -> list[str]:
    lines = [
        "| Op | IRON Active | MyLM Raw | Target Floor | IRON/MyLM |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_name = {cost.name: cost for cost in costs}
    iron = by_name["IRON active exact asm"].dynamic_ops
    mylm = by_name["MyLM raw Q4NX hot loop"].dynamic_ops
    target = by_name["Exact recomposed-coeff target floor"].dynamic_ops
    for op in OPS:
        lines.append(
            f"| `{op}` | {iron[op]} | {mylm[op]} | {target[op]} | {ratio(iron[op], mylm[op])} |"
        )
    return lines


def render_model_notes(costs: tuple[CostModel, ...]) -> list[str]:
    lines: list[str] = []
    for cost in costs:
        lines.append(f"### {cost.name}")
        lines.append("")
        lines.append(f"- Loop shape: `{cost.loop_shape}`")
        for note in cost.notes:
            lines.append(f"- {note}")
        lines.append("")
    return lines


def render_report(iron_asm: Path, mylm_disasm: Path, costs: tuple[CostModel, ...]) -> str:
    iron = costs[0].dynamic_ops
    mylm = costs[1].dynamic_ops
    lines = [
        "# Q4NX Dynamic Instruction Cost",
        "",
        f"- IRON asm: `{iron_asm}`",
        f"- MyLM disasm: `{mylm_disasm}`",
        "",
        "## Model Notes",
        "",
        *render_model_notes(costs),
        "## Dynamic Op Counts Per Q4NX Chunk",
        "",
        *render_cost_table(costs),
        "",
        "## Interpretation",
        "",
        f"- Active exact has `{iron['vmac.f']}` dynamic `vmac.f`, which is the expected 512 direct coefficient MACs.",
        f"- MyLM has `{mylm['vmac.f']}` dynamic `vmac.f`: 512 main MACs plus 16 group correction MACs.",
        f"- Active exact pays `{iron['vconv.bf16.fp32']}` dynamic `vconv.bf16.fp32` and "
        f"`{iron['vconv.fp32.bf16']}` dynamic `vconv.fp32.bf16`; MyLM pays "
        f"`{mylm['vconv.bf16.fp32']}` and `{mylm['vconv.fp32.bf16']}` in the hot loop.",
        "- Therefore the next exact-parity assembly work should not chase MAC count; it should reduce coefficient construction and conversion traffic while keeping zero inside the coefficient before MAC.",
        "- The MyLM `32+1 MAC/group` path remains a separate numerical contract because exp110 showed split zero accumulation is not exact.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iron-asm", type=Path, default=DEFAULT_IRON_ASM)
    parser.add_argument("--mylm-disasm", type=Path, default=DEFAULT_MYLM_DISASM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    costs = (
        active_iron_cost(args.iron_asm),
        mylm_cost(args.mylm_disasm),
        target_cost(),
    )
    args.output.write_text(render_report(args.iron_asm, args.mylm_disasm, costs))
    print(f"wrote {args.output}")
    for cost in costs:
        print(
            f"{cost.name}: vmac={cost.dynamic_ops['vmac.f']} "
            f"vconv.bf16.fp32={cost.dynamic_ops['vconv.bf16.fp32']} "
            f"vconv.fp32.bf16={cost.dynamic_ops['vconv.fp32.bf16']} "
            f"crupsmode={cost.dynamic_ops['crupsmode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
