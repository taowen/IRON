#!/usr/bin/env python3
"""Check the MyLM-style Q4NX body contract for the next main16 kernel."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_LAYER = REPO_ROOT / "qwen3-layer"
AIE_PROBE = REPO_ROOT / "experiments" / "aie_intrinsics_api_probe"
sys.path.insert(0, str(QWEN3_LAYER))
sys.path.insert(0, str(AIE_PROBE))

from analyze_mylm_main16_kernel import parse_disasm, summarize  # noqa: E402
from compare_q4nx_group_sum_reference import (  # noqa: E402
    diff_stats,
    make_activation,
    q4nx_group_sum_matvec,
)
from q4nx_reference import make_q4nx_chunk, q4nx_matvec_from_chunk  # noqa: E402

MYLM_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
IRON_ASM = REPO_ROOT / "qwen3-layer" / "main_projection_q4nx_asm.s"


@dataclass(frozen=True)
class BodyCounts:
    name: str
    vmac: int
    vextbcst16: int
    vextbcst32: int
    vunpack: int
    vups2x: int
    vups4x: int
    vconv_bf16_fp32: int
    vst: int
    crupsmode: int
    mov_s0: int


def _count_text_ops(name: str, text: str) -> BodyCounts:
    return BodyCounts(
        name=name,
        vmac=text.count("vmac.f"),
        vextbcst16=text.count("vextbcst.16"),
        vextbcst32=text.count("vextbcst.32"),
        vunpack=text.count("vunpack"),
        vups2x=text.count("vups.2x"),
        vups4x=text.count("vups.4x"),
        vconv_bf16_fp32=text.count("vconv.bf16.fp32"),
        vst=text.count("\tvst") + text.count(";		vst"),
        crupsmode=text.count("crupsmode"),
        mov_s0=text.count("mov	s0"),
    )


def _print_body_counts(counts: BodyCounts) -> None:
    print(f"{counts.name}:")
    print(f"  vmac.f={counts.vmac}")
    print(f"  vextbcst.16={counts.vextbcst16}")
    print(f"  vextbcst.32={counts.vextbcst32}")
    print(f"  vunpack={counts.vunpack}")
    print(f"  vups.2x={counts.vups2x}")
    print(f"  vups.4x={counts.vups4x}")
    print(f"  vconv.bf16.fp32={counts.vconv_bf16_fp32}")
    print(f"  vst={counts.vst}")
    print(f"  crupsmode={counts.crupsmode}")
    print(f"  mov_s0={counts.mov_s0}")


def _macro_body(text: str, name: str) -> str:
    marker = f".macro {name}"
    if marker not in text:
        raise ValueError(f"missing macro {name}")
    return text.split(marker, 1)[1].split(".endm", 1)[0]


def _scale_counts(name: str, counts: BodyCounts, factor: int) -> BodyCounts:
    return BodyCounts(
        name=name,
        vmac=counts.vmac * factor,
        vextbcst16=counts.vextbcst16 * factor,
        vextbcst32=counts.vextbcst32 * factor,
        vunpack=counts.vunpack * factor,
        vups2x=counts.vups2x * factor,
        vups4x=counts.vups4x * factor,
        vconv_bf16_fp32=counts.vconv_bf16_fp32 * factor,
        vst=counts.vst * factor,
        crupsmode=counts.crupsmode * factor,
        mov_s0=counts.mov_s0 * factor,
    )


def _validate_mylm_summary() -> list[str]:
    if not MYLM_DISASM.exists():
        return [f"missing MyLM disasm: {MYLM_DISASM}"]
    summary = summarize(parse_disasm(MYLM_DISASM))
    errors: list[str] = []
    static = summary.static_ops
    dynamic_vmac = static["vmac.f"] * summary.loop_count
    dynamic_vext = static["vextbcst.16"] * summary.loop_count
    dynamic_scratch = len(summary.scratch_loads) * summary.loop_count
    print("mylm_q4nx_body_contract:")
    print(f"  disasm={MYLM_DISASM}")
    print(f"  lc={summary.loop_count}")
    print(f"  static_vmac.f={static['vmac.f']}")
    print(f"  dynamic_vmac.f={dynamic_vmac}")
    print(f"  dynamic_vextbcst.16={dynamic_vext}")
    print(f"  dynamic_group_sum_loads={dynamic_scratch}")
    print(f"  static_vst={static['vst']}")
    print(f"  static_vconv.bf16.fp32={static['vconv.bf16.fp32']}")
    print(f"  static_vups.4x={static['vups.4x']}")
    print(f"  static_vunpack={static['vunpack']}")
    print(f"  phase_group_sum_stores={len(summary.phase_scratch.scratch_stores)}")
    if summary.loop_count != 2:
        errors.append(f"MyLM lc mismatch: {summary.loop_count} != 2")
    if dynamic_vmac != 528:
        errors.append(f"MyLM dynamic vmac mismatch: {dynamic_vmac} != 528")
    if dynamic_vext != 512:
        errors.append(f"MyLM dynamic vextbcst.16 mismatch: {dynamic_vext} != 512")
    if dynamic_scratch != 16:
        errors.append(f"MyLM group-sum load mismatch: {dynamic_scratch} != 16")
    if static["vst"] != 0:
        errors.append(f"MyLM hot loop has vst={static['vst']}")
    if static["vconv.bf16.fp32"] != 136:
        errors.append(f"MyLM hot loop vconv.bf16.fp32 mismatch: {static['vconv.bf16.fp32']} != 136")
    if static["vups.4x"] != 64:
        errors.append(f"MyLM hot loop vups.4x mismatch: {static['vups.4x']} != 64")
    if static["vunpack"] != 64:
        errors.append(f"MyLM hot loop vunpack mismatch: {static['vunpack']} != 64")
    if len(summary.phase_scratch.scratch_stores) != 8:
        errors.append(f"MyLM phase group-sum store mismatch: {len(summary.phase_scratch.scratch_stores)} != 8")
    return errors


def _validate_iron_current_gap() -> list[str]:
    text = IRON_ASM.read_text()
    exact4_counts = _count_text_ops("iron_q4_exact4_macro", _macro_body(text, "Q4_EXACT4_BLOCK"))
    group_direct = _macro_body(text, "Q4_EXACT32_GROUP_DIRECT")
    exact4_calls_per_group = group_direct.count("Q4_EXACT4_BLOCK")
    lane_passes = text.count("Q4_EXACT_LANE_ZOL .Lq4nx_chunk_accum_lane")
    expanded_counts = _scale_counts(
        "iron_current_static_expanded_estimate",
        exact4_counts,
        exact4_calls_per_group * lane_passes,
    )
    _print_body_counts(expanded_counts)
    print(f"  exact4_calls_per_group={exact4_calls_per_group}")
    print(f"  lane_passes={lane_passes}")
    errors: list[str] = []
    if expanded_counts.vconv_bf16_fp32 == 0:
        errors.append("current IRON body unexpectedly has no bf16 conversion traffic")
    if expanded_counts.crupsmode == 0 or expanded_counts.mov_s0 == 0:
        errors.append("current IRON body no longer exposes the repeated exact-rounding setup being replaced")
    if expanded_counts.vups4x != 0:
        errors.append("current IRON body already has vups.4x; update the experiment contract")
    return errors


def _validate_group_sum_numeric_delta() -> list[str]:
    rng = np.random.default_rng(20260531)
    all_stats = []
    for sample in range(8):
        packed = make_q4nx_chunk(rng)
        activation = make_activation(9000 + sample)
        expected = q4nx_matvec_from_chunk(packed, activation)
        got = q4nx_group_sum_matvec(packed, activation, round_group_sum=True)
        all_stats.append(diff_stats(f"sample{sample}", got, expected))

    max_abs = max(stat.max_abs for stat in all_stats)
    max_mismatches_1e_2 = max(stat.mismatches_1e_2 for stat in all_stats)
    mean_abs = float(np.mean([stat.mean_abs for stat in all_stats]))
    print("group_sum_vs_exact_reference:")
    print("  formula=bf16(q*scale)*activation + offset*bf16(group_sum)")
    print(f"  samples={len(all_stats)}")
    print(f"  max_abs={max_abs:.8f}")
    print(f"  mean_abs={mean_abs:.8f}")
    print(f"  max_mismatches_1e_2={max_mismatches_1e_2}")

    errors: list[str] = []
    if max_abs >= 0.01:
        errors.append(f"group-sum delta too large for migration gate: max_abs={max_abs:.8f}")
    if max_mismatches_1e_2 != 0:
        errors.append(f"group-sum has >1e-2 mismatches: {max_mismatches_1e_2}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(_validate_mylm_summary())
    errors.extend(_validate_iron_current_gap())
    errors.extend(_validate_group_sum_numeric_delta())
    if errors:
        print("FAIL:")
        for error in errors:
            print(f"  {error}")
        return 1
    print("PASS: MyLM-style Q4NX body contract is ready for a new implementation experiment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
