#!/usr/bin/env python3
"""Generate the main16 Q4NX source-assembly probes used by the role object."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "qwen3-layer/main_projection_q4nx_asm.s"
BROADCAST_REGISTERS = ("x0", "x1", "x4", "x6", "x8", "x10")
DEQUANT_REGISTERS = ("x2", "x3", "x5", "x7", "x9")
HEADER = (
    "\t.section\t.text.q4nx_accum_lane_asm_group_shape,\"ax\",@progbits",
    "\t.globl\tq4nx_accum_lane_asm_group_shape",
    "\t.p2align\t4",
    "\t.type\tq4nx_accum_lane_asm_group_shape,@function",
    "q4nx_accum_lane_asm_group_shape:",
    "\t// MyLM-style canonical middle Q4NX group shape for one 16-row lane.",
    "\t// This is a production-build assembly shape probe. It is intentionally",
    "\t// unreferenced by the numerical qwen3 decode path until the full lane body",
    "\t// is made bit-compatible with the current Q4NX reference.",
)
PREP = (
    "\tvldb\t x11, [p1], #0x40",
    "\tvldb\t wl2, [p5], #0x40",
    "\tvldb\t wl6, [p4], #0x40",
    "\tvlda\t x8, [p0], #0x40",
    "\tvunpack\t x1, wl0, unpacksign0",
    "\tvunpack\t x5, wl7, unpacksign0",
    "\tvunpack\t x9, wl8, unpacksign0",
    "\tvunpack\t x8, wh8, unpacksign0",
    "\tvunpack\t x10, wh7, unpacksign0",
    "\tvunpack\t x4, wl6, unpacksign0",
    "\tvunpack\t x6, wh6, unpacksign0",
    "\tvunpack\t x0, wh8, unpacksign0",
    "\tvups.4x\t dm2, x9, s0, upssign0",
    "\tvups.4x\t dm2, x8, s0, upssign0",
    "\tvups.4x\t dm1, x5, s0, upssign0",
    "\tvups.4x\t dm3, x10, s0, upssign0",
    "\tvups.4x\t dm4, x1, s0, upssign0",
    "\tvups.4x\t dm4, x4, s0, upssign0",
    "\tvups.4x\t dm1, x6, s0, upssign0",
    "\tvups.4x\t dm3, x1, s0, upssign0",
    "\tlda.s16\t r7, [p3], #0x2",
)
FOOTER = (
    "\tvbcst.16\t x4, r7",
    "\tvmac.f\t dm1, dm1, x5, x4, r4",
    "\tret\tlr",
    "\t.size\tq4nx_accum_lane_asm_group_shape, .-q4nx_accum_lane_asm_group_shape",
)
EXACT_LANE_MACRO = (
    "",
    "\t.macro Q4_EXACT4_BLOCK pack0, pack1, lane0, lane1, lane2, lane3",
    "\tnopa\t;\t\tvldb.128\t wh0, [p5, #\\pack0];\t\tnops\t;\t\tnopxm\t;\t\tnopv",
    "\tvunpack\tx6, wh0, unpacksign0",
    "\tvunpack\ty3, x6, unpacksign0;\t\tmov\tcrunpacksize, #0",
    "\tmovxm\tr0, #0x4b01",
    "\tmov\tcrupsmode, #0",
    "\tmova\tr1, #0;\t\tvbcst.16\t x8, r0",
    "\tmov\ts0, r1",
    "\tmov\tcrunpacksize, #1",
    "\tvconv.fp32.bf16\tcml0, x8",
    "\tvups.2x\tcml2, x6, s0, upssign0;\t\tvadd\tdm2, dm2, dm0, r1",
    "\tmova\tr0, #0x3c",
    "\tvsub.f\tdm2, dm2, dm0, r0",
    "\tvldb.128\t wh0, [p5, #\\pack1]",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tvunpack\tx6, wh0, unpacksign0",
    "\tvunpack\ty3, x6, unpacksign0;\t\tvconv.bf16.fp32\t x8, cml2",
    "\tnop",
    "\tvmul.f\tdm2, x8, x0, r0",
    "\tnop",
    "\tmov\tcrunpacksize, #0",
    "\tmov\tcrunpacksize, #1",
    "\tnop",
    "\tvups.2x\tcml2, x6, s0, upssign0;\t\tvadd\tdm3, dm2, dm0, r1",
    "\tvconv.bf16.fp32\t wh0, bmll2;\t\tvmov\twl6, wh8",
    "\tvsub.f\tdm0, dm3, dm0, r0",
    "\tvmul.f\tdm0, x6, x0, r0",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tvconv.bf16.fp32\t x8, cml0",
    "\tvconv.bf16.fp32\t wh0, bmll0;\t\tvconv.fp32.bf16\tbmll4, wh0",
    "\tvmov\twl6, wh8;\t\tvadd.f\tdm4, dm4, dm2, r0",
    "\tvconv.fp32.bf16\tbmll2, wl2;\t\tvmul.f\tdm0, x8, x0, r0",
    "\tvmul.f\tdm4, x6, x0, r0",
    "\tnop",
    "\tnop",
    "\tvadd.f\tdm3, dm0, dm2, r0",
    "\tvconv.bf16.fp32\t wl2, bmll4;\t\tvconv.fp32.bf16\tbmll0, wh0",
    "\tvconv.bf16.fp32\t wh2, bmll0",
    "\tvconv.bf16.fp32\t wh2, bmll4;\t\tvextbcst.16\t x0, x4, #\\lane0;\t\tvadd.f\tdm0, dm0, dm2, r0",
    "\tmova\tr1, #0x33c;\t\tvconv.fp32.bf16\tbmll0, wh2",
    "\tvmac.f\tdm1, dm1, x2, x0, r1",
    "\tvconv.bf16.fp32\t wl2, bmll3;\t\tvextbcst.16\t x0, x4, #\\lane1;\t\tvadd.f\tdm2, dm3, dm2, r0",
    "\tvconv.fp32.bf16\tbmll3, wh2",
    "\tvmac.f\tdm1, dm1, x2, x0, r1",
    "\tvconv.bf16.fp32\t wl0, bmll0;\t\tvextbcst.16\t x2, x4, #\\lane2",
    "\tnop",
    "\tvmac.f\tdm0, dm1, x0, x2, r1",
    "\tvconv.bf16.fp32\t wl0, bmll2;\t\tvextbcst.16\t x2, x4, #\\lane3",
    "\tnop",
    "\tvmac.f\tdm0, dm0, x0, x2, r1",
    "\t.endm",
    "",
    "\t.macro Q4_RELOAD_ARGS_PTR",
    "\tvldb\twl0, [p1, #0]",
    "\tvldb\twl2, [p2, #0]",
    "\tvldb\tx4, [p3, #0]",
    "\t.endm",
    "",
    "\t.macro Q4_SET_PACK_BASE_REG base_reg",
    "\tmovs\tm0, \\base_reg",
    "\tmov\tp5, p0",
    "\tpadda\t[p5], m0",
    "\t.endm",
    "",
    "\t.macro Q4_HANDOFF",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tvmov\tbmll1, bmll0",
    "\t.endm",
)
EXACT_LANE_HEADER = (
    "",
    "\t.section\t.text.q4nx_accum_lane_exact_body_shape,\"ax\",@progbits",
    "\t.globl\tq4nx_accum_lane_exact_body_shape",
    "\t.p2align\t4",
    "\t.type\tq4nx_accum_lane_exact_body_shape,@function",
    "q4nx_accum_lane_exact_body_shape:",
    "\t// Callable exact Q4NX lane candidate for the production role object.",
    "\t// ABI: p0=packed lane data, p1=scale lane, p2=offset lane,",
    "\t//      p3=activation bf16[256], p4=dst float[16].",
    "\t// It returns through the normal C ABI and does not release any lock.",
    "\tmova\tr1, #0",
    "\tmov\tcrrnd, #0xc",
    "\tvbcst.32\tx8, r1",
    "\tvmov\tbmll1, x8",
    "\tmova\tr10, #0",
    "\tmova\tr15, #8",
    "\tmova\tr14, #0x40",
    ".Lq4_exact_lane_loop_body:",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
)
EXACT_LANE_FOOTER = (
    "\tmova\tr7, #0x20",
    "\tmovs\tm0, r7",
    "\tpadda\t[p1], m0",
    "\tpadda\t[p2], m0",
    "\tmova\tr7, #0x40",
    "\tmovs\tm0, r7",
    "\tpadda\t[p3], m0",
    "\tadd\tr15, r15, #-1",
    "\tjnz\tr15, #.Lq4_exact_lane_loop_body",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tnop",
    "\tvst\tbmll1, [p4, #0]",
    "\tret\tlr",
    "\t.size\tq4nx_accum_lane_exact_body_shape, .-q4nx_accum_lane_exact_body_shape",
)


@dataclass(frozen=True)
class MacStep:
    lane: int
    broadcast_register: str
    dequant_register: str


def group_mac_steps() -> tuple[MacStep, ...]:
    return tuple(
        MacStep(
            lane=lane,
            broadcast_register=BROADCAST_REGISTERS[lane % len(BROADCAST_REGISTERS)],
            dequant_register=DEQUANT_REGISTERS[lane % len(DEQUANT_REGISTERS)],
        )
        for lane in range(32)
    )


def generate_assembly() -> str:
    lines: list[str] = []
    lines.extend(HEADER)
    lines.extend(PREP)
    for step in group_mac_steps():
        lines.append(
            f"\tvextbcst.16\t {step.broadcast_register}, x11, #{step.lane}"
        )
        lines.append(
            f"\tvmac.f\t dm1, dm1, {step.dequant_register}, {step.broadcast_register}, r4"
        )
    lines.extend(FOOTER)
    lines.extend(EXACT_LANE_MACRO)
    lines.extend(EXACT_LANE_HEADER)
    for lanes in range(0, 32, 4):
        lines.append("\tQ4_RELOAD_ARGS_PTR")
        lines.append("\tQ4_SET_PACK_BASE_REG r10")
        lines.append(
            f"\tQ4_EXACT4_BLOCK 0x000, 0x020, {lanes}, {lanes + 1}, {lanes + 2}, {lanes + 3}"
        )
        lines.append("\tQ4_HANDOFF")
        lines.append("\tadd\tr10, r10, r14")
    lines.extend(EXACT_LANE_FOOTER)
    return "\n".join(lines) + "\n"


def check_generated(output: Path) -> bool:
    return output.read_text() == generate_assembly()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    generated = generate_assembly()
    if args.write:
        args.output.write_text(generated)
    elif args.check:
        if check_generated(args.output):
            print(f"PASS: {args.output} matches generated main16 Q4NX assembly")
            return 0
        print(f"FAIL: {args.output} differs from generated main16 Q4NX assembly")
        return 1
    else:
        print(generated, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
