#!/usr/bin/env python3
"""Generate the main16 Q4NX source-assembly probe used by the role object."""

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
