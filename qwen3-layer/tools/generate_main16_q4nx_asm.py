#!/usr/bin/env python3
"""Generate the main16 Q4NX source assembly used by the role object."""

from __future__ import annotations

import argparse
from pathlib import Path

from main16_q4nx_asm_lib import Q4_EXACT_MACROS, main16_layer_scheduler_asm, q4_exact_function_asm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "qwen3-layer/main_projection_q4nx_asm.s"
Q4_SYMBOL = "q4nx_chunk_accum_asm_zol"


def generate_assembly() -> str:
    return (
        Q4_EXACT_MACROS.strip()
        + "\n\n"
        + q4_exact_function_asm(Q4_SYMBOL, save_restore=True)
        + "\n\n"
        + main16_layer_scheduler_asm()
    )


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
