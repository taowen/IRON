#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from iron.applications.qwen3_0_6b.qwen3_cpu import DEFAULT_MODEL, DEFAULT_PROMPT
from iron.applications.qwen3_0_6b.qwen3_preflight import Qwen3PreflightError

from runner import (
    build_real_qwen3_attention_case,
    compile_fixed_attention_stage,
    load_model_and_prompt,
    print_fixed_attention_run,
    run_fixed_attention_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 new-mega production runner.")
    parser.add_argument(
        "--stage",
        choices=("fixed-attention",),
        default="fixed-attention",
        help="Production stage to compile/run.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--rel-tol", type=float, default=0.08)
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_production"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir, model, input_ids = load_model_and_prompt(
        model_name=args.model,
        revision=args.revision,
        prompt=args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    if input_ids.shape[1] >= args.max_seq_len:
        raise ValueError("prompt must fit inside max_seq_len")

    case = build_real_qwen3_attention_case(
        model=model,
        input_ids=input_ids,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
    )
    try:
        op, preflight = compile_fixed_attention_stage(
            model=model,
            max_seq_len=args.max_seq_len,
            chunk_size=args.chunk_size,
            build_dir=args.build_dir,
        )
    except Qwen3PreflightError as err:
        print("stage: production fixed-attention")
        print(f"build_dir: {args.build_dir}")
        print("decision: rejected")
        print(f"root_cause: preflight failed: {err}")
        raise SystemExit(1) from err

    result = run_fixed_attention_stage(
        op=op,
        preflight=preflight,
        case=case,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        compile_only=args.compile_only,
    )

    print("stage: production fixed-attention")
    print(f"build_dir: {args.build_dir}")
    print(f"model_dir: {model_dir}")
    print_fixed_attention_run(result=result)
    if args.compile_only:
        return

    if result.errors == 0:
        print("decision: accepted")
        print(
            "root_cause: production fixed-cache attention matches the real "
            "Qwen3 layer-0 reference context with host-owned current K/V writeback."
        )
    else:
        print("decision: rejected")
        print(
            "root_cause: production fixed-cache attention context mismatched "
            "the Qwen3 reference tensor."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
