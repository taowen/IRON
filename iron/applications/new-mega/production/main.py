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
    build_phase_owned_case,
    compile_phase_owned_stage,
    load_model_and_prompt,
    print_phase_owned_run,
    run_phase_owned_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 new-mega production runner.")
    parser.add_argument(
        "--stage",
        choices=("phase-owned",),
        default="phase-owned",
        help="Production stage to compile/run.",
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--phase-packets-per-layer", type=int, default=11)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--attention-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--q-rows-per-packet", type=int, default=4)
    parser.add_argument(
        "--fabric-group-size",
        type=int,
        default=4,
        help="Maximum lanes per local broadcast/join fabric group.",
    )
    parser.add_argument(
        "--packet-elements",
        type=int,
        default=None,
        help=(
            "BF16 values per phase packet. Defaults to "
            "the largest real row-shard phase payload size."
        ),
    )
    parser.add_argument("--abs-tol", type=float, default=0.5)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_production_phase_owned"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        op, preflight = compile_phase_owned_stage(
            num_lanes=args.num_lanes,
            num_layers=args.num_layers,
            phase_packets_per_layer=args.phase_packets_per_layer,
            packet_elements=args.packet_elements,
            hidden_size=args.hidden_size,
            attention_size=args.attention_size,
            intermediate_size=args.intermediate_size,
            q_rows_per_packet=args.q_rows_per_packet,
            fabric_group_size=args.fabric_group_size,
            build_dir=args.build_dir,
        )
    except Qwen3PreflightError as err:
        print("stage: production phase-owned")
        print(f"build_dir: {args.build_dir}")
        print("decision: rejected")
        print(f"root_cause: preflight failed: {err}")
        raise SystemExit(1) from err

    case = None
    if not args.compile_only:
        model_dir, model, input_ids = load_model_and_prompt(
            model_name=args.model,
            revision=args.revision,
            prompt=args.prompt,
            raw_prompt=args.raw_prompt,
            enable_thinking=args.enable_thinking,
        )
        case = build_phase_owned_case(
            op,
            model_dir=model_dir,
            model=model,
            input_ids=input_ids,
            max_seq_len=args.max_seq_len,
        )
    result = run_phase_owned_stage(
        op=op,
        preflight=preflight,
        case=case,
        abs_tol=args.abs_tol,
        compile_only=args.compile_only,
    )

    print("stage: production phase-owned")
    print(f"build_dir: {args.build_dir}")
    print_phase_owned_run(result)
    if args.compile_only:
        return

    if result.errors == 0 and result.qwen3_errors == 0:
        print("decision: accepted")
        print(
            "root_cause: production is a single phase-owned topology: fixed lane "
            "Workers consume shared broadcast packets plus lane-local phase "
            "streams loaded from real Qwen3 weights, phase 0 computes real "
            "row-sharded Q outputs, o_proj computes real attention residual "
            "shards, gate_up computes real post-norm gate/up shards, down_proj "
            "computes real layer residual shards, next_layer_token updates "
            "tile-local hidden state, and resource use is bounded by lanes "
            "rather than by statically appended layer stages."
        )
    else:
        print("decision: rejected")
        print("root_cause: phase-owned real shard output did not match reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
