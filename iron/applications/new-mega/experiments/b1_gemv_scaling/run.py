#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import aie.utils as aie_utils

from iron.common.context import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemv.op import GEMV
from iron.operators.gemv.reference import generate_golden_reference

QWEN3_GEMV_SHAPES: dict[str, tuple[int, int]] = {
    "q": (2048, 1024),
    "k": (1024, 1024),
    "v": (1024, 1024),
    "o": (1024, 2048),
    "gate": (3072, 1024),
    "up": (3072, 1024),
    "down": (1024, 3072),
}


@dataclass
class GemvResult:
    shape_name: str
    m: int
    k: int
    columns: int
    tile_size_input: int
    tile_size_output: int
    status: str
    latency_us: float | None = None
    bandwidth_gbps: float | None = None
    gflops: float | None = None
    wall_s: float | None = None
    reason: str = ""


def _tile_size_output(m: int, columns: int, requested: int) -> int:
    rows_per_column = m // columns
    if rows_per_column % requested == 0:
        return requested
    divisors = [
        d
        for d in [512, 384, 256, 192, 128, 96, 64, 32, 16, 8, 4]
        if d <= rows_per_column
    ]
    for divisor in divisors:
        if rows_per_column % divisor == 0:
            return divisor
    raise ValueError(f"Could not choose tile_size_output for M={m}, columns={columns}")


def _run_one(
    *,
    context: AIEContext,
    shape_name: str,
    m: int,
    k: int,
    columns: int,
    tile_size_input: int,
    tile_size_output: int,
    warmup_iters: int,
    timed_iters: int,
    rel_tol: float,
    abs_tol: float,
) -> GemvResult:
    start = time.perf_counter()
    try:
        golden = generate_golden_reference(M=m, K=k, seed=42 + m + k + columns)
        operator = GEMV(
            M=m,
            K=k,
            num_aie_columns=columns,
            tile_size_input=tile_size_input,
            tile_size_output=tile_size_output,
            context=context,
        )
        errors, latency_us, bandwidth_gbps = run_test(
            operator,
            {"matrix": golden["A"].flatten(), "vector": golden["B"]},
            {"output": golden["C"]},
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        wall_s = time.perf_counter() - start
        if errors:
            return GemvResult(
                shape_name=shape_name,
                m=m,
                k=k,
                columns=columns,
                tile_size_input=tile_size_input,
                tile_size_output=tile_size_output,
                status="numeric-fail",
                latency_us=latency_us,
                bandwidth_gbps=bandwidth_gbps,
                gflops=(2.0 * m * k) / (latency_us * 1e-6) / 1e9,
                wall_s=wall_s,
                reason=f"errors={sum(len(v) for v in errors.values())}",
            )
        return GemvResult(
            shape_name=shape_name,
            m=m,
            k=k,
            columns=columns,
            tile_size_input=tile_size_input,
            tile_size_output=tile_size_output,
            status="ok",
            latency_us=latency_us,
            bandwidth_gbps=bandwidth_gbps,
            gflops=(2.0 * m * k) / (latency_us * 1e-6) / 1e9,
            wall_s=wall_s,
        )
    except Exception as exc:
        wall_s = time.perf_counter() - start
        return GemvResult(
            shape_name=shape_name,
            m=m,
            k=k,
            columns=columns,
            tile_size_input=tile_size_input,
            tile_size_output=tile_size_output,
            status="fail",
            wall_s=wall_s,
            reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 Qwen3 real-shape GEMV scaling.")
    parser.add_argument(
        "--build-dir", type=Path, default=Path("build_new_mega_b1_gemv_scaling")
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["q", "k", "v", "o", "gate", "up", "down"],
        choices=sorted(QWEN3_GEMV_SHAPES),
    )
    parser.add_argument("--columns", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--tile-size-input", type=int, default=4)
    parser.add_argument("--tile-size-output", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument("--rel-tol", type=float, default=0.04)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_columns = aie_utils.get_current_device().cols
    columns = [col for col in args.columns if col <= max_columns]
    if not columns:
        raise ValueError(
            f"No requested column count is available on this device ({max_columns} columns)"
        )

    context = AIEContext(build_dir=args.build_dir)
    results: list[GemvResult] = []
    for shape_name in args.shapes:
        m, k = QWEN3_GEMV_SHAPES[shape_name]
        for col in columns:
            if m % col != 0:
                results.append(
                    GemvResult(
                        shape_name=shape_name,
                        m=m,
                        k=k,
                        columns=col,
                        tile_size_input=args.tile_size_input,
                        tile_size_output=args.tile_size_output,
                        status="skip",
                        reason="M is not divisible by columns",
                    )
                )
                continue
            tso = _tile_size_output(m, col, args.tile_size_output)
            result = _run_one(
                context=context,
                shape_name=shape_name,
                m=m,
                k=k,
                columns=col,
                tile_size_input=args.tile_size_input,
                tile_size_output=tso,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                rel_tol=args.rel_tol,
                abs_tol=args.abs_tol,
            )
            results.append(result)

    print("experiment: B1 real-shape GEMV scaling")
    print(f"build_dir: {args.build_dir}")
    print(f"device_columns: {max_columns}")
    print(f"requested_shapes: {','.join(args.shapes)}")
    print(f"requested_columns: {','.join(str(col) for col in args.columns)}")
    print(f"timed_iters: {args.timed_iters}")
    print(
        "csv: shape,M,K,columns,tile_size_input,tile_size_output,status,"
        "latency_us,bandwidth_gbps,gflops,wall_s,reason"
    )
    for result in results:
        latency = "" if result.latency_us is None else f"{result.latency_us:.3f}"
        bandwidth = (
            "" if result.bandwidth_gbps is None else f"{result.bandwidth_gbps:.6f}"
        )
        gflops = "" if result.gflops is None else f"{result.gflops:.6f}"
        wall = "" if result.wall_s is None else f"{result.wall_s:.3f}"
        print(
            "result: "
            f"{result.shape_name},{result.m},{result.k},{result.columns},"
            f"{result.tile_size_input},{result.tile_size_output},{result.status},"
            f"{latency},{bandwidth},{gflops},{wall},{result.reason}"
        )

    ok_results = [result for result in results if result.status == "ok"]
    failed = [result for result in results if result.status not in {"ok", "skip"}]
    if failed:
        print("decision: rejected")
        print("root_cause: at least one real-shape GEMV configuration failed.")
        raise SystemExit(1)
    if not ok_results:
        print("decision: rejected")
        print("root_cause: no GEMV configuration ran.")
        raise SystemExit(1)
    print("decision: accepted")


if __name__ == "__main__":
    main()
