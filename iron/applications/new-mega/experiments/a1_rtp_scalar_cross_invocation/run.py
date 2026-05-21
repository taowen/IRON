#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from ml_dtypes import bfloat16
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.common.context import AIEContext
from iron.operators.softmax.op import Softmax


@dataclass
class VariantResult:
    rtp_vector_size: int
    mlir_path: Path
    bin_path: Path
    xclbin_path: Path
    npu_output: torch.Tensor
    cpu_output: torch.Tensor
    max_abs: float
    tail_sum: float


def _byte_diff_count(left: Path, right: Path) -> int | str:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    if len(left_bytes) != len(right_bytes):
        return f"size-mismatch:{len(left_bytes)}->{len(right_bytes)}"
    return sum(a != b for a, b in zip(left_bytes, right_bytes))


def _make_input(rows: int, cols: int) -> torch.Tensor:
    base = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    values = ((base % cols) - cols / 2) / 16.0
    return values.to(torch.bfloat16).flatten()


def _reference(
    input_tensor: torch.Tensor, rows: int, cols: int, rtp: int
) -> torch.Tensor:
    x = input_tensor.reshape(rows, cols).to(torch.float32).clone()
    x[:, rtp:] = -torch.inf
    return torch.softmax(x, dim=-1).to(torch.bfloat16).flatten()


def _run_variant(
    *,
    context: AIEContext,
    rows: int,
    cols: int,
    rtp_vector_size: int,
    input_tensor: torch.Tensor,
) -> VariantResult:
    op = Softmax(
        rows=rows,
        cols=cols,
        num_aie_columns=1,
        num_channels=1,
        rtp_vector_size=rtp_vector_size,
        context=context,
    )
    op.compile()
    op_func = op.get_callable()

    input_buf = XRTTensor.from_torch(input_tensor)
    output_buf = XRTTensor((rows * cols,), dtype=bfloat16)
    op_func(input_buf, output_buf)
    npu_output = output_buf.to_torch().detach().clone().flatten()
    cpu_output = _reference(input_tensor, rows, cols, rtp_vector_size)
    diff = (npu_output.to(torch.float32) - cpu_output.to(torch.float32)).abs()
    tail_sum = float(npu_output.reshape(rows, cols)[:, rtp_vector_size:].abs().sum())

    return VariantResult(
        rtp_vector_size=rtp_vector_size,
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        bin_path=Path(op.insts_artifact.filename),
        xclbin_path=Path(op.xclbin_artifact.filename),
        npu_output=npu_output,
        cpu_output=cpu_output,
        max_abs=float(diff.max()),
        tail_sum=tail_sum,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A1 RTP scalar cross-invocation probe using Softmax RTP."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build_new_mega_a1_rtp"))
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--rtp-a", type=int, default=32)
    parser.add_argument("--rtp-b", type=int, default=64)
    parser.add_argument("--abs-tol", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows % 16 != 0 or args.cols % 16 != 0:
        raise ValueError("rows and cols must be multiples of 16 for Softmax")
    if not (0 < args.rtp_a <= args.cols and 0 < args.rtp_b <= args.cols):
        raise ValueError("RTP values must be inside the row width")
    if args.rtp_a == args.rtp_b:
        raise ValueError("Use two different RTP values")

    context = AIEContext(build_dir=args.build_dir)
    input_tensor = _make_input(args.rows, args.cols)

    result_a = _run_variant(
        context=context,
        rows=args.rows,
        cols=args.cols,
        rtp_vector_size=args.rtp_a,
        input_tensor=input_tensor,
    )
    result_b = _run_variant(
        context=context,
        rows=args.rows,
        cols=args.cols,
        rtp_vector_size=args.rtp_b,
        input_tensor=input_tensor,
    )

    mlir_diff = _byte_diff_count(result_a.mlir_path, result_b.mlir_path)
    bin_diff = _byte_diff_count(result_a.bin_path, result_b.bin_path)
    xclbin_diff = _byte_diff_count(result_a.xclbin_path, result_b.xclbin_path)
    output_diff = float(
        (result_a.npu_output.to(torch.float32) - result_b.npu_output.to(torch.float32))
        .abs()
        .max()
    )

    a_match = result_a.max_abs <= args.abs_tol
    b_match = result_b.max_abs <= args.abs_tol
    same_artifact = mlir_diff == 0 and bin_diff == 0 and xclbin_diff == 0

    print("experiment: A1 RTP scalar cross-invocation proof")
    print(f"build_dir: {args.build_dir}")
    print(f"rows: {args.rows}")
    print(f"cols: {args.cols}")
    print(f"rtp_a: {args.rtp_a}")
    print(f"rtp_b: {args.rtp_b}")
    print(f"rtp{args.rtp_a}_max_abs: {result_a.max_abs:.6f}")
    print(f"rtp{args.rtp_a}_tail_sum: {result_a.tail_sum:.6f}")
    print(f"rtp{args.rtp_a}_cpu_match: {a_match}")
    print(f"rtp{args.rtp_b}_max_abs: {result_b.max_abs:.6f}")
    print(f"rtp{args.rtp_b}_tail_sum: {result_b.tail_sum:.6f}")
    print(f"rtp{args.rtp_b}_cpu_match: {b_match}")
    print(f"variant_output_max_abs: {output_diff:.6f}")
    print(f"mlir_diff_bytes: {mlir_diff}")
    print(f"runtime_bin_diff_bytes: {bin_diff}")
    print(f"xclbin_diff_bytes: {xclbin_diff}")
    print(f"same_artifact: {same_artifact}")

    if a_match and b_match and same_artifact:
        print("decision: accepted")
    elif a_match and b_match:
        print("decision: rejected")
        print(
            "root_cause: RTP changes behavior, but the RTP value is still "
            "encoded in generated artifacts rather than supplied as an "
            "invocation-time value."
        )
    else:
        print("decision: rejected")
        print("root_cause: NPU output did not match the CPU RTP reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
