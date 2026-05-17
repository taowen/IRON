#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F


def tensor_error_stats(
    output: torch.Tensor,
    expected: torch.Tensor,
    rel_tol: float,
    abs_tol: float,
) -> tuple[int, float, float, int | None, float | None, float | None]:
    output_f = output.flatten().to(torch.float32)
    expected_f = expected.flatten().to(torch.float32)
    compare_len = min(output_f.numel(), expected_f.numel())
    if output_f.numel() != expected_f.numel():
        first = compare_len
        errors = abs(output_f.numel() - expected_f.numel())
        return errors, float("inf"), float("inf"), first, None, None
    if compare_len == 0:
        return 0, 0.0, 0.0, None, None, None

    diff = (output_f[:compare_len] - expected_f[:compare_len]).abs()
    norm = (output_f[:compare_len].abs() + expected_f[:compare_len].abs()).clamp(
        max=torch.finfo(torch.float32).max
    )
    mask = diff >= torch.maximum(
        torch.tensor(abs_tol, dtype=torch.float32),
        rel_tol * norm,
    )
    errors = int(mask.sum().item())
    if errors:
        first = int(mask.nonzero(as_tuple=False)[0].item())
        first_expected = float(expected_f[first])
        first_output = float(output_f[first])
    else:
        first = None
        first_expected = None
        first_output = None
    return (
        errors,
        float(diff.max().item()),
        float(diff.mean().item()),
        first,
        first_expected,
        first_output,
    )


def print_tensor_check(
    label: str,
    output: torch.Tensor,
    expected: torch.Tensor,
    rel_tol: float,
    abs_tol: float,
) -> int:
    errors, max_abs, mean_abs, first, first_expected, first_output = tensor_error_stats(
        output,
        expected,
        rel_tol,
        abs_tol,
    )
    print(f"{label}_max_abs: {max_abs:.6f}")
    print(f"{label}_mean_abs: {mean_abs:.6f}")
    print(f"{label}_errors: {errors}")
    if first is not None:
        if first_expected is None or first_output is None:
            print(f"{label}_first_error: index={first} shape_mismatch")
        else:
            print(
                f"{label}_first_error: index={first} "
                f"expected={first_expected:.6f} got={first_output:.6f}"
            )
    return errors


def full_layer_local_invariants(
    inputs: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    ffn_out_local = F.linear(
        actual["ffn_hidden"].view(1, 1, -1),
        inputs["W_down"],
    ).flatten()
    return {
        "ffn_out_local": ffn_out_local.contiguous(),
        "layer_residual_local": (
            actual["attn_residual"].to(torch.float32)
            + actual["ffn_out"].to(torch.float32)
        )
        .to(dtype=actual["layer_residual"].dtype)
        .contiguous(),
    }
