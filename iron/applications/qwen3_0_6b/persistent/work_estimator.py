#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static work estimator for the Qwen3 persistent decode graph.

This tool is deliberately static: it reads generated MLIR, counts calls under
finite `scf.for` loops, and applies simple Qwen3-specific kernel cost models.
It is not a hardware trace replacement; use it to choose the next optimization
target before spending time on a new graph shape.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

_CONSTANT_RE = re.compile(
    r"(?P<name>%[A-Za-z0-9_.$-]+)\s*=\s*arith\.constant\s+"
    r"(?P<value>-?\d+)\s*:\s*(?:index|i32|i64)"
)
_FOR_RE = re.compile(
    r"scf\.for\s+%[A-Za-z0-9_.$-]+\s*=\s*(?P<lb>%[A-Za-z0-9_.$-]+)\s+"
    r"to\s+(?P<ub>%[A-Za-z0-9_.$-]+)\s+"
    r"step\s+(?P<step>%[A-Za-z0-9_.$-]+)\s*\{"
)
_CALL_RE = re.compile(
    r"func\.call\s+@(?P<name>[A-Za-z0-9_.$-]+)\((?P<args>.*?)\)\s*:\s*"
    r"\((?P<signature>.*?)\)\s*->\s*\(\)"
)
_MEMREF_RE = re.compile(r"memref<(?P<shape>[^>]+)>")


@dataclass(frozen=True)
class LoopFrame:
    indent: int
    trip_count: int


@dataclass
class KernelCount:
    name: str
    signature: str
    calls: int = 0
    macs: int = 0
    bf16_element_visits: int = 0


@dataclass
class CategoryCount:
    name: str
    calls: int = 0
    macs: int = 0
    bf16_element_visits: int = 0


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_constants(mlir_text: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    for match in _CONSTANT_RE.finditer(mlir_text):
        constants[match.group("name")] = int(match.group("value"))
    return constants


def _trip_count(match: re.Match[str], constants: dict[str, int]) -> int:
    lb = constants.get(match.group("lb"))
    ub = constants.get(match.group("ub"))
    step = constants.get(match.group("step"))
    if lb is None or ub is None or step is None:
        return 1
    if step <= 0:
        return 1
    if ub >= 2**62:
        return 1
    return max(0, (ub - lb + step - 1) // step)


def _memref_numels(signature: str) -> list[int]:
    numels: list[int] = []
    for match in _MEMREF_RE.finditer(signature):
        shape = match.group("shape")
        pieces = shape.split("x")
        dtype = pieces[-1]
        if dtype != "bf16":
            continue
        numel = 1
        for dim in pieces[:-1]:
            if not dim.isdigit():
                numel = 0
                break
            numel *= int(dim)
        if numel:
            numels.append(numel)
    return numels


def _first_matrix_shape(signature: str) -> tuple[int, int] | None:
    for match in _MEMREF_RE.finditer(signature):
        pieces = match.group("shape").split("x")
        if len(pieces) == 3 and pieces[-1] == "bf16":
            return int(pieces[0]), int(pieces[1])
    return None


def _constant_args(call_args: str, constants: dict[str, int]) -> list[int]:
    values: list[int] = []
    for raw in call_args.split(","):
        name = raw.strip()
        if name in constants:
            values.append(constants[name])
    return values


def _classify_kernel(name: str) -> str:
    if name in {"matvec_vectorized_bf16_bf16"}:
        return "QKV projection matvec"
    if name == "qwen3_o_proj_matvec_vectorized_bf16_bf16":
        return "O-proj matvec"
    if name == "qwen3_down_proj_matvec_vectorized_bf16_bf16":
        return "MLP down matvec"
    if name in {
        "qwen3_mlp_matvec4_rows_shard_bf16",
        "qwen3_mlp_gate_up_pair_matvec4_rows_shard_bf16",
        "qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16",
        "qwen3_mlp_gate_up_pair_silu8_rows_shard_bf16",
    }:
        return "MLP gate/up matvec"
    if name in {"rms_norm_bf16_vector", "eltwise_mul_bf16_vector"}:
        return "input RMSNorm"
    if name == "qwen3_weighted_rms_norm_bf16":
        return "MLP RMSNorm"
    if name == "qwen3_norm_rope_with_metadata_bf16":
        return "Q/K RoPE"
    if name in {"qwen3_pack_qk_pair_bf16"}:
        return "attention QK pack"
    if name == "qwen3_attention_scores_bf16":
        return "attention QK scores"
    if name in {"mask_bf16", "softmax_bf16"}:
        return "attention mask/softmax"
    if name in {"qwen3_merge_current_v_bf16"}:
        return "attention V cache merge"
    if name == "qwen3_attention_context_bf16":
        return "attention PV context"
    if name == "qwen3_pack_context_head_bf16":
        return "attention context pack"
    if name in {"eltwise_add_bf16_vector", "qwen3_add_full_slice_bf16"}:
        return "attention residual/join"
    if name in {"qwen3_silu_mul_shard_bf16"}:
        return "MLP SiLU/mul"
    if name in {
        "qwen3_copy_ffn_shard_to_full_bf16",
        "qwen3_add_full_slice_to_tile_bf16",
        "qwen3_copy_tile_to_full_bf16",
        "qwen3_copy_bf16",
    }:
        return "copy/join"
    return "other"


def _estimate_kernel_work(
    name: str,
    signature: str,
    call_args: str,
    constants: dict[str, int],
    *,
    active_seq_len: int,
) -> tuple[int, int]:
    """Return `(macs_per_call, rough_bf16_element_visits_per_call)`."""

    const_args = _constant_args(call_args, constants)
    memref_numels = _memref_numels(signature)

    if name in {
        "matvec_vectorized_bf16_bf16",
        "qwen3_o_proj_matvec_vectorized_bf16_bf16",
        "qwen3_down_proj_matvec_vectorized_bf16_bf16",
    }:
        matrix = _first_matrix_shape(signature)
        if matrix is None:
            return 0, sum(memref_numels)
        rows, dim_k = matrix
        macs = rows * dim_k
        return macs, rows * dim_k + dim_k + rows

    if name == "qwen3_mlp_matvec4_rows_shard_bf16":
        rows = const_args[0] if const_args else 4
        dim_k = 1024
        macs = rows * dim_k
        return macs, rows * dim_k + dim_k + rows

    if name in {
        "qwen3_mlp_gate_up_pair_matvec4_rows_shard_bf16",
        "qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16",
        "qwen3_mlp_gate_up_pair_silu8_rows_shard_bf16",
    }:
        rows = const_args[0] if const_args else 4
        dim_k = 1024
        macs = 2 * rows * dim_k
        return macs, 2 * rows * dim_k + dim_k + 2 * rows

    if name in {"qwen3_attention_scores_bf16", "qwen3_attention_context_bf16"}:
        head_dim = 128
        macs = active_seq_len * head_dim
        return macs, active_seq_len * (head_dim + 1) + head_dim

    if name in {"mask_bf16", "softmax_bf16"}:
        return 0, max(active_seq_len, const_args[-1] if const_args else 0)

    if const_args:
        size = const_args[-1]
        if name in {
            "qwen3_add_full_slice_bf16",
            "qwen3_add_full_slice_to_tile_bf16",
            "qwen3_copy_tile_to_full_bf16",
        }:
            return 0, 3 * size
        if name in {"qwen3_copy_bf16", "qwen3_copy_ffn_shard_to_full_bf16"}:
            return 0, 2 * size
        if name in {
            "rms_norm_bf16_vector",
            "qwen3_weighted_rms_norm_bf16",
            "eltwise_mul_bf16_vector",
            "eltwise_add_bf16_vector",
            "qwen3_silu_mul_shard_bf16",
        }:
            return 0, 3 * size

    return 0, sum(memref_numels)


def iter_weighted_calls(
    mlir_text: str, constants: dict[str, int]
) -> Iterable[tuple[str, str, str, int]]:
    loop_stack: list[LoopFrame] = []
    for line in mlir_text.splitlines():
        line_indent = _indent(line)
        while loop_stack and line_indent <= loop_stack[-1].indent:
            loop_stack.pop()

        loop = _FOR_RE.search(line)
        if loop is not None:
            loop_stack.append(
                LoopFrame(indent=line_indent, trip_count=_trip_count(loop, constants))
            )
            continue

        call = _CALL_RE.search(line)
        if call is None:
            continue

        multiplier = 1
        for frame in loop_stack:
            multiplier *= frame.trip_count
        yield (
            call.group("name"),
            call.group("signature"),
            call.group("args"),
            multiplier,
        )


def estimate_work(
    mlir_path: Path,
    *,
    layers: int,
    position: int,
) -> tuple[dict[str, KernelCount], dict[str, CategoryCount]]:
    mlir_text = mlir_path.read_text()
    constants = _parse_constants(mlir_text)
    active_seq_len = position + 1
    kernels: dict[str, KernelCount] = {}
    categories: dict[str, CategoryCount] = defaultdict(lambda: CategoryCount(name=""))

    for name, signature, call_args, multiplier in iter_weighted_calls(
        mlir_text, constants
    ):
        macs_per_call, elems_per_call = _estimate_kernel_work(
            name,
            signature,
            call_args,
            constants,
            active_seq_len=active_seq_len,
        )
        kernel = kernels.setdefault(name, KernelCount(name=name, signature=signature))
        kernel.calls += multiplier
        kernel.macs += macs_per_call * multiplier
        kernel.bf16_element_visits += elems_per_call * multiplier

        category_name = _classify_kernel(name)
        category = categories[category_name]
        category.name = category_name
        category.calls += multiplier
        category.macs += macs_per_call * multiplier
        category.bf16_element_visits += elems_per_call * multiplier

    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")
    return kernels, dict(categories)


def _fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_m(value: float) -> str:
    return f"{value / 1_000_000:.3f}M"


def print_markdown(
    *,
    mlir_path: Path,
    layers: int,
    position: int,
    kernels: dict[str, KernelCount],
    categories: dict[str, CategoryCount],
    show_kernels: bool,
) -> None:
    total_calls = sum(item.calls for item in kernels.values())
    total_macs = sum(item.macs for item in kernels.values())
    total_elems = sum(item.bf16_element_visits for item in kernels.values())

    print(f"MLIR: `{mlir_path}`")
    print()
    print("```text")
    print(f"layers: {layers}")
    print(f"position: {position}")
    print(f"active_seq_len: {position + 1}")
    print(f"kernel_calls_per_token: {_fmt_int(total_calls)}")
    print(f"estimated_macs_per_token: {_fmt_m(total_macs)}")
    print(f"rough_bf16_element_visits_per_token: {_fmt_int(total_elems)}")
    print("```")
    print()
    print(
        "| category | calls/token | calls/layer | est MACs/token | "
        "rough bf16 elem visits/token |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for item in sorted(
        categories.values(), key=lambda entry: (entry.macs, entry.calls), reverse=True
    ):
        print(
            f"| {item.name} | {_fmt_int(item.calls)} | "
            f"{item.calls / layers:.2f} | {_fmt_m(item.macs)} | "
            f"{_fmt_int(item.bf16_element_visits)} |"
        )

    if not show_kernels:
        return

    print()
    print(
        "| kernel | category | calls/token | calls/layer | est MACs/token | "
        "rough bf16 elem visits/token |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for item in sorted(kernels.values(), key=lambda entry: entry.macs, reverse=True):
        print(
            f"| `{item.name}` | {_classify_kernel(item.name)} | "
            f"{_fmt_int(item.calls)} | {item.calls / layers:.2f} | "
            f"{_fmt_m(item.macs)} | {_fmt_int(item.bf16_element_visits)} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate static work in a generated Qwen3 persistent MLIR graph"
    )
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--position", type=int, default=26)
    parser.add_argument(
        "--show-kernels",
        action="store_true",
        help="also print per-kernel rows after the category table",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kernels, categories = estimate_work(
        args.mlir,
        layers=args.layers,
        position=args.position,
    )
    print_markdown(
        mlir_path=args.mlir,
        layers=args.layers,
        position=args.position,
        kernels=kernels,
        categories=categories,
        show_kernels=args.show_kernels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
