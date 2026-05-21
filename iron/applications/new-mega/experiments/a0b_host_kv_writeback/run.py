#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext


@dataclass
class HostKVWritebackBlob(MLIROperator):
    max_seq_len: int = 256
    head_dim: int = 128
    chunk_size: int = 64
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        super().__init__(context=self.context)

    @property
    def packed_chunk_elements(self) -> int:
        return 2 * self.chunk_size * self.head_dim + self.chunk_size

    @property
    def num_chunks(self) -> int:
        return self.max_seq_len // self.chunk_size

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "host_kv_writeback_blob",
                (
                    aie_utils.get_current_device(),
                    self.max_seq_len,
                    self.head_dim,
                    self.chunk_size,
                ),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "host_kv_writeback.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "host_kv_writeback.cc")
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.head_dim,)),
            AIERuntimeArgSpec("in", (self.num_chunks * self.packed_chunk_elements,)),
            AIERuntimeArgSpec("out", (3 * self.head_dim,)),
        ]


@dataclass
class StepResult:
    position: int
    present_k_max_abs: float
    present_v_max_abs: float
    summary_max_abs: float
    host_writeback_us: float


def _current_for_position(position: int, head_dim: int) -> torch.Tensor:
    dim = torch.arange(head_dim, dtype=torch.float32)
    current = torch.sin(dim * 0.07 + position * 0.11) * 0.25
    current += torch.cos(dim * 0.03 - position * 0.05) * 0.125
    return current.to(torch.bfloat16).contiguous()


def _packed_offsets(
    position: int, head_dim: int, chunk_size: int, packed_chunk_elements: int
) -> tuple[int, int, int]:
    chunk = position // chunk_size
    row = position % chunk_size
    base = chunk * packed_chunk_elements
    k_offset = base + row * head_dim
    v_offset = base + chunk_size * head_dim + row * head_dim
    mask_offset = base + 2 * chunk_size * head_dim + row
    return k_offset, v_offset, mask_offset


def _expected_present(current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    current_f = current.to(torch.float32)
    return (
        (current_f + 0.25).to(torch.bfloat16),
        (current_f - 0.50).to(torch.bfloat16),
    )


def _expected_summary(k_cache: torch.Tensor, v_cache: torch.Tensor, position: int):
    summary = torch.zeros(k_cache.shape[1], dtype=torch.float32)
    for row in range(position):
        summary += k_cache[row].to(torch.float32)
        summary += 0.5 * v_cache[row].to(torch.float32)
    return summary.to(torch.bfloat16)


def _run_step(
    op_func,
    packed_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: int,
    head_dim: int,
    chunk_size: int,
    packed_chunk_elements: int,
) -> StepResult:
    current = _current_for_position(position, head_dim)
    current_buf = XRTTensor.from_torch(current.flatten().contiguous())
    cache_buf = XRTTensor.from_torch(packed_cache.flatten().contiguous())
    out_buf = XRTTensor((3 * head_dim,), dtype=bfloat16)

    op_func(current_buf, cache_buf, out_buf)
    out = out_buf.to_torch().detach().clone().flatten()
    actual_present_k = out[:head_dim]
    actual_present_v = out[head_dim : 2 * head_dim]
    actual_summary = out[2 * head_dim :]

    expected_present_k, expected_present_v = _expected_present(current)
    expected_summary = _expected_summary(k_cache, v_cache, position)

    k_diff = (
        actual_present_k.to(torch.float32) - expected_present_k.to(torch.float32)
    ).abs()
    v_diff = (
        actual_present_v.to(torch.float32) - expected_present_v.to(torch.float32)
    ).abs()
    summary_diff = (
        actual_summary.to(torch.float32) - expected_summary.to(torch.float32)
    ).abs()

    k_offset, v_offset, mask_offset = _packed_offsets(
        position, head_dim, chunk_size, packed_chunk_elements
    )
    start_ns = time.perf_counter_ns()
    packed_cache[k_offset : k_offset + head_dim] = actual_present_k
    packed_cache[v_offset : v_offset + head_dim] = actual_present_v
    packed_cache[mask_offset] = torch.tensor(1.0, dtype=torch.bfloat16)
    k_cache[position] = actual_present_k
    v_cache[position] = actual_present_v
    host_writeback_us = (time.perf_counter_ns() - start_ns) / 1000.0

    return StepResult(
        position=position,
        present_k_max_abs=float(k_diff.max()),
        present_v_max_abs=float(v_diff.max()),
        summary_max_abs=float(summary_diff.max()),
        host_writeback_us=host_writeback_us,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A0B host-side KV-cache writeback experiment."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_a0b_host_kv_writeback"),
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--abs-tol", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.steps > args.max_seq_len:
        raise ValueError("steps must be inside [1, max_seq_len]")

    context = AIEContext(build_dir=args.build_dir)
    op = HostKVWritebackBlob(
        max_seq_len=args.max_seq_len,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    op.compile()
    op_func = op.get_callable()

    packed_cache = torch.zeros(
        op.num_chunks * op.packed_chunk_elements, dtype=torch.bfloat16
    )
    k_cache = torch.zeros(args.max_seq_len, args.head_dim, dtype=torch.bfloat16)
    v_cache = torch.zeros(args.max_seq_len, args.head_dim, dtype=torch.bfloat16)

    results = [
        _run_step(
            op_func,
            packed_cache,
            k_cache,
            v_cache,
            position,
            args.head_dim,
            args.chunk_size,
            op.packed_chunk_elements,
        )
        for position in range(args.steps)
    ]

    max_k = max(result.present_k_max_abs for result in results)
    max_v = max(result.present_v_max_abs for result in results)
    max_summary = max(result.summary_max_abs for result in results)
    max_host_writeback_us = max(result.host_writeback_us for result in results)
    selected_positions = {0, 1, 2, args.chunk_size - 1, args.chunk_size, args.steps - 1}

    print("experiment: A0B host-side KV writeback")
    print(f"build_dir: {args.build_dir}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"head_dim: {args.head_dim}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"num_chunks: {op.num_chunks}")
    print(f"steps: {args.steps}")
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print("artifact_reused_for_all_steps: True")
    print("npu_writes_kv_cache: False")
    print(f"host_writeback_bytes_per_step: {2 * args.head_dim * 2}")
    for result in results:
        if result.position in selected_positions:
            print(
                "step_result: "
                f"position={result.position} "
                f"present_k_max_abs={result.present_k_max_abs:.6f} "
                f"present_v_max_abs={result.present_v_max_abs:.6f} "
                f"summary_max_abs={result.summary_max_abs:.6f} "
                f"host_writeback_us={result.host_writeback_us:.3f}"
            )
    print(f"max_present_k_abs: {max_k:.6f}")
    print(f"max_present_v_abs: {max_v:.6f}")
    print(f"max_summary_abs: {max_summary:.6f}")
    print(f"max_host_writeback_us: {max_host_writeback_us:.3f}")

    if max_k <= args.abs_tol and max_v <= args.abs_tol and max_summary <= args.abs_tol:
        print("decision: accepted")
        print(
            "root_cause: current-token KV cache writeback does not need an NPU "
            "dynamic offset. The blob can output fixed present K/V tensors, and "
            "the host can memcpy them into the persistent cache between dispatches."
        )
    else:
        print("decision: rejected")
        print(
            "root_cause: fixed-output present K/V or next-step cache readback mismatched."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
