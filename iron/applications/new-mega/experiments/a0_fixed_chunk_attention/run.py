#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
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
class FixedChunkAttention(MLIROperator):
    max_seq_len: int = 256
    head_dim: int = 128
    chunk_size: int = 64
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if self.head_dim % 32 != 0:
            raise ValueError("head_dim must be a multiple of 32")
        super().__init__(context=self.context)

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "fixed_chunk_attention",
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
                "chunked_attention.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "chunked_attention.cc")
                ],
                extra_flags=[
                    f"-DFIXED_CHUNK_HEAD_DIM={self.head_dim}",
                    f"-DFIXED_CHUNK_SIZE={self.chunk_size}",
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        packed_chunk_elements = 2 * self.chunk_size * self.head_dim + self.chunk_size
        num_chunks = self.max_seq_len // self.chunk_size
        return [
            AIERuntimeArgSpec("in", (self.head_dim,)),
            AIERuntimeArgSpec("in", (num_chunks * packed_chunk_elements,)),
            AIERuntimeArgSpec("out", (self.head_dim,)),
        ]


@dataclass
class PositionResult:
    position: int
    valid_chunks: int
    max_abs: float
    mean_abs: float
    cpu_match: bool


def _make_inputs(max_seq_len: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    q_idx = torch.arange(head_dim, dtype=torch.float32)
    q = (torch.sin(q_idx * 0.13) * 0.25 + torch.cos(q_idx * 0.07) * 0.05).to(
        torch.bfloat16
    )

    kv_idx = torch.arange(max_seq_len * head_dim, dtype=torch.float32).reshape(
        max_seq_len, head_dim
    )
    rows = torch.arange(max_seq_len, dtype=torch.float32).reshape(max_seq_len, 1)
    cols = torch.arange(head_dim, dtype=torch.float32).reshape(1, head_dim)
    k = torch.sin(kv_idx * 0.017) * 0.18 + torch.cos(rows * 0.11 + cols * 0.03) * 0.04
    v = torch.cos(kv_idx * 0.013) * 0.45 + torch.sin(rows * 0.19) * 0.05
    return (
        q.contiguous(),
        k.to(torch.bfloat16).contiguous(),
        v.to(torch.bfloat16).contiguous(),
    )


def _make_mask(max_seq_len: int, position: int) -> torch.Tensor:
    mask = torch.zeros(max_seq_len, dtype=torch.float32)
    mask[: position + 1] = 1.0
    return mask.to(torch.bfloat16).contiguous()


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32).reshape(mask.numel(), q.numel())
    v_f = v.to(torch.float32).reshape(mask.numel(), q.numel())
    scores = (k_f @ q_f) * (q.numel() ** -0.5)
    scores = scores.masked_fill(mask.to(torch.float32) <= 0.5, -torch.inf)
    weights = torch.softmax(scores, dim=0)
    return (weights @ v_f).to(torch.bfloat16)


def _pack_chunks(
    k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    max_seq_len, head_dim = k.shape
    chunks: list[torch.Tensor] = []
    for start in range(0, max_seq_len, chunk_size):
        end = start + chunk_size
        chunks.append(
            torch.cat(
                [
                    k[start:end].flatten(),
                    v[start:end].flatten(),
                    mask[start:end].flatten(),
                ]
            )
        )
    return torch.cat(chunks).to(torch.bfloat16).contiguous()


def _run_position(
    op_func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: int,
    max_seq_len: int,
    chunk_size: int,
    abs_tol: float,
) -> PositionResult:
    mask = _make_mask(max_seq_len, position)
    packed = _pack_chunks(k, v, mask, chunk_size)
    q_buf = XRTTensor.from_torch(q.flatten().contiguous())
    packed_buf = XRTTensor.from_torch(packed)
    out_buf = XRTTensor((q.numel(),), dtype=bfloat16)

    op_func(q_buf, packed_buf, out_buf)
    npu_output = out_buf.to_torch().detach().clone().flatten()
    cpu_output = _reference(q, k, v, mask).flatten()
    diff = (npu_output.to(torch.float32) - cpu_output.to(torch.float32)).abs()

    return PositionResult(
        position=position,
        valid_chunks=(position + chunk_size) // chunk_size,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        cpu_match=bool(float(diff.max()) <= abs_tol),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A0 fixed TAP chunked decode-attention experiment."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_a0_fixed_chunk_attention"),
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--abs-tol", type=float, default=0.06)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[0, 26, 63, 64, 127, 200, 255],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(pos < 0 or pos >= args.max_seq_len for pos in args.positions):
        raise ValueError("positions must be inside [0, max_seq_len)")

    context = AIEContext(build_dir=args.build_dir)
    op = FixedChunkAttention(
        max_seq_len=args.max_seq_len,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    op.compile()
    op_func = op.get_callable()
    q, k, v = _make_inputs(args.max_seq_len, args.head_dim)

    results = [
        _run_position(
            op_func,
            q,
            k,
            v,
            position,
            args.max_seq_len,
            args.chunk_size,
            args.abs_tol,
        )
        for position in args.positions
    ]

    all_match = all(result.cpu_match for result in results)

    print("experiment: A0 fixed chunk decode attention")
    print(f"build_dir: {args.build_dir}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"head_dim: {args.head_dim}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"num_chunks: {args.max_seq_len // args.chunk_size}")
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print("artifact_reused_for_all_positions: True")
    for result in results:
        print(
            "position_result: "
            f"position={result.position} "
            f"valid_chunks={result.valid_chunks} "
            f"max_abs={result.max_abs:.6f} "
            f"mean_abs={result.mean_abs:.6f} "
            f"cpu_match={result.cpu_match}"
        )

    if all_match:
        print("decision: accepted")
        print(
            "root_cause: position-specific attention artifacts are avoidable when "
            "KV/mask movement uses fixed max-sequence TAPs and the live position is "
            "encoded only in runtime mask data."
        )
    else:
        print("decision: rejected")
        print("root_cause: NPU output did not match the fixed-chunk CPU reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
