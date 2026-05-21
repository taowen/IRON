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
class C1TwoPhaseWorker(MLIROperator):
    size: int = 256
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__init__(context=self.context)

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "c1_two_phase_worker",
                (aie_utils.get_current_device(), self.size),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "two_phase_worker.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "two_phase_worker.cc")
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]


def _make_inputs(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(size, dtype=torch.float32)
    phase0 = (torch.sin(idx * 0.03) * 0.5 + torch.cos(idx * 0.11) * 0.25).to(
        torch.bfloat16
    )
    phase1 = (torch.cos(idx * 0.05) * 0.75 - torch.sin(idx * 0.13) * 0.125).to(
        torch.bfloat16
    )
    return phase0.contiguous(), phase1.contiguous()


def _reference(phase0: torch.Tensor, phase1: torch.Tensor) -> torch.Tensor:
    state = (2.0 * phase0.to(torch.float32) + 1.0).to(torch.bfloat16)
    out = state.to(torch.float32) + 3.0 * phase1.to(torch.float32) + 7.0
    return out.to(torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1 two-phase same-Worker proof.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_c1_two_phase_worker"),
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = AIEContext(build_dir=args.build_dir)
    op = C1TwoPhaseWorker(size=args.size, context=context)
    op.compile()
    op_func = op.get_callable()

    phase0, phase1 = _make_inputs(args.size)
    phase0_buf = XRTTensor.from_torch(phase0)
    phase1_buf = XRTTensor.from_torch(phase1)
    out_buf = XRTTensor((args.size,), dtype=bfloat16)

    result = op_func(phase0_buf, phase1_buf, out_buf)
    npu_out = out_buf.to_torch().detach().clone()
    expected = _reference(phase0, phase1)
    diff = (npu_out.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs = float(diff.max())
    errors = int((diff > args.abs_tol).sum().item())

    print("experiment: C1 two-phase same-Worker protocol")
    print(f"build_dir: {args.build_dir}")
    print(f"size: {args.size}")
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print("dispatch_count: 1")
    print("worker_phase_order: phase0_then_phase1")
    print(f"npu_time_us: {result.npu_time / 1000.0:.3f}")
    print(f"max_abs: {max_abs:.6f}")
    print(f"errors: {errors}")

    if errors == 0:
        print("decision: accepted")
        print(
            "root_cause: one Worker can execute two FIFO-guarded phases in one "
            "dispatch when both phases have balanced acquire/release tokens."
        )
    else:
        print("decision: rejected")
        print(
            "root_cause: two-phase output did not match the CPU phase-order reference."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
