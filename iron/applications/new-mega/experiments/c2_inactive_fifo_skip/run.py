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
class C2InactiveFifoSkip(MLIROperator):
    size: int = 256
    use_optional_phase: bool = True
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__init__(context=self.context)

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "c2_inactive_fifo_skip",
                (
                    aie_utils.get_current_device(),
                    self.size,
                    self.use_optional_phase,
                ),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "inactive_fifo_skip.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "inactive_fifo_skip.cc")
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        specs = [AIERuntimeArgSpec("in", (self.size,))]
        if self.use_optional_phase:
            specs.append(AIERuntimeArgSpec("in", (self.size,)))
        specs.append(AIERuntimeArgSpec("out", (self.size,)))
        return specs


@dataclass
class VariantResult:
    name: str
    use_optional_phase: bool
    arg_count: int
    max_abs: float
    errors: int
    mlir_path: Path
    bin_path: Path
    xclbin_path: Path


def _byte_diff_count(left: Path, right: Path) -> int | str:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    if len(left_bytes) != len(right_bytes):
        return f"size-mismatch:{len(left_bytes)}->{len(right_bytes)}"
    return sum(a != b for a, b in zip(left_bytes, right_bytes))


def _make_inputs(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(size, dtype=torch.float32)
    phase0 = (torch.sin(idx * 0.03) * 0.5 + torch.cos(idx * 0.11) * 0.25).to(
        torch.bfloat16
    )
    optional = (torch.cos(idx * 0.05) * 0.75 - torch.sin(idx * 0.13) * 0.125).to(
        torch.bfloat16
    )
    return phase0.contiguous(), optional.contiguous()


def _reference(
    phase0: torch.Tensor, optional: torch.Tensor, use_optional_phase: bool
) -> torch.Tensor:
    state = (2.0 * phase0.to(torch.float32) + 1.0).to(torch.bfloat16)
    out = state.to(torch.float32) + 7.0
    if use_optional_phase:
        out += 3.0 * optional.to(torch.float32)
    return out.to(torch.bfloat16)


def _run_variant(
    *,
    context: AIEContext,
    name: str,
    size: int,
    use_optional_phase: bool,
    abs_tol: float,
) -> VariantResult:
    op = C2InactiveFifoSkip(
        size=size,
        use_optional_phase=use_optional_phase,
        context=context,
    )
    op.compile()
    op_func = op.get_callable()
    phase0, optional = _make_inputs(size)
    out_buf = XRTTensor((size,), dtype=bfloat16)

    if use_optional_phase:
        op_func(XRTTensor.from_torch(phase0), XRTTensor.from_torch(optional), out_buf)
    else:
        op_func(XRTTensor.from_torch(phase0), out_buf)

    npu_out = out_buf.to_torch().detach().clone()
    expected = _reference(phase0, optional, use_optional_phase)
    diff = (npu_out.to(torch.float32) - expected.to(torch.float32)).abs()

    aie_utils.DefaultNPURuntime.cleanup()
    return VariantResult(
        name=name,
        use_optional_phase=use_optional_phase,
        arg_count=len(op.get_arg_spec()),
        max_abs=float(diff.max()),
        errors=int((diff > abs_tol).sum().item()),
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        bin_path=Path(op.insts_artifact.filename),
        xclbin_path=Path(op.xclbin_artifact.filename),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C2 inactive FIFO / phase skip protocol proof."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_c2_inactive_fifo_skip"),
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = AIEContext(build_dir=args.build_dir)
    active = _run_variant(
        context=context,
        name="active",
        size=args.size,
        use_optional_phase=True,
        abs_tol=args.abs_tol,
    )
    skip = _run_variant(
        context=context,
        name="skip",
        size=args.size,
        use_optional_phase=False,
        abs_tol=args.abs_tol,
    )

    mlir_diff = _byte_diff_count(active.mlir_path, skip.mlir_path)
    bin_diff = _byte_diff_count(active.bin_path, skip.bin_path)
    xclbin_diff = _byte_diff_count(active.xclbin_path, skip.xclbin_path)
    same_artifact = mlir_diff == 0 and bin_diff == 0 and xclbin_diff == 0
    all_match = active.errors == 0 and skip.errors == 0

    print("experiment: C2 inactive FIFO / phase skip protocol")
    print(f"build_dir: {args.build_dir}")
    print(f"size: {args.size}")
    for result in [active, skip]:
        print(
            "variant_result: "
            f"name={result.name} "
            f"use_optional_phase={result.use_optional_phase} "
            f"arg_count={result.arg_count} "
            f"max_abs={result.max_abs:.6f} "
            f"errors={result.errors}"
        )
    print(f"mlir_diff_bytes: {mlir_diff}")
    print(f"runtime_bin_diff_bytes: {bin_diff}")
    print(f"xclbin_diff_bytes: {xclbin_diff}")
    print(f"same_artifact: {same_artifact}")
    print("skip_uses_dummy_optional_fifo: False")

    if all_match and not same_artifact:
        print("decision: partially-accepted")
        print(
            "root_cause: inactive FIFO tokens can be removed only by compiling a "
            "different static graph/ABI. This avoids dummy DMA, but it does not "
            "prove same-artifact dynamic phase skipping."
        )
    elif all_match:
        print("decision: accepted")
    else:
        print("decision: rejected")
        print("root_cause: active or skip variant failed numerical verification.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
