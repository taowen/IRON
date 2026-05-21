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

from iron.applications.qwen3_0_6b.qwen3_preflight import (
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)
from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext

PHASE_LABELS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "attn_chunk_0",
    "attn_chunk_1",
    "attn_chunk_2",
    "attn_chunk_3",
)


@dataclass
class D0StaticPhaseOwnership(MLIROperator):
    num_lanes: int = 8
    num_phase_packets: int = len(PHASE_LABELS)
    packet_elements: int = 16896
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__init__(context=self.context)

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "d0_static_phase_ownership",
                (
                    aie_utils.get_current_device(),
                    self.num_lanes,
                    self.num_phase_packets,
                    self.packet_elements,
                ),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "static_phase_skeleton.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "static_phase_skeleton.cc")
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec(
                "in",
                (self.num_lanes * self.num_phase_packets * self.packet_elements,),
            ),
            AIERuntimeArgSpec("out", (self.num_lanes * 8,)),
        ]


def _make_packets(
    num_lanes: int,
    num_phase_packets: int,
    packet_elements: int,
) -> torch.Tensor:
    total = num_lanes * num_phase_packets * packet_elements
    values = torch.arange(total, dtype=torch.float32)
    values = (torch.remainder(values, 17) - 8.0) / 4096.0
    return values.to(torch.bfloat16).contiguous()


def _reference(
    packets: torch.Tensor,
    num_lanes: int,
    num_phase_packets: int,
    packet_elements: int,
) -> torch.Tensor:
    lane_span = num_phase_packets * packet_elements
    expected = torch.empty((num_lanes,), dtype=torch.float32)
    packets_f32 = packets.to(torch.float32)
    for lane in range(num_lanes):
        acc = torch.zeros((), dtype=torch.float32)
        start = lane * lane_span
        for phase in range(num_phase_packets):
            phase_start = start + phase * packet_elements
            phase_end = phase_start + packet_elements
            acc += packets_f32[phase_start:phase_end].sum(dtype=torch.float32)
        expected[lane] = acc
    return expected.to(torch.bfloat16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D0 static phase ownership resource skeleton."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_d0_static_phase_ownership"),
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--num-phase-packets", type=int, default=len(PHASE_LABELS))
    parser.add_argument(
        "--packet-elements",
        type=int,
        default=16896,
        help="BF16 elements per lane-local phase packet.",
    )
    parser.add_argument("--abs-tol", type=float, default=0.25)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def _print_preflight(result) -> None:
    print(f"preflight_runtime_memrefs: {result.runtime_memrefs}")
    print(f"preflight_arg_specs: {result.arg_specs}")
    print(f"preflight_metadata_host_bos: {result.metadata_host_bos}")
    print(f"preflight_compute_cores: {result.compute_cores}")
    print(f"preflight_max_fifo_buffered_bytes: {result.max_fifo_buffered_bytes}")
    print(f"preflight_total_dma_tasks: {result.total_dma_tasks}")
    print(f"preflight_max_dma_tasks_per_fifo: {result.max_dma_tasks_per_fifo}")
    print(f"preflight_max_compute_tile_inputs: {result.max_compute_tile_inputs}")
    print(f"preflight_max_compute_tile_outputs: {result.max_compute_tile_outputs}")
    print(f"preflight_non_advancing_acquires: {result.non_advancing_acquires}")


def main() -> None:
    args = parse_args()
    context = AIEContext(build_dir=args.build_dir)
    op = D0StaticPhaseOwnership(
        num_lanes=args.num_lanes,
        num_phase_packets=args.num_phase_packets,
        packet_elements=args.packet_elements,
        context=context,
    )
    op.compile()
    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)

    try:
        preflight = run_persistent_artifact_preflight(
            mlir_path=mlir_path,
            arg_specs=len(op.get_arg_spec()),
        )
    except Qwen3PreflightError as err:
        print("experiment: D0 static phase ownership skeleton")
        print(f"build_dir: {args.build_dir}")
        print(f"mlir: {mlir_path}")
        print("decision: rejected")
        print(f"root_cause: preflight failed: {err}")
        raise SystemExit(1) from err

    packet_bytes = args.packet_elements * 2
    lane_packet_bytes = args.num_phase_packets * packet_bytes
    total_input_bytes = args.num_lanes * lane_packet_bytes

    print("experiment: D0 static phase ownership skeleton")
    print(f"build_dir: {args.build_dir}")
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print(f"mlir: {mlir_path}")
    print(f"num_lanes: {args.num_lanes}")
    print(f"num_phase_packets: {args.num_phase_packets}")
    print(f"phase_labels: {','.join(PHASE_LABELS[: args.num_phase_packets])}")
    print(f"packet_elements: {args.packet_elements}")
    print(f"packet_bytes: {packet_bytes}")
    print(f"lane_packet_stream_bytes: {lane_packet_bytes}")
    print(f"total_input_bytes: {total_input_bytes}")
    print(f"packet_fits_l1_64k: {packet_bytes <= 64 * 1024}")
    _print_preflight(preflight)

    if args.skip_run:
        print("decision: compile-only")
        return

    packets = _make_packets(
        args.num_lanes,
        args.num_phase_packets,
        args.packet_elements,
    )
    out_buf = XRTTensor((args.num_lanes * 8,), dtype=bfloat16)
    result = op.get_callable()(XRTTensor.from_torch(packets), out_buf)

    npu_out = out_buf.to_torch().detach().clone()[0::8]
    expected = _reference(
        packets,
        args.num_lanes,
        args.num_phase_packets,
        args.packet_elements,
    )
    diff = (npu_out.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs = float(diff.max())
    errors = int((diff > args.abs_tol).sum().item())

    print(f"npu_time_us: {result.npu_time / 1000.0:.3f}")
    print(f"max_abs: {max_abs:.6f}")
    print(f"errors: {errors}")
    print("resource_implication: one packed input FIFO and one output FIFO per lane")
    print(
        "utilization_implication: resource-safe skeleton only; math parallelism unproven"
    )

    if errors == 0:
        print("decision: accepted")
        print(
            "root_cause: lane-local packed phase streams pass compile, preflight, "
            "and NPU execution with bounded input/output FIFO endpoints."
        )
    else:
        print("decision: rejected")
        print("root_cause: NPU accumulation output did not match the CPU reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
