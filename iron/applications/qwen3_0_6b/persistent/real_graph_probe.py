#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile/preflight probes for real Qwen3 persistent graph stages."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

from iron.applications.qwen3_0_6b.persistent.ops_core import (  # noqa: E402
    Qwen3PersistentInputRMSNormQKV,
)
from iron.applications.qwen3_0_6b.persistent.ops_mlp import (  # noqa: E402
    Qwen3PersistentPostAttnMLPDownResidual,
    Qwen3PersistentPostAttnRMSNormFullMLP,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
)
from iron.applications.qwen3_0_6b.persistent.ops_nlayer import (  # noqa: E402
    Qwen3PersistentNLayerFinalOnly,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext  # noqa: E402

STAGES = {
    "qkv",
    "mlp-gate-up",
    "mlp-down",
    "full-mlp",
    "n-layer-final-only",
}

_PLACEMENT_TRACE_COUNTS = None


def _reset_placement_trace():
    """Install a small SequentialPlacer endpoint trace and reset counters."""
    global _PLACEMENT_TRACE_COUNTS
    _PLACEMENT_TRACE_COUNTS = {
        "runtime_output": 0,
        "runtime_input": 0,
        "other_output": 0,
        "other_input": 0,
    }

    from aie.iron.placers import SequentialPlacer  # noqa: PLC0415
    from aie.iron.runtime.endpoint import RuntimeEndpoint  # noqa: PLC0415

    if getattr(SequentialPlacer, "_qwen3_trace_installed", False):
        return _PLACEMENT_TRACE_COUNTS

    original_place_endpoint = SequentialPlacer._place_endpoint

    def traced_place_endpoint(
        self,
        ofe,
        tiles,
        common_col,
        channels,
        device,
        output=False,
        link_tiles=[],
        link_channels={},
    ):
        if isinstance(ofe, RuntimeEndpoint):
            key = "runtime_output" if output else "runtime_input"
        else:
            key = "other_output" if output else "other_input"
        try:
            result = original_place_endpoint(
                self,
                ofe,
                tiles,
                common_col,
                channels,
                device,
                output,
                link_tiles,
                link_channels,
            )
            if _PLACEMENT_TRACE_COUNTS is not None:
                _PLACEMENT_TRACE_COUNTS[key] += 1
            return result
        except Exception as exc:
            print(f"placement_trace_fail_counts: {_PLACEMENT_TRACE_COUNTS}")
            print(f"placement_trace_fail_key: {key}")
            print(f"placement_trace_fail_type: {type(ofe).__name__}")
            print(f"placement_trace_fail_repr: {ofe!r}")
            print(f"placement_trace_fail_output: {output}")
            print(f"placement_trace_fail_common_col: {common_col}")
            print(f"placement_trace_fail_remaining_tiles: {tiles}")
            print(
                "placement_trace_fail_channels_used: "
                f"{ {str(k): sum(c for _, c in v) for k, v in channels.items()} }"
            )
            print(f"placement_trace_fail_exception: {type(exc).__name__}: {exc}")
            raise

    SequentialPlacer._qwen3_original_place_endpoint = original_place_endpoint
    SequentialPlacer._place_endpoint = traced_place_endpoint
    SequentialPlacer._qwen3_trace_installed = True
    return _PLACEMENT_TRACE_COUNTS


def _stage_kwargs(args, stage: str, columns: int) -> tuple[type, dict]:
    common_attention = dict(
        hidden_size=args.hidden_size,
        q_size=args.q_size,
        kv_size=args.kv_size,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        position=args.position,
        num_aie_columns=columns,
        tile_size_input=args.tile_size_input,
        tile_size_output=args.head_dim,
    )
    if stage == "qkv":
        return (
            Qwen3PersistentInputRMSNormQKV,
            dict(
                hidden_size=args.hidden_size,
                q_size=args.q_size,
                kv_size=args.kv_size,
                num_aie_columns=columns,
                tile_size_input=args.tile_size_input,
                tile_size_output=args.qkv_tile_size_output,
                epsilon=args.epsilon,
            ),
        )
    if stage == "mlp-gate-up":
        return (
            Qwen3PersistentPostAttnRMSNormMLPGateUp,
            dict(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_aie_columns=columns,
                tile_size_input=args.tile_size_input,
                tile_size_output=args.mlp_gate_up_tile_size_output,
                epsilon=args.epsilon,
            ),
        )
    if stage == "mlp-down":
        return (
            Qwen3PersistentPostAttnMLPDownResidual,
            dict(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_aie_columns=columns,
                tile_size_input=args.tile_size_input,
                tile_size_output=args.mlp_down_tile_size_output,
            ),
        )
    if stage == "full-mlp":
        return (
            Qwen3PersistentPostAttnRMSNormFullMLP,
            dict(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_aie_columns=columns,
                tile_size_input=args.tile_size_input,
                tile_size_output=args.mlp_down_tile_size_output,
                epsilon=args.epsilon,
            ),
        )
    if stage == "n-layer-final-only":
        return (
            Qwen3PersistentNLayerFinalOnly,
            dict(
                **common_attention,
                intermediate_size=args.intermediate_size,
                layer_iterations=args.layer_iterations,
                attention_columns=args.attention_columns,
                mlp_gate_up_columns=args.mlp_gate_up_columns,
                mlp_gate_up_pair_rows=args.mlp_gate_up_pair_rows,
                mlp_gate_up_direct_silu=args.mlp_gate_up_direct_silu,
                mlp_gate_up_row_group=args.mlp_gate_up_row_group,
                attention_probe_only=args.attention_probe_only,
            ),
        )
    raise ValueError(f"unknown stage: {stage}")


def _write_mlir_without_aiecc(op, mlir_path: Path):
    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    mlir_path.write_text(str(op.get_mlir_artifact().generator()))


def compile_probe(args, stage: str, columns: int) -> bool:
    build_dir = args.build_dir / f"{stage.replace('-', '_')}_cols{columns}"
    if stage == "n-layer-final-only":
        build_dir = build_dir.with_name(
            f"{build_dir.name}_layers{args.layer_iterations}"
        )
    if args.clean_build and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    phase = "preflight" if args.preflight_only else "compile"
    placement_counts = _reset_placement_trace() if args.trace_placement else None
    try:
        cls, kwargs = _stage_kwargs(args, stage, columns)
        context = AIEContext(build_dir=str(build_dir))
        op = cls(context=context, **kwargs)
        mlir_path = build_dir / op.get_mlir_artifact().filename
        if args.preflight_only:
            _write_mlir_without_aiecc(op, mlir_path)
        else:
            op.compile()
            mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
        preflight = run_persistent_artifact_preflight(
            mlir_path=mlir_path,
            arg_specs=len(op.get_arg_spec()),
            max_dma_tasks_per_fifo=args.max_dma_tasks_per_fifo,
            max_compute_cores=args.max_compute_cores,
        )
        elapsed = time.perf_counter() - start
        print(
            "real_graph_probe: ok "
            f"stage={stage} cols={columns} layers={args.layer_iterations} "
            f"phase={phase} elapsed_s={elapsed:.3f} "
            f"compute_cores={preflight.compute_cores} "
            f"total_dma_tasks={preflight.total_dma_tasks} "
            f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
            f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
            f"max_tile_inputs={preflight.max_compute_tile_inputs} "
            f"max_tile_outputs={preflight.max_compute_tile_outputs}"
        )
        if placement_counts is not None:
            print(f"placement_trace_counts: {placement_counts}")
        return True
    except (RuntimeError, Qwen3PreflightError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        print(
            "real_graph_probe: fail "
            f"stage={stage} cols={columns} layers={args.layer_iterations} "
            f"phase={phase} elapsed_s={elapsed:.3f} "
            f"error={type(exc).__name__}: {exc}"
        )
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile/preflight real Qwen3 persistent graph stages"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["qkv", "mlp-gate-up", "n-layer-final-only"],
        choices=sorted(STAGES),
    )
    parser.add_argument("--columns", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--attention-columns",
        type=int,
        default=1,
        help=(
            "Attention head-shard columns for n-layer-final-only probes. "
            "The accepted path is currently 1; value 2 is an early-fail probe "
            "until the attention2 layout is implemented."
        ),
    )
    parser.add_argument(
        "--attention-probe-only",
        action="store_true",
        help=(
            "For n-layer-final-only, stop after attention/O-proj/residual and "
            "skip MLP workers/fills. This is a resource-isolated attention2 "
            "bring-up probe."
        ),
    )
    parser.add_argument(
        "--mlp-gate-up-columns",
        type=int,
        default=0,
        help=(
            "For n-layer-final-only, override gate/up MLP columns. 0 keeps the "
            "operator default; 1 with --columns 2 probes down-only MLP2; 3 "
            "runs the experimental three-way gate/up branch."
        ),
    )
    parser.add_argument(
        "--mlp-gate-up-pair-rows",
        action="store_true",
        help=(
            "For n-layer-final-only MLP2 probes, expect gate/up rows packed as "
            "4 gate rows followed by their 4 matching up rows. This validates "
            "the boundary needed by a future fused gate+up kernel."
        ),
    )
    parser.add_argument(
        "--mlp-gate-up-direct-silu",
        action="store_true",
        help=(
            "For n-layer-final-only MLP2 probes, use the paired-row kernel "
            "variant that writes SiLU(gate)*up directly. Requires "
            "--mlp-gate-up-pair-rows."
        ),
    )
    parser.add_argument(
        "--mlp-gate-up-row-group",
        type=int,
        default=4,
        choices=(4, 8),
        help="Paired direct-SiLU gate/up row group for n-layer-final-only probes.",
    )
    parser.add_argument("--layer-iterations", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--q-size", type=int, default=2048)
    parser.add_argument("--kv-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--position", type=int, default=6)
    parser.add_argument("--tile-size-input", type=int, default=4)
    parser.add_argument("--qkv-tile-size-output", type=int, default=128)
    parser.add_argument("--mlp-gate-up-tile-size-output", type=int, default=384)
    parser.add_argument("--mlp-down-tile-size-output", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--max-dma-tasks-per-fifo",
        type=int,
        default=8,
        help="preflight limit from the Qwen3 persistent graph BD allocator probe",
    )
    parser.add_argument("--max-compute-cores", type=int, default=64)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="generate MLIR and run static preflight without invoking aiecc",
    )
    parser.add_argument(
        "--trace-placement",
        action="store_true",
        help=(
            "Trace SequentialPlacer endpoint counts and print endpoint details "
            "when placement fails."
        ),
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="return exit code 0 for exploratory matrices with expected failures",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_qwen3_real_graph_probe"),
    )
    parser.add_argument("--clean-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    failed = False
    for stage in args.stages:
        for columns in args.columns:
            failed = not compile_probe(args, stage, columns) or failed
    if args.allow_failures:
        failed = False
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
