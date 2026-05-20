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

from iron.applications.qwen3_0_6b.persistent.ops import (  # noqa: E402
    Qwen3PersistentInputRMSNormQKV,
    Qwen3PersistentNLayerFinalOnly,
    Qwen3PersistentPostAttnMLPDownResidual,
    Qwen3PersistentPostAttnRMSNormFullMLP,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
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
