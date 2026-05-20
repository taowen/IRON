#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic IRON persistent-graph scaling probes.

This intentionally avoids model weights and external kernels.  It isolates the
Runtime/TAP/ObjectFifo resource shape that matters for large decode graphs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
from ml_dtypes import bfloat16

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

from aie.dialects.aiex import TensorAccessPattern  # noqa: E402
from aie.iron import ObjectFifo, Program, Runtime  # noqa: E402
from aie.iron.placers import SequentialPlacer  # noqa: E402

from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)
from iron.common import (  # noqa: E402
    AIERuntimeArgSpec,
    DesignGenerator,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)
from iron.common.context import AIEContext  # noqa: E402

PATTERNS = {"separate", "repeat", "grouped"}


def layer_groups(layers: int, group_layers: int) -> list[tuple[int, int]]:
    if layers < 1:
        raise ValueError(f"layers must be positive, got {layers}")
    if group_layers < 1:
        raise ValueError(f"group_layers must be positive, got {group_layers}")
    return [
        (start, min(group_layers, layers - start))
        for start in range(0, layers, group_layers)
    ]


def _cache_tap(
    total_numel: int,
    cache_size: int,
    max_seq_len: int,
    kv_size: int,
    head_dim: int,
    kv_heads: int,
    position: int,
    layer_start: int,
    layer_count: int,
    *,
    values: bool,
):
    base = layer_start * cache_size + position * head_dim
    if values:
        base += kv_size * max_seq_len
    return TensorAccessPattern(
        (total_numel,),
        base,
        [layer_count, 1, kv_heads, head_dim],
        [cache_size, 0, max_seq_len * head_dim, 1],
    )


def persistent_graph_probe_design(
    dev,
    layers: int,
    kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    position: int,
    pattern: str,
    group_layers: int,
):
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, got {pattern}")
    if not (0 <= position < max_seq_len):
        raise ValueError(f"position must be in [0, {max_seq_len}), got {position}")

    dtype = bfloat16
    kv_size = kv_heads * head_dim
    cache_size = 2 * kv_size * max_seq_len
    total_numel = layers * cache_size

    cache_ty = np.ndarray[(total_numel,), np.dtype[dtype]]
    head_ty = np.ndarray[(head_dim,), np.dtype[dtype]]

    in_fifo = ObjectFifo(head_ty, name="probe_current_kv_in", depth=2)
    out_fifo = in_fifo.cons().forward(name="probe_current_kv_out", depth=2)

    def emit_transfer(rt, inp, out, layer_start, layer_count, tg):
        input_tap = _cache_tap(
            total_numel,
            cache_size,
            max_seq_len,
            kv_size,
            head_dim,
            kv_heads,
            position,
            layer_start,
            layer_count,
            values=False,
        )
        output_tap = _cache_tap(
            total_numel,
            cache_size,
            max_seq_len,
            kv_size,
            head_dim,
            kv_heads,
            position,
            layer_start,
            layer_count,
            values=True,
        )
        rt.fill(in_fifo.prod(), inp, input_tap, task_group=tg)
        rt.drain(out_fifo.cons(), out, output_tap, wait=True, task_group=tg)

    rt = Runtime()
    with rt.sequence(cache_ty, cache_ty) as (inp, out):
        tg = rt.task_group()
        if pattern == "separate":
            for layer_idx in range(layers):
                emit_transfer(rt, inp, out, layer_idx, 1, tg)
        elif pattern == "repeat":
            emit_transfer(rt, inp, out, 0, layers, tg)
        else:
            for layer_start, layer_count in layer_groups(layers, group_layers):
                emit_transfer(rt, inp, out, layer_start, layer_count, tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


class PersistentGraphProbe(MLIROperator):
    """Compile-only synthetic probe for persistent graph resource scaling."""

    def __init__(
        self,
        *,
        layers: int = 1,
        kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 256,
        position: int = 6,
        pattern: str = "separate",
        group_layers: int = 4,
        context=None,
    ):
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.position = position
        self.pattern = pattern
        self.group_layers = group_layers
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(PATTERNS)}")
        MLIROperator.__init__(self, context=context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device()
        return (
            "PersistentGraphProbe_"
            f"layers{self.layers}_kvh{self.kv_heads}_hd{self.head_dim}_"
            f"msl{self.max_seq_len}_pos{self.position}_"
            f"pattern{self.pattern}_group{self.group_layers}_"
            f"{dev.resolve().name}"
        )

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                Path(__file__),
                "persistent_graph_probe_design",
                (
                    aie_utils.get_current_device(),
                    self.layers,
                    self.kv_heads,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.pattern,
                    self.group_layers,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def get_arg_spec(self):
        kv_size = self.kv_heads * self.head_dim
        cache_size = 2 * kv_size * self.max_seq_len
        return [
            AIERuntimeArgSpec("in", (self.layers * cache_size,)),
            AIERuntimeArgSpec("out", (self.layers * cache_size,)),
        ]


def compile_probe(args, layers: int, pattern: str):
    build_dir = args.build_dir / f"{pattern}_layers{layers}_group{args.group_layers}"
    if args.clean_build and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    context = AIEContext(build_dir=str(build_dir))
    op = PersistentGraphProbe(
        layers=layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        position=args.position,
        pattern=pattern,
        group_layers=args.group_layers,
        context=context,
    )
    start = time.perf_counter()
    mlir_path = build_dir / op.get_mlir_artifact().filename
    try:
        mlir_path.write_text(str(op.get_mlir_artifact().generator()))
        preflight = run_persistent_artifact_preflight(
            mlir_path=mlir_path,
            arg_specs=len(op.get_arg_spec()),
            max_dma_tasks_per_fifo=args.max_dma_tasks_per_fifo,
        )
        if args.preflight_only:
            compile_s = time.perf_counter() - start
            print(
                "probe_result: ok "
                f"pattern={pattern} layers={layers} group_layers={args.group_layers} "
                f"phase=preflight compile_s={compile_s:.3f} "
                f"runtime_memrefs={preflight.runtime_memrefs} "
                f"compute_cores={preflight.compute_cores} "
                f"total_dma_tasks={preflight.total_dma_tasks} "
                f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
                f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes}"
            )
            return True
        op.compile()
        compile_s = time.perf_counter() - start
        preflight = run_persistent_artifact_preflight(
            mlir_path=mlir_path,
            arg_specs=len(op.get_arg_spec()),
            max_dma_tasks_per_fifo=args.max_dma_tasks_per_fifo,
        )
        print(
            "probe_result: ok "
            f"pattern={pattern} layers={layers} group_layers={args.group_layers} "
            f"compile_s={compile_s:.3f} runtime_memrefs={preflight.runtime_memrefs} "
            f"compute_cores={preflight.compute_cores} "
            f"total_dma_tasks={preflight.total_dma_tasks} "
            f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
            f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes}"
        )
        return True
    except (RuntimeError, Qwen3PreflightError, ValueError) as exc:
        compile_s = time.perf_counter() - start
        print(
            "probe_result: fail "
            f"pattern={pattern} layers={layers} group_layers={args.group_layers} "
            f"compile_s={compile_s:.3f} error={type(exc).__name__}: {exc}"
        )
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile synthetic persistent graph scaling probes"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["separate", "repeat", "grouped"],
        choices=sorted(PATTERNS),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--group-layers", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--position", type=int, default=6)
    parser.add_argument(
        "--max-dma-tasks-per-fifo",
        type=int,
        default=8,
        help="preflight limit from the Qwen3 persistent graph BD allocator probe",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="generate MLIR and run static preflight without invoking aiecc",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_qwen3_persistent_graph_probe"),
    )
    parser.add_argument("--clean-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    failed = False
    for pattern in args.patterns:
        for layers in args.layers:
            failed = not compile_probe(args, layers, pattern) or failed
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
