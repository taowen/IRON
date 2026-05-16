#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static preflight checks for the Qwen3 persistent megakernel bring-up."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


class Qwen3PreflightError(RuntimeError):
    """Raised when a compiled Qwen3 artifact fails a pre-runtime check."""


@dataclass(frozen=True)
class ObjectFifoInfo:
    name: str
    producer: str
    consumers: tuple[str, ...]
    depth: int
    memref: str
    object_bytes: int | None

    @property
    def buffered_bytes(self) -> int | None:
        if self.object_bytes is None:
            return None
        return self.object_bytes * self.depth


@dataclass(frozen=True)
class PersistentPreflightResult:
    runtime_memrefs: int
    arg_specs: int
    metadata_host_bos: int | None
    max_fifo_buffered_bytes: int
    max_dma_tasks_per_fifo: int
    max_compute_tile_inputs: int
    max_compute_tile_outputs: int


_RUNTIME_SEQUENCE_RE = re.compile(
    r"aie\.runtime_sequence\((?P<args>.*?)\)\s*\{",
    re.MULTILINE | re.DOTALL,
)
_OBJECTFIFO_RE = re.compile(
    r"aie\.objectfifo @(?P<name>[^\(]+)"
    r"\((?P<producer>%[^,]+), \{(?P<consumers>[^\}]*)\}, "
    r"(?P<depth>\d+) : i32\) : !aie\.objectfifo<memref<(?P<memref>[^>]+)>>",
    re.MULTILINE,
)
_DMA_TASK_RE = re.compile(r"dma_configure_task_for @(?P<fifo>[A-Za-z0-9_.$-]+)")
_TILE_RE = re.compile(r"%tile_(?P<col>\d+)_(?P<row>\d+)")


def count_runtime_sequence_memrefs(mlir_text: str) -> int:
    matches = list(_RUNTIME_SEQUENCE_RE.finditer(mlir_text))
    if not matches:
        raise Qwen3PreflightError("MLIR has no aie.runtime_sequence block")
    if len(matches) > 1:
        raise Qwen3PreflightError(
            f"MLIR has {len(matches)} aie.runtime_sequence blocks; expected 1"
        )
    return len(re.findall(r"%arg\d+\s*:\s*memref<", matches[0].group("args")))


def count_main_kernel_host_bos(metadata: dict[str, Any]) -> int:
    kernels = metadata.get("ps-kernels", {}).get("kernels", [])
    if not kernels:
        raise Qwen3PreflightError("main_kernels.json has no ps-kernels.kernels entry")
    arguments = kernels[0].get("arguments", [])
    return sum(
        1
        for arg in arguments
        if arg.get("memory-connection") == "HOST"
        and re.fullmatch(r"bo\d+", str(arg.get("name", "")))
    )


def main_kernels_json_for_mlir(mlir_path: Path) -> Path:
    return Path(str(mlir_path) + ".prj") / "main_kernels.json"


def parse_objectfifos(mlir_text: str) -> list[ObjectFifoInfo]:
    fifos: list[ObjectFifoInfo] = []
    for match in _OBJECTFIFO_RE.finditer(mlir_text):
        consumers = tuple(
            item.strip() for item in match.group("consumers").split(",") if item.strip()
        )
        memref = match.group("memref")
        fifos.append(
            ObjectFifoInfo(
                name=match.group("name"),
                producer=match.group("producer").strip(),
                consumers=consumers,
                depth=int(match.group("depth")),
                memref=memref,
                object_bytes=memref_object_bytes(memref),
            )
        )
    return fifos


def memref_object_bytes(memref: str) -> int | None:
    parts = memref.split("x")
    if not parts:
        return None
    dtype = parts[-1]
    dtype_bytes = {
        "bf16": 2,
        "f16": 2,
        "i8": 1,
        "si8": 1,
        "ui8": 1,
        "i32": 4,
        "si32": 4,
        "ui32": 4,
        "f32": 4,
    }.get(dtype)
    if dtype_bytes is None:
        return None
    numel = 1
    for dim in parts[:-1]:
        if not dim.isdigit():
            return None
        numel *= int(dim)
    return numel * dtype_bytes


def count_dma_tasks_by_fifo(mlir_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _DMA_TASK_RE.finditer(mlir_text):
        fifo = match.group("fifo")
        counts[fifo] = counts.get(fifo, 0) + 1
    return counts


def compute_tile_endpoint_counts(
    fifos: list[ObjectFifoInfo],
) -> tuple[dict[str, int], dict[str, int]]:
    inputs: dict[str, int] = {}
    outputs: dict[str, int] = {}
    for fifo in fifos:
        if is_compute_tile(fifo.producer):
            outputs[fifo.producer] = outputs.get(fifo.producer, 0) + 1
        for consumer in fifo.consumers:
            if is_compute_tile(consumer):
                inputs[consumer] = inputs.get(consumer, 0) + 1
    return inputs, outputs


def is_compute_tile(tile_name: str) -> bool:
    match = _TILE_RE.fullmatch(tile_name.strip())
    if match is None:
        return False
    return int(match.group("row")) >= 2


def run_persistent_artifact_preflight(
    *,
    mlir_path: Path,
    arg_specs: int,
    max_l1_fifo_buffered_bytes: int = 64 * 1024,
    max_compute_tile_input_fifos: int = 2,
    max_compute_tile_output_fifos: int = 2,
    max_dma_tasks_per_fifo: int = 32,
) -> PersistentPreflightResult:
    """Fail before runtime when generated artifacts match known bad patterns."""

    if not mlir_path.exists():
        raise Qwen3PreflightError(f"MLIR artifact does not exist: {mlir_path}")

    mlir_text = mlir_path.read_text()
    runtime_memrefs = count_runtime_sequence_memrefs(mlir_text)
    if runtime_memrefs != arg_specs:
        raise Qwen3PreflightError(
            "Runtime ABI mismatch: MLIR runtime_sequence has "
            f"{runtime_memrefs} memref arguments, but operator arg spec has "
            f"{arg_specs}. This can make XRT set_arg validate the wrong BO."
        )

    metadata_path = main_kernels_json_for_mlir(mlir_path)
    metadata_host_bos: int | None = None
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata_host_bos = count_main_kernel_host_bos(json.load(f))
        if runtime_memrefs > metadata_host_bos:
            raise Qwen3PreflightError(
                "Runtime BO metadata mismatch: MLIR runtime_sequence has "
                f"{runtime_memrefs} memref arguments, but {metadata_path} exposes "
                f"only {metadata_host_bos} HOST bo* arguments. This is the class "
                "of failure that previously reached XRT BO validation as a segfault."
            )

    fifos = parse_objectfifos(mlir_text)
    oversized = [
        fifo
        for fifo in fifos
        if fifo.buffered_bytes is not None
        and fifo.buffered_bytes > max_l1_fifo_buffered_bytes
        and (
            is_compute_tile(fifo.producer) or any(map(is_compute_tile, fifo.consumers))
        )
    ]
    if oversized:
        fifo = max(oversized, key=lambda item: item.buffered_bytes or 0)
        raise Qwen3PreflightError(
            "ObjectFIFO L1 budget mismatch: "
            f"{fifo.name} carries memref<{fifo.memref}> at depth {fifo.depth}, "
            f"buffered_bytes={fifo.buffered_bytes}, limit={max_l1_fifo_buffered_bytes}. "
            "Stream the tensor in smaller blocks before compiling/running."
        )

    tile_inputs, tile_outputs = compute_tile_endpoint_counts(fifos)
    bad_inputs = {
        tile: count
        for tile, count in tile_inputs.items()
        if count > max_compute_tile_input_fifos
    }
    if bad_inputs:
        tile, count = max(bad_inputs.items(), key=lambda item: item[1])
        raise Qwen3PreflightError(
            f"Compute tile {tile} has {count} input ObjectFIFOs; "
            f"limit={max_compute_tile_input_fifos}. Pack or stage inputs before "
            "changing external-kernel math."
        )
    bad_outputs = {
        tile: count
        for tile, count in tile_outputs.items()
        if count > max_compute_tile_output_fifos
    }
    if bad_outputs:
        tile, count = max(bad_outputs.items(), key=lambda item: item[1])
        raise Qwen3PreflightError(
            f"Compute tile {tile} has {count} output ObjectFIFOs; "
            f"limit={max_compute_tile_output_fifos}. Use broadcast/split/join instead "
            "of duplicating producer outputs."
        )

    dma_counts = count_dma_tasks_by_fifo(mlir_text)
    bad_dma = {
        fifo: count
        for fifo, count in dma_counts.items()
        if count > max_dma_tasks_per_fifo
    }
    if bad_dma:
        fifo, count = max(bad_dma.items(), key=lambda item: item[1])
        raise Qwen3PreflightError(
            f"FIFO {fifo} has {count} DMA tasks; limit={max_dma_tasks_per_fifo}. "
            "Prefer one legal multidimensional TAP or reuse data on tile."
        )

    return PersistentPreflightResult(
        runtime_memrefs=runtime_memrefs,
        arg_specs=arg_specs,
        metadata_host_bos=metadata_host_bos,
        max_fifo_buffered_bytes=max(
            (fifo.buffered_bytes or 0 for fifo in fifos),
            default=0,
        ),
        max_dma_tasks_per_fifo=max(dma_counts.values(), default=0),
        max_compute_tile_inputs=max(tile_inputs.values(), default=0),
        max_compute_tile_outputs=max(tile_outputs.values(), default=0),
    )
