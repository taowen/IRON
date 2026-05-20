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
    compute_cores: int
    max_fifo_buffered_bytes: int
    total_dma_tasks: int
    max_dma_tasks_per_fifo: int
    max_compute_tile_inputs: int
    max_compute_tile_outputs: int
    non_advancing_acquires: int


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
_OBJECTFIFO_ACQUIRE_RE = re.compile(
    r"aie\.objectfifo\.acquire @(?P<fifo>[A-Za-z0-9_.$-]+)"
    r"\((?P<port>Produce|Consume), (?P<size>\d+)\)"
)
_OBJECTFIFO_RELEASE_RE = re.compile(
    r"aie\.objectfifo\.release @(?P<fifo>[A-Za-z0-9_.$-]+)"
    r"\((?P<port>Produce|Consume), (?P<size>\d+)\)"
)
_KERNEL_DECL_RE = re.compile(
    r"func\.func private @(?P<name>[A-Za-z0-9_.$-]+)"
    r"\((?P<args>.*?)\) attributes \{link_with = \"(?P<link>[^\"]+)\"\}",
    re.MULTILINE | re.DOTALL,
)


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


def find_non_advancing_acquires(mlir_text: str) -> list[str]:
    """Find ObjectFIFO acquires that cannot advance to a new FIFO object.

    `aie.objectfifo.acquire` requests access to a held set of `size` objects. If
    the process already holds that many objects from the same FIFO/port, the
    acquire does not take another lock and a returned subview can alias an
    already-held object.
    """

    held: dict[tuple[str, str], int] = {}
    issues: list[str] = []
    for lineno, line in enumerate(mlir_text.splitlines(), start=1):
        acquire = _OBJECTFIFO_ACQUIRE_RE.search(line)
        if acquire:
            key = (acquire.group("fifo"), acquire.group("port"))
            size = int(acquire.group("size"))
            already_held = held.get(key, 0)
            if already_held >= size:
                issues.append(
                    f"line {lineno}: acquire @{key[0]}({key[1]}, {size}) "
                    f"while {already_held} object(s) are already held"
                )
            held[key] = max(already_held, size)
            continue

        release = _OBJECTFIFO_RELEASE_RE.search(line)
        if release:
            key = (release.group("fifo"), release.group("port"))
            size = int(release.group("size"))
            held[key] = max(0, held.get(key, 0) - size)
    return issues


def find_gemv_symbol_abi_issues(mlir_text: str) -> list[str]:
    """Find GEMV declarations that hide distinct DIM_K ABIs behind one symbol."""

    issues: list[str] = []
    declarations: dict[str, set[tuple[str, str]]] = {}
    for match in _KERNEL_DECL_RE.finditer(mlir_text):
        name = match.group("name")
        args = " ".join(match.group("args").split())
        link = match.group("link")
        declarations.setdefault(name, set()).add((args, link))

    for name, variants in declarations.items():
        if len(variants) > 1:
            rendered = ", ".join(f"{link}: ({args})" for args, link in sorted(variants))
            issues.append(f"kernel @{name} has multiple ABI/link variants: {rendered}")

    for args, link in declarations.get("matvec_vectorized_bf16_bf16", set()):
        if "memref<4x2048xbf16>" in args:
            issues.append(
                "O projection GEMV uses the generic @matvec_vectorized_bf16_bf16 "
                f"symbol via {link}; use a distinct qwen3_o_proj_* symbol for "
                "the DIM_K=2048 object."
            )

    for args, link in declarations.get(
        "qwen3_o_proj_matvec_vectorized_bf16_bf16", set()
    ):
        if "memref<4x2048xbf16>" not in args or "memref<2048xbf16>" not in args:
            issues.append(
                "O projection GEMV ABI mismatch: "
                f"@qwen3_o_proj_matvec_vectorized_bf16_bf16 from {link} has "
                f"arguments ({args}); expected DIM_K=2048 matrix/vector memrefs."
            )

    return issues


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
    max_compute_cores: int = 32,
    max_dma_tasks_per_fifo: int = 8,
) -> PersistentPreflightResult:
    """Fail before runtime when generated artifacts match known bad patterns."""

    if not mlir_path.exists():
        raise Qwen3PreflightError(f"MLIR artifact does not exist: {mlir_path}")

    mlir_text = mlir_path.read_text()
    compute_cores = mlir_text.count("aie.core(")
    if compute_cores > max_compute_cores:
        raise Qwen3PreflightError(
            f"Compute worker budget mismatch: MLIR has {compute_cores} aie.core ops, "
            f"limit={max_compute_cores}. Fuse adjacent stages or drop debug-only "
            "workers before changing kernel math."
        )

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
            "The NPU lowering BD allocator has failed at 9 tasks on a single "
            "FIFO in the Qwen3 persistent graph probe. Prefer one legal "
            "multidimensional TAP or reuse data on tile."
        )

    non_advancing_acquires = find_non_advancing_acquires(mlir_text)
    if non_advancing_acquires:
        first = non_advancing_acquires[0]
        raise Qwen3PreflightError(
            "ObjectFIFO acquire does not advance to a new object: "
            f"{first}. Use acquire(2) with indexed subviews, separate FIFOs, "
            "or a packed FIFO object before using the returned values as "
            "independent buffers."
        )

    gemv_symbol_issues = find_gemv_symbol_abi_issues(mlir_text)
    if gemv_symbol_issues:
        raise Qwen3PreflightError(
            "External kernel GEMV ABI mismatch: " + gemv_symbol_issues[0]
        )

    return PersistentPreflightResult(
        runtime_memrefs=runtime_memrefs,
        arg_specs=arg_specs,
        metadata_host_bos=metadata_host_bos,
        compute_cores=compute_cores,
        max_fifo_buffered_bytes=max(
            (fifo.buffered_bytes or 0 for fifo in fifos),
            default=0,
        ),
        total_dma_tasks=sum(dma_counts.values()),
        max_dma_tasks_per_fifo=max(dma_counts.values(), default=0),
        max_compute_tile_inputs=max(tile_inputs.values(), default=0),
        max_compute_tile_outputs=max(tile_outputs.values(), default=0),
        non_advancing_acquires=0,
    )
