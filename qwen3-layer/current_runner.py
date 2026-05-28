"""Run the current full-schedule qwen3-layer NPU backend."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import aie.utils as aie_utils
import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from ml_dtypes import bfloat16

from npu_generate import (
    ACT_SLICE_BF16,
    CHUNK_BF16,
    COLUMN_OUTPUT_BF16,
    CONTEXT_LEN,
    EDGE_COLUMNS,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    MAIN_COLUMNS,
    M_PER_TILE,
    NUM_PHASES,
    OUT_RECORD_BF16,
    OUT_TOTAL_I32,
    PATCH_BF16_BY_PHASE,
    PATCH_DESCRIPTORS_PER_COLUMN,
    PATCHES_PER_COLUMN,
    PHASE_BLOCKS,
    PHASE_CHUNKS,
    PHASE_INPUT_DIMS,
    PHASE_NAMES,
    PHASE_OUTPUT_DIMS,
    PHASE_WEIGHT_BF16,
    RECORD_DWORDS,
    ROWS_PER_PATCH,
    ROWS_PER_COLUMN,
    TOTAL_PATCHES,
    TOTAL_LOGICAL_BLOCKS,
    TOTAL_WEIGHT_I32,
    WEIGHT_BD_BATCH,
    _patch_packet_id,
    generate_mlir,
)
from npu_reference import expected_output, make_packed_weights
EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "qwen3_layer.cc"
    obj = EXPERIMENT_DIR / "qwen3_layer.o"
    cmd = [
        str(clang),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    print("  Compiling qwen3_layer.cc...")
    run_command(cmd)


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    cmd = [
        str(aiecc),
        "-v",
        "-j1",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--peano",
        str(peano_dir),
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("  Compiling MLIR...")
    run_command(cmd)


def check_mlir_structure(mlir_text: str) -> list[str]:
    weight_batches = (
        PATCH_DESCRIPTORS_PER_COLUMN + WEIGHT_BD_BATCH - 1
    ) // WEIGHT_BD_BATCH
    weight_descriptor_count = len(MAIN_COLUMNS) * PATCH_DESCRIPTORS_PER_COLUMN
    runtime_descriptor_count = len(MAIN_COLUMNS) + weight_descriptor_count
    push_count = len(MAIN_COLUMNS) + len(MAIN_COLUMNS) * weight_batches
    sync_count = len(MAIN_COLUMNS) + len(MAIN_COLUMNS) * weight_batches
    linked_bd_count = len(MAIN_COLUMNS) * (
        PATCH_DESCRIPTORS_PER_COLUMN - weight_batches
    )
    expected_counts = {
        "aie.flow(": len(MAIN_COLUMNS) * (ROWS_PER_COLUMN * 3 + 1),
        "aie.packet_flow": len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + PATCHES_PER_COLUMN),
        "aie.packet_dest<%mt": len(MAIN_COLUMNS)
        * (ROWS_PER_COLUMN + PATCHES_PER_COLUMN),
        "aie.memtile_dma": len(MAIN_COLUMNS),
        "aie.core(": len(MAIN_COLUMNS) * ROWS_PER_COLUMN * 2,
        "aie.mem(": len(MAIN_COLUMNS) * ROWS_PER_COLUMN * 2,
        "aiex.npu.writebd": runtime_descriptor_count,
        "aiex.npu.address_patch": runtime_descriptor_count,
        "aiex.npu.push_queue": push_count,
        "aiex.npu.sync": sync_count,
        "use_next_bd = 1": linked_bd_count,
        "enable_packet = 1": weight_descriptor_count,
        "func.call @clear_summary": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "func.call @emit_seed_sideband": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "func.call @emit_next_block_sideband": len(MAIN_COLUMNS)
        * ROWS_PER_COLUMN
        * NUM_PHASES,
        "func.call @accumulate_block_summary": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "func.call @edge_make_block_slice": len(MAIN_COLUMNS)
        * ROWS_PER_COLUMN
        * NUM_PHASES
        * 2,
        "func.call @q4nx_chunk_accum_slice": len(MAIN_COLUMNS)
        * ROWS_PER_COLUMN
        * NUM_PHASES
        * 2,
        "func.call @q4nx_flush_output": len(MAIN_COLUMNS)
        * ROWS_PER_COLUMN
        * NUM_PHASES,
        "func.call @flush_schedule_output_with_header": len(MAIN_COLUMNS)
        * ROWS_PER_COLUMN,
    }
    errors: list[str] = []
    for marker, expected in expected_counts.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")

    match = re.search(r"aie\.runtime_sequence\(([^)]+)\)", mlir_text)
    if match is None:
        errors.append("No runtime_sequence found")
    else:
        runtime_args = [arg.strip() for arg in match.group(1).split(",")]
        if runtime_args != [
            f"%weights: memref<{TOTAL_WEIGHT_I32}xi32>",
            f"%output: memref<{OUT_TOTAL_I32}xi32>",
        ]:
            errors.append(f"Unexpected runtime args: {runtime_args}")

    required_markers = (
        f"memref<{RECORD_DWORDS}xi32>",
        f"memref<{ACT_SLICE_BF16}xbf16>",
        f"memref<{CHUNK_BF16}xbf16>",
        f"memref<{OUT_RECORD_BF16}xbf16>",
        f"memref<{ROWS_PER_PATCH * CHUNK_BF16}xbf16>",
        f"memref<{COLUMN_OUTPUT_BF16}xbf16>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUT_TOTAL_I32}xi32>",
        "aie.tile(2, 1)",
        "aie.tile(5, 5)",
        f"aie.tile({EDGE_COLUMNS[0]}, 2)",
        f"aie.tile({EDGE_COLUMNS[-1]}, 5)",
        "aie.flow(%m0_0, DMA : 0, %edge0_0, DMA : 0)",
        "aie.flow(%edge0_0, DMA : 0, %m0_0, DMA : 1)",
        f"aie.packet_flow({_patch_packet_id(0, 0)})",
        f"aie.packet_flow({_patch_packet_id(0, 1)})",
        "aie.packet_source<%shim0, DMA : 0>",
        "aie.packet_dest<%mt0, DMA : 0>",
        "aie.packet_dest<%mt0, DMA : 1>",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>",
        f"%p6_chunks = arith.constant {PHASE_CHUNKS[-1]} : index",
        f"%p5_blocks = arith.constant {PHASE_BLOCKS[5]} : index",
        f"packet_id = {_patch_packet_id(0, 0)} : i32",
        f"packet_id = {_patch_packet_id(0, 1)} : i32",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")

    forbidden_markers = (
        f"memref<{HIDDEN_DIM}xbf16>",
        f"memref<{INTERMEDIATE_DIM}xbf16>",
        "edge_make_attention_shard",
        "q4nx_chunk_accum_offset",
        "expand_attention",
        "dma_configure_task_for",
        "aie.shim_dma_allocation",
        f"memref<{ROWS_PER_COLUMN * CHUNK_BF16}xbf16>",
        "aie.flow(%shim0, DMA : 0, %mt0, DMA : 0)",
    )
    for marker in forbidden_markers:
        if marker in mlir_text:
            errors.append(f"Forbidden full-tensor/old marker found: {marker}")

    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    forbidden_runtime_names = (
        "%m0_",
        "%mt0_",
        "%edge",
        "%record",
        "%act_ping",
        "%wt_ping",
    )
    for marker in forbidden_runtime_names:
        if marker in runtime_body:
            errors.append(f"Internal tensor leaked into runtime sequence: {marker}")
    return errors


def build_kernel() -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  STRUCTURE FAIL: {error}" for error in errors))

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_backend_structure() -> bool:
    mlir_text = generate_mlir()
    errors = check_mlir_structure(mlir_text)
    if errors:
        for error in errors:
            print(f"  STRUCTURE FAIL: {error}")
        return False
    print("  PASS: runnable MLIR structure matches qwen3-layer executable contract")
    return True


def build_only() -> bool:
    compile_kernel()
    xclbin_path, insts_path = build_kernel()
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def _records_to_ordered_values(records_bf16: np.ndarray, label: str) -> np.ndarray:
    records = records_bf16.reshape(-1, OUT_RECORD_BF16)
    expected_records = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    if records.shape[0] != expected_records:
        raise RuntimeError(
            f"{label}: expected {expected_records} records, got {records.shape[0]}"
        )

    ordered = np.empty((expected_records, M_PER_TILE), dtype=bfloat16)
    seen: set[tuple[int, int]] = set()
    for record_idx, record in enumerate(records):
        group = int(round(float(record[0])))
        row = int(round(float(record[1])))
        key = (group, row)
        if not (0 <= group < len(MAIN_COLUMNS) and 0 <= row < ROWS_PER_COLUMN):
            raise RuntimeError(
                f"{label}: bad record header at {record_idx}: group={group}, row={row}"
            )
        if key in seen:
            raise RuntimeError(
                f"{label}: duplicate record header group={group}, row={row}"
            )
        seen.add(key)
        ordered[group * ROWS_PER_COLUMN + row] = record[2 : 2 + M_PER_TILE]

    missing = [
        (group, row)
        for group in range(len(MAIN_COLUMNS))
        for row in range(ROWS_PER_COLUMN)
        if (group, row) not in seen
    ]
    if missing:
        raise RuntimeError(f"{label}: missing record headers {missing}")
    return ordered.reshape(-1)


def _record_headers(records_bf16: np.ndarray) -> list[tuple[int, int]]:
    records = records_bf16.reshape(-1, OUT_RECORD_BF16)
    return [
        (int(round(float(record[0]))), int(round(float(record[1]))))
        for record in records
    ]


def run_on_npu() -> bool:
    print("=" * 78)
    print("qwen3-layer: runnable NPU integration backend")
    print("=" * 78)
    print(f"  main columns: {MAIN_COLUMNS}, edge columns: {EDGE_COLUMNS}")
    print(f"  rows per column: {ROWS_PER_COLUMN}")
    print(f"  phases: {PHASE_NAMES}")
    print(f"  phase input dims: {PHASE_INPUT_DIMS}")
    print(f"  phase output dims: {PHASE_OUTPUT_DIMS}")
    print(f"  phase blocks: {PHASE_BLOCKS}")
    print(f"  phase chunks: {PHASE_CHUNKS}")
    print(f"  logical output blocks: {TOTAL_LOGICAL_BLOCKS}")
    print(f"  MyLM/Qwen exact patch count: {TOTAL_PATCHES}")
    print(f"  patch descriptors per main column: {PATCH_DESCRIPTORS_PER_COLUMN}")
    print(f"  sideband per handoff: {RECORD_DWORDS} i32")
    print(f"  per tile: M={M_PER_TILE}, slice={ACT_SLICE_BF16} bf16")
    print(f"  Q4NX chunk: {CHUNK_BF16 * 2} bytes")
    print(f"  per-phase patch sizes: {[size * 2 for size in PATCH_BF16_BY_PHASE]}")
    print(f"  per-phase total streams: {[size * 2 for size in PHASE_WEIGHT_BF16]}")
    print(
        "  runtime weight source: packetized exact patch descriptors, "
        f"{WEIGHT_BD_BATCH} BDs per linked batch"
    )
    print(
        "  O phase activation source: canonical Attn[32][128] layout -> "
        f"16x256 O slices, L={CONTEXT_LEN} online-softmax weighted-V producer"
    )
    print(f"  debug output record: 2 bf16 header + {M_PER_TILE} bf16 values")
    print(f"  runtime weights/output: {TOTAL_WEIGHT_I32}/{OUT_TOTAL_I32} i32")
    print()

    compile_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    print("  Preparing Q4NX weights and CPU reference...")
    packed = make_packed_weights()
    expected = expected_output(packed)
    weights_i32 = np.frombuffer(packed.tobytes(), dtype=np.int32)
    if weights_i32.shape[0] != TOTAL_WEIGHT_I32:
        raise RuntimeError(
            f"weight i32 mismatch: {weights_i32.shape[0]} != {TOTAL_WEIGHT_I32}"
        )

    weights_buf = XRTTensor.from_torch(
        torch.from_numpy(weights_i32.copy()).to(torch.int32)
    )
    output_buf = XRTTensor((OUT_TOTAL_I32,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [weights_buf, output_buf])
    got_i32 = output_buf.to_torch().numpy().astype(np.int32)
    got = np.frombuffer(got_i32.tobytes(), dtype=bfloat16)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  expected headers: {_record_headers(expected)}")
    print(f"  got headers:      {_record_headers(got)}")

    expected_values = _records_to_ordered_values(expected, "expected")
    got_values = _records_to_ordered_values(got, "got")
    print(f"  expected values[0:8]: {expected_values[:8].tolist()}")
    print(f"  got values[0:8]:      {got_values[:8].tolist()}")

    expected_f32 = expected_values.astype(np.float32)
    got_f32 = got_values.astype(np.float32)
    abs_err = np.abs(expected_f32 - got_f32)
    rel_err = abs_err / np.maximum(np.abs(expected_f32), 1e-6)
    abs_tol = 0.5
    rel_tol = 0.16
    mask = abs_err > np.maximum(abs_tol, rel_tol * np.abs(expected_f32))
    mismatch = np.where(mask)[0]
    print(
        f"  max_abs_err={float(abs_err.max()):.6f}, max_rel_err={float(rel_err.max()):.6f}"
    )

    if mismatch.size == 0:
        print(
            "  PASS: qwen3-layer ran the Q/K/V/O/up/gate/down NPU schedule "
            "with packetized patch queues and the global QKV/O layout contract."
        )
        return True

    print(f"  FAIL: mismatches={mismatch.size}/{expected_values.shape[0]}")
    for group in range(len(MAIN_COLUMNS)):
        print(f"  group {group} row samples:")
        for row in range(ROWS_PER_COLUMN):
            start = (group * ROWS_PER_COLUMN + row) * M_PER_TILE
            end = start + 4
            print(
                f"    row {row}: expected={expected_values[start:end].tolist()} "
                f"got={got_values[start:end].tolist()}"
            )
    for idx in mismatch[:32]:
        print(
            f"    out[{idx}]: expected={expected_f32[idx]:.6f} got={got_f32[idx]:.6f} err={abs_err[idx]:.6f}"
        )
    return False
