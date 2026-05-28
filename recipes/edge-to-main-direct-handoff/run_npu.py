#!/usr/bin/env python3
"""Run the edge-to-main direct handoff recipe on real NPU."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import aie.utils as aie_utils
import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from ml_dtypes import bfloat16

from generate import (
    CHUNK_BF16,
    COLUMN_OUTPUT_BF16,
    K,
    K_CHUNKS,
    MAIN_COLUMNS,
    M_PER_TILE,
    OUT_RECORD_BF16,
    OUT_TOTAL_I32,
    PATCHES_PER_COLUMN,
    PATCH_BF16,
    ROWS_PER_COLUMN,
    TOTAL_OUTPUT_BF16,
    TOTAL_WEIGHT_I32,
    _patch_packet_id,
    generate_mlir,
)
from reference import expected_output, make_packed_weights, packed_as_i32

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "projection_nblock.cc"
    obj = EXPERIMENT_DIR / "projection_nblock.o"
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
    print("  Compiling projection_nblock.cc...")
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
    expected_tiles = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    expected_flows_per_column = ROWS_PER_COLUMN * 2 + 1
    expected_packet_flows = expected_tiles + len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    expected_counts = {
        "aie.flow(": len(MAIN_COLUMNS) * expected_flows_per_column,
        "aie.packet_flow": expected_packet_flows,
        "aie.memtile_dma": len(MAIN_COLUMNS),
        "aie.core(": expected_tiles * 2,
        "aie.mem(": expected_tiles * 2,
        "aiex.npu.writebd": len(MAIN_COLUMNS) * 3,
        "aiex.npu.address_patch": len(MAIN_COLUMNS) * 3,
        "aiex.npu.push_queue": len(MAIN_COLUMNS) * 2,
        "aiex.npu.sync": len(MAIN_COLUMNS) * 2,
        "use_next_bd = 1": len(MAIN_COLUMNS),
        "enable_packet = 1": len(MAIN_COLUMNS) * PATCHES_PER_COLUMN,
        "func.call @edge_replay_attention_slice": expected_tiles * 2,
        "func.call @q4nx_chunk_accum_slice": expected_tiles * 2,
        "func.call @flush_projection_output": expected_tiles,
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
        f"memref<{2 * CHUNK_BF16}xbf16>",
        f"memref<{CHUNK_BF16}xbf16>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUT_TOTAL_I32}xi32>",
        "mt0_patch0_ping_empty",
        "mt0_patch0_ping_full",
        "mt0_patch0_pong_empty",
        "mt0_patch0_pong_full",
        "mt0_patch1_ping_empty",
        "mt0_patch1_ping_full",
        "mt0_patch1_pong_empty",
        "mt0_patch1_pong_full",
        "aie.flow(%edge0_0, DMA : 0, %m0_0, DMA : 0)",
        "aie.flow(%mt0, DMA : 0, %m0_0, DMA : 1)",
        "packet_dest<%mt0, DMA : 2>",
        "aie.dma_start(S2MM, 2, ^out0, ^drain_start)",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>",
        f"aie.packet_flow({_patch_packet_id(0, 0)})",
        f"aie.packet_flow({_patch_packet_id(0, 1)})",
        "aie.packet_source<%shim0, DMA : 0>",
        "aie.packet_dest<%mt0, DMA : 0>",
        "aie.packet_dest<%mt0, DMA : 1>",
        "aie.flow(%mt0, DMA : 5, %shim0, DMA : 0)",
        "aie.dma_start(S2MM, 0, ^patch0_split_ping, ^patch1_start)",
        "aie.dma_start(S2MM, 1, ^patch1_split_ping, ^row0_start)",
        "aiex.npu.push_queue(2, 0, MM2S : 0)",
        "aiex.npu.push_queue(2, 0, S2MM : 0)",
        f"packet_id = {_patch_packet_id(0, 0)} : i32",
        f"packet_id = {_patch_packet_id(0, 1)} : i32",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")

    forbidden_markers = (
        "mt0_patch0_empty",
        "mt0_patch0_full",
        "memref<81920xbf16>",
        "aie.flow(%shim0, DMA : 0, %mt0, DMA : 0)",
        "aie.flow(%shim0, DMA : 1, %mt0, DMA : 1)",
        "aiex.npu.push_queue(2, 0, MM2S : 1)",
        "aiex.npu.push_queue(2, 0, S2MM : 1)",
    )
    for marker in forbidden_markers:
        if marker in mlir_text:
            errors.append(f"Forbidden shared patch lock marker present: {marker}")
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


def _records_to_ordered_values(records_bf16: np.ndarray, label: str) -> np.ndarray:
    records = records_bf16.reshape(-1, OUT_RECORD_BF16)
    expected_records = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
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
    print("Recipe: Edge To Main Direct Handoff")
    print("=" * 78)
    print(f"  main columns: {MAIN_COLUMNS}, rows per column: {ROWS_PER_COLUMN}")
    print(f"  K={K}, K_chunks={K_CHUNKS}")
    print(f"  Q4NX chunk: {CHUNK_BF16 * 2} bytes")
    print(f"  MyLM patch: {PATCH_BF16 * 2} bytes (0x{PATCH_BF16 * 2:x})")
    print(f"  column output rows: {len(MAIN_COLUMNS) * ROWS_PER_COLUMN * M_PER_TILE}")
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

    print("  Preparing packetized weights and edge-slice CPU reference...")
    packed = make_packed_weights()
    expected = expected_output(packed)
    weights_i32 = packed_as_i32(packed)
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
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected headers: {_record_headers(expected)}")
    print(f"  got headers:      {_record_headers(got)}")

    expected_values = _records_to_ordered_values(expected, "expected")
    got_values = _records_to_ordered_values(got, "got")
    expected_f32 = expected_values.astype(np.float32)
    got_f32 = got_values.astype(np.float32)
    abs_err = np.abs(expected_f32 - got_f32)
    rel_err = abs_err / np.maximum(np.abs(expected_f32), 1e-6)
    abs_tol = 0.5
    rel_tol = 0.18
    mismatch = np.where(abs_err > np.maximum(abs_tol, rel_tol * np.abs(expected_f32)))[
        0
    ]
    print(f"  expected values[0:8]: {expected_values[:8].tolist()}")
    print(f"  got values[0:8]:      {got_values[:8].tolist()}")
    print(
        f"  max_abs_err={float(abs_err.max()):.6f}, max_rel_err={float(rel_err.max()):.6f}"
    )

    if mismatch.size == 0:
        print(
            "  PASS: attention-result slices stream directly into packetized O phase."
        )
        return True

    print(
        f"  FAIL: mismatches={mismatch.size}/{TOTAL_OUTPUT_BF16 - len(MAIN_COLUMNS) * ROWS_PER_COLUMN * 2}"
    )
    for idx in mismatch[:32]:
        print(
            f"    out[{idx}]: expected={expected_f32[idx]:.6f} got={got_f32[idx]:.6f} err={abs_err[idx]:.6f}"
        )
    return False


def main() -> bool:
    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    return run_on_npu()


if __name__ == "__main__":
    try:
        success = main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()
    raise SystemExit(0 if success else 1)
