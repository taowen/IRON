#!/usr/bin/env python3
"""Run exp48 on real NPU and verify full-K Q4NX phase replay on main16."""

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
    ACT_BF16,
    CHUNK_BF16,
    COLUMN_OUTPUT_BF16,
    COLUMN_WEIGHT_BF16,
    FAT_CHUNK_BF16,
    HIDDEN_DIM,
    K_CHUNK,
    MAIN_COLUMNS,
    M_PER_TILE,
    NUM_CHUNKS,
    NUM_PHASES,
    OUT_TOTAL_I32,
    PHASE_NAMES,
    PHASE_WEIGHT_BF16,
    ROWS_PER_COLUMN,
    TOTAL_OUTPUT_BF16,
    TOTAL_WEIGHT_I32,
    generate_mlir,
)
from reference import expected_output, make_packed_weights

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "q4nx_phase_replay.cc"
    obj = EXPERIMENT_DIR / "q4nx_phase_replay.o"
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
    print("  Compiling q4nx_phase_replay.cc...")
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
    expected_counts = {
        "aie.flow(": len(MAIN_COLUMNS) * (ROWS_PER_COLUMN + 2),
        "aie.packet_flow": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "aie.packet_dest<%mt": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "aie.memtile_dma": len(MAIN_COLUMNS),
        "aie.core(": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "aie.mem(": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "aiex.npu.writebd": len(MAIN_COLUMNS) * (NUM_PHASES + 1),
        "aiex.npu.address_patch": len(MAIN_COLUMNS) * (NUM_PHASES + 1),
        "aiex.npu.push_queue": len(MAIN_COLUMNS) * (NUM_PHASES + 1),
        "aiex.npu.sync": len(MAIN_COLUMNS),
        "func.call @init_activation": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
        "func.call @q4nx_flush_output": len(MAIN_COLUMNS) * ROWS_PER_COLUMN * NUM_PHASES,
        "func.call @combine_q4nx_phases": len(MAIN_COLUMNS) * ROWS_PER_COLUMN,
    }
    errors: list[str] = []
    for marker, expected in expected_counts.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")

    q4_calls = mlir_text.count("func.call @q4nx_chunk_accum_offset")
    expected_q4_calls = len(MAIN_COLUMNS) * ROWS_PER_COLUMN * NUM_PHASES * 2
    if q4_calls != expected_q4_calls:
        errors.append(f"q4nx chunk call branches: expected {expected_q4_calls}, got {q4_calls}")

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
        f"memref<{ACT_BF16}xbf16>",
        f"memref<{CHUNK_BF16}xbf16>",
        f"memref<{FAT_CHUNK_BF16}xbf16>",
        f"memref<{COLUMN_OUTPUT_BF16}xbf16>",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        f"memref<{OUT_TOTAL_I32}xi32>",
        "aie.tile(2, 1)",
        "aie.tile(5, 5)",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")

    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    forbidden_runtime_names = ("%m0_", "%mt0_", "%wt_ping", "%gate", "%up", "%down", "%act")
    for marker in forbidden_runtime_names:
        if marker in runtime_body:
            errors.append(f"Internal tensor leaked into runtime sequence: {marker}")
    if "dma_configure_task_for" in mlir_text:
        errors.append("Expected raw writebd runtime, found dma_configure_task_for")
    if "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found shim_dma_allocation")
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


def bf16_to_i32_view(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.astype(bfloat16).view(np.uint8).tobytes(), dtype=np.int32)


def run_on_npu() -> bool:
    print("=" * 78)
    print("Experiment 48: Main16 Full-K Q4NX Phase Replay")
    print("=" * 78)
    print(f"  main columns: {MAIN_COLUMNS}, rows per column: {ROWS_PER_COLUMN}")
    print(f"  phases: {' -> '.join(PHASE_NAMES)}")
    print(f"  per tile: M={M_PER_TILE}, K={HIDDEN_DIM}, K_CHUNK={K_CHUNK}, chunks={NUM_CHUNKS}")
    print(f"  Q4NX chunk: {CHUNK_BF16 * 2} bytes, fat chunk: {FAT_CHUNK_BF16 * 2} bytes")
    print(f"  phase stream per column: {PHASE_WEIGHT_BF16 * 2} bytes")
    print(f"  column stream: {COLUMN_WEIGHT_BF16 * 2} bytes")
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
        raise RuntimeError(f"weight i32 mismatch: {weights_i32.shape[0]} != {TOTAL_WEIGHT_I32}")

    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights_i32.copy()).to(torch.int32))
    output_buf = XRTTensor((OUT_TOTAL_I32,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [weights_buf, output_buf])
    got_i32 = output_buf.to_torch().numpy().astype(np.int32)
    got = np.frombuffer(got_i32.tobytes(), dtype=bfloat16)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    expected_f32 = expected.astype(np.float32)
    got_f32 = got.astype(np.float32)
    abs_err = np.abs(expected_f32 - got_f32)
    rel_err = abs_err / np.maximum(np.abs(expected_f32), 1e-6)
    abs_tol = 0.5
    rel_tol = 0.15
    mask = abs_err > np.maximum(abs_tol, rel_tol * np.abs(expected_f32))
    mismatch = np.where(mask)[0]
    print(f"  max_abs_err={float(abs_err.max()):.6f}, max_rel_err={float(rel_err.max()):.6f}")

    if mismatch.size == 0:
        print("  PASS: main16 full-K Q4NX O/gate/up/down replay verified.")
        return True

    print(f"  FAIL: mismatches={mismatch.size}/{TOTAL_OUTPUT_BF16}")
    for idx in mismatch[:32]:
        print(f"    out[{idx}]: expected={expected_f32[idx]:.6f} got={got_f32[idx]:.6f} err={abs_err[idx]:.6f}")
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
