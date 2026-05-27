#!/usr/bin/env python3
"""Run exp45 on real NPU and verify main16 -> edge -> main16 O-phase handoff."""

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

from generate import (
    COLUMN_DWORDS,
    RECORD_DWORDS,
    ROWS_PER_COLUMN,
    SHARD_DWORDS,
    TOTAL_DWORDS,
    generate_mlir,
)
from reference import edge_attention_from_record, expected_output, make_record, make_weights

EXPERIMENT_DIR = Path(__file__).parent
MAIN_GROUPS = 4


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "edge_return_o_phase.cc"
    obj = EXPERIMENT_DIR / "edge_return_o_phase.o"
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
    print("  Compiling edge_return_o_phase.cc...")
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
        "aie.flow(": 56,
        "aie.packet_flow": 16,
        "aie.packet_dest<%mt": 16,
        "aie.memtile_dma": 4,
        "aie.core(": 32,
        "aie.mem(": 32,
        "aiex.npu.writebd": 8,
        "aiex.npu.address_patch": 8,
        "aiex.npu.push_queue": 8,
        "aiex.npu.sync": 4,
        "func.call @emit_main_record": 16,
        "func.call @edge_make_attention_row": 16,
        "func.call @main_o_phase": 16,
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
            f"%weights: memref<{TOTAL_DWORDS}xi32>",
            f"%output: memref<{TOTAL_DWORDS}xi32>",
        ]:
            errors.append(f"Unexpected runtime args: {runtime_args}")

    required_markers = (
        f"memref<{SHARD_DWORDS}xi32>",
        f"memref<{RECORD_DWORDS}xi32>",
        f"memref<{COLUMN_DWORDS}xi32>",
        f"memref<{TOTAL_DWORDS}xi32>",
        "aie.tile(2, 1)",
        "aie.tile(5, 5)",
        "aie.tile(0, 2)",
        "aie.tile(7, 5)",
        "aie.flow(%m0_0, DMA : 0, %edge0_0, DMA : 0)",
        "aie.flow(%edge0_0, DMA : 0, %m0_0, DMA : 1)",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>",
        "packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")

    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    forbidden_runtime_names = ("%record", "%attention", "%edge", "%m0_", "%mt0_")
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
        message = "\n".join(f"  STRUCTURE FAIL: {error}" for error in errors)
        raise RuntimeError(message)

    compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def run_on_npu() -> bool:
    print("=" * 72)
    print("Experiment 45: Main16 Edge Return O Phase")
    print("=" * 72)
    print(f"  main groups: {MAIN_GROUPS}, rows per group: {ROWS_PER_COLUMN}")
    print(f"  internal sideband: 16 x {RECORD_DWORDS} i32 records")
    print(f"  internal edge attention: 16 x {SHARD_DWORDS} i32 shards")
    print(f"  runtime weights/output: {TOTAL_DWORDS}/{TOTAL_DWORDS} i32")
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

    weights = make_weights()
    expected = expected_output(weights)
    group0_attention = edge_attention_from_record(0, 0, make_record(0, 0))

    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [weights_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  group0 record0: {make_record(0, 0).tolist()}")
    print(f"  group0 attention[0:8]: {group0_attention[:8].tolist()}")
    print(f"  output[0:8]: {got[:8].tolist()}")

    mismatches = np.where(got != expected)[0]
    if mismatches.size == 0:
        print("  PASS: main16 -> edge -> main16 O-phase handoff verified.")
        return True

    print(f"  FAIL: mismatches={mismatches.size}/{TOTAL_DWORDS}")
    for idx in mismatches[:32]:
        print(f"    out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
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
