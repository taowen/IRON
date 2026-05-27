#!/usr/bin/env python3
"""Run exp41 on real NPU and verify direct shape-A -> shape-B handoff."""

from __future__ import annotations

import os
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
    CURRENT_DWORDS,
    HISTORY_DWORDS,
    OUTPUT_DWORDS,
    STATE_DWORDS,
    generate_mlir,
)
from reference import (
    expected_output,
    make_current_payload,
    make_history_a_payload,
    make_history_b_payload,
    make_state,
)

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_kernel() -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "edge_handoff.cc"
    obj = EXPERIMENT_DIR / "edge_handoff.o"
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
    print("  Compiling edge_handoff.cc...")
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
    checks = {
        "aie.flow(": 5,
        "aie.flow(%shape_a, DMA : 0, %shape_b, DMA : 0)": 1,
        "aie.runtime_sequence": 1,
        "aiex.npu.writebd": 4,
        "aiex.npu.address_patch": 4,
        "aiex.npu.push_queue": 4,
        "aiex.npu.sync": 1,
        "func.call @shape_a_make_state": 1,
        "func.call @shape_b_consume_state": 1,
    }
    errors: list[str] = []
    for marker, expected in checks.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")
    required_markers = (
        f"memref<{CURRENT_DWORDS}xi32>",
        f"memref<{HISTORY_DWORDS}xi32>",
        f"memref<{STATE_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        "%current: memref<512xi32>, %history_a: memref<2048xi32>, "
        "%history_b: memref<2048xi32>, %output: memref<512xi32>",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")
    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    if f"%state: memref<{STATE_DWORDS}xi32>" in runtime_body:
        errors.append("The 17-dword state must not be a runtime buffer")
    if "aie.packet_flow" in mlir_text:
        errors.append("Expected direct tile-to-tile flow, found packet_flow")
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
    print("Experiment 41: Edge Shape A -> Shape B Handoff Probe")
    print("=" * 72)
    print(f"  current: {CURRENT_DWORDS} i32")
    print(f"  histories: 2 x {HISTORY_DWORDS} i32")
    print(f"  internal state: {STATE_DWORDS} i32, not host-visible")
    print(f"  output: {OUTPUT_DWORDS} i32")
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

    current = make_current_payload()
    history_a = make_history_a_payload()
    history_b = make_history_b_payload()
    expected = expected_output()
    state_ref = make_state(current, history_a)

    current_buf = XRTTensor.from_torch(torch.from_numpy(current.copy()).to(torch.int32))
    history_a_buf = XRTTensor.from_torch(torch.from_numpy(history_a.copy()).to(torch.int32))
    history_b_buf = XRTTensor.from_torch(torch.from_numpy(history_b.copy()).to(torch.int32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(
        handle, [current_buf, history_a_buf, history_b_buf, output_buf]
    )
    got = output_buf.to_torch().numpy().astype(np.int32)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  reference state[0:5]: {state_ref[:5].tolist()}")
    print(f"  output[0:8]: {got[:8].tolist()}")

    mismatches = np.where(got != expected)[0]
    if mismatches.size == 0:
        print("  PASS: direct shape-A -> shape-B state handoff verified.")
        return True

    print(f"  FAIL: mismatches={mismatches.size}/{OUTPUT_DWORDS}")
    for idx in mismatches[:16]:
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
