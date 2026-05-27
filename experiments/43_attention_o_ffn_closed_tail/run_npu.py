#!/usr/bin/env python3
"""Run exp43 on real NPU and verify attention + O projection + FFN closed tail."""

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
    ATTENTION_DWORDS,
    CURRENT_DWORDS,
    FFN_WEIGHT_DWORDS,
    HALF_DWORDS,
    HISTORY_DWORDS,
    OUTPUT_DWORDS,
    O_WEIGHT_DWORDS,
    generate_mlir,
)
from reference import (
    expected_output,
    make_attention_output,
    make_current_payload,
    make_ffn_weight_payload,
    make_history_payload,
    make_o_weight_payload,
    o_project,
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
    src = EXPERIMENT_DIR / "closed_tail.cc"
    obj = EXPERIMENT_DIR / "closed_tail.o"
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
    print("  Compiling closed_tail.cc...")
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
        "aie.flow(": 7,
        "aie.flow(%edge_attn, DMA : 0, %o_proj, DMA : 0)": 1,
        "aie.flow(%o_proj, DMA : 0, %ffn_tail, DMA : 0)": 1,
        "aie.runtime_sequence": 1,
        "aiex.npu.writebd": 5,
        "aiex.npu.address_patch": 5,
        "aiex.npu.push_queue": 5,
        "aiex.npu.sync": 1,
        "func.call @edge_make_attention": 1,
        "func.call @o_project_attention": 1,
        "func.call @ffn_tail": 1,
    }
    errors: list[str] = []
    for marker, expected in checks.items():
        actual = mlir_text.count(marker)
        if actual != expected:
            errors.append(f"{marker}: expected {expected}, got {actual}")

    required_markers = (
        f"memref<{CURRENT_DWORDS}xi32>",
        f"memref<{HISTORY_DWORDS}xi32>",
        f"memref<{ATTENTION_DWORDS}xi32>",
        f"memref<{HALF_DWORDS}xi32>",
        f"memref<{O_WEIGHT_DWORDS}xi32>",
        f"memref<{FFN_WEIGHT_DWORDS}xi32>",
        f"memref<{OUTPUT_DWORDS}xi32>",
        "%current: memref<512xi32>, %history: memref<2048xi32>, "
        "%o_weight_bo: memref<2048xi32>, %ffn_weight_bo: memref<4096xi32>, "
        "%output: memref<512xi32>",
        "%ffn_gate",
        "%ffn_up",
        "%ffn_swiglu",
    )
    for marker in required_markers:
        if marker not in mlir_text:
            errors.append(f"Missing required marker: {marker}")

    runtime_body = mlir_text.split("aie.runtime_sequence", 1)[1]
    forbidden_runtime_names = ("%attention", "%o_output", "%gate", "%up", "%swiglu")
    for marker in forbidden_runtime_names:
        if marker in runtime_body:
            errors.append(f"Internal tensor leaked into runtime sequence: {marker}")
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
    print("Experiment 43: Attention + O Projection + FFN Closed Tail")
    print("=" * 72)
    print(f"  current/history: {CURRENT_DWORDS}/{HISTORY_DWORDS} i32")
    print(f"  internal attention and O output: {ATTENTION_DWORDS} i32 each, split into 2 x {HALF_DWORDS}")
    print(f"  O/FFN weights: {O_WEIGHT_DWORDS}/{FFN_WEIGHT_DWORDS} i32")
    print(f"  final output: {OUTPUT_DWORDS} i32")
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
    history = make_history_payload()
    o_weight = make_o_weight_payload()
    ffn_weight = make_ffn_weight_payload()
    attention_ref = make_attention_output(current, history)
    o_ref = o_project(attention_ref, o_weight)
    expected = expected_output()

    current_buf = XRTTensor.from_torch(torch.from_numpy(current.copy()).to(torch.int32))
    history_buf = XRTTensor.from_torch(torch.from_numpy(history.copy()).to(torch.int32))
    o_weight_buf = XRTTensor.from_torch(torch.from_numpy(o_weight.copy()).to(torch.int32))
    ffn_weight_buf = XRTTensor.from_torch(torch.from_numpy(ffn_weight.copy()).to(torch.int32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(
        handle, [current_buf, history_buf, o_weight_buf, ffn_weight_buf, output_buf]
    )
    got = output_buf.to_torch().numpy().astype(np.int32)
    npu_time_us = result.npu_time / 1e3
    print(f"  NPU time: {npu_time_us:.1f} us")
    print(f"  reference attention[0:8]: {attention_ref[:8].tolist()}")
    print(f"  reference o_output[0:8]: {o_ref[:8].tolist()}")
    print(f"  output[0:8]: {got[:8].tolist()}")

    mismatches = np.where(got != expected)[0]
    if mismatches.size == 0:
        print("  PASS: attention -> O projection -> FFN closed tail verified.")
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
