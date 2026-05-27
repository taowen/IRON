#!/usr/bin/env python3
"""Run exp25 on real NPU and verify edge/KV ring checksums."""

import os
import re
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

from generate import CHECKSUM_DWORDS, NUM_PLANES, generate_mlir
from reference import expected_output, make_kv_cache

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "aiex.npu.writebd": 8,
        "aiex.npu.address_patch": 8,
        "aiex.npu.push_queue": 8,
        "aiex.npu.sync": 4,
        "aie.flow": 12,
        "aie.memtile_dma": 2,
        "aie.core(": 4,
        "aie.mem(": 4,
    }
    for needle, expected in expected_counts.items():
        actual = mlir_text.count(needle)
        if actual != expected:
            errors.append(f"Expected {expected} {needle}, got {actual}")

    if "dma_configure_task_for" in mlir_text:
        errors.append("Expected raw writebd runtime, found dma_configure_task_for")
    if "aie.shim_dma_allocation" in mlir_text:
        errors.append("Expected raw writebd runtime, found shim_dma_allocation")

    match = re.search(r"aie\.runtime_sequence\(([^)]+)\)", mlir_text)
    if match is None:
        errors.append("No runtime_sequence found")
    else:
        args = match.group(1).split(",")
        if len(args) != 2:
            errors.append(f"Expected 2 runtime args, got {len(args)}")

    for marker in ["checksum_zero", "checksum_accum", "HALF_TILE_DWORDS"]:
        if marker == "HALF_TILE_DWORDS":
            continue
        if marker not in mlir_text:
            errors.append(f"Missing marker: {marker}")
    return errors


def compile_kernel() -> bool:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / "checksum.cc"
    obj = EXPERIMENT_DIR / "checksum.o"
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
    print("  Compiling checksum.cc...")
    return os.system(" ".join(cmd)) == 0


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> bool:
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
    return os.system(" ".join(cmd)) == 0


def run_case(context_len: int) -> bool:
    print(f"\n--- L={context_len} ---")
    build_dir = EXPERIMENT_DIR / "build" / f"L{context_len}"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate_mlir(context_len)
    mlir_path.write_text(mlir_text)
    errors = check_mlir_structure(mlir_text)
    if errors:
        for error in errors:
            print(f"  STRUCTURE FAIL: {error}")
        return False

    if not compile_mlir(mlir_path, xclbin_path, insts_path):
        print("  FAIL: MLIR compilation error")
        return False

    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    kv_cache = make_kv_cache(context_len)
    expected = expected_output(context_len)
    kv_buf = XRTTensor.from_torch(torch.from_numpy(kv_cache.copy()).to(torch.int32))
    out_buf = XRTTensor((NUM_PLANES * CHECKSUM_DWORDS,), dtype=np.int32)

    result = aie_utils.DefaultNPURuntime.run(handle, [kv_buf, out_buf])
    npu_time_us = result.npu_time / 1e3
    got = out_buf.to_torch().numpy().astype(np.int32)
    mismatches = np.where(got != expected)[0]
    if mismatches.size == 0:
        print(f"  PASS  time={npu_time_us:.1f}us")
        print(f"    output={got.reshape(NUM_PLANES, CHECKSUM_DWORDS).tolist()}")
        return True

    print(f"  FAIL  time={npu_time_us:.1f}us mismatches={mismatches.size}")
    for idx in mismatches[:12]:
        print(f"    out[{idx}]: expected={expected[idx]} got={got[idx]}")
    print(f"    got={got.reshape(NUM_PLANES, CHECKSUM_DWORDS).tolist()}")
    print(f"    exp={expected.reshape(NUM_PLANES, CHECKSUM_DWORDS).tolist()}")
    return False


def main() -> bool:
    print("=" * 72)
    print("Experiment 25: MyLM Edge BD Ring Runnable Skeleton")
    print("=" * 72)
    print("Compiling kernel...")
    if not compile_kernel():
        return False

    print(f"NPU device: {aie_utils.DefaultNPURuntime.device()}")
    all_passed = True
    for context_len in [17, 31, 32, 128]:
        all_passed &= run_case(context_len)

    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: edge/KV static-ring checksum skeleton verified on NPU.")
    else:
        print("FAIL: at least one context length failed.")
    print("=" * 72)
    return all_passed


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
