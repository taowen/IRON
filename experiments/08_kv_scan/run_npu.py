#!/usr/bin/env python3
"""
Experiment 08: FastFlowLM KV Scan Mechanism

Proves:
1. ONE runtime descriptor per stream (no per-tile host tasks)
2. Static BD ring auto-advances through ceil(L/16) KV tiles
3. Memtile splits 0x4000-byte KV tile into K+V streams via offset
4. Worker handles non-16-aligned L via tail masking
5. Numerical closure: simplified attention out[h][d] += dot(q[h],k[t,h])*v[t,h][d]
"""

import sys
import os
import re
from pathlib import Path
from math import ceil

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
import aie.utils as aie_utils

from generate import generate_mlir, NUM_HEADS, HEAD_DIM, TOKENS_PER_TILE, KV_TILE_DWORDS
from reference import make_kv_input, attention_reference

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str, L: int) -> list:
    """Static checker: verify split ring invariants in generated MLIR."""
    errors = []
    num_tiles = ceil(L / TOKENS_PER_TILE)

    # 1. Exactly 2 dma_configure_task_for (input + output)
    n_tasks = mlir_text.count("dma_configure_task_for")
    if n_tasks != 2:
        errors.append(f"Expected 2 dma_configure_task_for, got {n_tasks}")

    # 2. Exactly 2 dma_start_task
    n_starts = mlir_text.count("dma_start_task")
    if n_starts != 2:
        errors.append(f"Expected 2 dma_start_task, got {n_starts}")

    # 3. Check memtile has 3 DMA channels: S2MM 0, MM2S 0, MM2S 1
    if "dma_start(S2MM, 0" not in mlir_text:
        errors.append("Missing memtile S2MM ch0")
    if "dma_start(MM2S, 0" not in mlir_text:
        errors.append("Missing memtile MM2S ch0")
    if "dma_start(MM2S, 1" not in mlir_text:
        errors.append("Missing memtile MM2S ch1")

    # 4. Check offset split: MM2S ch1 BDs should use offset=2048
    # Look for dma_bd with offset 2048
    offset_bds = re.findall(r'dma_bd\(%buf_p\w+ : memref<4096xi32>, (\d+), (\d+)\)', mlir_text)
    offsets_found = [int(o) for o, _ in offset_bds]
    if 2048 not in offsets_found:
        errors.append(f"No BD with offset=2048 found (offsets: {offsets_found})")

    # 5. Lock init values for split protocol
    lock_inits = re.findall(r'init = (\d+).*?sym_name = "(\w+)"', mlir_text)
    lock_map = {name: int(val) for val, name in lock_inits}
    for name in ["ping_rdy", "pong_rdy"]:
        if lock_map.get(name) != 2:
            errors.append(f"Lock {name} init should be 2, got {lock_map.get(name)}")
    for name in ["ping_done", "pong_done"]:
        if lock_map.get(name) != 0:
            errors.append(f"Lock {name} init should be 0, got {lock_map.get(name)}")

    # 6. BD ring structure: check next_bd_id forms cycles
    bd_pairs = re.findall(r'bd_id = (\d+).*?next_bd_id = (\d+)', mlir_text)
    if len(bd_pairs) < 10:
        errors.append(f"Expected >=10 BD ring entries (memtile 6 + core 4), got {len(bd_pairs)}")

    return errors


def compile_kernel(build_dir):
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    src = EXPERIMENT_DIR / "kv_attention.cc"
    obj = EXPERIMENT_DIR / "kv_attention.o"

    cmd = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c", str(src), "-o", str(obj),
    ]
    print(f"  Compiling kv_attention.cc...")
    ret = os.system(" ".join(cmd))
    return ret == 0


def compile_mlir(build_dir, mlir_path):
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    cmd = [
        str(aiecc), "-v", "-j1",
        "--no-compile-host", "--no-xchesscc", "--no-xbridge",
        "--peano", str(peano_dir),
        "--aie-generate-xclbin", f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print(f"  Compiling MLIR...")
    ret = os.system(" ".join(cmd))
    return ret == 0, xclbin_path, insts_path


def run_test(L: int):
    """Build, load, and run for a given effective length L."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    label = f"L={L}, tiles={num_tiles}, last_valid={last_valid}"
    print(f"\n--- {label} ---")

    build_dir = EXPERIMENT_DIR / "build" / f"L{L}"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Generate MLIR
    mlir_text = generate_mlir(L)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Static structure check
    errors = check_mlir_structure(mlir_text, L)
    if errors:
        for e in errors:
            print(f"  STRUCTURE FAIL: {e}")
        return False

    # Compile MLIR
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        print("  FAIL: MLIR compilation error")
        return False

    # Load
    from aie.utils.npukernel import NPUKernel
    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    # Prepare data
    kv_data = make_kv_input(L)
    expected = attention_reference(L)

    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    input_t = torch.from_numpy(kv_data.copy()).to(torch.int32)
    input_buf = XRTTensor.from_torch(input_t)
    output_buf = XRTTensor((128,), dtype=np.int32)

    # Run
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    npu_time_us = result.npu_time / 1e3

    # Check output
    output_torch = output_buf.to_torch()
    npu_output = output_torch.numpy().astype(np.int32)

    mismatches = []
    for i in range(128):
        if npu_output[i] != expected[i]:
            mismatches.append(i)

    passed = len(mismatches) == 0
    if passed:
        print(f"  PASS  time={npu_time_us:.1f}us  output[0:4]={npu_output[:4]}")
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  {len(mismatches)}/128 elements wrong")
        for i in mismatches[:8]:
            print(f"    out[{i}]: expected={expected[i]} got={npu_output[i]}")

    return passed


def main():
    print("=" * 70)
    print("Experiment 08: FastFlowLM KV Scan Mechanism")
    print("=" * 70)
    print()

    # Compile kernel once
    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    print("Compiling kernel...")
    if not compile_kernel(build_dir):
        print("FAILED: Kernel compilation error")
        return False

    dev = aie_utils.DefaultNPURuntime.device()
    print(f"NPU device: {dev}")

    all_passed = True

    # Test all L values
    test_cases = [1, 15, 16, 17, 31, 32, 79]
    for L in test_cases:
        passed = run_test(L)
        all_passed &= passed

    print()
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All tests pass.")
        print("  1. Single descriptor per stream (no per-tile host tasks)")
        print("  2. BD ring auto-advances through variable tile counts")
        print("  3. Memtile splits 0x4000 KV tile → K + V streams")
        print("  4. Tail masking correct for non-16-aligned L")
        print("  5. Attention computation matches CPU reference exactly")
    else:
        print("FAIL: Some tests failed.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    sys.exit(0 if success else 1)
