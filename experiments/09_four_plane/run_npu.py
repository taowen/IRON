#!/usr/bin/env python3
"""
Experiment 09: Four-Plane Fanout with GQA Head-Group Mapping

Proves:
1. 4 independent shim channels (k03/k47 on col0, v03/v47 on col1)
2. 2 memtiles with dual BD rings (ring A + ring B per memtile)
3. Cross-column routing (worker0 gets K from col0, V from col1)
4. GQA head-group separation (worker0=heads 0-3, worker1=heads 4-7)
5. Numerical closure against CPU reference for both workers
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

from generate import generate_mlir, NUM_KV_HEADS_PER_GROUP, HEAD_DIM, TOKENS_PER_TILE, PLANE_TILE_DWORDS
from reference import make_plane_input, attention_reference

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str, L: int) -> list:
    """Static checker: verify 4-plane fanout invariants in generated MLIR."""
    errors = []
    num_tiles = ceil(L / TOKENS_PER_TILE)

    # 1. Exactly 6 dma_configure_task_for (4 input + 2 output)
    n_tasks = mlir_text.count("dma_configure_task_for")
    if n_tasks != 6:
        errors.append(f"Expected 6 dma_configure_task_for, got {n_tasks}")

    # 2. Exactly 6 dma_start_task (all 6 tasks are started)
    n_starts = mlir_text.count("dma_start_task")
    if n_starts != 6:
        errors.append(f"Expected 6 dma_start_task, got {n_starts}")

    # 3. Check 10 flows
    n_flows = mlir_text.count("aie.flow")
    if n_flows != 10:
        errors.append(f"Expected 10 aie.flow, got {n_flows}")

    # 4. Check 2 memtile_dma blocks
    n_memtile_dma = mlir_text.count("aie.memtile_dma")
    if n_memtile_dma != 2:
        errors.append(f"Expected 2 aie.memtile_dma, got {n_memtile_dma}")

    # 5. Check BD ID 24/25/26/27 present (ch1 must use these ranges)
    for bd_id in [24, 25, 26, 27]:
        if f"bd_id = {bd_id}" not in mlir_text:
            errors.append(f"Missing bd_id = {bd_id} (ch1 range)")

    # 6. Check both head_offset values present
    if "arith.constant 0 : i32" not in mlir_text:
        errors.append("Missing head_offset=0 for worker0")
    if "arith.constant 4 : i32" not in mlir_text:
        errors.append("Missing head_offset=4 for worker1")

    # 7. Check 2 core blocks
    n_cores = mlir_text.count("aie.core(")
    if n_cores != 2:
        errors.append(f"Expected 2 aie.core, got {n_cores}")

    # 8. Check 4 shim_dma_allocation for input + 2 for output
    n_alloc = mlir_text.count("aie.shim_dma_allocation")
    if n_alloc != 6:
        errors.append(f"Expected 6 shim_dma_allocation, got {n_alloc}")

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

    # Prepare data: 4 input planes + 2 output buffers
    k03_data = make_plane_input(L, 'k03')
    v03_data = make_plane_input(L, 'v03')
    k47_data = make_plane_input(L, 'k47')
    v47_data = make_plane_input(L, 'v47')
    expected = attention_reference(L)

    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    k03_buf = XRTTensor.from_torch(torch.from_numpy(k03_data.copy()).to(torch.int32))
    v03_buf = XRTTensor.from_torch(torch.from_numpy(v03_data.copy()).to(torch.int32))
    k47_buf = XRTTensor.from_torch(torch.from_numpy(k47_data.copy()).to(torch.int32))
    v47_buf = XRTTensor.from_torch(torch.from_numpy(v47_data.copy()).to(torch.int32))
    out_buf = XRTTensor((256,), dtype=np.int32)

    # Run (5 buffer args — XRT kernel limit)
    result = aie_utils.DefaultNPURuntime.run(handle, [k03_buf, v03_buf, k47_buf, v47_buf, out_buf])
    npu_time_us = result.npu_time / 1e3

    # Check output
    npu_output = out_buf.to_torch().numpy().astype(np.int32)

    mismatches = []
    for i in range(256):
        if npu_output[i] != expected[i]:
            mismatches.append(i)

    passed = len(mismatches) == 0
    if passed:
        print(f"  PASS  time={npu_time_us:.1f}us  w0[0:4]={npu_output[:4]}  w1[0:4]={npu_output[128:132]}")
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  {len(mismatches)}/256 elements wrong")
        for i in mismatches[:8]:
            worker = "w0" if i < 128 else "w1"
            local_i = i if i < 128 else i - 128
            print(f"    {worker}[{local_i}]: expected={expected[i]} got={npu_output[i]}")

    return passed


def main():
    print("=" * 70)
    print("Experiment 09: Four-Plane Fanout with GQA Head-Group Mapping")
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
        print("  1. 4 independent shim channels (one per plane)")
        print("  2. 2 memtiles with dual BD rings (ring A + ring B)")
        print("  3. Cross-column routing (K from col0, V from col1)")
        print("  4. GQA head-group separation verified")
        print("  5. Both workers match CPU reference exactly")
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
