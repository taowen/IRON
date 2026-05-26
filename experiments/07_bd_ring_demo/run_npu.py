#!/usr/bin/env python3
"""
Experiment 07a: Memtile BD Ring Demo

Proves:
1. Static BD ring in memtile (ping-pong, bd0→bd1→bd0)
2. Runtime: ONE shim descriptor per stream, no per-chunk tasks
3. Core consumes N chunks via lock acquire/release
4. Per-chunk checksum output verifies ordering and ping/pong correctness
5. Works for odd/even num_chunks including 1, 3, 5, 17

NOTE: Runtime uses dma_configure_task_for (07a).
      07b will use npu.writebd/address_patch/push_queue/sync.
"""

import sys
import os
import re
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
import aie.utils as aie_utils

from generate import generate_mlir, CHUNK_ELEMS_SMOKE, CHUNK_ELEMS_REAL
from reference import make_input, checksums_reference

EXPERIMENT_DIR = Path(__file__).parent


def check_mlir_structure(mlir_text: str, num_chunks: int, chunk_elems: int) -> list:
    """Static checker: verify BD ring invariants in generated MLIR."""
    errors = []

    # 1. Exactly 2 dma_configure_task_for (input + output)
    n_tasks = mlir_text.count("dma_configure_task_for")
    if n_tasks != 2:
        errors.append(f"Expected 2 dma_configure_task_for, got {n_tasks}")

    # 2. Memtile S2MM ring: bd0→bd1→bd0
    mt_s2mm_bds = re.findall(r'bd_id = (\d+).*?next_bd_id = (\d+)', mlir_text)
    if len(mt_s2mm_bds) < 2:
        errors.append(f"Expected >=2 BD entries with next_bd_id, got {len(mt_s2mm_bds)}")
    else:
        # Check first ring: 0→1, 1→0
        if mt_s2mm_bds[0] != ('0', '1'):
            errors.append(f"First BD ring entry: expected (0,1), got {mt_s2mm_bds[0]}")
        if mt_s2mm_bds[1] != ('1', '0'):
            errors.append(f"Second BD ring entry: expected (1,0), got {mt_s2mm_bds[1]}")
        # Check memtile MM2S ring: 2→3, 3→2
        if len(mt_s2mm_bds) >= 4:
            if mt_s2mm_bds[2] != ('2', '3'):
                errors.append(f"Memtile MM2S bd2: expected (2,3), got {mt_s2mm_bds[2]}")
            if mt_s2mm_bds[3] != ('3', '2'):
                errors.append(f"Memtile MM2S bd3: expected (3,2), got {mt_s2mm_bds[3]}")
        # Check core S2MM ring: 0→1, 1→0
        if len(mt_s2mm_bds) >= 6:
            if mt_s2mm_bds[4] != ('0', '1'):
                errors.append(f"Core S2MM bd0: expected (0,1), got {mt_s2mm_bds[4]}")
            if mt_s2mm_bds[5] != ('1', '0'):
                errors.append(f"Core S2MM bd1: expected (1,0), got {mt_s2mm_bds[5]}")

    # 3. Lock init values
    lock_inits = re.findall(r'init = (\d+).*?sym_name = "(\w+)"', mlir_text)
    lock_map = {name: int(val) for val, name in lock_inits}
    for name in ["mt_empty", "core_empty"]:
        if lock_map.get(name) != 2:
            errors.append(f"Lock {name} init should be 2, got {lock_map.get(name)}")
    for name in ["mt_full", "core_full"]:
        if lock_map.get(name) != 0:
            errors.append(f"Lock {name} init should be 0, got {lock_map.get(name)}")

    # 4. No per-chunk patterns (no repeated dma_start_task beyond 2)
    n_starts = mlir_text.count("dma_start_task")
    if n_starts != 2:
        errors.append(f"Expected 2 dma_start_task, got {n_starts}")

    return errors


def compile_kernel(build_dir):
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    clang = peano_dir / "bin" / "clang++"

    src = EXPERIMENT_DIR / "checksum.cc"
    obj = EXPERIMENT_DIR / "checksum.o"

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
    print(f"  Compiling checksum.cc...")
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


def run_test(num_chunks, chunk_elems):
    """Build, load, and run for a given configuration."""
    label = f"num_chunks={num_chunks}, chunk_elems={chunk_elems}"
    print(f"\n--- {label} ---")

    build_dir = EXPERIMENT_DIR / "build" / f"n{num_chunks}_c{chunk_elems}"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Generate MLIR
    mlir_text = generate_mlir(num_chunks, chunk_elems)
    mlir_path = build_dir / "design.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Static structure check
    errors = check_mlir_structure(mlir_text, num_chunks, chunk_elems)
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
    input_data = make_input(num_chunks, chunk_elems)
    expected = checksums_reference(input_data, num_chunks, chunk_elems)

    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    input_t = torch.from_numpy(input_data.copy()).to(torch.int32)
    input_buf = XRTTensor.from_torch(input_t)
    output_buf = XRTTensor((num_chunks,), dtype=np.int32)

    # Run
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    npu_time_us = result.npu_time / 1e3

    # Check per-chunk checksums
    output_torch = output_buf.to_torch()
    npu_output = output_torch.numpy().astype(np.int32)

    mismatches = []
    for i in range(num_chunks):
        if npu_output[i] != expected[i]:
            mismatches.append(i)

    passed = len(mismatches) == 0
    if passed:
        print(f"  PASS  time={npu_time_us:.1f}us  all {num_chunks} chunk checksums match")
    else:
        print(f"  FAIL  time={npu_time_us:.1f}us  {len(mismatches)}/{num_chunks} chunks wrong")
        for i in mismatches[:5]:
            print(f"    chunk[{i}]: expected={expected[i]} got={npu_output[i]}")

    return passed


def main():
    print("=" * 70)
    print("Experiment 07a: Memtile BD Ring Demo")
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

    # Smoke tests (small chunks, fast compile)
    print(f"\n{'='*40} SMOKE (chunk={CHUNK_ELEMS_SMOKE}) {'='*40}")
    for n in [1, 2, 3, 4, 5, 8, 17]:
        passed = run_test(n, CHUNK_ELEMS_SMOKE)
        all_passed &= passed

    # Real-size tests (4096 int32 = 0x4000 bytes, matches FastFlowLM KV tile)
    print(f"\n{'='*40} REAL (chunk={CHUNK_ELEMS_REAL}) {'='*40}")
    for n in [1, 3, 4, 8]:
        passed = run_test(n, CHUNK_ELEMS_REAL)
        all_passed &= passed

    print()
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All tests pass.")
        print("  Static memtile BD ring verified for odd/even chunks")
        print("  Per-chunk checksums confirm ordering + ping/pong correctness")
        print(f"  Tested with chunk sizes: {CHUNK_ELEMS_SMOKE} and {CHUNK_ELEMS_REAL} int32")
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
