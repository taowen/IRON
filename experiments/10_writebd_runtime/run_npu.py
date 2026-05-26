#!/usr/bin/env python3
"""
Experiment 10: writebd Runtime with Single KV Cache BO

Proves:
1. Single KV cache BO — 4 planes at offsets into one buffer (not 4 separate BOs)
2. writebd + address_patch + push_queue — raw NPU instruction format
3. sync for completion — aiex.npu.sync on S2MM output channels
4. Numerical equivalence — same data path as exp 09, proving runtime form is transparent
"""

import sys
import os
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
    """Static checker: verify writebd runtime invariants in generated MLIR."""
    errors = []

    # 1. No dma_configure_task_for (fully replaced)
    n_old = mlir_text.count("dma_configure_task_for")
    if n_old != 0:
        errors.append(f"Expected 0 dma_configure_task_for, got {n_old}")

    # 2. Exactly 6 writebd (4 input + 2 output)
    n_writebd = mlir_text.count("aiex.npu.writebd")
    if n_writebd != 6:
        errors.append(f"Expected 6 aiex.npu.writebd, got {n_writebd}")

    # 3. Exactly 6 address_patch
    n_patch = mlir_text.count("aiex.npu.address_patch")
    if n_patch != 6:
        errors.append(f"Expected 6 aiex.npu.address_patch, got {n_patch}")

    # 4. Exactly 6 push_queue
    n_push = mlir_text.count("aiex.npu.push_queue")
    if n_push != 6:
        errors.append(f"Expected 6 aiex.npu.push_queue, got {n_push}")

    # 5. Exactly 2 sync
    n_sync = mlir_text.count("aiex.npu.sync")
    if n_sync != 2:
        errors.append(f"Expected 2 aiex.npu.sync, got {n_sync}")

    # 6. Only 2 runtime_sequence args
    import re
    rt_match = re.search(r'aie\.runtime_sequence\(([^)]+)\)', mlir_text)
    if rt_match:
        args = rt_match.group(1).split(',')
        if len(args) != 2:
            errors.append(f"Expected 2 runtime_sequence args, got {len(args)}")
    else:
        errors.append("No runtime_sequence found")

    # 7. Check 10 flows (same as exp 09)
    n_flows = mlir_text.count("aie.flow")
    if n_flows != 10:
        errors.append(f"Expected 10 aie.flow, got {n_flows}")

    # 8. Check 2 memtile_dma blocks
    n_memtile_dma = mlir_text.count("aie.memtile_dma")
    if n_memtile_dma != 2:
        errors.append(f"Expected 2 aie.memtile_dma, got {n_memtile_dma}")

    # 9. Check BD ID 24/25/26/27 present (memtile ch1 range)
    for bd_id in [24, 25, 26, 27]:
        if f"bd_id = {bd_id}" not in mlir_text:
            errors.append(f"Missing bd_id = {bd_id} (ch1 range)")

    # 10. No shim_dma_allocation (writebd replaces it)
    n_alloc = mlir_text.count("aie.shim_dma_allocation")
    if n_alloc != 0:
        errors.append(f"Expected 0 shim_dma_allocation, got {n_alloc}")

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
    total_plane_elems = num_tiles * PLANE_TILE_DWORDS
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

    # Prepare data: pack 4 planes into single KV cache BO
    k03_data = make_plane_input(L, 'k03')
    v03_data = make_plane_input(L, 'v03')
    k47_data = make_plane_input(L, 'k47')
    v47_data = make_plane_input(L, 'v47')

    # Layout: k03 | v03 | k47 | v47 (matches generate.py offsets)
    kv_cache_data = np.concatenate([k03_data, v03_data, k47_data, v47_data])
    expected = attention_reference(L)

    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    kv_cache_buf = XRTTensor.from_torch(torch.from_numpy(kv_cache_data.copy()).to(torch.int32))
    out_buf = XRTTensor((256,), dtype=np.int32)

    # Run (2 buffer args — single KV cache BO + output)
    result = aie_utils.DefaultNPURuntime.run(handle, [kv_cache_buf, out_buf])
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
    print("Experiment 10: writebd Runtime with Single KV Cache BO")
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
        print("  1. Single KV cache BO with 4 plane offsets")
        print("  2. writebd + address_patch + push_queue (raw NPU form)")
        print("  3. sync for S2MM completion")
        print("  4. Numerical equivalence with exp 09 reference")
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
