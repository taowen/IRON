#!/usr/bin/env python3
"""NPU runner for P11: integration boundary testing.

Demonstrates three-level verification:
1. Structure check (MLIR markers)
2. Poison check (no 0xDEADBEEF remains in output)
3. Value check (mathematical correctness)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

PATTERN_DIR = Path(__file__).parent
sys.path.insert(0, str(PATTERN_DIR.parent))
sys.path.insert(0, str(PATTERN_DIR))

import generate
import npu_build
from reference import (
    CASE_NAME,
    DATA_DWORDS,
    POISON_VALUE,
    check_poison,
    check_values,
    make_input,
)


def build_kernel() -> tuple[Path, Path]:
    build_dir = PATTERN_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    npu_build.compile_aie_object(PATTERN_DIR, "kernel.cc", "kernel.o")

    mlir_text = generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  STRUCTURE FAIL: {e}" for e in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    print("  Level 1: Structure check")
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for e in errors:
            print(f"    FAIL: {e}")
        return False
    print("    PASS: MLIR contains expected tiles, flows, locks, kernel link")
    return True


def run() -> bool:
    print("=" * 70)
    print("Pattern 11: Integration Boundary Testing")
    print("=" * 70)
    print("  Three-level verification: structure → poison → value")
    print()

    # Level 1: structure check
    print("  Level 1: Structure check")
    mlir_text = generate.generate_mlir()
    struct_errors = generate.validate_generated_mlir(mlir_text)
    if struct_errors:
        for e in struct_errors:
            print(f"    FAIL: {e}")
        return False
    print("    PASS")

    xclbin_path, insts_path = build_kernel()
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    input_data = make_input()
    input_buf = XRTTensor(input_data.view(np.uint32))

    # Pre-fill output with POISON to detect untouched elements
    poison_array = np.full(DATA_DWORDS, POISON_VALUE, dtype=np.int32)
    output_buf = XRTTensor(poison_array.view(np.uint32))

    print("  Running on NPU (output pre-poisoned with 0x7EADBEEF)...")
    result = npu_build.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    # Level 2: poison check
    print("  Level 2: Poison check")
    poison_errors = check_poison(got)
    if poison_errors:
        for e in poison_errors:
            print(f"    FAIL: {e}")
        print("    DMA path failed: some output slots were never written by NPU")
        return False
    print("    PASS: no poison remains (all slots overwritten by NPU)")

    # Level 3: value check
    print("  Level 3: Value check")
    value_errors = check_values(got, input_data)
    if value_errors:
        for e in value_errors:
            print(f"    FAIL: {e}")
        print("    Math error: DMA delivered data but computation is wrong")
        return False

    from reference import expected_output
    expected = expected_output(input_data)
    print(f"    expected[0:4]: {expected[:4].tolist()}")
    print(f"    got[0:4]:      {got[:4].tolist()}")
    print("    PASS: output = input*3 + 7 for all elements")

    print()
    print("  ALL THREE LEVELS PASS")
    return True


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        return check_only()
    if args.build_only:
        build_kernel()
        print("  PASS: built successfully")
        return True
    print(f"NPU device: {npu_build.device()}")
    return run()


if __name__ == "__main__":
    try:
        success = main()
    except Exception:
        traceback.print_exc()
        success = False
    finally:
        npu_build.cleanup()
    raise SystemExit(0 if success else 1)
