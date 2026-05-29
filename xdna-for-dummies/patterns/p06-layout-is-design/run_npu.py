#!/usr/bin/env python3
"""NPU runner for P6: layout is design."""

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
from reference import CASE_NAME, OUTPUT_DWORDS, TOTAL_ELEMENTS, make_data_interleaved, pack_even_odd, validate_output


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
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASS: MLIR structure valid")
    return True


def run() -> bool:
    print("=" * 70)
    print("Pattern 6: Layout Is Part of the Design")
    print("=" * 70)
    print("  Problem: tile0 needs even elements [0,2,4,...], tile1 needs odd [1,3,5,...]")
    print("  In natural layout [0,1,2,3,...], evens and odds are INTERLEAVED (non-contiguous)")
    print()
    print("  Solution: pre-pack as [all_evens | all_odds]")
    print("    → 1 BD per tile, contiguous, no 2D stride needed")
    print("  Without pre-pack: would need d0_stride=2 to skip every other element")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    data = make_data_interleaved()
    packed = pack_even_odd(data)
    input_buf = XRTTensor(packed.view(np.uint32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.uint32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    from reference import expected_output
    expected = expected_output(data)
    print(f"  tile0 (evens sum): expected={expected[0]} got={got[0]}")
    print(f"  tile1 (odds sum):  expected={expected[1]} got={got[1]}")

    errors = validate_output(got, data)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    print("  PASS: pre-packed even/odd layout → simple 1D BDs, correct result")
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
