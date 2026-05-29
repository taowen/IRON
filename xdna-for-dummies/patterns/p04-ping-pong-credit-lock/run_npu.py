#!/usr/bin/env python3
"""NPU runner for P4: ping-pong credit lock."""

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
from reference import CASE_NAME, OUTPUT_DWORDS, expected_output, validate_output


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
    print("  PASS: MLIR structure valid (ping-pong producer -> consumer -> host)")
    return True


def build_only() -> bool:
    xclbin_path, insts_path = build_kernel()
    print(f"  PASS: built {xclbin_path}")
    return True


def run() -> bool:
    print("=" * 70)
    print(f"Pattern 4: Ping-Pong + Credit Lock")
    print("=" * 70)
    print(f"  producer(2,2) --[ping/pong]--> consumer(2,3) --[ping/pong]--> host")
    print(f"  {OUTPUT_DWORDS} dwords output = {generate.NUM_BATCHES} batches x {generate.CHUNK_DWORDS} dwords")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.uint32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    expected = expected_output()
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} mismatches")
        for e in errors:
            print(f"    {e}")
        return False

    print("  PASS: ping-pong producer -> consumer -> host verified")
    return True


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        return check_only()
    if args.build_only:
        return build_only()
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
