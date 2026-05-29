#!/usr/bin/env python3
"""NPU runner for P08: append/scan separation.

Two DMA phases separated by npu.sync:
  Phase 1: writer tile → shim S2MM → cache BO at position-dependent offset
  npu.sync (ensures append visible)
  Phase 2: shim MM2S → entire cache BO → scanner tile → sum
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
    APPEND_POSITION,
    CACHE_DWORDS,
    CASE_NAME,
    OUTPUT_DWORDS,
    make_cache,
    validate_output,
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
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASS: MLIR has two-phase structure (append BD → sync → scan BD)")
    return True


def run() -> bool:
    print("=" * 70)
    print("Pattern 8: Append/Scan Separation (descriptor-level)")
    print("=" * 70)
    print(f"  Phase 1: writer tile → shim S2MM → cache BO at offset {APPEND_POSITION*4} bytes")
    print(f"  npu.sync (ensures append visible in BO)")
    print(f"  Phase 2: shim MM2S → entire cache BO ({CACHE_DWORDS} dw) → scanner tile → sum")
    print(f"  Cache positions {APPEND_POSITION}..{APPEND_POSITION+3} poisoned to verify ordering")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    cache = make_cache()
    cache_buf = XRTTensor(cache.view(np.uint32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.uint32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [cache_buf, output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    # Read back cache to verify append actually wrote
    cache_after = cache_buf.to_torch().numpy().view(np.int32)
    print(f"  cache[{APPEND_POSITION}:{APPEND_POSITION+4}] after: {cache_after[APPEND_POSITION:APPEND_POSITION+4].tolist()}")

    from reference import expected_output, append_values
    expected = expected_output(cache)
    print(f"  expected sum: {expected[0]}")
    print(f"  got sum:      {got[0]}")

    errors = validate_output(got, cache)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    print("  PASS: Phase 1 append → sync → Phase 2 scan sees appended data")
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
