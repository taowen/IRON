#!/usr/bin/env python3
"""NPU runner for P09: RTP + descriptor patch.

Demonstrates two layers of runtime parameters:
  Layer 1 (RTP): tile reads scalar offset
  Layer 2 (Descriptor): BD buffer_offset selects input slice

Both change between runs without recompiling xclbin.
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
from reference import CASE_NAME, SLICE_DWORDS, TOTAL_INPUT_DWORDS, make_input, validate_output


def build_and_run(input_data: np.ndarray, rtp_value: int, slice_idx: int) -> tuple[np.ndarray, float]:
    """Build with specific (rtp_value, slice_idx) and run."""
    import subprocess, io, contextlib
    build_dir = PATTERN_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rtp{rtp_value}_slice{slice_idx}"
    mlir_path = build_dir / f"design_{tag}.mlir"
    xclbin_path = build_dir / f"design_{tag}.xclbin"
    insts_path = build_dir / f"design_{tag}.bin"

    with contextlib.redirect_stdout(io.StringIO()):
        npu_build.compile_aie_object(PATTERN_DIR, "kernel.cc", "kernel.o")
        mlir_text = generate.generate_mlir(rtp_value=rtp_value, slice_idx=slice_idx)
        mlir_path.write_text(mlir_text)
        npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)

    handle = npu_build.load_kernel(xclbin_path, insts_path)
    input_buf = XRTTensor(input_data.view(np.uint32))
    output_buf = XRTTensor((SLICE_DWORDS,), dtype=np.uint32)
    result = npu_build.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    return got, result.npu_time / 1e3


def check_only() -> bool:
    mlir_text = generate.generate_mlir(rtp_value=42, slice_idx=2)
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASS: MLIR has rtp_write + runtime_start lock + buffer_offset descriptor patch")
    return True


def run(rtp_value: int = 42, slice_idx: int = 2) -> bool:
    print("=" * 70)
    print("Pattern 9: RTP + Descriptor Patch")
    print("=" * 70)
    print("  Layer 1 (RTP): rtp_write sets additive offset, core reads after lock")
    print("  Layer 2 (Descriptor): writebd buffer_offset selects input slice")
    print(f"  This run: rtp={rtp_value}, slice={slice_idx} (offset={slice_idx*SLICE_DWORDS*4} bytes)")
    print()

    input_data = make_input()
    got, npu_us = build_and_run(input_data, rtp_value, slice_idx)
    print(f"  NPU time: {npu_us:.1f} us")

    from reference import expected_output
    expected = expected_output(input_data, slice_idx, rtp_value)
    print(f"  expected[0:4]: {expected[:4].tolist()}")
    print(f"  got[0:4]:      {got[:4].tolist()}")

    errors = validate_output(got, input_data, slice_idx, rtp_value)
    if errors:
        print(f"  FAIL: {errors}")
        return False

    print(f"  PASS: input[{slice_idx*SLICE_DWORDS}:{(slice_idx+1)*SLICE_DWORDS}] + {rtp_value}")
    print(f"  (Run again with --rtp and --slice to try different parameters)")
    return True


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rtp", type=int, default=42, help="RTP additive offset")
    parser.add_argument("--slice", type=int, default=2, help="Input slice index (0-3)")
    args = parser.parse_args()

    if args.check_only:
        return check_only()
    if args.build_only:
        build_dir = PATTERN_DIR / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        npu_build.compile_aie_object(PATTERN_DIR, "kernel.cc", "kernel.o")
        mlir_text = generate.generate_mlir(rtp_value=args.rtp, slice_idx=args.slice)
        mlir_path = build_dir / "design.mlir"
        mlir_path.write_text(mlir_text)
        npu_build.compile_mlir(mlir_path, build_dir / "design.xclbin", build_dir / "design.bin")
        print("  PASS: built successfully")
        return True
    print(f"NPU device: {npu_build.device()}")
    return run(rtp_value=args.rtp, slice_idx=args.slice)


if __name__ == "__main__":
    try:
        success = main()
    except Exception:
        traceback.print_exc()
        success = False
    raise SystemExit(0 if success else 1)
