#!/usr/bin/env python3
"""NPU runner for P5: channel ownership + packet ID."""

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
from reference import CASE_NAME, DATA_DWORDS, OUTPUT_DWORDS, make_input, validate_output


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
    print("  PASS: MLIR contains packet_flow with shared source channel and separate destinations")
    return True


def run() -> bool:
    print("=" * 70)
    print("Pattern 5: Channel Ownership + Packet ID")
    print("=" * 70)
    print("  Producer OWNS MM2S ch0 exclusively")
    print("  Same physical channel carries TWO logical streams:")
    print("    packet_id=0 → worker0 (adds +10)")
    print("    packet_id=1 → worker1 (adds +20)")
    print("  Workers filter by packet ID, each sees only its data")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    input_data = make_input()
    input_buf = XRTTensor(input_data.view(np.uint32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.uint32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    from reference import expected_output
    expected = expected_output(input_data)
    print(f"  worker0 (pkt0): expected={expected[:4].tolist()} got={got[:4].tolist()}")
    print(f"  worker1 (pkt1): expected={expected[DATA_DWORDS:DATA_DWORDS+4].tolist()} got={got[DATA_DWORDS:DATA_DWORDS+4].tolist()}")

    errors = validate_output(got, input_data)
    if errors:
        print(f"  FAIL: {len(errors)} mismatches")
        for e in errors:
            print(f"    {e}")
        return False

    print("  PASS: single channel owner, packet ID demuxes to correct consumers")
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
