#!/usr/bin/env python3
"""NPU runner for P1: tile-local work units.

2 tiles each own a fixed output slice (4 rows), receive weight+activation
chunks, accumulate locally, then emit a fixed-format record (header+payload).
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
    ACT_DWORDS,
    CASE_NAME,
    NUM_TILES,
    OUTPUT_DWORDS,
    RECORD_DWORDS,
    ROWS_PER_TILE,
    TOTAL_WEIGHT_DWORDS,
    make_activation,
    make_weights,
    pack_weights_chunk_major,
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
    print("  PASS: MLIR structure valid (2 tiles, record ABI, weight+act streams)")
    return True


def run() -> bool:
    print("=" * 70)
    print("Pattern 1: Tile-Local Work Units")
    print("=" * 70)
    print(f"  {NUM_TILES} tiles, each owns {ROWS_PER_TILE} output rows")
    print(f"  Each tile: weight chunk ({generate.WEIGHT_CHUNK_DWORDS} dw) + act chunk ({generate.CHUNK_COLS} dw) → MAC → accumulate")
    print(f"  Output: {RECORD_DWORDS}-dword record per tile (1 header + {ROWS_PER_TILE} payload)")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    weights = make_weights()
    activation = make_activation()
    packed_weights = pack_weights_chunk_major(weights)

    wt_buf = XRTTensor(packed_weights.view(np.uint32))
    act_buf = XRTTensor(activation.view(np.uint32))
    out_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.uint32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [wt_buf, act_buf, out_buf])
    got = out_buf.to_torch().numpy().view(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    from reference import expected_output
    expected = expected_output(weights, activation)
    for tile in range(NUM_TILES):
        base = tile * RECORD_DWORDS
        print(f"  tile{tile} record: header=0x{got[base]:08X} payload={got[base+1:base+1+ROWS_PER_TILE].tolist()}")

    errors = validate_output(got, weights, activation)
    if errors:
        print(f"  FAIL: {len(errors)} mismatches")
        for e in errors:
            print(f"    {e}")
        return False

    print("  PASS: both tiles emitted correct records (header + matvec payload)")
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
