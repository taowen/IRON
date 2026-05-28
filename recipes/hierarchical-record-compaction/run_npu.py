#!/usr/bin/env python3
"""NPU runner for hierarchical record compaction."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import generate
import npu_build
from reference import (
    CASE_NAME,
    CONSUMER_OUTPUT_DWORDS,
    expected_output,
    route_summary,
    validate_output,
)

EXPERIMENT_DIR = Path(__file__).parent


def build_kernel() -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / CASE_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  COMPACTION STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  COMPACTION STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: MLIR contains producer records -> column compact -> "
        "global compact -> paired consumer"
    )
    return True


def build_only() -> bool:
    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel()
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run() -> bool:
    print("=" * 78)
    print(f"Recipe: Hierarchical Record Compaction")
    print("=" * 78)
    for line in route_summary():
        print(f"  {line}")
    print(
        "  route: producer tiles -> column compactors -> global compactor -> "
        "paired consumer -> host"
    )
    print()

    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    expected = expected_output()
    output_buf = XRTTensor((CONSUMER_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} compaction mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print(
        "  PASS: independent record streams compacted hierarchically and "
        "arrived at the consumer as paired low/high halves"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="generate MLIR and check structure without compiling",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="compile kernel, MLIR, NPU instructions, and xclbin without running",
    )
    return parser.parse_args()


def main() -> bool:
    args = parse_args()
    if args.check_only and args.build_only:
        raise ValueError("--check-only and --build-only are mutually exclusive")
    if args.check_only:
        return check_only()
    if args.build_only:
        return build_only()
    print(f"NPU device: {npu_build.device()}")
    return run()


if __name__ == "__main__":
    try:
        success = main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        success = False
    finally:
        npu_build.cleanup()
    raise SystemExit(0 if success else 1)
