"""NPU runner for main16 -> row1/c1r1 -> c6r2 FFN bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import npu_build
import swiglu_generate
from swiglu_reference import (
    CASE_NAME,
    SWIGLU_OUTPUT_DWORDS,
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

    mlir_text = swiglu_generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = swiglu_generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  SWIGLU STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = swiglu_generate.generate_mlir()
    errors = swiglu_generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  SWIGLU STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: ffn-upgate-c6r2-bridge MLIR contains "
        "main16 records -> row1 column compact -> c1r1 global compact -> c6r2"
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
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    for line in route_summary():
        print(f"  {line}")
    print(
        "  route: main16 stubs -> c2r1..c5r1 column compact -> "
        "c1r1 global compact -> c6r2 -> c6r1 -> host"
    )
    print()

    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    expected = expected_output()
    output_buf = XRTTensor((SWIGLU_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} ffn-upgate-c6r2 mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print(
        "  PASS: main16 up/gate records compacted through row1/c1r1 and "
        "arrived at c6r2 with low=up, high=gate"
    )
    return True
