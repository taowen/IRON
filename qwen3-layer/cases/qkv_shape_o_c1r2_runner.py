"""NPU runner for qkv-shape-o-c1r2-bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import npu_build
from cases import qkv_shape_o_c1r2_generate as generate
from cases.qkv_shape_o_c1r2_reference import (
    CASE_NAME,
    OUTPUT_DWORDS,
    expected_output,
    route_summary,
    validate_output,
)

EXPERIMENT_DIR = Path(__file__).parent.parent


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
        raise RuntimeError("\n".join(f"  QKV STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  QKV STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: qkv-shape-o-c1r2-bridge MLIR contains "
        "main16 Q/K/V compact -> c1r3 -> Shape-A/B -> packet2 -> O compact -> c1r2"
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
    print()

    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    expected = expected_output()
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected: {expected.tolist()}")
    print(f"  got:      {got.tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} qkv bridge mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: main16 Q/K/V producer reached c1r2 through Shape-A/B and O compact")
    return True
