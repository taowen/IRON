"""NPU runner for the c1r2 full-vector replay integration case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import c1r2_generate
import npu_build
from c1r2_reference import (
    CASE_NAME,
    OUTPUT_DWORDS,
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

    mlir_text = c1r2_generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = c1r2_generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  C1R2 STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = c1r2_generate.generate_mlir()
    errors = c1r2_generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  C1R2 STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: c1r2-o-upgate-bridge MLIR contains "
        "O compact -> c1r2 -> packet0 replay -> c1r1/main16 -> c6r2"
    )
    return True


def build_only() -> bool:
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
        "  route: main16 O record -> row1/c1r1 compact -> c1r2 -> "
        "c1r1 packet0 bridge -> main16 up/gate summary -> row1/c1r1 -> c6r2"
    )
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    expected = expected_output()
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} c1r2 bridge mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print(
        "  PASS: c1r2 replayed 48 packet0 full-vector payloads through "
        "c1r1 and the resulting up/gate records reached c6r2"
    )
    return True
