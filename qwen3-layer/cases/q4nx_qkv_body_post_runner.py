"""NPU runner for q4nx-qkv-body-post-bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import npu_build
from cases import q4nx_qkv_body_post_generate as generate
from cases.q4nx_qkv_body_post_reference import (
    CASE_NAME,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    TOTAL_WEIGHT_I32,
    expected_output,
    hidden_as_i32,
    make_packed_weights,
    packed_as_i32,
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
        raise RuntimeError("\n".join(f"  Q4NX QKV BODY POST STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  Q4NX QKV BODY POST STRUCTURE FAIL: {error}")
        return False
    print("  PASS: q4nx-qkv-body-post-bridge MLIR feeds c1r3 with Q4NX Q/K/V bodies")
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
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    packed = make_packed_weights()
    hidden = hidden_as_i32()
    weights = packed_as_i32(packed)
    expected = expected_output(packed)
    if hidden.shape[0] != HIDDEN_DWORDS:
        raise RuntimeError(f"bad hidden shape: {hidden.shape[0]}")
    if weights.shape[0] != TOTAL_WEIGHT_I32:
        raise RuntimeError(f"bad weight shape: {weights.shape[0]}")

    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden.copy()).to(torch.int32))
    weight_buf = XRTTensor.from_torch(torch.from_numpy(weights.copy()).to(torch.int32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [hidden_buf, weight_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")
    print(f"  expected[-8:]: {expected[-8:].tolist()}")
    print(f"  got[-8:]:      {got[-8:].tolist()}")

    errors = validate_output(expected, got)
    if errors:
        print(f"  FAIL: {len(errors)} q4nx qkv body post mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: Q4NX Q/K/V body reached c1r3 postprocess and host output on NPU")
    return True
