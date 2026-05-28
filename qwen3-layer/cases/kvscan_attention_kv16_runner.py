"""NPU runner for kvscan-attention-kv16-o-bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import npu_build
from cases import kvscan_attention_kv16_generate as generate
from cases.kvscan_attention_kv16_reference import (
    CASE_NAME,
    KV_CACHE_SIDE_DWORDS,
    Q_DWORDS,
    TOTAL_SUMMARY_DWORDS,
    expected_output,
    make_kv_cache_payload,
    make_q_payload,
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
        raise RuntimeError("\n".join(f"  KVSCAN ATTENTION STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only() -> bool:
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  KVSCAN ATTENTION STRUCTURE FAIL: {error}")
        return False
    print("  PASS: kvscan-attention-kv16-o-bridge MLIR contains shim-sliced KV scan")
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
        "  route: host Q + logical KV cache sides -> shim BD slices -> "
        "c0r1/c7r1 row1 KV scan -> kv16 Shape-A/B -> packet2 O bridge"
    )
    print()

    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    q_payload = make_q_payload()
    kv_left = make_kv_cache_payload(0)
    kv_right = make_kv_cache_payload(1)
    if q_payload.shape != (Q_DWORDS,):
        raise ValueError(f"q payload shape mismatch: {q_payload.shape}")
    if kv_left.shape != (KV_CACHE_SIDE_DWORDS,) or kv_right.shape != (KV_CACHE_SIDE_DWORDS,):
        raise ValueError(f"kv cache payload shape mismatch: {kv_left.shape}/{kv_right.shape}")

    expected = expected_output()
    q_buf = XRTTensor.from_torch(torch.from_numpy(q_payload.copy()).to(torch.int32))
    kv_left_buf = XRTTensor.from_torch(torch.from_numpy(kv_left.copy()).to(torch.int32))
    kv_right_buf = XRTTensor.from_torch(torch.from_numpy(kv_right.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_SUMMARY_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [q_buf, kv_left_buf, kv_right_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(got)
    if errors:
        print(f"  FAIL: {len(errors)} kvscan attention mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: shim-sliced KV cache scan reached kv16 attention and main16 O")
    return True
