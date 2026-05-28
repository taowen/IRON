"""NPU runner for c6r1 -> c1r1 -> main16 bridge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import bridge_generate
import npu_build
from bridge_reference import (
    TOTAL_SUMMARY_DWORDS,
    bridge_case,
    expected_output,
    make_payload,
    validate_output,
)

EXPERIMENT_DIR = Path(__file__).parent


def build_kernel(case_name: str) -> tuple[Path, Path]:
    case = bridge_case(case_name)
    build_dir = EXPERIMENT_DIR / "build" / case.name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = bridge_generate.generate_mlir(case)
    mlir_path.write_text(mlir_text)
    errors = bridge_generate.validate_generated_mlir(mlir_text, case)
    if errors:
        raise RuntimeError("\n".join(f"  BRIDGE STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(case_name: str) -> bool:
    case = bridge_case(case_name)
    mlir_text = bridge_generate.generate_mlir(case)
    errors = bridge_generate.validate_generated_mlir(mlir_text, case)
    if errors:
        for error in errors:
            print(f"  BRIDGE STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: "
        f"{case.name} MLIR contains c6r1 packet{case.packet_id} -> "
        "c1r1 DMA4 -> DMA1 -> main16 multicast"
    )
    return True


def build_only(case_name: str) -> bool:
    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel(case_name)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run(case_name: str) -> bool:
    case = bridge_case(case_name)
    print("=" * 78)
    print(f"qwen3-layer: {case.name}")
    print("=" * 78)
    print(f"  packet_id: {case.packet_id}")
    print(f"  payload_dwords: {case.payload_dwords}")
    print(f"  bridge_iterations: {case.bridge_iterations}")
    print(f"  main_chunks: {case.main_chunks}")
    print("  route: c6r1 packet source -> c1r1 DMA4 -> c1r1 DMA1 -> main16 DMA0")
    print()

    npu_build.compile_bridge_kernel()
    xclbin_path, insts_path = build_kernel(case.name)

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    payload = make_payload(case)
    expected = expected_output(case)
    payload_buf = XRTTensor.from_torch(torch.from_numpy(payload.copy()).to(torch.int32))
    output_buf = XRTTensor((TOTAL_SUMMARY_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [payload_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")

    errors = validate_output(case, got)
    if errors:
        print(f"  FAIL: {len(errors)} bridge mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print(
        "  PASS: "
        f"{case.name} delivered {case.main_chunks} chunks to all 16 main tiles"
    )
    return True
