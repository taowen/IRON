#!/usr/bin/env python3
"""NPU runner for packet-source to multicast-bridge cases."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

import generate
import npu_build
from reference import (
    BRIDGE_CASES,
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

    mlir_text = generate.generate_mlir(case)
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text, case)
    if errors:
        raise RuntimeError("\n".join(f"  BRIDGE STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(case_name: str) -> bool:
    case = bridge_case(case_name)
    mlir_text = generate.generate_mlir(case)
    errors = generate.validate_generated_mlir(mlir_text, case)
    if errors:
        for error in errors:
            print(f"  BRIDGE STRUCTURE FAIL: {error}")
        return False
    print(
        "  PASS: "
        f"{case.name} MLIR contains packet{case.packet_id} source -> "
        "bridge ping/pong -> worker multicast"
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
    print(f"Recipe: Packet To Multicast Bridge: {case.name}")
    print("=" * 78)
    print(f"  packet_id: {case.packet_id}")
    print(f"  payload_dwords: {case.payload_dwords}")
    print(f"  bridge_iterations: {case.bridge_iterations}")
    print(f"  main_chunks: {case.main_chunks}")
    print("  route: packet source -> bridge ping/pong -> multicast worker inputs")
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
        f"{case.name} delivered {case.main_chunks} chunks to all 16 workers"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(case.name for case in BRIDGE_CASES),
        default=BRIDGE_CASES[0].name,
        help="bridge payload shape to run",
    )
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
        return check_only(args.case)
    if args.build_only:
        return build_only(args.case)
    print(f"NPU device: {npu_build.device()}")
    return run(args.case)


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
