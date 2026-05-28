#!/usr/bin/env python3
"""Run qwen3-layer NPU integration cases."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import bridge_runner
import c1r2_runner
import current_runner
import npu_build
import shape_runner
import swiglu_runner
from bridge_reference import BRIDGE_CASES
from c1r2_reference import CASE_NAME as C1R2_CASE_NAME
from shape_reference import CASE_NAME as SHAPE_CASE_NAME
from swiglu_reference import CASE_NAME as SWIGLU_CASE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "current",
            C1R2_CASE_NAME,
            SHAPE_CASE_NAME,
            SWIGLU_CASE_NAME,
        )
        + tuple(case.name for case in BRIDGE_CASES),
        default="current",
        help="NPU integration case to run",
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


def run_case(args: argparse.Namespace) -> bool:
    if args.check_only and args.build_only:
        raise ValueError("--check-only and --build-only are mutually exclusive")

    if args.case == "current":
        if args.check_only:
            return current_runner.check_backend_structure()
        if args.build_only:
            return current_runner.build_only()
        print(f"NPU device: {npu_build.device()}")
        return current_runner.run_on_npu()

    if args.case == C1R2_CASE_NAME:
        if args.check_only:
            return c1r2_runner.check_only()
        if args.build_only:
            return c1r2_runner.build_only()
        print(f"NPU device: {npu_build.device()}")
        return c1r2_runner.run()

    if args.case == SHAPE_CASE_NAME:
        if args.check_only:
            return shape_runner.check_only()
        if args.build_only:
            return shape_runner.build_only()
        print(f"NPU device: {npu_build.device()}")
        return shape_runner.run()

    if args.case == SWIGLU_CASE_NAME:
        if args.check_only:
            return swiglu_runner.check_only()
        if args.build_only:
            return swiglu_runner.build_only()
        print(f"NPU device: {npu_build.device()}")
        return swiglu_runner.run()

    if args.check_only:
        return bridge_runner.check_only(args.case)
    if args.build_only:
        return bridge_runner.build_only(args.case)
    print(f"NPU device: {npu_build.device()}")
    return bridge_runner.run(args.case)


if __name__ == "__main__":
    try:
        success = run_case(parse_args())
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        success = False
    finally:
        npu_build.cleanup()
    raise SystemExit(0 if success else 1)
