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

import npu_build
from cases import registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=registry.CASE_NAMES,
        default=registry.DEFAULT_CASE_NAME,
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
    parser.add_argument(
        "--current-token",
        type=int,
        default=None,
        help="current decode token for token-aware cases",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="path to MyLM Qwen3-8B-NPU2 model directory for real qwen3-8b cases",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Qwen3 layer index for real qwen3-8b cases",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="download missing Qwen3-8B-NPU2 model files before running real qwen3-8b cases",
    )
    return parser.parse_args()


def run_case(args: argparse.Namespace) -> bool:
    if args.check_only and args.build_only:
        raise ValueError("--check-only and --build-only are mutually exclusive")

    if args.check_only:
        return registry.check_only(
            args.case,
            args.current_token,
            args.model_path,
            args.layer,
            args.download_model,
        )
    if args.build_only:
        return registry.build_only(
            args.case,
            args.current_token,
            args.model_path,
            args.layer,
            args.download_model,
        )
    return registry.run(
        args.case,
        args.current_token,
        args.model_path,
        args.layer,
        args.download_model,
    )


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
