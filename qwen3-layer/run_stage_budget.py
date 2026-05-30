#!/usr/bin/env python3
"""Run active qwen3-layer stage budget measurements on NPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STAGE_CASES = {
    "row1-weight": "row1-weight-stream-perf",
    "main16-compute": "main16-q4nx-compute-perf",
    "c1r2": "qwen3-8b-c1r2-input-norm-replay",
    "qkv": "qwen3-8b-qkv-cache-write-bridge",
    "attention": "full-layer-attention-o-bf16",
    "full": "qwen3-8b-decode-layer",
}
MODEL_AWARE_STAGES = ("c1r2", "qkv", "attention", "full")
TOKEN_AWARE_STAGES = ("qkv", "attention", "full")
DEFAULT_TOKENS = (31, 91)
DEFAULT_STAGES = ("c1r2", "qkv", "attention", "full")


@dataclass(frozen=True)
class CaseRun:
    stage: str
    case_name: str
    token: int | None
    returncode: int
    npu_time_us: str
    budget_lines: tuple[str, ...]
    output: str


def _csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("empty comma-separated list")
    return items


def _parse_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item) for item in _csv_items(value))
    for token in tokens:
        if token < 0:
            raise ValueError("tokens must be non-negative")
    return tokens


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = _csv_items(value)
    for stage in stages:
        if stage not in STAGE_CASES:
            raise ValueError(f"unknown stage {stage}; expected one of {','.join(STAGE_CASES)}")
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default=",".join(str(token) for token in DEFAULT_TOKENS))
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--case-timeout-sec", type=int, default=600)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _run_case(
    stage: str,
    token: int | None,
    model_path: Path | None,
    layer: int,
    download_model: bool,
    timeout_sec: int,
) -> CaseRun:
    run_npu = Path(__file__).with_name("run_npu.py")
    case_name = STAGE_CASES[stage]
    cmd = [
        sys.executable,
        str(run_npu),
        "--case",
        case_name,
    ]
    if stage in MODEL_AWARE_STAGES:
        cmd.extend(("--layer", str(layer)))
    if token is not None:
        cmd.extend(("--current-token", str(token)))
    if model_path is not None and stage in MODEL_AWARE_STAGES:
        cmd.extend(("--model-path", str(model_path)))
    if download_model and stage in MODEL_AWARE_STAGES:
        cmd.append("--download-model")

    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return CaseRun(
            stage=stage,
            case_name=case_name,
            token=token,
            returncode=124,
            npu_time_us="timeout",
            budget_lines=(),
            output=output + f"\nTIMEOUT after {timeout_sec}s: {' '.join(cmd)}\n",
        )
    output = result.stdout + result.stderr
    budget_lines = tuple(
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(("stage_budget:", "perf_budget:"))
    )
    npu_time_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("NPU time:")
    ]
    npu_time_us = npu_time_lines[-1].split(":", 1)[1].strip() if npu_time_lines else "n/a"
    return CaseRun(
        stage=stage,
        case_name=case_name,
        token=token,
        returncode=result.returncode,
        npu_time_us=npu_time_us,
        budget_lines=budget_lines,
        output=output,
    )


def _scheduled_runs(stages: tuple[str, ...], tokens: tuple[int, ...]) -> tuple[tuple[str, int | None], ...]:
    runs: list[tuple[str, int | None]] = []
    for stage in stages:
        if stage in TOKEN_AWARE_STAGES:
            runs.extend((stage, token) for token in tokens)
        else:
            runs.append((stage, None))
    return tuple(runs)


def main() -> int:
    args = parse_args()
    tokens = _parse_tokens(args.tokens)
    stages = _parse_stages(args.stages)
    runs = _scheduled_runs(stages, tokens)
    failures = 0

    print("qwen3-layer stage budget", flush=True)
    print(f"  stages={','.join(stages)}", flush=True)
    print(f"  tokens={','.join(str(token) for token in tokens)}", flush=True)
    print(f"  layer={args.layer}", flush=True)

    for stage, token in runs:
        result = _run_case(stage, token, args.model_path, args.layer, args.download_model, args.case_timeout_sec)
        token_label = "n/a" if result.token is None else str(result.token)
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(
            f"{status} stage={result.stage} case={result.case_name} "
            f"token={token_label} npu_time={result.npu_time_us}",
            flush=True,
        )
        for line in result.budget_lines:
            print(f"  {line}", flush=True)
        if result.returncode != 0:
            failures += 1
            print(result.output, flush=True)
        elif args.verbose:
            print(result.output, flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
