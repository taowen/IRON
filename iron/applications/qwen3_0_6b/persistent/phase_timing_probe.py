#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Qwen3 persistent phase-sensitivity timing probes.

The probes are intentionally built from existing verified stages. They are not
cycle-accurate hardware trace, but they provide a repeatable dynamic signal for
which phase is worth optimizing next.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN = REPO_ROOT / "iron" / "applications" / "qwen3_0_6b" / "persistent" / "main.py"

NPU_TIME_RE = re.compile(r"^npu_time_us:\s*(?P<value>[0-9.]+)\s*$", re.MULTILINE)
PREFLIGHT_RE = re.compile(r"^preflight:\s*(?P<value>.*)$", re.MULTILINE)
OPERATOR_RE = re.compile(r"^operator_name:\s*(?P<value>.*)$", re.MULTILINE)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    description: str
    build_suffix: str
    args: tuple[str, ...]
    strict_current_graph: bool


@dataclass
class ProbeResult:
    spec: ProbeSpec
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    npu_times_us: list[float]
    warm_times_us: list[float]
    operator_name: str | None
    preflight: str | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.npu_times_us)

    def summary(self) -> dict[str, float | int | str | None]:
        if not self.warm_times_us:
            return {
                "name": self.spec.name,
                "runs": len(self.npu_times_us),
                "warm_runs": 0,
                "mean_us": None,
                "median_us": None,
                "min_us": None,
                "max_us": None,
            }
        return {
            "name": self.spec.name,
            "runs": len(self.npu_times_us),
            "warm_runs": len(self.warm_times_us),
            "mean_us": statistics.fmean(self.warm_times_us),
            "median_us": statistics.median(self.warm_times_us),
            "min_us": min(self.warm_times_us),
            "max_us": max(self.warm_times_us),
        }


def _common_prompt_args(args: argparse.Namespace) -> list[str]:
    prompt_args: list[str] = []
    if args.raw_prompt:
        prompt_args.append("--raw-prompt")
    if args.prompt is not None:
        prompt_args.extend(["--prompt", args.prompt])
    return prompt_args


def probe_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(
            name="full_layer_current_l1",
            description="current one-layer graph with attention2, MLP2, and paired gate/up rows",
            build_suffix="full_l1",
            strict_current_graph=True,
            args=(
                "--stage",
                "n-layer-final-only",
                "--verify",
                "--layer-chunk-size",
                "1",
                "--num-aie-columns",
                "2",
                "--attention-columns",
                "2",
                "--mlp-gate-up-columns",
                "2",
                "--mlp-gate-up-pair-rows",
            ),
        ),
        ProbeSpec(
            name="attention_only_current_l1",
            description="same one-layer graph stopped after attention/O-proj/residual",
            build_suffix="attention_l1",
            strict_current_graph=True,
            args=(
                "--stage",
                "n-layer-final-only",
                "--verify",
                "--layer-chunk-size",
                "1",
                "--num-aie-columns",
                "2",
                "--attention-columns",
                "2",
                "--mlp-gate-up-columns",
                "2",
                "--mlp-gate-up-pair-rows",
                "--attention-probe-only",
            ),
        ),
        ProbeSpec(
            name="qkv_standalone_l1",
            description="standalone input RMSNorm + QKV projection stage",
            build_suffix="qkv_l1",
            strict_current_graph=False,
            args=(
                "--stage",
                "input-rmsnorm-qkv",
                "--verify",
                "--num-aie-columns",
                "2",
            ),
        ),
        ProbeSpec(
            name="mlp_full_standalone_l1",
            description="standalone post-attention RMSNorm + full MLP stage",
            build_suffix="full_mlp_l1",
            strict_current_graph=False,
            args=(
                "--stage",
                "post-attn-rmsnorm-full-mlp",
                "--verify",
                "--num-aie-columns",
                "2",
            ),
        ),
        ProbeSpec(
            name="mlp_gate_up_standalone_l1",
            description="standalone post-attention RMSNorm + gate/up + SiLU/mul stage",
            build_suffix="gateup_l1",
            strict_current_graph=False,
            args=(
                "--stage",
                "post-attn-rmsnorm-mlp-gate-up",
                "--verify",
                "--num-aie-columns",
                "2",
            ),
        ),
        ProbeSpec(
            name="mlp_down_standalone_l1",
            description="standalone MLP down projection + residual-add stage",
            build_suffix="down_l1",
            strict_current_graph=False,
            args=(
                "--stage",
                "post-attn-mlp-down-residual",
                "--verify",
                "--num-aie-columns",
                "2",
            ),
        ),
    ]


def run_probe(
    spec: ProbeSpec,
    args: argparse.Namespace,
) -> ProbeResult:
    build_dir = f"{args.build_dir_prefix}_{spec.build_suffix}"
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        str(MAIN),
        *spec.args,
        "--verify-repeat",
        str(args.repeat),
        "--build-dir",
        build_dir,
        *_common_prompt_args(args),
    ]
    if args.mlp_gate_up_direct_silu and spec.strict_current_graph:
        cmd.append("--mlp-gate-up-direct-silu")
    if args.mlp_gate_up_row_group != 4 and spec.strict_current_graph:
        cmd.extend(["--mlp-gate-up-row-group", str(args.mlp_gate_up_row_group)])
    if args.clean_build:
        cmd.append("--clean-build")

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    npu_times = [
        float(match.group("value")) for match in NPU_TIME_RE.finditer(completed.stdout)
    ]
    warm_times = npu_times[min(args.warmup_drop, len(npu_times)) :]
    operator = OPERATOR_RE.search(completed.stdout)
    preflight = PREFLIGHT_RE.search(completed.stdout)
    return ProbeResult(
        spec=spec,
        command=cmd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        npu_times_us=npu_times,
        warm_times_us=warm_times,
        operator_name=operator.group("value") if operator else None,
        preflight=preflight.group("value") if preflight else None,
    )


def _fmt_us(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000.0:.3f} ms"


def print_markdown(results: list[ProbeResult], warmup_drop: int) -> None:
    print("# Qwen3 Phase Timing Probe")
    print()
    print("```text")
    print(f"warmup_drop: {warmup_drop}")
    print("timing_source: npu_time_us reported by the existing verified stage runners")
    print("```")
    print()
    print(
        "| probe | current graph | warm mean | warm median | warm min | warm max | runs |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    summaries = {result.spec.name: result.summary() for result in results}
    for result in results:
        summary = summaries[result.spec.name]
        print(
            f"| {result.spec.name} | "
            f"{'yes' if result.spec.strict_current_graph else 'no'} | "
            f"{_fmt_us(summary['mean_us'])} | "
            f"{_fmt_us(summary['median_us'])} | "
            f"{_fmt_us(summary['min_us'])} | "
            f"{_fmt_us(summary['max_us'])} | "
            f"{summary['warm_runs']}/{summary['runs']} |"
        )

    full = summaries.get("full_layer_current_l1", {})
    attention = summaries.get("attention_only_current_l1", {})
    qkv = summaries.get("qkv_standalone_l1", {})
    full_mean = full.get("mean_us")
    attention_mean = attention.get("mean_us")
    full_median = full.get("median_us")
    attention_median = attention.get("median_us")
    full_min = full.get("min_us")
    attention_min = attention.get("min_us")
    qkv_mean = qkv.get("mean_us")
    qkv_median = qkv.get("median_us")
    qkv_min = qkv.get("min_us")

    print()
    print("## Derived Signals")
    print()
    print("```text")
    if isinstance(full_mean, float) and isinstance(attention_mean, float):
        mlp_increment = full_mean - attention_mean
        print(f"current full-layer warm mean:      {full_mean / 1000.0:.3f} ms")
        print(f"current attention-only warm mean:  {attention_mean / 1000.0:.3f} ms")
        print(f"current MLP-side mean increment:   {mlp_increment / 1000.0:.3f} ms")
        print(
            f"MLP mean-increment share:          {mlp_increment / full_mean * 100.0:.1f}%"
        )
    if isinstance(full_median, float) and isinstance(attention_median, float):
        mlp_increment = full_median - attention_median
        print(f"current full-layer warm median:    {full_median / 1000.0:.3f} ms")
        print(f"current attention-only median:     {attention_median / 1000.0:.3f} ms")
        print(f"current MLP-side median increment: {mlp_increment / 1000.0:.3f} ms")
        print(
            f"MLP median-increment share:        "
            f"{mlp_increment / full_median * 100.0:.1f}%"
        )
    if isinstance(full_min, float) and isinstance(attention_min, float):
        mlp_increment = full_min - attention_min
        print(f"current full-layer warm min:       {full_min / 1000.0:.3f} ms")
        print(f"current attention-only warm min:   {attention_min / 1000.0:.3f} ms")
        print(f"current MLP-side min increment:    {mlp_increment / 1000.0:.3f} ms")
    else:
        print("current MLP-side increment: unavailable")
    if isinstance(attention_mean, float) and isinstance(qkv_mean, float):
        print(f"standalone QKV warm mean:          {qkv_mean / 1000.0:.3f} ms")
        print(
            "attention-only mean minus standalone QKV mean: "
            f"{(attention_mean - qkv_mean) / 1000.0:.3f} ms"
        )
    if isinstance(attention_median, float) and isinstance(qkv_median, float):
        print(f"standalone QKV warm median:        {qkv_median / 1000.0:.3f} ms")
        print(
            "attention-only median minus standalone QKV median: "
            f"{(attention_median - qkv_median) / 1000.0:.3f} ms"
        )
    if isinstance(attention_min, float) and isinstance(qkv_min, float):
        print(f"standalone QKV warm min:           {qkv_min / 1000.0:.3f} ms")
        print(
            "attention-only min minus standalone QKV min: "
            f"{(attention_min - qkv_min) / 1000.0:.3f} ms"
        )
    print("```")
    print()
    print("Notes:")
    print()
    print("- `current graph=yes` probes preserve the n-layer-final-only graph family.")
    print(
        "- Standalone probes are diagnostic signals; they are not an additive timing split."
    )
    print("- Use warm min/median when a standalone probe has high run-to-run variance.")


def write_json(path: Path, results: list[ProbeResult], warmup_drop: int) -> None:
    payload = {
        "warmup_drop": warmup_drop,
        "results": [
            {
                "name": result.spec.name,
                "description": result.spec.description,
                "strict_current_graph": result.spec.strict_current_graph,
                "command": result.command,
                "returncode": result.returncode,
                "operator_name": result.operator_name,
                "preflight": result.preflight,
                "npu_times_us": result.npu_times_us,
                "warm_times_us": result.warm_times_us,
                "summary": result.summary(),
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verified Qwen3 persistent phase timing probes"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup-drop", type=int, default=1)
    parser.add_argument("--build-dir-prefix", default="build_qwen3_phase_probe")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument(
        "--mlp-gate-up-direct-silu",
        action="store_true",
        help="Run current-graph probes with direct hidden-producing gate/up+SiLU",
    )
    parser.add_argument(
        "--mlp-gate-up-row-group",
        type=int,
        default=4,
        choices=(4, 8),
        help="Run current-graph probes with the selected paired gate/up row group.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=[spec.name for spec in probe_specs()],
        action="append",
        help="Run only the named probe; can be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = probe_specs()
    if args.only:
        requested = set(args.only)
        selected = [spec for spec in selected if spec.name in requested]

    results = [run_probe(spec, args) for spec in selected]
    failed = [result for result in results if not result.ok]
    if failed:
        for result in failed:
            print(f"probe failed: {result.spec.name}", file=sys.stderr)
            print("command:", " ".join(result.command), file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        return 1

    print_markdown(results, args.warmup_drop)
    if args.json_output is not None:
        write_json(args.json_output, results, args.warmup_drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
