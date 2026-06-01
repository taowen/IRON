#!/usr/bin/env python3
"""Observe the MyLM main16 dispatcher header sequence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP132_RUN = REPO_ROOT / "experiments/132_mylm_main16_dispatcher_stub_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "dispatcher_header_sequence.json"
REPORT = EXPERIMENT_DIR / "dispatcher_header_sequence.md"
QKV_CONTROL = 1


@dataclass(frozen=True)
class Scenario:
    name: str
    records: int
    stream_chunks: int
    expected_headers: tuple[str, ...]


def load_exp132() -> Any:
    spec = importlib.util.spec_from_file_location("mylm_exp132", EXP132_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp132 helper: {EXP132_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPERIMENT_DIR = EXPERIMENT_DIR
    module.BUILD_DIR = BUILD_DIR
    return module


EXP132 = load_exp132()


def headers_20() -> tuple[str, ...]:
    return ("0x1",) * 12 + ("0x4",) * 8


def headers_68() -> tuple[str, ...]:
    return headers_20() + ("0x8",) * 48


def headers_76() -> tuple[str, ...]:
    return headers_68() + ("0x4",) * 8


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("dispatcher_wait_20_records", 20, 12 * 16 + 8 * 16, headers_20()),
        Scenario("dispatcher_wait_68_records", 68, 12 * 16 + 8 * 16 + 48 * 16, headers_68()),
        Scenario("dispatcher_wait_76_records", 76, 12 * 16 + 8 * 16 + 48 * 16 + 8 * 48, headers_76()),
    )


def runs(headers: tuple[str, ...] | list[str]) -> list[dict[str, int | str]]:
    if not headers:
        return []
    out: list[dict[str, int | str]] = []
    current = headers[0]
    count = 1
    for header in headers[1:]:
        if header == current:
            count += 1
        else:
            out.append({"header": current, "count": count})
            current = header
            count = 1
    out.append({"header": current, "count": count})
    return out


def runs_text(headers: tuple[str, ...] | list[str]) -> str:
    return " + ".join(f"{run['count']} x {run['header']}" for run in runs(headers))


def run_scenario(scenario: Scenario, runtime_timeout: int, xrt_timeout: int) -> dict[str, Any]:
    variant = EXP132.Variant(scenario.name, QKV_CONTROL)
    activation_dwords = EXP132.EXP130.activation_dwords
    weight_dwords = EXP132.EXP130.weight_dwords
    EXP132.EXP130.activation_dwords = (
        lambda _records, chunks=scenario.stream_chunks: chunks * EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    )
    EXP132.EXP130.weight_dwords = (
        lambda _records, chunks=scenario.stream_chunks: chunks * EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    )
    try:
        result = EXP132.run_variant(variant, scenario.records, runtime_timeout, xrt_timeout)
    finally:
        EXP132.EXP130.activation_dwords = activation_dwords
        EXP132.EXP130.weight_dwords = weight_dwords
    got_headers = tuple(result.get("record_headers") or ())
    passed = result["status"] == "record_observed" and got_headers == scenario.expected_headers
    return {
        **result,
        "records_requested": scenario.records,
        "stream_chunks": scenario.stream_chunks,
        "expected_headers": scenario.expected_headers,
        "expected_runs": runs(scenario.expected_headers),
        "got_runs": runs(got_headers),
        "passed": passed,
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Dispatcher Header Sequence",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dispatcher control value: `{manifest['qkv_control']}`",
        "",
        "## Results",
        "",
        "| scenario | records | stream chunks | status | expected | observed | pass | topology after |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for result in manifest["scenarios"]:
        topology = result.get("topology_after", {}).get("topology")
        expected = runs_text(result["expected_headers"])
        observed = runs_text(result.get("record_headers") or [])
        lines.append(
            f"| `{result['name']}` | `{result['records_requested']}` | `{result['stream_chunks']}` | "
            f"`{result['status']}` | "
            f"`{expected}` | `{observed}` | `{result['passed']}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if manifest["status"] == "passed":
        lines.append(
            "The raw MyLM main16 dispatcher emitted the expected full phase-header "
            "sequence: Q/K/V `0x1`, O `0x4`, up/gate `0x8`, and down `0x4`."
        )
    else:
        lines.append(
            "At least one prefix did not match the expected dispatcher header sequence. "
            "Inspect the observed run lengths before using the phase contract."
        )
    lines.extend(["", "## Runtime Output Preview", ""])
    for result in manifest["scenarios"]:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "```text",
                result.get("runtime_output", ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = default_scenarios()
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    results: list[dict[str, Any]] = []
    stopped_early = False
    for scenario in scenarios:
        result = run_scenario(scenario, args.runtime_timeout, args.xrt_timeout)
        results.append(result)
        topology_after = result.get("topology_after", {})
        if topology_after.get("status") != "ok":
            stopped_early = True
            break
        if args.stop_on_fail and not result["passed"]:
            stopped_early = True
            break
    status = "passed" if all(result["passed"] for result in results) and not stopped_early else "failed"
    return {
        "status": status,
        "stopped_early": stopped_early,
        "qkv_control": QKV_CONTROL,
        "scenarios": results,
        "source": {
            "stub_helper": str(EXP132_RUN.relative_to(REPO_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=20)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Dispatcher Header Sequence\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest) + "\n")
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
