#!/usr/bin/env python3
"""Confirm MyLM Q/K/V dispatcher path record count with exp132's stub."""

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
MANIFEST = EXPERIMENT_DIR / "mylm_main16_qkv_record_count_probe.json"
REPORT = EXPERIMENT_DIR / "mylm_main16_qkv_record_count_probe.md"
QKV_CONTROL = 1
QKV_RECORDS = 12


@dataclass(frozen=True)
class Scenario:
    name: str
    records: int
    expected_status: str
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


def default_scenarios(include_negative: bool) -> tuple[Scenario, ...]:
    scenarios = [
        Scenario("qkv_wait_12_records", QKV_RECORDS, "record_observed", ("0x1",) * QKV_RECORDS),
    ]
    if include_negative:
        scenarios.append(
            Scenario(
                "qkv_wait_13_records",
                QKV_RECORDS + 1,
                "record_observed",
                ("0x1",) * QKV_RECORDS + ("0x4",),
            )
        )
    return tuple(scenarios)


def run_scenario(scenario: Scenario, runtime_timeout: int, xrt_timeout: int) -> dict[str, Any]:
    variant = EXP132.Variant(scenario.name, QKV_CONTROL)
    result = EXP132.run_variant(variant, scenario.records, runtime_timeout, xrt_timeout)
    headers = result.get("record_headers") or []
    passed = result["status"] == scenario.expected_status and tuple(headers) == scenario.expected_headers
    return {
        **result,
        "records_requested": scenario.records,
        "expected_status": scenario.expected_status,
        "expected_headers": scenario.expected_headers,
        "passed": passed,
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# MyLM Main16 QKV Record-Count Probe",
        "",
        f"- Status: `{manifest['status']}`",
        f"- QKV control value: `{QKV_CONTROL}`",
        "",
        "## Results",
        "",
        "| scenario | requested | status | unique headers | pass | topology after |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for result in manifest["scenarios"]:
        topology = result.get("topology_after", {}).get("topology")
        lines.append(
            f"| `{result['name']}` | `{result['records_requested']}` | `{result['status']}` | "
            f"`{result.get('unique_record_headers', [])}` | `{result['passed']}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    pass12 = next(
        (result for result in manifest["scenarios"] if result["name"] == "qkv_wait_12_records"),
        None,
    )
    pass13 = next(
        (result for result in manifest["scenarios"] if result["name"] == "qkv_wait_13_records"),
        None,
    )
    if pass12 and pass12["passed"]:
        lines.append("The MyLM Q/K/V body emitted 12 observable compact records with header `0x1`.")
    if pass13 and pass13["passed"]:
        lines.append(
            "The 13th observed record has header `0x4`, so `control=1` enters the "
            "dispatcher sequence as Q/K/V first, then transitions to the next O-like body."
        )
    elif pass13:
        lines.append("The 13-record probe did not match expectation; inspect runtime output before relying on the boundary.")
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
    return "\n".join(lines) + "\n"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    stopped_early = False
    for scenario in default_scenarios(args.include_negative):
        result = run_scenario(scenario, args.runtime_timeout, args.xrt_timeout)
        results.append(result)
        topology_after = result.get("topology_after", {})
        if topology_after.get("status") != "ok":
            stopped_early = True
            break
        if result["status"] == "xrt_get_info_failed":
            stopped_early = True
            break
    status = "passed" if all(result["passed"] for result in results) and not stopped_early else "failed"
    return {
        "status": status,
        "stopped_early": stopped_early,
        "qkv_control": QKV_CONTROL,
        "qkv_records": QKV_RECORDS,
        "scenarios": results,
        "source": {
            "stub_helper": str(EXP132_RUN.relative_to(REPO_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=8)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--include-negative", action=argparse.BooleanOptionalAction, default=True)
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
            "# MyLM Main16 QKV Record-Count Probe\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
