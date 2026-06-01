#!/usr/bin/env python3
"""Check MyLM main16 stream consumption at phase boundaries."""

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
MANIFEST = EXPERIMENT_DIR / "stream_consumption_boundary.json"
REPORT = EXPERIMENT_DIR / "stream_consumption_boundary.md"
QKV_CONTROL = 1


@dataclass(frozen=True)
class Scenario:
    name: str
    records: int
    stream_chunks: int
    expected_status: str
    expected_headers: tuple[str, ...]
    meaning: str


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


def headers_qkv() -> tuple[str, ...]:
    return ("0x1",) * 12


def headers_o() -> tuple[str, ...]:
    return headers_qkv() + ("0x4",) * 8


def headers_upgate() -> tuple[str, ...]:
    return headers_o() + ("0x8",) * 48


def headers_full() -> tuple[str, ...]:
    return headers_upgate() + ("0x4",) * 8


def exact_scenario(name: str, records: int, stream_chunks: int, headers: tuple[str, ...]) -> Scenario:
    return Scenario(name, records, stream_chunks, "record_observed", headers, "exact stream boundary")


def underfed_scenario(name: str, records: int, stream_chunks: int) -> Scenario:
    return Scenario(name, records, stream_chunks, "timeout_waiting_for_record", (), "one chunk underfed")


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        exact_scenario("qkv_exact_192_chunks", 12, 192, headers_qkv()),
        exact_scenario("o_exact_320_chunks", 20, 320, headers_o()),
        exact_scenario("upgate_exact_1088_chunks", 68, 1088, headers_upgate()),
        exact_scenario("down_exact_1472_chunks", 76, 1472, headers_full()),
    )


def underfed_scenarios() -> tuple[Scenario, ...]:
    return (
        underfed_scenario("qkv_underfed_191_chunks", 12, 191),
        underfed_scenario("o_underfed_319_chunks", 20, 319),
        underfed_scenario("upgate_underfed_1087_chunks", 68, 1087),
        underfed_scenario("down_underfed_1471_chunks", 76, 1471),
        underfed_scenario("full_naive_1216_chunks", 76, 76 * 16),
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
    if not headers:
        return ""
    return " + ".join(f"{run['count']} x {run['header']}" for run in runs(headers))


def with_stream_chunks(scenario: Scenario, runtime_timeout: int, xrt_timeout: int) -> dict[str, Any]:
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
    if scenario.expected_status == "record_observed":
        passed = result["status"] == scenario.expected_status and got_headers == scenario.expected_headers
    else:
        passed = result["status"] == scenario.expected_status
    return {
        **result,
        "records_requested": scenario.records,
        "stream_chunks": scenario.stream_chunks,
        "expected_status": scenario.expected_status,
        "expected_headers": scenario.expected_headers,
        "expected_runs": runs(scenario.expected_headers),
        "got_runs": runs(got_headers),
        "meaning": scenario.meaning,
        "passed": passed,
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Stream Consumption Boundary",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dispatcher control value: `{manifest['qkv_control']}`",
        "",
        "## Results",
        "",
        "| scenario | records | chunks | expected status | actual status | expected headers | observed headers | pass | topology |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for result in manifest["scenarios"]:
        topology = result.get("topology_after", {}).get("topology")
        lines.append(
            f"| `{result['name']}` | `{result['records_requested']}` | `{result['stream_chunks']}` | "
            f"`{result['expected_status']}` | `{result['status']}` | "
            f"`{runs_text(result['expected_headers'])}` | `{runs_text(result.get('record_headers') or [])}` | "
            f"`{result['passed']}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if manifest["status"] == "passed":
        lines.extend(
            [
                "The stream consumption boundaries are now explicit:",
                "",
                "```text",
                "Q/K/V   consumes 12 * 16 = 192 chunks.",
                "O       consumes  8 * 16 = 128 chunks after Q/K/V.",
                "up/gate consumes 48 * 16 = 768 chunks after O.",
                "down    consumes  8 * 48 = 384 chunks after up/gate.",
                "full    consumes 1472 chunks total.",
                "```",
                "",
                "A full-layer scheduler that feeds `records * 16` chunks will underfeed down.",
            ]
        )
    else:
        lines.append("At least one stream boundary did not match expectation; inspect the scenario table.")
    if manifest["include_underfed"]:
        lines.append(
            "Underfed scenarios intentionally timeout. A timeout can leave the raw-core runtime "
            "session unsuitable for immediate follow-up runs even when `xrt-smi examine` still "
            "shows topology `6x8`; run those cases separately and recover with "
            "`sudo systemctl restart amdxdna-pinned.service` before continuing."
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
    return "\n".join(lines) + "\n"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = default_scenarios()
    if args.include_underfed:
        scenarios = scenarios + underfed_scenarios()
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    results: list[dict[str, Any]] = []
    stopped_early = False
    for scenario in scenarios:
        result = with_stream_chunks(scenario, args.runtime_timeout, args.xrt_timeout)
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
        "include_underfed": args.include_underfed,
        "scenarios": results,
        "source": {
            "stub_helper": str(EXP132_RUN.relative_to(REPO_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=8)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--include-underfed", action="store_true")
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
            "# Stream Consumption Boundary\n\nExperiment failed.\n\n```text\n"
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
