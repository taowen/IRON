#!/usr/bin/env python3
"""Probe direct entry into MyLM main16 phase bodies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP132_RUN = REPO_ROOT / "experiments/132_mylm_main16_dispatcher_stub_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "phase_body_direct_entry.json"
REPORT = EXPERIMENT_DIR / "phase_body_direct_entry.md"


@dataclass(frozen=True)
class Scenario:
    name: str
    target: int
    header: int
    records: int
    stream_chunks: int
    register_profile: str


def load_exp132() -> ModuleType:
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


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("qkv_body_1870_full", 0x1870, 0x1, 12, 192, "normal"),
        Scenario("o_body_1e80_full", 0x1E80, 0x4, 8, 128, "normal"),
        Scenario("upgate_body_2490_full", 0x2490, 0x8, 48, 768, "normal"),
        Scenario("down_body_2aa0_full", 0x2AA0, 0x4, 8, 384, "normal"),
    )


def profile_register_setup(profile: str) -> tuple[str, ...]:
    if profile == "normal":
        return (
            "  movxm p0, #0x78000",
            "  movxm p1, #0x78200",
            "  movxm p2, #0x72800",
            "  movxm p3, #0x75400",
            "  movxm p4, #0x73c00",
            "  movxm p5, #0x75400",
            "  movxm p6, #0x73c00",
            "  movxm p7, #0x75400",
        )
    if profile == "down":
        return (
            "  movxm p0, #0x78000",
            "  movxm p1, #0x78200",
            "  movxm p2, #0x72800",
            "  movxm p3, #0x75400",
            "  movxm p4, #0x7c000",
            "  movxm p5, #0x75400",
            "  movxm p6, #0x7c000",
            "  movxm p7, #0x74000",
        )
    raise ValueError(f"unknown register profile: {profile}")


def phase_stub_asm(scenario: Scenario) -> str:
    lines = [
        '  .section .text,"ax",@progbits',
        "  .globl __start",
        "  .type __start,@function",
        "  .p2align 4",
        "__start:",
        "  movxm sp, #0x70000",
        "  paddxm [sp], #0x80",
        "  movxm r16, #0x78200",
        "  st r16, [sp, #-4]",
        "  mov p0, r16",
        "  mova r17, #1",
        "  st r17, [p0, #0]",
        *profile_register_setup(scenario.register_profile),
        f"  movx r0, #{scenario.header}",
        "  movx r1, #-0x2",
        "  mova r11, #0",
        "  mova r13, #0",
        "  mova r14, #0",
        "  movx r15, #-0x2",
        f"  jl #0x{scenario.target:x}",
        "  nop",
        "  nop",
        "  nop",
        "  nop",
        "  nop",
        "  done",
        ".Lspin:",
        "  j #.Lspin",
        "  nop",
        "  nop",
        "  nop",
        "  nop",
        "  nop",
        "  .size __start, .-__start",
        "",
    ]
    return "\n".join(lines)


def runs(headers: list[str]) -> list[dict[str, int | str]]:
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


def run_scenario(scenario: Scenario, runtime_timeout: int, xrt_timeout: int) -> dict:
    old_stub_asm = EXP132.stub_asm
    old_activation_dwords = EXP132.EXP130.activation_dwords
    old_weight_dwords = EXP132.EXP130.weight_dwords
    EXP132.stub_asm = lambda _control_value: phase_stub_asm(scenario)
    EXP132.EXP130.activation_dwords = (
        lambda _records: scenario.stream_chunks * EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    )
    EXP132.EXP130.weight_dwords = (
        lambda _records: scenario.stream_chunks * EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    )
    try:
        variant = EXP132.Variant(scenario.name, scenario.header)
        result = EXP132.run_variant(variant, scenario.records, runtime_timeout, xrt_timeout)
    finally:
        EXP132.stub_asm = old_stub_asm
        EXP132.EXP130.activation_dwords = old_activation_dwords
        EXP132.EXP130.weight_dwords = old_weight_dwords
    headers = result.get("record_headers") or []
    expected = f"0x{scenario.header:x}"
    expected_headers = [expected] * scenario.records
    passed = result["status"] == "record_observed" and headers == expected_headers
    return {
        **result,
        "target": hex(scenario.target),
        "expected_header": expected,
        "expected_headers": expected_headers,
        "records_requested": scenario.records,
        "stream_chunks": scenario.stream_chunks,
        "register_profile": scenario.register_profile,
        "got_runs": runs(headers),
        "passed": passed,
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Phase Body Direct Entry",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Results",
        "",
        "| scenario | target | profile | records | chunks | status | expected | observed | pass | topology |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for result in manifest["scenarios"]:
        topology = result.get("topology_after", {}).get("topology")
        observed = result.get("record_headers") or []
        lines.append(
            f"| `{result['name']}` | `{result['target']}` | `{result['register_profile']}` | "
            f"`{result['records_requested']}` | `{result['stream_chunks']}` | `{result['status']}` | "
            f"`{result['expected_header']}` | `{observed}` | `{result['passed']}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if manifest["status"] == "passed":
        lines.append(
            "Each probed phase body can run its full phase-local record count when "
            "called directly with the copied dispatcher-style register profile. "
            "This means QKV/O/upgate/down phase bodies are not dispatcher-only black boxes."
        )
    else:
        lines.append(
            "At least one phase body did not emit the expected full phase record set under the "
            "current copied register profile. Treat this as a register/control-state "
            "mismatch, not as evidence that the phase body cannot be isolated."
        )
    lines.extend(
        [
            "",
            "Direct-entry timeout recovery:",
            "",
            "```bash",
            "sudo systemctl restart amdxdna-pinned.service",
            "```",
            "",
            "## Runtime Output Preview",
            "",
        ]
    )
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


def build_manifest(args: argparse.Namespace) -> dict:
    scenarios = default_scenarios()
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    results: list[dict] = []
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
    status = "passed" if results and all(result["passed"] for result in results) and not stopped_early else "failed"
    return {
        "status": status,
        "stopped_early": stopped_early,
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
            "# Phase Body Direct Entry\n\nExperiment failed.\n\n```text\n"
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
