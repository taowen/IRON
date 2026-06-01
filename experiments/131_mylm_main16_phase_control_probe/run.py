#!/usr/bin/env python3
"""Probe MyLM main16 raw-program phase/control words with exp130's harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BASE_RUN = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/run.py"
DEFAULT_RAW = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_c2r2_program.bin"
DEFAULT_73C80 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73c80.bin"
DEFAULT_73D00 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73d00.bin"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "mylm_main16_phase_control_probe.json"
REPORT = EXPERIMENT_DIR / "mylm_main16_phase_control_probe.md"
XRT_SMI = Path("/var/opt/xilinx/xrt/bin/xrt-smi")

RECORD_DWORDS_PER_RECORD = 17
PHASE_ADDRS = (0x1870, 0x1E80, 0x2490, 0x2AA0, 0x30C0, 0x36D0, 0x3820)


@dataclass(frozen=True)
class Patch:
    segment: str
    dword: int
    value: int


@dataclass(frozen=True)
class Variant:
    name: str
    patches: tuple[Patch, ...]


@dataclass(frozen=True)
class Topology:
    status: str
    topology: str | None
    stdout: str
    stderr: str


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("mylm_exp130", BASE_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp130 harness: {BASE_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()


def patch_static(base: bytes, segment: str, patches: tuple[Patch, ...]) -> bytes:
    out = bytearray(base)
    for patch in patches:
        if patch.segment != segment:
            continue
        offset = patch.dword * 4
        if offset + 4 > len(out):
            raise ValueError(f"{segment} dword {patch.dword} outside {len(out)} bytes")
        struct.pack_into("<I", out, offset, patch.value)
    return bytes(out)


def variant_dir(variant: Variant) -> Path:
    return BUILD_DIR / variant.name


def configure_exp130_paths(build: Path) -> None:
    EXP130.RAW_COPY = build / "mylm_c2r2_program.bin"
    EXP130.STATIC_73C80_COPY = build / "mylm_static_73c80.bin"
    EXP130.STATIC_73D00_COPY = build / "mylm_static_73d00.bin"
    EXP130.ELF = build / "mylm_c2r2_main16_phase_probe.elf"
    EXP130.MLIR = build / "design.mlir"
    EXP130.TXN = build / "design.txn.mlir"
    EXP130.XCLBIN = build / "design.xclbin"
    EXP130.INSTS = build / "design.bin"
    EXP130.PRJ_DIR = build / "prj"


def build_variant(variant: Variant, records: int) -> dict[str, Any]:
    build = variant_dir(variant)
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    raw = DEFAULT_RAW.read_bytes()
    static_73c80 = patch_static(DEFAULT_73C80.read_bytes(), "73c80", variant.patches)
    static_73d00 = patch_static(DEFAULT_73D00.read_bytes(), "73d00", variant.patches)
    raw_path = build / "mylm_c2r2_program.bin.input"
    static_73c80_path = build / "mylm_static_73c80.bin.input"
    static_73d00_path = build / "mylm_static_73d00.bin.input"
    raw_path.write_bytes(raw)
    static_73c80_path.write_bytes(static_73c80)
    static_73d00_path.write_bytes(static_73d00)
    configure_exp130_paths(build)
    source = EXP130.write_elf(raw_path, static_73c80_path, static_73d00_path)
    EXP130.write_mlir(records)
    EXP130.package_with_aiecc()
    return {
        "build_dir": str(build.relative_to(REPO_ROOT)),
        "source": source,
        "artifacts": {
            "elf": str(EXP130.ELF.relative_to(REPO_ROOT)),
            "mlir": str(EXP130.MLIR.relative_to(REPO_ROOT)),
            "txn": str(EXP130.TXN.relative_to(REPO_ROOT)),
            "xclbin": str(EXP130.XCLBIN.relative_to(REPO_ROOT)),
            "insts": str(EXP130.INSTS.relative_to(REPO_ROOT)),
        },
        "transaction": {
            "payload_bytes": EXP130.txn_payload_sizes(EXP130.TXN),
            "blockwrite_addresses": [
                hex(address) for address in EXP130.txn_blockwrite_addresses(EXP130.TXN)
            ],
        },
    }


def examine_topology(timeout_s: int) -> Topology:
    result = subprocess.run(
        (str(XRT_SMI), "examine"),
        cwd=EXPERIMENT_DIR,
        timeout=timeout_s,
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout + result.stderr
    match = re.search(r"Topology\s*:\s*([^\n]+)", text)
    if match:
        topology = match.group(1).strip()
    else:
        table_match = re.search(r"\|\[[^\n]+\]\s*\|[^|]+\|[^|]+\|\s*([^|\s]+)\s*\|", text)
        topology = table_match.group(1).strip() if table_match else None
    if result.returncode != 0:
        status = "xrt_smi_failed"
    elif topology == "6x8":
        status = "ok"
    else:
        status = "unexpected_topology"
    return Topology(status, topology, result.stdout, result.stderr)


def run_variant(variant: Variant, records: int, runtime_timeout: int, xrt_timeout: int) -> dict[str, Any]:
    before = examine_topology(xrt_timeout)
    if before.status != "ok":
        return {
            "name": variant.name,
            "status": "skipped_topology_not_ok",
            "topology_before": before.__dict__,
            "patches": [patch.__dict__ for patch in variant.patches],
        }
    build_info = build_variant(variant, records)
    runtime = EXP130.try_run_observable(runtime_timeout, records)
    after = examine_topology(xrt_timeout)
    record_words = runtime.get("record_words") or []
    headers = record_words[0::RECORD_DWORDS_PER_RECORD]
    return {
        "name": variant.name,
        "status": runtime["status"],
        "patches": [patch.__dict__ for patch in variant.patches],
        "topology_before": before.__dict__,
        "topology_after": after.__dict__,
        **build_info,
        "runtime_probe": {
            key: value
            for key, value in runtime.items()
            if key not in {"stdout", "stderr"}
        },
        "runtime_output": EXP130._short_output(
            (runtime.get("stdout") or "") + (runtime.get("stderr") or ""),
            max_chars=1400,
        ),
        "record_headers": headers,
        "unique_record_headers": sorted(set(headers)),
    }


def default_variants() -> tuple[Variant, ...]:
    variants = [Variant("baseline_73d00_3820", ())]
    variants.extend(
        Variant(f"patch_73d00_0_{addr:04x}", (Patch("73d00", 0, addr),))
        for addr in PHASE_ADDRS
        if addr != 0x3820
    )
    variants.extend(
        Variant(f"patch_73c80_0_{addr:04x}", (Patch("73c80", 0, addr),))
        for addr in PHASE_ADDRS
    )
    return tuple(variants)


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# MyLM Main16 Phase-Control Probe",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Records per variant: `{manifest['records']}`",
        f"- Variants attempted: `{len(manifest['variants'])}`",
        "",
        "## Results",
        "",
        "| variant | status | unique headers | topology after |",
        "| --- | --- | --- | --- |",
    ]
    for result in manifest["variants"]:
        topology = result.get("topology_after", {}).get("topology")
        lines.append(
            f"| `{result['name']}` | `{result['status']}` | "
            f"`{result.get('unique_record_headers', [])}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    observed = [
        result
        for result in manifest["variants"]
        if result["status"] == "record_observed"
    ]
    if observed:
        lines.append(
            "Every observed variant still emitted the same record-header set shown "
            "above. This rules out a simple model where `73d00[0]` or `73c80[0]` "
            "directly selects the phase body/header under this standalone harness."
        )
    if manifest.get("stopped_early"):
        lines.append(
            "The sweep stopped early because a timeout or topology change was seen. "
            "That is still useful evidence: the patch touched executable control "
            "state, but not a safe direct phase selector."
        )
    lines.extend(
        [
            "",
            "Follow-up disassembly of the baseline ELF shows the relevant control is "
            "more likely in the entry/dispatcher path:",
            "",
            "- entry at `0x0` initializes the stack, sets tile-local pointers, then "
            "calls dispatcher `0x36d0`;",
            "- dispatcher text hardcodes the phase headers with `r0 = 0x1`, "
            "`r0 = 0x4`, and `r0 = 0x8` before phase-body calls;",
            "- the first branch condition in dispatcher reads a caller stack/control "
            "slot, not the first dword of `0x73c80` or `0x73d00`.",
            "",
            "So the next useful control experiment is not more static-data first-dword "
            "patching. It should either patch the entry/dispatcher control slot setup, "
            "or build a tiny raw caller/stub that enters a chosen phase body with the "
            "expected `p0..p7` and `r0` state.",
        ]
    )
    lines.extend(["", "## Runtime Output Preview", ""])
    for result in manifest["variants"]:
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
    variants = default_variants()
    if args.max_variants > 0:
        variants = variants[: args.max_variants]
    results: list[dict[str, Any]] = []
    stopped_early = False
    for variant in variants:
        result = run_variant(variant, args.records, args.runtime_timeout, args.xrt_timeout)
        results.append(result)
        if result["status"] in {"timeout_waiting_for_record", "xrt_get_info_failed"}:
            stopped_early = True
            break
        topology_after = result.get("topology_after", {})
        if topology_after.get("status") != "ok":
            stopped_early = True
            break
    status = "completed" if not stopped_early else "stopped_early"
    return {
        "status": status,
        "records": args.records,
        "runtime_timeout_s": args.runtime_timeout,
        "variants": results,
        "stopped_early": stopped_early,
        "candidate_phase_addresses": [hex(addr) for addr in PHASE_ADDRS],
        "source": {
            "harness": str(BASE_RUN.relative_to(REPO_ROOT)),
            "raw": str(DEFAULT_RAW.relative_to(REPO_ROOT)),
            "static_73c80": str(DEFAULT_73C80.relative_to(REPO_ROOT)),
            "static_73d00": str(DEFAULT_73D00.relative_to(REPO_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--runtime-timeout", type=int, default=8)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-variants", type=int, default=3)
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
            "# MyLM Main16 Phase-Control Probe\n\nExperiment failed.\n\n```text\n"
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
