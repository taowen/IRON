#!/usr/bin/env python3
"""Separate activation and weight chunk contribution axes for MyLM Q4NX."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP008_RUN = REPO_ROOT / "main16-exps/008_q4nx_payload_formula_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_activation_weight_axis_map.json"
REPORT = EXPERIMENT_DIR / "q4nx_activation_weight_axis_map.md"


@dataclass(frozen=True)
class AxisScenario:
    name: str
    axis: str
    activation_start: int
    activation_count: int
    weight_start: int
    weight_count: int


@dataclass(frozen=True)
class AxisResult:
    name: str
    axis: str
    status: str
    first_record_word: str
    first_record_value: float
    nonzero_records: tuple[int, ...]
    npu_time: str | None
    runtime_output: str


def load_exp008() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp008_for_axis_map", EXP008_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 008 helper: {EXP008_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP008 = load_exp008()


def configure_helpers() -> None:
    EXP008.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP008.BUILD_DIR = BUILD_DIR


def word_to_float(word_text: str) -> float:
    word = int(word_text, 16)
    return EXP008.bf16_to_float((word >> 16) & 0xFFFF)


def parse_record_words(stdout: str) -> tuple[int, ...]:
    match = re.search(r"^record=([^\n]+)", stdout, re.MULTILINE)
    if match is None:
        return ()
    return tuple(int(word, 16) for word in match.group(1).split(","))


def unique_payload_word_per_record(words: tuple[int, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for record_index in range(len(words) // EXP008.RECORD_DWORDS):
        start = record_index * EXP008.RECORD_DWORDS + 1
        payload = words[start : start + EXP008.RECORD_DWORDS - 1]
        unique = sorted(set(payload))
        out.append(hex(unique[0] & 0xFFFFFFFF) if len(unique) == 1 else "mixed")
    return tuple(out)


def nonzero_records(words: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(index for index, word in enumerate(words) if int(word, 16) != 0)


def scenario_json(scenario: AxisScenario) -> str:
    return json.dumps(
        {
            "activation_pair": EXP008.bf16_pair(0x3F80),
            "scale_pair": EXP008.bf16_pair(0x3C80),
            "nibble": 1,
            "activation_start": scenario.activation_start,
            "activation_count": scenario.activation_count,
            "weight_start": scenario.weight_start,
            "weight_count": scenario.weight_count,
        },
        sort_keys=True,
    )


def run_axis_scenario(scenario: AxisScenario, runtime_timeout: int) -> AxisResult:
    env = os.environ.copy()
    env["PATH"] = f"{EXP008.EXP005.EXP132.EXP130.XRT_BIN}:{env.get('PATH', '')}"
    out_dwords = EXP008.EXP005.EXP132.EXP130.output_dwords(EXP008.RECORDS)
    act_dwords = EXP008.CHUNKS * EXP008.EXP005.EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    wt_dwords = EXP008.CHUNKS * EXP008.EXP005.EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    payload = scenario_json(scenario)
    code = f"""
import json
import os
import time
import traceback
os.environ['PATH'] = {env["PATH"]!r}
import numpy as np
import torch
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

SCALE_DWORDS = {EXP008.SCALE_DWORDS}
ZERO_DWORDS = {EXP008.ZERO_DWORDS}
Q4_CHUNK_DWORDS = {EXP008.Q4_CHUNK_DWORDS}
ACT_DWORDS_PER_CHUNK = {EXP008.EXP005.EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK}
SCENARIO = json.loads({payload!r})

def make_activation():
    activation = np.zeros(({act_dwords},), dtype=np.int32)
    start = int(SCENARIO['activation_start'])
    end = start + int(SCENARIO['activation_count'])
    pair = np.int32(int(SCENARIO['activation_pair']))
    for chunk in range(start, end):
        base = chunk * ACT_DWORDS_PER_CHUNK
        activation[base:base + ACT_DWORDS_PER_CHUNK] = pair
    return activation

def make_weights():
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    start = int(SCENARIO['weight_start'])
    end = start + int(SCENARIO['weight_count'])
    scale_pair = np.int32(int(SCENARIO['scale_pair']))
    q4_word = np.int32((int(SCENARIO['nibble']) & 0xf) * 0x11111111)
    chunks = weights.reshape((-1, Q4_CHUNK_DWORDS))
    for chunk_index in range(start, end):
        chunk = chunks[chunk_index]
        chunk[:SCALE_DWORDS] = scale_pair
        chunk[SCALE_DWORDS:SCALE_DWORDS + ZERO_DWORDS] = np.int32(0)
        chunk[SCALE_DWORDS + ZERO_DWORDS:] = q4_word
    return weights

try:
    aie_utils.DefaultNPURuntime.cleanup()
except Exception:
    pass

try:
    kernel = NPUKernel(
        xclbin_path={str(EXP008.EXP005.EXP132.EXP130.XCLBIN)!r},
        kernel_name='MLIR_AIE',
        insts_path={str(EXP008.EXP005.EXP132.EXP130.INSTS)!r},
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    out_buf = XRTTensor(({out_dwords},), dtype=np.int32)
    activation_buf = XRTTensor.from_torch(torch.from_numpy(make_activation()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(make_weights()).to(torch.int32))
    start_time = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf, weights_buf, activation_buf])
    elapsed = time.time() - start_time
    got = out_buf.to_torch().numpy().astype(np.int32)
    print('run_ok')
    print('elapsed=' + str(elapsed))
    print('npu_time=' + str(getattr(result, 'npu_time', None)))
    print('record=' + ','.join(hex(int(x) & 0xffffffff) for x in got.tolist()))
except Exception:
    print('exception_begin')
    traceback.print_exc()
    raise
finally:
    try:
        aie_utils.DefaultNPURuntime.cleanup()
    except Exception:
        traceback.print_exc()
"""
    try:
        result = EXP008.EXP005.EXP132.EXP130.run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=runtime_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = EXP008.EXP005.EXP132.EXP130.as_text(exc.stdout) + EXP008.EXP005.EXP132.EXP130.as_text(exc.stderr)
        return AxisResult(
            scenario.name,
            scenario.axis,
            "timeout_waiting_for_record",
            "0x0",
            0.0,
            (),
            None,
            EXP008.EXP005.EXP132.EXP130._short_output(output, max_chars=1800),
        )
    words = parse_record_words(result.stdout)
    record_words = unique_payload_word_per_record(words)
    first = record_words[0] if record_words else "0x0"
    status = "record_observed" if result.returncode == 0 and len(words) == out_dwords else "runtime_failed"
    npu_match = re.search(r"^npu_time=([^\n]+)", result.stdout, re.MULTILINE)
    output = result.stdout + result.stderr
    if "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    return AxisResult(
        scenario.name,
        scenario.axis,
        status,
        first,
        word_to_float(first),
        nonzero_records(record_words),
        npu_match.group(1) if npu_match is not None else None,
        EXP008.EXP005.EXP132.EXP130._short_output(output, max_chars=1200),
    )


def scenarios() -> tuple[AxisScenario, ...]:
    rows: list[AxisScenario] = []
    for chunk in range(EXP008.CHUNKS_PER_RECORD):
        rows.append(AxisScenario(f"activation_chunk_{chunk:02d}", "activation", chunk, 1, 0, EXP008.CHUNKS_PER_RECORD))
    for chunk in range(EXP008.CHUNKS_PER_RECORD):
        rows.append(AxisScenario(f"weight_chunk_{chunk:02d}", "weight", 0, EXP008.CHUNKS_PER_RECORD, chunk, 1))
    return tuple(rows)


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    before = EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    build = EXP008.build_qkv_direct_xclbin()
    results = [run_axis_scenario(scenario, args.runtime_timeout) for scenario in scenarios()]
    after = EXP008.topology_status(args.xrt_timeout)
    complete = after["status"] == "ok" and all(result.status == "record_observed" for result in results)
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build,
        "results": [
            {
                "name": result.name,
                "axis": result.axis,
                "status": result.status,
                "first_record_word": result.first_record_word,
                "first_record_value": result.first_record_value,
                "nonzero_records": list(result.nonzero_records),
                "npu_time": result.npu_time,
                "runtime_output": result.runtime_output,
            }
            for result in results
        ],
    }


def render_axis_table(lines: list[str], manifest: dict, axis: str) -> None:
    lines.extend(
        [
            f"## {axis.title()} Axis",
            "",
            "| chunk | word | value | nonzero records | npu_time |",
            "| ---: | --- | ---: | --- | --- |",
        ]
    )
    for result in manifest.get("results", []):
        if result["axis"] != axis:
            continue
        chunk = int(result["name"].rsplit("_", 1)[1])
        lines.append(
            f"| `{chunk}` | `{result['first_record_word']}` | "
            f"`{result['first_record_value']}` | `{result['nonzero_records']}` | "
            f"`{result['npu_time']}` |"
        )
    lines.append("")


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Activation/Weight Axis Map",
        "",
        f"- Status: `{manifest['status']}`",
        "- Setup: activation bf16 `1.0`, scale bf16 `1/64`, q4 nibble `1`",
        "",
    ]
    render_axis_table(lines, manifest, "activation")
    render_axis_table(lines, manifest, "weight")
    lines.extend(
        [
            "## Interpretation",
            "",
            "If the activation axis is dense but the weight axis is even-only, the "
            "chunk contribution effect belongs to the Q4NX weight stream/layout. If "
            "both axes are even-only, it is a paired ping/pong schedule property.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Activation/Weight Axis Map\n\nExperiment failed.\n\n```text\n"
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

