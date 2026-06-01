#!/usr/bin/env python3
"""Probe the scalar Q4NX payload formula of MyLM direct Q/K/V body."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP005_RUN = REPO_ROOT / "main16-exps/005_nonzero_payload_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_payload_formula_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_payload_formula_probe.md"


def load_exp005() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp005_for_formula", EXP005_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 005 helper: {EXP005_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP005 = load_exp005()

RECORDS = EXP005.QKV_RECORDS
CHUNKS = EXP005.QKV_CHUNKS
CHUNKS_PER_RECORD = 16
EFFECTIVE_BF16_PER_CHUNK = 128
RECORD_DWORDS = EXP005.RECORD_DWORDS
SCALE_DWORDS = EXP005.SCALE_DWORDS
ZERO_DWORDS = EXP005.ZERO_DWORDS
Q4_DATA_DWORDS = EXP005.Q4_DATA_DWORDS
Q4_CHUNK_DWORDS = EXP005.Q4_CHUNK_DWORDS


@dataclass(frozen=True)
class Scenario:
    name: str
    activation_bf16: int
    scale_bf16: int
    nibble: int
    active_start: int
    active_count: int


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    status: str
    headers: tuple[str, ...]
    actual_record_words: tuple[str, ...]
    expected_record_words: tuple[str, ...]
    formula_match: bool
    payload_nonzero: int
    payload_hash: str
    npu_time: str | None
    runtime_output: str


def bf16_pair(bf16: int) -> int:
    return ((bf16 & 0xFFFF) << 16) | (bf16 & 0xFFFF)


def bf16_to_float(bf16: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (bf16 & 0xFFFF) << 16))[0]


def float_to_bf16(value: float) -> int:
    raw = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    lsb = (raw >> 16) & 1
    return ((raw + 0x7FFF + lsb) >> 16) & 0xFFFF


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("zero_all", 0x0000, 0x0000, 0, 0, 0),
        Scenario("scale_1_64_nibble_1", 0x3F80, 0x3C80, 1, 0, CHUNKS),
        Scenario("scale_1_64_nibble_2", 0x3F80, 0x3C80, 2, 0, CHUNKS),
        Scenario("scale_1_128_nibble_1", 0x3F80, 0x3C00, 1, 0, CHUNKS),
        Scenario("scale_1_32_nibble_1", 0x3F80, 0x3D00, 1, 0, CHUNKS),
        Scenario("activation_2_scale_1_64_nibble_1", 0x4000, 0x3C80, 1, 0, CHUNKS),
        Scenario("first_record_only", 0x3F80, 0x3C80, 1, 0, CHUNKS_PER_RECORD),
        Scenario("first_chunk_only", 0x3F80, 0x3C80, 1, 0, 1),
    )


def configure_helpers() -> None:
    EXP005.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP005.BUILD_DIR = BUILD_DIR


def active_chunks_for_record(scenario: Scenario, record_index: int) -> int:
    record_start = record_index * CHUNKS_PER_RECORD
    record_end = record_start + CHUNKS_PER_RECORD
    active_start = scenario.active_start
    active_end = scenario.active_start + scenario.active_count
    return max(0, min(record_end, active_end) - max(record_start, active_start))


def expected_record_words(scenario: Scenario) -> tuple[str, ...]:
    activation = bf16_to_float(scenario.activation_bf16)
    scale = bf16_to_float(scenario.scale_bf16)
    words: list[str] = []
    for record_index in range(RECORDS):
        active = active_chunks_for_record(scenario, record_index)
        value = active * EFFECTIVE_BF16_PER_CHUNK * activation * scale * scenario.nibble
        bf16 = float_to_bf16(value)
        words.append(hex(bf16_pair(bf16)))
    return tuple(words)


def build_qkv_direct_xclbin() -> dict:
    configure_helpers()
    return EXP005.build_qkv_direct_xclbin()


def topology_status(timeout_s: int) -> dict:
    return EXP005.topology_status(timeout_s)


def scenario_json(scenario: Scenario) -> str:
    return json.dumps(
        {
            "name": scenario.name,
            "activation_pair": bf16_pair(scenario.activation_bf16),
            "scale_pair": bf16_pair(scenario.scale_bf16),
            "nibble": scenario.nibble,
            "active_start": scenario.active_start,
            "active_count": scenario.active_count,
        },
        sort_keys=True,
    )


def parse_record_words(stdout: str) -> tuple[int, ...]:
    match = re.search(r"^record=([^\n]+)", stdout, re.MULTILINE)
    if match is None:
        return ()
    return tuple(int(word, 16) for word in match.group(1).split(","))


def unique_payload_word_per_record(words: tuple[int, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for record_index in range(len(words) // RECORD_DWORDS):
        start = record_index * RECORD_DWORDS + 1
        payload = words[start : start + RECORD_DWORDS - 1]
        unique = sorted(set(payload))
        out.append(hex(unique[0] & 0xFFFFFFFF) if len(unique) == 1 else "mixed")
    return tuple(out)


def run_scenario(scenario: Scenario, runtime_timeout: int) -> ScenarioResult:
    env = os.environ.copy()
    env["PATH"] = f"{EXP005.EXP132.EXP130.XRT_BIN}:{env.get('PATH', '')}"
    out_dwords = EXP005.EXP132.EXP130.output_dwords(RECORDS)
    act_dwords = CHUNKS * EXP005.EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    wt_dwords = CHUNKS * EXP005.EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    scenario_text = scenario_json(scenario)
    code = f"""
import hashlib
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

RECORD_DWORDS = {RECORD_DWORDS}
SCALE_DWORDS = {SCALE_DWORDS}
ZERO_DWORDS = {ZERO_DWORDS}
Q4_DATA_DWORDS = {Q4_DATA_DWORDS}
Q4_CHUNK_DWORDS = {Q4_CHUNK_DWORDS}
ACT_DWORDS_PER_CHUNK = {EXP005.EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK}
WT_DWORDS_PER_CHUNK = {EXP005.EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK}
SCENARIO = json.loads({scenario_text!r})

def make_activation():
    activation = np.zeros(({act_dwords},), dtype=np.int32)
    start = int(SCENARIO['active_start'])
    end = start + int(SCENARIO['active_count'])
    pair = np.int32(int(SCENARIO['activation_pair']))
    for chunk in range(start, end):
        base = chunk * ACT_DWORDS_PER_CHUNK
        activation[base:base + ACT_DWORDS_PER_CHUNK] = pair
    return activation

def make_weights():
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    start = int(SCENARIO['active_start'])
    end = start + int(SCENARIO['active_count'])
    scale_pair = np.int32(int(SCENARIO['scale_pair']))
    nibble = int(SCENARIO['nibble']) & 0xf
    q4_word = np.int32(nibble * 0x11111111)
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
        xclbin_path={str(EXP005.EXP132.EXP130.XCLBIN)!r},
        kernel_name='MLIR_AIE',
        insts_path={str(EXP005.EXP132.EXP130.INSTS)!r},
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    out_buf = XRTTensor(({out_dwords},), dtype=np.int32)
    activation_buf = XRTTensor.from_torch(torch.from_numpy(make_activation()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(make_weights()).to(torch.int32))
    start_time = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf, weights_buf, activation_buf])
    elapsed = time.time() - start_time
    got = out_buf.to_torch().numpy().astype(np.int32)
    payload = got.reshape((-1, RECORD_DWORDS))[:, 1:].copy()
    print('run_ok')
    print('elapsed=' + str(elapsed))
    print('npu_time=' + str(getattr(result, 'npu_time', None)))
    print('headers=' + ','.join(hex(int(x) & 0xffffffff) for x in got[0::RECORD_DWORDS].tolist()))
    print('payload_nonzero=' + str(int(np.count_nonzero(payload))))
    print('payload_hash=' + hashlib.sha256(payload.tobytes()).hexdigest())
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
    started = time.time()
    try:
        result = EXP005.EXP132.EXP130.run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=runtime_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = EXP005.EXP132.EXP130.as_text(exc.stdout) + EXP005.EXP132.EXP130.as_text(exc.stderr)
        return ScenarioResult(
            scenario.name,
            "timeout_waiting_for_record",
            (),
            (),
            expected_record_words(scenario),
            False,
            0,
            "",
            None,
            EXP005.EXP132.EXP130._short_output(output, max_chars=1800),
        )
    stdout = result.stdout
    stderr = result.stderr
    words = parse_record_words(stdout)
    headers = tuple(hex(word & 0xFFFFFFFF) for word in words[0::RECORD_DWORDS])
    actual = unique_payload_word_per_record(words)
    expected = expected_record_words(scenario)
    payload_values = []
    for record_index in range(len(words) // RECORD_DWORDS):
        start = record_index * RECORD_DWORDS + 1
        payload_values.extend(words[start : start + RECORD_DWORDS - 1])
    status = "record_observed" if result.returncode == 0 and len(words) == out_dwords else "runtime_failed"
    output = stdout + stderr
    if "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    payload_hash_match = re.search(r"^payload_hash=([0-9a-f]+)", stdout, re.MULTILINE)
    nonzero_match = re.search(r"^payload_nonzero=(\d+)", stdout, re.MULTILINE)
    npu_match = re.search(r"^npu_time=([^\n]+)", stdout, re.MULTILINE)
    payload_bytes = b"".join(int(value & 0xFFFFFFFF).to_bytes(4, "little") for value in payload_values)
    return ScenarioResult(
        scenario.name,
        status,
        headers,
        actual,
        expected,
        status == "record_observed" and headers == tuple(["0x1"] * RECORDS) and actual == expected,
        int(nonzero_match.group(1)) if nonzero_match is not None else 0,
        payload_hash_match.group(1) if payload_hash_match is not None else hashlib.sha256(payload_bytes).hexdigest(),
        npu_match.group(1) if npu_match is not None else None,
        EXP005.EXP132.EXP130._short_output(output, max_chars=1800),
    )


def result_row(result: ScenarioResult) -> dict:
    return {
        "name": result.name,
        "status": result.status,
        "headers": list(result.headers),
        "actual_record_words": list(result.actual_record_words),
        "expected_record_words": list(result.expected_record_words),
        "formula_match": result.formula_match,
        "payload_nonzero": result.payload_nonzero,
        "payload_hash": result.payload_hash,
        "npu_time": result.npu_time,
        "runtime_output": result.runtime_output,
    }


def build_manifest(args: argparse.Namespace) -> dict:
    before = topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {
            "status": "skipped_topology_not_ok",
            "topology_before": before,
            "scenarios": [],
        }
    build_info = build_qkv_direct_xclbin()
    chosen = scenarios()
    if args.max_scenarios > 0:
        chosen = chosen[: args.max_scenarios]
    results = [run_scenario(scenario, args.runtime_timeout) for scenario in chosen]
    after = topology_status(args.xrt_timeout)
    passed = after["status"] == "ok" and all(result.formula_match for result in results)
    return {
        "status": "passed" if passed else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build_info,
        "contract": {
            "records": RECORDS,
            "chunks": CHUNKS,
            "chunks_per_record": CHUNKS_PER_RECORD,
            "effective_bf16_per_chunk": EFFECTIVE_BF16_PER_CHUNK,
            "formula": "active_chunks_per_record * 128 * activation_bf16 * scale_bf16 * q4_nibble",
        },
        "scenarios": [
            {
                "name": scenario.name,
                "activation_bf16": hex(scenario.activation_bf16),
                "scale_bf16": hex(scenario.scale_bf16),
                "nibble": scenario.nibble,
                "active_start": scenario.active_start,
                "active_count": scenario.active_count,
                "expected_record_words": list(expected_record_words(scenario)),
            }
            for scenario in chosen
        ],
        "results": [result_row(result) for result in results],
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Payload Formula Probe",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Formula: `{manifest.get('contract', {}).get('formula', '')}`",
        "",
        "## Results",
        "",
        "| scenario | status | formula match | actual record words | expected record words | nonzero payload words | npu_time |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for result in manifest.get("results", []):
        actual = result["actual_record_words"]
        expected = result["expected_record_words"]
        lines.append(
            f"| `{result['name']}` | `{result['status']}` | `{result['formula_match']}` | "
            f"`{actual[:4]}{'...' if len(actual) > 4 else ''}` | "
            f"`{expected[:4]}{'...' if len(expected) > 4 else ''}` | "
            f"`{result['payload_nonzero']}` | `{result['npu_time']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if manifest["status"] == "passed":
        lines.extend(
            [
                "The synthetic direct-QKV payloads match the scalar formula with 128 "
                "effective bf16 products per active stream chunk. With 16 chunks per "
                "record, uniform activation=1, scale=1/64, and nibble=1 produce bf16 "
                "`32` (`0x4200`) in every payload lane.",
                "",
                "The first-record-only and first-chunk-only cases also match. That "
                "anchors the record-major chunk mapping: each Q/K/V record consumes 16 "
                "chunks, and one active chunk contributes bf16 `2` (`0x4000`) under the "
                "same activation/scale/nibble setup.",
            ]
        )
    else:
        lines.append(
            "At least one synthetic case failed the scalar formula. Inspect the "
            "per-scenario actual/expected words before using this as a reference."
        )
    lines.extend(
        [
            "",
            "## Runtime Output Preview",
            "",
        ]
    )
    for result in manifest.get("results", []):
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "```text",
                result["runtime_output"],
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-scenarios", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Payload Formula Probe\n\nExperiment failed.\n\n```text\n"
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

