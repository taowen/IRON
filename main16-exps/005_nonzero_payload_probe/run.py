#!/usr/bin/env python3
"""Probe non-zero payload behavior of the MyLM Q/K/V phase body."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP132_RUN = REPO_ROOT / "experiments/132_mylm_main16_dispatcher_stub_probe/run.py"
EXP003_RUN = REPO_ROOT / "main16-exps/003_phase_body_direct_entry/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "nonzero_payload_probe.json"
REPORT = EXPERIMENT_DIR / "nonzero_payload_probe.md"

QKV_TARGET = 0x1870
QKV_HEADER = 0x1
QKV_RECORDS = 12
QKV_CHUNKS = 192
RECORD_DWORDS = 17
SCALE_DWORDS = 128
ZERO_DWORDS = 128
Q4_DATA_DWORDS = 1024
Q4_CHUNK_DWORDS = SCALE_DWORDS + ZERO_DWORDS + Q4_DATA_DWORDS
BF16_ONE_PAIR = 0x3F803F80
BF16_SCALE_PAIR = 0x3C803C80


@dataclass(frozen=True)
class PayloadScenario:
    name: str
    activation_pattern: str
    weight_pattern: str


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP132 = load_module("mylm_exp132_for_nonzero_payload", EXP132_RUN)
EXP003 = load_module("mylm_exp003_for_nonzero_payload", EXP003_RUN)


def scenarios() -> tuple[PayloadScenario, ...]:
    return (
        PayloadScenario("zero_all", "zero", "zero"),
        PayloadScenario("activation_only", "ones", "zero"),
        PayloadScenario("q4_ones", "ones", "q4_ones"),
        PayloadScenario("q4_twos", "ones", "q4_twos"),
    )


def configure_helpers() -> None:
    EXP132.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP132.BUILD_DIR = BUILD_DIR


def build_qkv_direct_xclbin() -> dict:
    configure_helpers()
    scenario = EXP003.Scenario(
        "qkv_nonzero_payload_direct",
        QKV_TARGET,
        QKV_HEADER,
        QKV_RECORDS,
        QKV_CHUNKS,
        "normal",
    )
    old_stub_asm = EXP132.stub_asm
    old_activation_dwords = EXP132.EXP130.activation_dwords
    old_weight_dwords = EXP132.EXP130.weight_dwords
    EXP132.stub_asm = lambda _control_value: EXP003.phase_stub_asm(scenario)
    EXP132.EXP130.activation_dwords = (
        lambda _records: QKV_CHUNKS * EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    )
    EXP132.EXP130.weight_dwords = (
        lambda _records: QKV_CHUNKS * EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    )
    try:
        variant = EXP132.Variant("qkv_nonzero_payload_direct", QKV_HEADER)
        return EXP132.build_variant(variant, QKV_RECORDS)
    finally:
        EXP132.stub_asm = old_stub_asm
        EXP132.EXP130.activation_dwords = old_activation_dwords
        EXP132.EXP130.weight_dwords = old_weight_dwords


def topology_status(timeout_s: int) -> dict:
    result = EXP132.examine_topology(timeout_s)
    return result.__dict__


def run_payload_scenario(scenario: PayloadScenario, runtime_timeout: int) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{EXP132.EXP130.XRT_BIN}:{env.get('PATH', '')}"
    out_dwords = EXP132.EXP130.output_dwords(QKV_RECORDS)
    act_dwords = QKV_CHUNKS * EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    wt_dwords = QKV_CHUNKS * EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    code = f"""
import hashlib
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
BF16_ONE_PAIR = {BF16_ONE_PAIR}
BF16_SCALE_PAIR = {BF16_SCALE_PAIR}

def make_activation(pattern):
    activation = np.zeros(({act_dwords},), dtype=np.int32)
    if pattern == 'ones':
        activation[:] = np.int32(BF16_ONE_PAIR)
    elif pattern == 'ramp':
        for index in range(activation.size):
            activation[index] = np.int32(0x3f800000 | ((index * 17) & 0xffff))
    elif pattern != 'zero':
        raise ValueError('unknown activation pattern: ' + pattern)
    return activation

def fill_q4_chunk(chunk, nibble):
    chunk[:SCALE_DWORDS] = np.int32(BF16_SCALE_PAIR)
    chunk[SCALE_DWORDS:SCALE_DWORDS + ZERO_DWORDS] = np.int32(0)
    value = (nibble & 0xf) * 0x11111111
    chunk[SCALE_DWORDS + ZERO_DWORDS:] = np.int32(value)

def make_weights(pattern):
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    if pattern == 'zero':
        return weights
    if pattern == 'q4_ones':
        nibble = 1
    elif pattern == 'q4_twos':
        nibble = 2
    else:
        raise ValueError('unknown weight pattern: ' + pattern)
    chunks = weights.reshape((-1, Q4_CHUNK_DWORDS))
    for chunk in chunks:
        fill_q4_chunk(chunk, nibble)
    return weights

try:
    aie_utils.DefaultNPURuntime.cleanup()
except Exception:
    pass

try:
    kernel = NPUKernel(
        xclbin_path={str(EXP132.EXP130.XCLBIN)!r},
        kernel_name='MLIR_AIE',
        insts_path={str(EXP132.EXP130.INSTS)!r},
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    out_buf = XRTTensor(({out_dwords},), dtype=np.int32)
    activation = make_activation({scenario.activation_pattern!r})
    weights = make_weights({scenario.weight_pattern!r})
    activation_buf = XRTTensor.from_torch(torch.from_numpy(activation).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights).to(torch.int32))
    start = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf, weights_buf, activation_buf])
    elapsed = time.time() - start
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
        result = EXP132.EXP130.run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=runtime_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": scenario.name,
            "activation_pattern": scenario.activation_pattern,
            "weight_pattern": scenario.weight_pattern,
            "status": "timeout_waiting_for_record",
            "elapsed_s": time.time() - started,
            "headers": [],
            "unique_headers": [],
            "payload_nonzero": None,
            "payload_hash": None,
            "npu_time": None,
            "runtime_output": EXP132.EXP130._short_output(
                EXP132.EXP130.as_text(exc.stdout) + EXP132.EXP130.as_text(exc.stderr),
                max_chars=1800,
            ),
        }
    elapsed = time.time() - started
    stdout = result.stdout
    stderr = result.stderr
    output = stdout + stderr
    record_match = re.search(r"^record=([^\n]+)", stdout, re.MULTILINE)
    words = record_match.group(1).split(",") if record_match else []
    headers = words[0::RECORD_DWORDS]
    payload_words = []
    for record_index in range(len(words) // RECORD_DWORDS):
        start = record_index * RECORD_DWORDS + 1
        payload_words.extend(words[start : start + RECORD_DWORDS - 1])
    payload_bytes = "\n".join(payload_words).encode("ascii")
    status = "record_observed" if result.returncode == 0 and len(words) == out_dwords else "runtime_failed"
    if "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    return {
        "name": scenario.name,
        "activation_pattern": scenario.activation_pattern,
        "weight_pattern": scenario.weight_pattern,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "headers": headers,
        "unique_headers": sorted(set(headers)),
        "unique_payload_words": sorted(set(payload_words)),
        "payload_nonzero": int(re.search(r"^payload_nonzero=(\d+)", stdout, re.MULTILINE).group(1))
        if re.search(r"^payload_nonzero=(\d+)", stdout, re.MULTILINE)
        else None,
        "payload_hash": re.search(r"^payload_hash=([0-9a-f]+)", stdout, re.MULTILINE).group(1)
        if re.search(r"^payload_hash=([0-9a-f]+)", stdout, re.MULTILINE)
        else hashlib.sha256(payload_bytes).hexdigest(),
        "npu_time": re.search(r"^npu_time=([^\n]+)", stdout, re.MULTILINE).group(1)
        if re.search(r"^npu_time=([^\n]+)", stdout, re.MULTILINE)
        else None,
        "runtime_output": EXP132.EXP130._short_output(output, max_chars=1800),
    }


def compare_against_zero(results: list[dict]) -> list[dict]:
    zero_hash = None
    for result in results:
        if result["name"] == "zero_all":
            zero_hash = result.get("payload_hash")
    comparisons: list[dict] = []
    for result in results:
        comparisons.append(
            {
                "name": result["name"],
                "payload_differs_from_zero": zero_hash is not None
                and result.get("payload_hash") != zero_hash,
            }
        )
    return comparisons


def render_report(manifest: dict) -> str:
    lines = [
        "# Non-Zero Payload Probe",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Phase body: `0x{QKV_TARGET:x}` Q/K/V direct entry",
        f"- Records: `{QKV_RECORDS}`",
        f"- Stream chunks: `{QKV_CHUNKS}`",
        "",
        "## Results",
        "",
        "| scenario | activation | weight | status | unique headers | payload words | nonzero payload words | differs from zero | npu_time |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    diff_by_name = {
        item["name"]: item["payload_differs_from_zero"]
        for item in manifest["comparisons"]
    }
    for result in manifest["scenarios"]:
        lines.append(
            f"| `{result['name']}` | `{result['activation_pattern']}` | `{result['weight_pattern']}` | "
            f"`{result['status']}` | `{result['unique_headers']}` | "
            f"`{result.get('unique_payload_words', [])[:4]}` | "
            f"`{result['payload_nonzero']}` | `{diff_by_name.get(result['name'])}` | "
            f"`{result.get('npu_time')}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if manifest["status"] == "passed" and any(
        item["payload_differs_from_zero"]
        for item in manifest["comparisons"]
        if item["name"] != "zero_all"
    ):
        lines.append(
            "The direct Q/K/V body is not just emitting fixed headers: at least one "
            "deterministic non-zero activation/Q4NX weight stream changes the compact "
            "record payload. That makes the raw MyLM numeric path observable under the "
            "IRON harness."
        )
    else:
        lines.append(
            "The payload did not change under the current deterministic inputs, or the "
            "runtime did not complete. The next step is to check whether the direct-entry "
            "register profile points the Q4NX body at a different local input/control "
            "window than the host-fed DMA0/DMA1 buffers."
        )
    lines.extend(
        [
            "",
            "Timeout recovery:",
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
    before = topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {
            "status": "skipped_topology_not_ok",
            "topology_before": before,
            "scenarios": [],
            "comparisons": [],
        }
    build_info = build_qkv_direct_xclbin()
    chosen = scenarios()
    if args.max_scenarios > 0:
        chosen = chosen[: args.max_scenarios]
    results = [run_payload_scenario(scenario, args.runtime_timeout) for scenario in chosen]
    after = topology_status(args.xrt_timeout)
    expected_headers = [f"0x{QKV_HEADER:x}"] * QKV_RECORDS
    passed = (
        after["status"] == "ok"
        and all(result["status"] == "record_observed" for result in results)
        and all(result["headers"] == expected_headers for result in results)
        and any(
            result.get("payload_hash") != results[0].get("payload_hash")
            for result in results[1:]
        )
    )
    return {
        "status": "passed" if passed else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build_info,
        "scenarios": results,
        "comparisons": compare_against_zero(results),
        "contract": {
            "phase_body": hex(QKV_TARGET),
            "header": hex(QKV_HEADER),
            "records": QKV_RECORDS,
            "stream_chunks": QKV_CHUNKS,
            "activation_dwords": QKV_CHUNKS * EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK,
            "weight_dwords": QKV_CHUNKS * EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK,
        },
    }


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
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Non-Zero Payload Probe\n\nExperiment failed.\n\n```text\n"
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
