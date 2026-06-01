#!/usr/bin/env python3
"""Probe q4 nibble behavior across active stream chunk pairs."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_stream_parity_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_stream_parity_probe.md"
RECORD_CHUNKS = 16


@dataclass(frozen=True)
class ParityCase:
    name: str
    active_chunk: int
    q4_index: int
    q4_word: int
    group: str


@dataclass(frozen=True)
class ParityResult:
    name: str
    group: str
    active_chunk: int
    status: str
    payload_words: tuple[str, ...]
    nonzero_lanes: tuple[int, ...]
    lane_values: tuple[float, ...]
    npu_time: str | None
    runtime_output: str


def load_exp008() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp008_for_stream_parity", EXP008_RUN)
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


def record0_payload(words: tuple[int, ...]) -> tuple[str, ...]:
    if len(words) < EXP008.RECORD_DWORDS:
        return ()
    return tuple(hex(word & 0xFFFFFFFF) for word in words[1:EXP008.RECORD_DWORDS])


def parity_cases() -> tuple[ParityCase, ...]:
    cases: list[ParityCase] = []
    for active_chunk in range(RECORD_CHUNKS):
        cases.append(
            ParityCase(
                f"chunk_{active_chunk:02d}_q4word0000_allnibbles",
                active_chunk,
                0,
                0x11111111,
                "chunk_baseline",
            )
        )
    for active_chunk in range(0, RECORD_CHUNKS, 2):
        for nibble_index in range(8):
            cases.append(
                ParityCase(
                    f"chunk_{active_chunk:02d}_q4word0000_n{nibble_index}",
                    active_chunk,
                    0,
                    1 << (4 * nibble_index),
                    "even_chunk_nibble",
                )
            )
    for active_chunk in (1, 3):
        for nibble_index in (0, 1, 3, 5, 7):
            cases.append(
                ParityCase(
                    f"chunk_{active_chunk:02d}_q4word0000_n{nibble_index}",
                    active_chunk,
                    0,
                    1 << (4 * nibble_index),
                    "odd_chunk_nibble",
                )
            )
    return tuple(cases)


def case_json(case: ParityCase) -> str:
    return json.dumps(
        {
            "active_chunk": case.active_chunk,
            "q4_index": case.q4_index,
            "activation_pair": EXP008.bf16_pair(0x3F80),
            "scale_pair": EXP008.bf16_pair(0x3C80),
            "q4_word": case.q4_word,
        },
        sort_keys=True,
    )


def run_case(case: ParityCase, runtime_timeout: int) -> ParityResult:
    env = os.environ.copy()
    env["PATH"] = f"{EXP008.EXP005.EXP132.EXP130.XRT_BIN}:{env.get('PATH', '')}"
    out_dwords = EXP008.EXP005.EXP132.EXP130.output_dwords(EXP008.RECORDS)
    act_dwords = EXP008.CHUNKS * EXP008.EXP005.EXP132.EXP130.ACTIVATION_DWORDS_PER_CHUNK
    wt_dwords = EXP008.CHUNKS * EXP008.EXP005.EXP132.EXP130.WEIGHT_DWORDS_PER_CHUNK
    payload = case_json(case)
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
    start = int(SCENARIO['active_chunk']) * ACT_DWORDS_PER_CHUNK
    activation[start:start + ACT_DWORDS_PER_CHUNK] = np.int32(int(SCENARIO['activation_pair']))
    return activation

def make_weights():
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    chunks = weights.reshape((-1, Q4_CHUNK_DWORDS))
    chunk = chunks[int(SCENARIO['active_chunk'])]
    chunk[:SCALE_DWORDS] = np.int32(int(SCENARIO['scale_pair']))
    q4_base = SCALE_DWORDS + ZERO_DWORDS
    chunk[q4_base + int(SCENARIO['q4_index'])] = np.int32(int(SCENARIO['q4_word']))
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
        return ParityResult(
            case.name,
            case.group,
            case.active_chunk,
            "timeout_waiting_for_record",
            (),
            (),
            (),
            None,
            EXP008.EXP005.EXP132.EXP130._short_output(output, max_chars=1200),
        )
    words = parse_record_words(result.stdout)
    payload_words = record0_payload(words)
    status = "record_observed" if result.returncode == 0 and len(words) == out_dwords else "runtime_failed"
    output = result.stdout + result.stderr
    if "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    npu_match = re.search(r"^npu_time=([^\n]+)", result.stdout, re.MULTILINE)
    lane_values = tuple(word_to_float(word) for word in payload_words)
    nonzero = tuple(index for index, value in enumerate(lane_values) if value != 0.0)
    return ParityResult(
        case.name,
        case.group,
        case.active_chunk,
        status,
        payload_words,
        nonzero,
        lane_values,
        npu_match.group(1) if npu_match is not None else None,
        EXP008.EXP005.EXP132.EXP130._short_output(output, max_chars=1200),
    )


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    before = EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    build = EXP008.build_qkv_direct_xclbin()
    cases = parity_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    results = [run_case(case, args.runtime_timeout) for case in cases]
    after = EXP008.topology_status(args.xrt_timeout)
    complete = after["status"] == "ok" and all(result.status == "record_observed" for result in results)
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build,
        "setup": {
            "q4_index": 0,
            "activation_bf16": "0x3f80",
            "scale_bf16": "0x3c80",
        },
        "results": [
            {
                "name": result.name,
                "group": result.group,
                "active_chunk": result.active_chunk,
                "status": result.status,
                "payload_words": list(result.payload_words),
                "nonzero_lanes": list(result.nonzero_lanes),
                "lane_values": list(result.lane_values),
                "npu_time": result.npu_time,
                "runtime_output": result.runtime_output,
            }
            for result in results
        ],
    }


def short_values(values: list[float]) -> str:
    prefix = ", ".join(f"{value:g}" for value in values[:8])
    suffix = "..." if len(values) > 8 else ""
    return "[" + prefix + suffix + "]"


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Stream Parity Probe",
        "",
        f"- Status: `{manifest['status']}`",
        "",
        "## Results",
        "",
        "| case | group | chunk | status | nonzero lanes | lane values prefix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in manifest.get("results", []):
        lines.append(
            f"| `{result['name']}` | `{result['group']}` | `{result['active_chunk']}` | "
            f"`{result['status']}` | `{result['nonzero_lanes']}` | "
            f"`{short_values(result['lane_values'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment checks whether q4 nibble positions that are inactive "
            "for chunk0 become active on other stream chunks. A production Q4NX "
            "replacement needs this parity table before it can pack/load q4 data "
            "correctly.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-timeout", type=int, default=12)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Stream Parity Probe\n\nExperiment failed.\n\n```text\n"
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
