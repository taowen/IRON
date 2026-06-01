#!/usr/bin/env python3
"""Probe q4 half boundary, nibble order, and zero-slot mapping."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_layout_boundary_nibble_zero_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_layout_boundary_nibble_zero_probe.md"


@dataclass(frozen=True)
class ProbeCase:
    name: str
    scale_indices: tuple[int, ...]
    zero_indices: tuple[int, ...]
    q4_indices: tuple[int, ...]
    q4_word: int
    zero_pair: int
    group: str


@dataclass(frozen=True)
class ProbeResult:
    name: str
    group: str
    status: str
    payload_words: tuple[str, ...]
    nonzero_lanes: tuple[int, ...]
    lane_values: tuple[float, ...]
    npu_time: str | None
    runtime_output: str


def load_exp008() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp008_for_layout_boundary", EXP008_RUN)
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


def all_scale() -> tuple[int, ...]:
    return tuple(range(EXP008.SCALE_DWORDS))


def all_q4() -> tuple[int, ...]:
    return tuple(range(EXP008.Q4_DATA_DWORDS))


def probe_cases() -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []

    boundary_indices = (
        384,
        385,
        448,
        449,
        480,
        481,
        496,
        497,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
    )
    cases.extend(
        ProbeCase(
            f"boundary_q4word_{index:04d}",
            all_scale(),
            (),
            (index,),
            0x11111111,
            0,
            "boundary",
        )
        for index in boundary_indices
    )

    for q4_index in (0, 1, 512, 513):
        for nibble_index in range(8):
            cases.append(
                ProbeCase(
                    f"nibble_q4word_{q4_index:04d}_n{nibble_index}",
                    all_scale(),
                    (),
                    (q4_index,),
                    1 << (4 * nibble_index),
                    0,
                    "nibble",
                )
            )

    zero_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 96, 127)
    cases.extend(
        ProbeCase(
            f"zero_{index:03d}",
            all_scale(),
            (index,),
            all_q4(),
            0x11111111,
            EXP008.bf16_pair(0x3C80),
            "zero",
        )
        for index in zero_indices
    )

    return tuple(cases)


def case_json(case: ProbeCase) -> str:
    return json.dumps(
        {
            "scale_indices": list(case.scale_indices),
            "zero_indices": list(case.zero_indices),
            "q4_indices": list(case.q4_indices),
            "activation_pair": EXP008.bf16_pair(0x3F80),
            "scale_pair": EXP008.bf16_pair(0x3C80),
            "zero_pair": case.zero_pair,
            "q4_word": case.q4_word,
        },
        sort_keys=True,
    )


def run_case(case: ProbeCase, runtime_timeout: int) -> ProbeResult:
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
    activation[:ACT_DWORDS_PER_CHUNK] = np.int32(int(SCENARIO['activation_pair']))
    return activation

def make_weights():
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    chunks = weights.reshape((-1, Q4_CHUNK_DWORDS))
    chunk = chunks[0]
    scale_pair = np.int32(int(SCENARIO['scale_pair']))
    for index in SCENARIO['scale_indices']:
        chunk[int(index)] = scale_pair
    zero_pair = np.int32(int(SCENARIO['zero_pair']))
    zero_base = SCALE_DWORDS
    for index in SCENARIO['zero_indices']:
        chunk[zero_base + int(index)] = zero_pair
    q4_word = np.int32(int(SCENARIO['q4_word']))
    q4_base = SCALE_DWORDS + ZERO_DWORDS
    for index in SCENARIO['q4_indices']:
        chunk[q4_base + int(index)] = q4_word
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
        return ProbeResult(
            case.name,
            case.group,
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
    return ProbeResult(
        case.name,
        case.group,
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
    cases = probe_cases()
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
            "activation_chunk": 0,
            "weight_chunk": 0,
            "activation_bf16": "0x3f80",
            "scale_bf16": "0x3c80",
            "zero_bf16": "0x3c80",
        },
        "results": [
            {
                "name": result.name,
                "group": result.group,
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
        "# Q4NX Layout Boundary, Nibble, and Zero Probe",
        "",
        f"- Status: `{manifest['status']}`",
        "- Active pair: activation chunk `0`, weight chunk `0`",
        "",
        "## Results",
        "",
        "| case | group | status | nonzero lanes | lane values prefix | npu_time |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in manifest.get("results", []):
        lines.append(
            f"| `{result['name']}` | `{result['group']}` | `{result['status']}` | "
            f"`{result['nonzero_lanes']}` | `{short_values(result['lane_values'])}` | "
            f"`{result['npu_time']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Boundary cases locate the q4 low/high output-half split. Nibble cases "
            "show which payload lanes each 4-bit field affects. Zero cases test "
            "whether the middle 128 dwords participate in this synthetic MyLM "
            "Q4NX contract.",
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
            "# Q4NX Layout Boundary, Nibble, and Zero Probe\n\nExperiment failed.\n\n```text\n"
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
