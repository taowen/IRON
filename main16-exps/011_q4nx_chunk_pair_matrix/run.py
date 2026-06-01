#!/usr/bin/env python3
"""Probe activation/weight chunk pairing for MyLM Q4NX direct Q/K/V."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP010_RUN = REPO_ROOT / "main16-exps/010_q4nx_activation_weight_axis_map/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_chunk_pair_matrix.json"
REPORT = EXPERIMENT_DIR / "q4nx_chunk_pair_matrix.md"
MATRIX_CHUNKS = tuple(range(4))


def load_exp010() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp010_for_pair_matrix", EXP010_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment 010 helper: {EXP010_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP010 = load_exp010()


def configure_helpers() -> None:
    EXP010.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP010.BUILD_DIR = BUILD_DIR
    EXP010.EXP008.EXPERIMENT_DIR = EXPERIMENT_DIR
    EXP010.EXP008.BUILD_DIR = BUILD_DIR


def run_pair(activation_chunk: int, weight_chunk: int, runtime_timeout: int):
    scenario = EXP010.AxisScenario(
        f"a{activation_chunk}_w{weight_chunk}",
        "pair",
        activation_chunk,
        1,
        weight_chunk,
        1,
    )
    return EXP010.run_axis_scenario(scenario, runtime_timeout)


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    before = EXP010.EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    build = EXP010.EXP008.build_qkv_direct_xclbin()
    cells = []
    for activation_chunk in MATRIX_CHUNKS:
        for weight_chunk in MATRIX_CHUNKS:
            result = run_pair(activation_chunk, weight_chunk, args.runtime_timeout)
            cells.append(
                {
                    "activation_chunk": activation_chunk,
                    "weight_chunk": weight_chunk,
                    "status": result.status,
                    "first_record_word": result.first_record_word,
                    "first_record_value": result.first_record_value,
                    "nonzero_records": list(result.nonzero_records),
                    "npu_time": result.npu_time,
                }
            )
    after = EXP010.EXP008.topology_status(args.xrt_timeout)
    complete = after["status"] == "ok" and all(cell["status"] == "record_observed" for cell in cells)
    nonzero_pairs = [
        [cell["activation_chunk"], cell["weight_chunk"]]
        for cell in cells
        if cell["first_record_value"] != 0.0
    ]
    return {
        "status": "passed" if complete else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build,
        "chunks": list(MATRIX_CHUNKS),
        "cells": cells,
        "nonzero_pairs": nonzero_pairs,
    }


def render_report(manifest: dict) -> str:
    chunks = manifest.get("chunks", [])
    by_pair = {
        (cell["activation_chunk"], cell["weight_chunk"]): cell
        for cell in manifest.get("cells", [])
    }
    lines = [
        "# Q4NX Chunk Pair Matrix",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Non-zero pairs: `{manifest.get('nonzero_pairs', [])}`",
        "",
        "## Matrix",
        "",
        "| activation \\ weight | " + " | ".join(f"`{chunk}`" for chunk in chunks) + " |",
        "| --- | " + " | ".join("---:" for _ in chunks) + " |",
    ]
    for activation_chunk in chunks:
        values = []
        for weight_chunk in chunks:
            cell = by_pair[(activation_chunk, weight_chunk)]
            values.append(f"`{cell['first_record_value']}`")
        lines.append(f"| `{activation_chunk}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The non-zero cells show which activation/weight chunk pairs actually "
            "enter the Q4NX dot for record0 under the direct phase-body harness.",
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
            "# Q4NX Chunk Pair Matrix\n\nExperiment failed.\n\n```text\n"
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

