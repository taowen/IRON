#!/usr/bin/env python3
"""Map per-chunk contribution for one MyLM Q/K/V record."""

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
EXP008_RUN = REPO_ROOT / "main16-exps/008_q4nx_payload_formula_probe/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_chunk_contribution_map.json"
REPORT = EXPERIMENT_DIR / "q4nx_chunk_contribution_map.md"


@dataclass(frozen=True)
class ChunkRow:
    chunk: int
    status: str
    first_record_word: str
    first_record_value: float
    nonzero_records: tuple[int, ...]
    npu_time: str | None


def load_exp008() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp008_for_chunk_map", EXP008_RUN)
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


def nonzero_records(words: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(index for index, word in enumerate(words) if int(word, 16) != 0)


def run_chunk(chunk: int, runtime_timeout: int) -> ChunkRow:
    scenario = EXP008.Scenario(
        f"chunk_{chunk:02d}",
        0x3F80,
        0x3C80,
        1,
        chunk,
        1,
    )
    result = EXP008.run_scenario(scenario, runtime_timeout)
    first = result.actual_record_words[0] if result.actual_record_words else "0x0"
    return ChunkRow(
        chunk=chunk,
        status=result.status,
        first_record_word=first,
        first_record_value=word_to_float(first),
        nonzero_records=nonzero_records(result.actual_record_words),
        npu_time=result.npu_time,
    )


def run_all_chunks(runtime_timeout: int):
    scenario = EXP008.Scenario("all_record0_chunks", 0x3F80, 0x3C80, 1, 0, EXP008.CHUNKS_PER_RECORD)
    return EXP008.run_scenario(scenario, runtime_timeout)


def build_manifest(args: argparse.Namespace) -> dict:
    configure_helpers()
    before = EXP008.topology_status(args.xrt_timeout)
    if before["status"] != "ok":
        return {"status": "skipped_topology_not_ok", "topology_before": before}
    build = EXP008.build_qkv_direct_xclbin()
    rows = [run_chunk(chunk, args.runtime_timeout) for chunk in range(EXP008.CHUNKS_PER_RECORD)]
    all_result = run_all_chunks(args.runtime_timeout)
    after = EXP008.topology_status(args.xrt_timeout)
    observed_sum = sum(row.first_record_value for row in rows)
    all_word = all_result.actual_record_words[0] if all_result.actual_record_words else "0x0"
    all_value = word_to_float(all_word)
    completed = all(row.status == "record_observed" for row in rows) and all_result.status == "record_observed"
    return {
        "status": "passed" if completed and after["status"] == "ok" else "failed",
        "topology_before": before,
        "topology_after": after,
        "build": build,
        "setup": {
            "activation_bf16": "0x3f80",
            "scale_bf16": "0x3c80",
            "nibble": 1,
            "record": 0,
        },
        "rows": [
            {
                "chunk": row.chunk,
                "status": row.status,
                "first_record_word": row.first_record_word,
                "first_record_value": row.first_record_value,
                "nonzero_records": list(row.nonzero_records),
                "npu_time": row.npu_time,
            }
            for row in rows
        ],
        "aggregate": {
            "sum_single_chunk_values": observed_sum,
            "all_record0_word": all_word,
            "all_record0_value": all_value,
            "sum_matches_all_record0": observed_sum == all_value,
        },
    }


def render_report(manifest: dict) -> str:
    lines = [
        "# Q4NX Chunk Contribution Map",
        "",
        f"- Status: `{manifest['status']}`",
        "- Setup: activation bf16 `1.0`, scale bf16 `1/64`, q4 nibble `1`",
        "",
        "## Per-Chunk Contribution",
        "",
        "| chunk | status | first record word | first record value | nonzero records | npu_time |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in manifest.get("rows", []):
        lines.append(
            f"| `{row['chunk']}` | `{row['status']}` | `{row['first_record_word']}` | "
            f"`{row['first_record_value']}` | `{row['nonzero_records']}` | `{row['npu_time']}` |"
        )
    aggregate = manifest.get("aggregate", {})
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Sum of single-chunk values: `{aggregate.get('sum_single_chunk_values')}`",
            f"- All record0 chunks value: `{aggregate.get('all_record0_value')}`",
            f"- Sum matches all-record0: `{aggregate.get('sum_matches_all_record0')}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if manifest["status"] == "passed":
        lines.append(
            "This map explains why experiment 008's one-active-chunk formula failed. "
            "The scalar full-record formula is not made of 16 equal 128-element chunk "
            "contributions. Use this map as the next clue for decoding the Q4NX "
            "microkernel's intra-record chunk/lane schedule."
        )
    else:
        lines.append("The contribution map did not complete cleanly.")
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
            "# Q4NX Chunk Contribution Map\n\nExperiment failed.\n\n```text\n"
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

