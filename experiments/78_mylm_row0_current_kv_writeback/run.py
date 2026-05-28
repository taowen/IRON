#!/usr/bin/env python3
"""Audit packet14/15 row0 current K/V cache-writeback evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_MYLM = Path.home() / "projects" / "MyLM"
DEFAULT_OUT = Path("/tmp/iron_exp78_mylm_row0_current_kv_writeback")
EXP70_OUT = Path("/tmp/iron_exp70_mylm_path_replay")


def run(
    cmd: list[str | Path],
    *,
    cwd: Path,
    stdout_path: Path | None = None,
) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    if stdout_path is None:
        subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)
        return
    with stdout_path.open("w") as f:
        subprocess.run([str(part) for part in cmd], cwd=cwd, stdout=f, check=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def mylm_tool(mylm: Path, name: str) -> Path:
    return require(mylm / "tools" / "re" / name)


def prepare_exp70(reuse: bool) -> None:
    args = ["--out", EXP70_OUT]
    if reuse:
        args.append("--reuse")
    run(
        [".venv/bin/python", "experiments/70_mylm_stream_switch_physical_path_replay/run.py"] + args,
        cwd=Path.cwd(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm", type=Path, default=DEFAULT_MYLM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    prepare_exp70(args.reuse)
    args.out.mkdir(parents=True, exist_ok=True)

    layer_dir = EXP70_OUT / "layer"
    cdo_csv = require(layer_dir / "layer_cdo.csv")
    bd_csv = require(layer_dir / "layer_bd.csv")
    trace_csv = require(layer_dir / "qwen3_layer_L31.trace.csv")

    summary = args.out / "row0_current_kv_writeback_summary.txt"
    run(
        [
            "python3",
            mylm_tool(args.mylm, "aie_row0_current_kv_writeback_summary.py"),
            cdo_csv,
            "--bd-csv",
            bd_csv,
            "--trace-csv",
            trace_csv,
        ],
        cwd=args.mylm,
        stdout_path=summary,
    )

    print(summary.read_text())
    print(f"artifacts: {args.out}")


if __name__ == "__main__":
    main()
