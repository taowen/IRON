#!/usr/bin/env python3
"""Audit MyLM Shape-B hidden payload size and accumulator evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_MYLM = Path.home() / "projects" / "MyLM"
DEFAULT_OUT = Path("/tmp/iron_exp81_mylm_shape_b_hidden_payload")
EXP70_OUT = Path("/tmp/iron_exp70_mylm_path_replay")
EDGE_TILES = "c0r2,c0r3,c0r4,c0r5,c7r2,c7r3,c7r4,c7r5"


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
    bd_csv = require(layer_dir / "layer_bd.csv")
    program_dir = require(layer_dir / "programs")
    disasm_dir = program_dir / "disasm"

    run(
        [
            "python3",
            mylm_tool(args.mylm, "aie_program_disassemble.py"),
            "--program-dir",
            program_dir,
            "--tiles",
            EDGE_TILES,
        ],
        cwd=args.mylm,
    )

    summary = args.out / "shape_b_hidden_payload_inference.txt"
    run(
        [
            "python3",
            mylm_tool(args.mylm, "aie_shape_b_hidden_payload_infer.py"),
            "--bd-csv",
            bd_csv,
            "--disasm-dir",
            disasm_dir,
        ],
        cwd=args.mylm,
        stdout_path=summary,
    )

    print(summary.read_text())
    print(f"artifacts: {args.out}")


if __name__ == "__main__":
    main()
