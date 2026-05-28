#!/usr/bin/env python3
"""Replay the physical packet and circuit paths decoded from MyLM layer.xclbin."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_MYLM = Path.home() / "projects" / "MyLM"
DEFAULT_OUT = Path("/tmp/iron_exp70_mylm_path_replay")


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


def path_tool(mylm: Path) -> Path:
    return require(mylm / "tools" / "re" / "aie_physical_path_trace.py")


def dump_layer_contract(mylm: Path, out: Path, reuse: bool) -> None:
    layer_dir = out / "layer"
    if (
        reuse
        and (layer_dir / "layer_cdo.csv").exists()
        and (layer_dir / "layer_bd.csv").exists()
    ):
        return
    run(
        [
            "bash",
            require(mylm / "tools" / "re" / "dump_qwen3_layer_contract.sh"),
            "31",
            "4096",
            layer_dir,
        ],
        cwd=mylm,
    )


def write_trace(
    mylm: Path,
    cdo_csv: Path,
    bd_csv: Path,
    out_path: Path,
    args: list[str],
) -> None:
    run(
        ["python3", path_tool(mylm), cdo_csv, "--bd-csv", bd_csv] + args,
        cwd=mylm,
        stdout_path=out_path,
    )


def print_file(title: str, path: Path) -> None:
    print(title + ":")
    for line in path.read_text().splitlines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm", type=Path, default=DEFAULT_MYLM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    dump_layer_contract(args.mylm, args.out, args.reuse)
    layer_dir = args.out / "layer"
    cdo_csv = require(layer_dir / "layer_cdo.csv")
    bd_csv = require(layer_dir / "layer_bd.csv")

    packet_14_15 = args.out / "packet_14_15.txt"
    packet_0_2 = args.out / "packet_0_2.txt"
    circuit_c1r3 = args.out / "circuit_c1r3.txt"
    reverse_shape_a = args.out / "reverse_shape_a_current.txt"

    write_trace(
        args.mylm,
        cdo_csv,
        bd_csv,
        packet_14_15,
        ["--packet", "14", "--packet", "15", "--max-hops", "24"],
    )
    write_trace(
        args.mylm,
        cdo_csv,
        bd_csv,
        packet_0_2,
        ["--packet", "0", "--packet", "2", "--max-hops", "24"],
    )
    write_trace(
        args.mylm,
        cdo_csv,
        bd_csv,
        circuit_c1r3,
        ["--circuit", "c1r3:DMA_0", "--max-hops", "24"],
    )
    write_trace(
        args.mylm,
        cdo_csv,
        bd_csv,
        reverse_shape_a,
        [
            "--reverse-circuit",
            "c0r2:DMA_0",
            "--reverse-circuit",
            "c7r2:DMA_0",
            "--max-hops",
            "18",
        ],
    )

    print_file("packet14/15", packet_14_15)
    print_file("c1r3 2048-dword circuit", circuit_c1r3)
    print_file("shape-A current reverse", reverse_shape_a)
    print(f"artifacts: {args.out}")


if __name__ == "__main__":
    main()
