#!/usr/bin/env python3
"""Regenerate core-program payloads in an existing transaction MLIR.

This is the narrow toolchain step needed for raw main16 experiments: keep the
verified topology/runtime sequence, replace or edit the core ELFs under the
aiecc project directory, then rebuild the transaction core-program blockwrite
payloads from those ELFs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DEFAULT_AIE_OPT = Path(".venv/lib/python3.12/site-packages/mlir_aie/bin/aie-opt")
CONFIG_GLOBAL_RE = re.compile(r'^\s*memref\.global "private" constant @config_blockwrite_data_\d+\b')
CONFIGURE_RUNTIME_RE = re.compile(r"^\s*aie\.runtime_sequence @configure\(\)")


def _block_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def strip_transaction_payloads(txn_text: str) -> str:
    lines = txn_text.splitlines()
    kept: list[str] = []
    in_configure = False
    configure_depth = 0

    for line in lines:
        if in_configure:
            configure_depth += _block_delta(line)
            if configure_depth <= 0:
                in_configure = False
            continue

        if CONFIG_GLOBAL_RE.match(line):
            continue

        if CONFIGURE_RUNTIME_RE.match(line):
            in_configure = True
            configure_depth = _block_delta(line)
            if configure_depth <= 0:
                in_configure = False
            continue

        kept.append(line)

    return "\n".join(kept) + "\n"


def repack_transaction(txn: Path, elf_dir: Path, output_txn: Path, source_output: Path, aie_opt: Path) -> None:
    source_output.write_text(strip_transaction_payloads(txn.read_text()))
    cmd = [
        str(aie_opt),
        f"--convert-aie-to-transaction=elf-dir={elf_dir}",
        "-o",
        str(output_txn),
        str(source_output),
    ]
    subprocess.run(cmd, check=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txn", required=True, type=Path)
    parser.add_argument("--elf-dir", required=True, type=Path)
    parser.add_argument("--output-txn", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--aie-opt", default=DEFAULT_AIE_OPT, type=Path)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    repack_transaction(
        txn=args.txn,
        elf_dir=args.elf_dir,
        output_txn=args.output_txn,
        source_output=args.source_output,
        aie_opt=args.aie_opt,
    )
    print(f"repack_source={args.source_output}")
    print(f"repacked_txn={args.output_txn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
