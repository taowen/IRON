#!/usr/bin/env python3
"""Package an external-ELF AIE design without recompiling core programs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from externalize_core_programs import externalize_core_programs
from inspect_core_program_txn import DEFAULT_MYLM_PROGRAM_IMAGES, inspect


DEFAULT_AIECC = Path(".venv/lib/python3.12/site-packages/mlir_aie/bin/aiecc")
DEFAULT_LLVM_SIZE = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-size")
DEFAULT_PEANO = Path(".venv/lib/python3.12/site-packages/llvm-aie")
DEFAULT_XRT_BIN = Path("/var/opt/xilinx/xrt/bin")
MAIN16_ROLE_OBJECT = "main_projection_q4nx_fast.o"
MAIN16_MIN_TEXT_BYTES = 4096
MAIN16_TILES = tuple((col, row) for col in range(2, 6) for row in range(2, 6))
MAIN16_SUMMARY_RE = re.compile(
    rf"role={re.escape(MAIN16_ROLE_OBJECT)} cores=(?P<cores>\d+) "
    r"text_bytes=(?P<text_bytes>[0-9,]+) matching_config_blocks=(?P<blocks>\d+)"
)


@dataclass(frozen=True)
class PackageOutputs:
    externalized_mlir: Path
    project_dir: Path
    transaction_mlir: Path
    npu_insts: Path
    xclbin: Path
    inspection: Path


def _text_size(llvm_size: Path, elf: Path) -> int:
    result = subprocess.run(
        (str(llvm_size), "-A", str(elf)),
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text size in {elf}")


def _main16_elf(project_dir: Path, col: int, row: int) -> Path:
    return project_dir / f"main_core_{col}_{row}.elf"


def _main16_text_sizes(project_dir: Path, llvm_size: Path) -> dict[tuple[int, int], int]:
    sizes: dict[tuple[int, int], int] = {}
    for col, row in MAIN16_TILES:
        elf = _main16_elf(project_dir, col, row)
        if not elf.exists():
            raise FileNotFoundError(f"missing main16 ELF: {elf}")
        size = _text_size(llvm_size, elf)
        if size < MAIN16_MIN_TEXT_BYTES:
            raise ValueError(f"main16 ELF is too small for a real projection program: {elf} .text={size}")
        sizes[(col, row)] = size
    return sizes


def _copy_main16_replacements(main16_elf_dir: Path, project_dir: Path) -> None:
    for col, row in MAIN16_TILES:
        name = f"main_core_{col}_{row}.elf"
        src = main16_elf_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing replacement main16 ELF: {src}")
        shutil.copy2(src, project_dir / name)


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory exists, pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _run_aiecc(outputs: PackageOutputs, aiecc: Path, peano: Path, xrt_bin: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = f"{xrt_bin}:{env.get('PATH', '')}"
    cmd = [
        str(aiecc),
        "-j1",
        "--no-compile",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--alloc-scheme=basic-sequential",
        "--peano",
        str(peano),
        "--aie-generate-xclbin",
        f"--xclbin-name={outputs.xclbin}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-txn",
        f"--txn-name={outputs.transaction_mlir}",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={outputs.npu_insts}",
        f"--tmpdir={outputs.project_dir}",
        str(outputs.externalized_mlir),
    ]
    subprocess.run(cmd, check=True, env=env)


def _validate_outputs(outputs: PackageOutputs) -> None:
    for path in (outputs.transaction_mlir, outputs.npu_insts, outputs.xclbin):
        if not path.exists():
            raise FileNotFoundError(f"expected package output missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"package output is empty: {path}")


def _validate_main16_payloads(inspection_text: str) -> None:
    match = MAIN16_SUMMARY_RE.search(inspection_text)
    if match is None:
        raise ValueError(f"missing {MAIN16_ROLE_OBJECT} role summary in transaction inspection")
    cores = int(match.group("cores"))
    blocks = int(match.group("blocks"))
    if cores != len(MAIN16_TILES):
        raise ValueError(f"main16 role core count mismatch: expected={len(MAIN16_TILES)} got={cores}")
    if blocks != len(MAIN16_TILES):
        raise ValueError(f"main16 payload count mismatch: expected={len(MAIN16_TILES)} got={blocks}")


def package_externalized_design(
    input_mlir: Path,
    donor_transaction_mlir: Path,
    donor_project_dir: Path,
    output_dir: Path,
    force: bool,
    main16_elf_dir: Path | None,
    aiecc: Path,
    peano: Path,
    llvm_size: Path,
    xrt_bin: Path,
    mylm_program_images: Path,
) -> PackageOutputs:
    _prepare_output_dir(output_dir, force)
    outputs = PackageOutputs(
        externalized_mlir=output_dir / "design.externalized.mlir",
        project_dir=output_dir / "prj",
        transaction_mlir=output_dir / "design.txn.mlir",
        npu_insts=output_dir / "design.bin",
        xclbin=output_dir / "design.xclbin",
        inspection=output_dir / "core_program_inspection.txt",
    )
    outputs.externalized_mlir.write_text(
        externalize_core_programs(
            input_mlir=input_mlir.read_text(),
            transaction_mlir=donor_transaction_mlir.read_text(),
            elf_dir=outputs.project_dir,
        )
    )
    shutil.copytree(donor_project_dir, outputs.project_dir)
    if main16_elf_dir is not None:
        _copy_main16_replacements(main16_elf_dir, outputs.project_dir)

    before_sizes = _main16_text_sizes(outputs.project_dir, llvm_size)
    _run_aiecc(outputs, aiecc, peano, xrt_bin)
    _validate_outputs(outputs)
    after_sizes = _main16_text_sizes(outputs.project_dir, llvm_size)
    if after_sizes != before_sizes:
        raise ValueError(f"main16 ELF text sizes changed during --no-compile packaging: {before_sizes} -> {after_sizes}")

    inspection_text = inspect(outputs.transaction_mlir, llvm_size, mylm_program_images)
    _validate_main16_payloads(inspection_text)
    outputs.inspection.write_text(inspection_text)
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mlir", required=True, type=Path)
    parser.add_argument("--donor-transaction-mlir", required=True, type=Path)
    parser.add_argument("--donor-project-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--main16-elf-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aiecc", type=Path, default=DEFAULT_AIECC)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--llvm-size", type=Path, default=DEFAULT_LLVM_SIZE)
    parser.add_argument("--xrt-bin", type=Path, default=DEFAULT_XRT_BIN)
    parser.add_argument("--mylm-program-images", type=Path, default=DEFAULT_MYLM_PROGRAM_IMAGES)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    outputs = package_externalized_design(
        input_mlir=args.input_mlir,
        donor_transaction_mlir=args.donor_transaction_mlir,
        donor_project_dir=args.donor_project_dir,
        output_dir=args.output_dir,
        force=args.force,
        main16_elf_dir=args.main16_elf_dir,
        aiecc=args.aiecc,
        peano=args.peano,
        llvm_size=args.llvm_size,
        xrt_bin=args.xrt_bin,
        mylm_program_images=args.mylm_program_images,
    )
    print(f"externalized_mlir={outputs.externalized_mlir}")
    print(f"project_dir={outputs.project_dir}")
    print(f"transaction_mlir={outputs.transaction_mlir}")
    print(f"npu_insts={outputs.npu_insts}")
    print(f"xclbin={outputs.xclbin}")
    print(f"inspection={outputs.inspection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
