#!/usr/bin/env python3
"""Try packaging a qwen3-layer design with replacement main16 raw programs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from inspect_core_program_txn import DEFAULT_MYLM_PROGRAM_IMAGES, inspect
from package_externalized_design import DEFAULT_AIECC, DEFAULT_LLVM_SIZE, DEFAULT_PEANO, DEFAULT_XRT_BIN
from wrap_raw_aie_program import wrap_raw_program


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DONOR_PROJECT = REPO_ROOT / "design.mlir.prj"
DEFAULT_RAW_MAIN16 = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_program.bin"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "qwen3-layer/build/main16-raw-replacement-try"
MAIN16_TILES = tuple((col, row) for col in range(2, 6) for row in range(2, 6))
CORE_START_RE = re.compile(r"^(?P<indent>\s*)%core_(?P<col>\d+)_(?P<row>\d+) = aie\.core\(%tile_\d+_\d+\) \{")
LINK_FILES_RE = re.compile(r"link_files = \[(?P<link_files>[^\]]*)\]")


@dataclass(frozen=True)
class ReplacementManifest:
    status: str
    donor_project: str
    raw_main16_program: str
    output_dir: str
    externalized_mlir: str
    transaction_mlir: str
    npu_insts: str
    xclbin: str
    inspection: str
    replaced_main16_elfs: int
    raw_program_bytes: int
    errors: tuple[str, ...]


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def text_size(llvm_size: Path, elf: Path) -> int:
    result = subprocess.run((str(llvm_size), "-A", str(elf)), check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text size in {elf}")


def block_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def link_files_attr(line: str) -> str:
    match = LINK_FILES_RE.search(line)
    if match is None:
        return "[]"
    return "[" + match.group("link_files") + "]"


def prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory exists, pass --force: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_project(donor_project: Path, project_dir: Path) -> None:
    if not donor_project.exists():
        raise FileNotFoundError(donor_project)
    if not (donor_project / "input_with_addresses.mlir").exists():
        raise FileNotFoundError(donor_project / "input_with_addresses.mlir")
    shutil.copytree(donor_project, project_dir)


def replace_main16_elfs(project_dir: Path, raw_program: Path) -> int:
    raw = raw_program.read_bytes()
    wrapped = wrap_raw_program(raw)
    count = 0
    for col, row in MAIN16_TILES:
        elf = project_dir / f"main_core_{col}_{row}.elf"
        if not elf.exists():
            raise FileNotFoundError(f"missing donor main16 ELF: {elf}")
        elf.write_bytes(wrapped)
        count += 1
    return count


def externalized_mlir(input_mlir: str, project_dir: Path) -> str:
    lines = input_mlir.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = CORE_START_RE.match(line)
        if match is None:
            out.append(line)
            index += 1
            continue

        indent = match.group("indent")
        col = int(match.group("col"))
        row = int(match.group("row"))
        depth = block_delta(line)
        index += 1
        closing_line = ""
        while index < len(lines) and depth > 0:
            closing_line = lines[index]
            depth += block_delta(closing_line)
            index += 1
        if not closing_line:
            raise ValueError(f"unterminated core block c{col}r{row}")
        elf = project_dir / f"main_core_{col}_{row}.elf"
        if not elf.exists():
            raise FileNotFoundError(f"missing core ELF: {elf}")
        out.append(line)
        out.append(f"{indent}  aie.end")
        out.append(f'{indent}}} {{elf_file = "{elf}", link_files = {link_files_attr(closing_line)}}}')
    return "\n".join(out) + "\n"


def run_aiecc(
    externalized_mlir_path: Path,
    project_dir: Path,
    transaction_mlir: Path,
    npu_insts: Path,
    xclbin: Path,
    aiecc: Path,
    peano: Path,
    xrt_bin: Path,
) -> None:
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
        f"--xclbin-name={xclbin}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-txn",
        f"--txn-name={transaction_mlir}",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={npu_insts}",
        f"--tmpdir={project_dir}",
        str(externalized_mlir_path),
    ]
    run_command(cmd, env)


def validate_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"empty output: {path}")


def try_replace_main16(
    donor_project: Path,
    raw_main16_program: Path,
    output_dir: Path,
    force: bool,
    aiecc: Path,
    peano: Path,
    llvm_size: Path,
    xrt_bin: Path,
    mylm_program_images: Path,
) -> ReplacementManifest:
    prepare_output(output_dir, force)
    project_dir = output_dir / "prj"
    externalized_mlir_path = output_dir / "design.externalized.mlir"
    transaction_mlir = output_dir / "design.txn.mlir"
    npu_insts = output_dir / "design.bin"
    xclbin = output_dir / "design.xclbin"
    inspection = output_dir / "core_program_inspection.txt"
    errors: list[str] = []

    copy_project(donor_project, project_dir)
    raw_bytes = raw_main16_program.read_bytes()
    replaced = replace_main16_elfs(project_dir, raw_main16_program)
    sizes = tuple(sorted({text_size(llvm_size, project_dir / f"main_core_{col}_{row}.elf") for col, row in MAIN16_TILES}))
    if sizes != (len(raw_bytes),):
        errors.append(f"wrapped main16 text size mismatch: expected={len(raw_bytes)} got={sizes}")

    source_mlir = project_dir / "input_with_addresses.mlir"
    externalized_mlir_path.write_text(externalized_mlir(source_mlir.read_text(), project_dir))
    run_aiecc(externalized_mlir_path, project_dir, transaction_mlir, npu_insts, xclbin, aiecc, peano, xrt_bin)
    validate_outputs((transaction_mlir, npu_insts, xclbin))
    inspection_text = inspect(transaction_mlir, llvm_size, mylm_program_images)
    inspection.write_text(inspection_text)
    if "role=main_projection_q4nx_fast.o cores=16 text_bytes=14868 matching_config_blocks=16" not in inspection_text:
        errors.append("transaction does not contain 16 replacement main16 payloads of 14868 bytes")

    return ReplacementManifest(
        "passed" if not errors else "failed",
        str(donor_project),
        str(raw_main16_program),
        str(output_dir),
        str(externalized_mlir_path),
        str(transaction_mlir),
        str(npu_insts),
        str(xclbin),
        str(inspection),
        replaced,
        len(raw_bytes),
        tuple(errors),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-project", type=Path, default=DEFAULT_DONOR_PROJECT)
    parser.add_argument("--raw-main16-program", type=Path, default=DEFAULT_RAW_MAIN16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aiecc", type=Path, default=DEFAULT_AIECC)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--llvm-size", type=Path, default=DEFAULT_LLVM_SIZE)
    parser.add_argument("--xrt-bin", type=Path, default=DEFAULT_XRT_BIN)
    parser.add_argument("--mylm-program-images", type=Path, default=DEFAULT_MYLM_PROGRAM_IMAGES)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = try_replace_main16(
        donor_project=args.donor_project,
        raw_main16_program=args.raw_main16_program,
        output_dir=args.output_dir,
        force=args.force,
        aiecc=args.aiecc,
        peano=args.peano,
        llvm_size=args.llvm_size,
        xrt_bin=args.xrt_bin,
        mylm_program_images=args.mylm_program_images,
    )
    manifest_path = Path(manifest.output_dir) / "replacement_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    print(f"status={manifest.status}")
    print(f"manifest={manifest_path}")
    print(f"externalized_mlir={manifest.externalized_mlir}")
    print(f"transaction_mlir={manifest.transaction_mlir}")
    print(f"npu_insts={manifest.npu_insts}")
    print(f"xclbin={manifest.xclbin}")
    print(f"inspection={manifest.inspection}")
    if manifest.errors:
        print("errors:")
        for error in manifest.errors:
            print(f"  - {error}")
    return 0 if manifest.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
