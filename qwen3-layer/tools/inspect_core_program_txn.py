#!/usr/bin/env python3
"""Inspect how aiecc embeds AIE core programs in transaction MLIR."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TXN = Path("/tmp/iron_txn_probe_fresh/design.txn.mlir")
DEFAULT_LLVM_SIZE = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-size")
DEFAULT_MYLM_PROGRAM_IMAGES = Path("/tmp/mylm_solidify_L31/programs/program_images.tsv")
MAIN16_ROLE_OBJECT = "main_projection_q4nx_fast.o"

CONFIG_GLOBAL_RE = re.compile(
    r'memref\.global "private" constant @(?P<name>config_blockwrite_data_\d+) : '
    r"memref<(?P<dwords>\d+)xi32>"
)
CORE_ELF_RE = re.compile(
    r"%core_(?P<col>\d+)_(?P<row>\d+) = aie\.core\(%tile_\d+_\d+\) \{\s*"
    r"aie\.end\s*\}\s*\{elf_file = \"(?P<elf>[^\"]+)\", link_files = \[(?P<link_files>[^\]]*)\]\}",
    re.MULTILINE,
)
LINK_FILE_RE = re.compile(r'"([^"]+)"')


@dataclass(frozen=True)
class ConfigGlobal:
    name: str
    dwords: int

    @property
    def bytes(self) -> int:
        return self.dwords * 4


@dataclass(frozen=True)
class CoreProgram:
    col: int
    row: int
    elf: Path
    link_files: tuple[Path, ...]
    text_bytes: int

    @property
    def tile(self) -> str:
        return f"c{self.col}r{self.row}"

    @property
    def role(self) -> str:
        if not self.link_files:
            return "<no-link-files>"
        return ",".join(path.name for path in self.link_files)


@dataclass(frozen=True)
class RoleSummary:
    role: str
    core_count: int
    text_bytes: tuple[int, ...]
    matching_config_blocks: int
    tiles: tuple[str, ...]


@dataclass(frozen=True)
class MyLMProgramSummary:
    main16_count: int
    main16_bytes: tuple[int, ...]


def _parse_config_globals(txn_text: str) -> tuple[ConfigGlobal, ...]:
    return tuple(
        ConfigGlobal(name=match.group("name"), dwords=int(match.group("dwords")))
        for match in CONFIG_GLOBAL_RE.finditer(txn_text)
    )


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


def _parse_cores(txn_text: str, llvm_size: Path) -> tuple[CoreProgram, ...]:
    cores: list[CoreProgram] = []
    for match in CORE_ELF_RE.finditer(txn_text):
        elf = Path(match.group("elf"))
        link_files = tuple(Path(item) for item in LINK_FILE_RE.findall(match.group("link_files")))
        cores.append(
            CoreProgram(
                col=int(match.group("col")),
                row=int(match.group("row")),
                elf=elf,
                link_files=link_files,
                text_bytes=_text_size(llvm_size, elf),
            )
        )
    return tuple(sorted(cores, key=lambda core: (core.col, core.row)))


def _role_summaries(
    cores: tuple[CoreProgram, ...],
    config_globals: tuple[ConfigGlobal, ...],
) -> tuple[RoleSummary, ...]:
    block_sizes = Counter(item.bytes for item in config_globals)
    by_role: dict[str, list[CoreProgram]] = {}
    for core in cores:
        by_role.setdefault(core.role, []).append(core)

    summaries: list[RoleSummary] = []
    for role, role_cores in sorted(by_role.items()):
        text_bytes = tuple(sorted({core.text_bytes for core in role_cores}))
        matching_blocks = sum(block_sizes[size] for size in text_bytes)
        summaries.append(
            RoleSummary(
                role=role,
                core_count=len(role_cores),
                text_bytes=text_bytes,
                matching_config_blocks=matching_blocks,
                tiles=tuple(core.tile for core in role_cores),
            )
        )
    return tuple(summaries)


def _load_mylm_summary(path: Path) -> MyLMProgramSummary:
    if not path.exists():
        return MyLMProgramSummary(main16_count=0, main16_bytes=())
    byte_sizes: list[int] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            col = int(row["col"])
            tile_row = int(row["row"])
            if 2 <= col <= 5 and 2 <= tile_row <= 5:
                byte_sizes.append(int(row["bytes"]))
    return MyLMProgramSummary(
        main16_count=len(byte_sizes),
        main16_bytes=tuple(sorted(set(byte_sizes))),
    )


def _format_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def inspect(txn: Path, llvm_size: Path, mylm_program_images: Path) -> str:
    txn_text = txn.read_text()
    config_globals = _parse_config_globals(txn_text)
    cores = _parse_cores(txn_text, llvm_size)
    summaries = _role_summaries(cores, config_globals)
    mylm = _load_mylm_summary(mylm_program_images)

    block_size_counts = Counter(item.bytes for item in config_globals)
    lines = [
        "core_program_txn_inspection:",
        f"  txn={txn}",
        f"  cores={len(cores)}",
        f"  config_blockwrite_globals={len(config_globals)}",
        "  config_blockwrite_payload_sizes:",
    ]
    for size, count in sorted(block_size_counts.items()):
        lines.append(f"    bytes={size} count={count}")

    lines.append("  role_summaries:")
    for summary in summaries:
        lines.append(
            f"    role={summary.role} cores={summary.core_count} "
            f"text_bytes={_format_ints(summary.text_bytes)} "
            f"matching_config_blocks={summary.matching_config_blocks} "
            f"tiles={','.join(summary.tiles)}"
        )

    if mylm.main16_count:
        lines.extend(
            [
                "  mylm_reference:",
                f"    main16_images={mylm.main16_count}",
                f"    main16_bytes={_format_ints(mylm.main16_bytes)}",
            ]
        )

    main16 = tuple(summary for summary in summaries if summary.role == MAIN16_ROLE_OBJECT)
    if len(main16) == 1:
        summary = main16[0]
        lines.extend(
            [
                "  main16_replacement_implication:",
                "    aiecc writes each compiled core ELF back to an aie.core elf_file attribute.",
                "    transaction generation lowers each core .text into config_blockwrite_data payloads.",
                "    in-place transaction patching cannot grow a payload without regenerating the transaction.",
                "    a raw scheduled main16 path should therefore generate replacement ELFs, then rerun xclbin/transaction packaging.",
                f"    current_main16_text_bytes={_format_ints(summary.text_bytes)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txn", type=Path, default=DEFAULT_TXN)
    parser.add_argument("--llvm-size", type=Path, default=DEFAULT_LLVM_SIZE)
    parser.add_argument("--mylm-program-images", type=Path, default=DEFAULT_MYLM_PROGRAM_IMAGES)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(inspect(args.txn, args.llvm_size, args.mylm_program_images), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
