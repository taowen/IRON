#!/usr/bin/env python3
"""Replace source MLIR core bodies with ELF-backed external core programs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TILE_RE = re.compile(r"^\s*%(?P<name>\w+) = aie\.tile\((?P<col>\d+), (?P<row>\d+)\)")
SOURCE_CORE_RE = re.compile(r"^(?P<indent>\s*)%(?P<core>\w+) = aie\.core\(%(?P<tile>\w+)\) \{")
TXN_CORE_RE = re.compile(
    r"%core_(?P<col>\d+)_(?P<row>\d+) = aie\.core\(%tile_\d+_\d+\) \{\s*"
    r"aie\.end\s*\}\s*\{elf_file = \"(?P<elf>[^\"]+)\", link_files = \[(?P<link_files>[^\]]*)\]\}",
    re.MULTILINE,
)
LINK_FILE_RE = re.compile(r'"([^"]+)"')


@dataclass(frozen=True)
class TileCoord:
    col: int
    row: int


@dataclass(frozen=True)
class CoreProgramAttr:
    elf: Path
    link_files: tuple[Path, ...]

    def mlir_attr(self, elf_dir: Path | None = None) -> str:
        elf = elf_dir / self.elf.name if elf_dir is not None else self.elf
        links = ", ".join(f'"{path}"' for path in self.link_files)
        return f'elf_file = "{elf}", link_files = [{links}]'


def _parse_source_tiles(mlir_text: str) -> dict[str, TileCoord]:
    tiles: dict[str, TileCoord] = {}
    for line in mlir_text.splitlines():
        match = TILE_RE.match(line)
        if match is not None:
            tiles[match.group("name")] = TileCoord(
                col=int(match.group("col")),
                row=int(match.group("row")),
            )
    return tiles


def _parse_transaction_core_attrs(txn_text: str) -> dict[TileCoord, CoreProgramAttr]:
    attrs: dict[TileCoord, CoreProgramAttr] = {}
    for match in TXN_CORE_RE.finditer(txn_text):
        coord = TileCoord(col=int(match.group("col")), row=int(match.group("row")))
        link_files = tuple(Path(item) for item in LINK_FILE_RE.findall(match.group("link_files")))
        attrs[coord] = CoreProgramAttr(elf=Path(match.group("elf")), link_files=link_files)
    return attrs


def externalize_core_programs(input_mlir: str, transaction_mlir: str, elf_dir: Path | None = None) -> str:
    tiles = _parse_source_tiles(input_mlir)
    attrs = _parse_transaction_core_attrs(transaction_mlir)
    lines = input_mlir.splitlines()
    externalized: list[str] = []
    used_coords: set[TileCoord] = set()
    index = 0

    while index < len(lines):
        line = lines[index]
        match = SOURCE_CORE_RE.match(line)
        if match is None:
            externalized.append(line)
            index += 1
            continue

        tile_name = match.group("tile")
        if tile_name not in tiles:
            raise ValueError(f"core references unknown tile: {tile_name}")
        coord = tiles[tile_name]
        if coord not in attrs:
            raise ValueError(f"transaction has no external ELF for c{coord.col}r{coord.row}")

        indent = match.group("indent")
        externalized.append(f'{indent}%{match.group("core")} = aie.core(%{tile_name}) {{')
        externalized.append(f"{indent}  aie.end")
        externalized.append(f"{indent}}} {{{attrs[coord].mlir_attr(elf_dir)}}}")
        used_coords.add(coord)

        depth = line.count("{") - line.count("}")
        index += 1
        while index < len(lines) and depth > 0:
            depth += lines[index].count("{") - lines[index].count("}")
            index += 1

    unused = sorted(set(attrs) - used_coords, key=lambda coord: (coord.col, coord.row))
    if unused:
        names = ", ".join(f"c{coord.col}r{coord.row}" for coord in unused)
        raise ValueError(f"transaction has external ELFs not present in source MLIR: {names}")

    return "\n".join(externalized) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mlir", required=True, type=Path)
    parser.add_argument("--transaction-mlir", required=True, type=Path)
    parser.add_argument("--output-mlir", required=True, type=Path)
    parser.add_argument("--elf-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.output_mlir.write_text(
        externalize_core_programs(
            input_mlir=args.input_mlir.read_text(),
            transaction_mlir=args.transaction_mlir.read_text(),
            elf_dir=args.elf_dir,
        )
    )
    print(f"externalized_mlir={args.output_mlir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
