#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two Qwen3 persistent artifacts compiled for different positions."""

from __future__ import annotations

import argparse
import difflib
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DmaBd:
    fifo: str
    offset: int
    length: int
    line: str


def _path_has_prj_parent(path: Path) -> bool:
    return any(part.endswith(".mlir.prj") for part in path.parts)


def _find_one(root: Path, pattern: str, *, outside_prj: bool = False) -> Path | None:
    if root.is_file():
        return root if root.match(pattern) else None
    matches = []
    for path in root.rglob(pattern):
        if outside_prj and _path_has_prj_parent(path):
            continue
        matches.append(path)
    if not matches:
        return None
    return sorted(matches)[0]


def _find_mlir(root: Path) -> Path:
    if root.is_file():
        return root
    matches = [
        path
        for path in root.rglob("*.mlir")
        if not _path_has_prj_parent(path) and path.name != "input_with_addresses.mlir"
    ]
    if len(matches) != 1:
        joined = "\n".join(str(path) for path in sorted(matches)[:20])
        raise RuntimeError(
            f"expected exactly one top-level MLIR under {root}, found "
            f"{len(matches)}:\n{joined}"
        )
    return matches[0]


def _find_prj(mlir: Path) -> Path | None:
    candidate = Path(str(mlir) + ".prj")
    return candidate if candidate.is_dir() else None


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _changed_mlir_lines(left: Path, right: Path) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(
            _read_lines(left),
            _read_lines(right),
            fromfile=str(left),
            tofile=str(right),
            n=0,
        )
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]


def _classify_mlir_line(line: str) -> str:
    if "aie.dma_bd" in line:
        return "dma_bd"
    if "scf.for" in line:
        return "loop_bound"
    if "func.call @qwen3_attention_scores_bf16" in line:
        return "attention_scores_position"
    if "func.call @qwen3_attention_context_bf16" in line:
        return "attention_context_position"
    if "func.call @qwen3_merge_current_v_bf16" in line:
        return "merge_v_position"
    if "func.call @mask_bf16" in line:
        return "mask_length"
    if "arith.constant" in line:
        return "constant"
    return "other"


def _parse_dma_bds(path: Path) -> list[DmaBd]:
    fifo = ""
    bds = []
    fifo_re = re.compile(r"aiex\.dma_configure_task_for @([A-Za-z0-9_]+)")
    bd_re = re.compile(r"memref<[^>]+>,\s*([0-9]+),\s*([0-9]+),")
    for raw_line in _read_lines(path):
        line = raw_line.strip()
        fifo_match = fifo_re.search(line)
        if fifo_match:
            fifo = fifo_match.group(1)
            continue
        if "aie.dma_bd" not in line:
            continue
        bd_match = bd_re.search(line)
        if bd_match is None:
            continue
        bds.append(
            DmaBd(
                fifo=fifo,
                offset=int(bd_match.group(1)),
                length=int(bd_match.group(2)),
                line=line,
            )
        )
    return bds


def _byte_diffs(left: Path, right: Path) -> list[int]:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    if len(left_bytes) != len(right_bytes):
        raise RuntimeError(
            f"size mismatch: {left}={len(left_bytes)} {right}={len(right_bytes)}"
        )
    return [
        idx
        for idx, (left_byte, right_byte) in enumerate(zip(left_bytes, right_bytes))
        if left_byte != right_byte
    ]


def _u32_word_diffs(left: Path, right: Path) -> list[tuple[int, int, int]]:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    offsets = sorted({idx & ~0x3 for idx in _byte_diffs(left, right)})
    diffs = []
    for offset in offsets:
        if offset + 4 > len(left_bytes):
            continue
        left_word = struct.unpack_from("<I", left_bytes, offset)[0]
        right_word = struct.unpack_from("<I", right_bytes, offset)[0]
        if left_word != right_word:
            diffs.append((offset, left_word, right_word))
    return diffs


def _count_diff_bytes(left: Path, right: Path) -> int:
    return len(_byte_diffs(left, right))


def _print_mlir_summary(left: Path, right: Path) -> None:
    left_lines = _read_lines(left)
    right_lines = _read_lines(right)
    changed = _changed_mlir_lines(left, right)
    categories = Counter(_classify_mlir_line(line) for line in changed)
    print("mlir:")
    print(f"  left={left}")
    print(f"  right={right}")
    print(f"  line_count={len(left_lines)}/{len(right_lines)}")
    print(f"  changed_diff_lines={len(changed)}")
    for name, count in sorted(categories.items()):
        print(f"  category.{name}={count}")

    left_bds = _parse_dma_bds(left)
    right_bds = _parse_dma_bds(right)
    print(f"  dma_bd_count={len(left_bds)}/{len(right_bds)}")
    for idx, (left_bd, right_bd) in enumerate(zip(left_bds, right_bds)):
        if left_bd == right_bd:
            continue
        same_fifo = left_bd.fifo == right_bd.fifo
        print(
            "  dma_bd_change:"
            f" index={idx}"
            f" fifo={left_bd.fifo if same_fifo else left_bd.fifo + '->' + right_bd.fifo}"
            f" offset={left_bd.offset}->{right_bd.offset}"
            f" offset_delta={right_bd.offset - left_bd.offset}"
            f" length={left_bd.length}->{right_bd.length}"
        )


def _print_binary_summary(name: str, left: Path | None, right: Path | None) -> None:
    print(f"{name}:")
    if left is None or right is None:
        print("  missing")
        return
    print(f"  left={left}")
    print(f"  right={right}")
    print(f"  size={left.stat().st_size}/{right.stat().st_size}")
    if left.stat().st_size != right.stat().st_size:
        print("  changed_bytes=size-mismatch")
        return
    changed_bytes = _count_diff_bytes(left, right)
    print(f"  changed_bytes={changed_bytes}")
    if changed_bytes and name == "runtime_bin":
        for offset, left_word, right_word in _u32_word_diffs(left, right):
            print(
                "  u32_patch_candidate:"
                f" offset={offset}"
                f" value={left_word}->{right_word}"
                f" delta={right_word - left_word}"
            )


def _print_prj_summary(left_prj: Path | None, right_prj: Path | None) -> None:
    print("project:")
    if left_prj is None or right_prj is None:
        print("  missing")
        return
    important = (
        "main_mem_topology.json",
        "main_kernels.json",
        "main_aie_partition.json",
        "main_aie_cdo_init.bin",
        "main_aie_cdo_enable.bin",
        "main_aie_cdo_elfs.bin",
        "main.pdi",
    )
    for name in important:
        left = left_prj / name
        right = right_prj / name
        if not left.exists() or not right.exists():
            print(f"  {name}=missing")
            continue
        status = "identical" if left.read_bytes() == right.read_bytes() else "different"
        suffix = ""
        if status == "different" and left.stat().st_size == right.stat().st_size:
            suffix = f" changed_bytes={_count_diff_bytes(left, right)}"
        print(f"  {name}={status}{suffix}")

    changed_elves = []
    for left_elf in sorted(left_prj.glob("main_core_*.elf")):
        right_elf = right_prj / left_elf.name
        if not right_elf.exists() or left_elf.read_bytes() == right_elf.read_bytes():
            continue
        changed_elves.append((left_elf.name, _count_diff_bytes(left_elf, right_elf)))
    print(f"  changed_core_elves={len(changed_elves)}")
    for name, changed_bytes in changed_elves:
        print(f"  core_elf_change: {name} changed_bytes={changed_bytes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two Qwen3 persistent build artifacts compiled for different "
            "decode positions."
        )
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_mlir = _find_mlir(args.left)
    right_mlir = _find_mlir(args.right)
    left_prj = _find_prj(left_mlir)
    right_prj = _find_prj(right_mlir)
    left_bin = _find_one(args.left, "*.bin", outside_prj=True)
    right_bin = _find_one(args.right, "*.bin", outside_prj=True)
    left_xclbin = _find_one(args.left, "*.xclbin", outside_prj=True)
    right_xclbin = _find_one(args.right, "*.xclbin", outside_prj=True)

    _print_mlir_summary(left_mlir, right_mlir)
    _print_binary_summary("runtime_bin", left_bin, right_bin)
    _print_binary_summary("xclbin", left_xclbin, right_xclbin)
    _print_prj_summary(left_prj, right_prj)


if __name__ == "__main__":
    main()
