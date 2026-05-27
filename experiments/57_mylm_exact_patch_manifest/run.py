#!/usr/bin/env python3
"""Validate the exact MyLM-style Qwen3 patch manifest."""

from __future__ import annotations

from collections import Counter

from schedule import (
    BD_SLOTS_BY_BANK,
    MAIN_COLUMNS,
    PATCH_PAIRS_PER_COLUMN,
    PHASE_INPUT_DIMS,
    PHASE_NAMES,
    PHASE_PATCH_COUNTS,
    TOTAL_PATCHES,
    expected_total_weight_bytes,
    generate_patch_manifest,
    patch_size_bytes,
    phase_patch_range,
)


def check_manifest() -> None:
    entries = generate_patch_manifest()
    if len(entries) != TOTAL_PATCHES:
        raise RuntimeError(f"patch count mismatch: {len(entries)} != {TOTAL_PATCHES}")

    expected_offset = 0
    for entry in entries:
        if entry.patch_index != entries.index(entry):
            raise RuntimeError(f"patch_index mismatch at {entry}")
        if entry.offset_bytes != expected_offset:
            raise RuntimeError(f"offset mismatch at patch {entry.patch_index}: {entry.offset_bytes} != {expected_offset}")
        expected_offset += entry.size_bytes

    if expected_offset != expected_total_weight_bytes():
        raise RuntimeError(f"total bytes mismatch: {expected_offset} != {expected_total_weight_bytes()}")

    phase_counts = Counter(entry.phase for entry in entries)
    for phase, expected_count in zip(PHASE_NAMES, PHASE_PATCH_COUNTS, strict=True):
        actual = phase_counts[phase]
        if actual != expected_count:
            raise RuntimeError(f"{phase} patch count mismatch: {actual} != {expected_count}")

    for phase_index, phase in enumerate(PHASE_NAMES):
        expected_size = patch_size_bytes(PHASE_INPUT_DIMS[phase_index])
        for entry in entries:
            if entry.phase == phase and entry.size_bytes != expected_size:
                raise RuntimeError(f"{phase} size mismatch at patch {entry.patch_index}")

    patches_per_n_block = len(MAIN_COLUMNS) * PATCH_PAIRS_PER_COLUMN
    for first in range(0, len(entries), patches_per_n_block):
        block_entries = entries[first:first + patches_per_n_block]
        if len(block_entries) < patches_per_n_block:
            raise RuntimeError("partial N-block found")
        phase = block_entries[0].phase
        block = block_entries[0].block
        bank = block_entries[0].bank
        expected_slots = BD_SLOTS_BY_BANK[bank]
        cursor = 0
        for column in MAIN_COLUMNS:
            for pair in range(PATCH_PAIRS_PER_COLUMN):
                entry = block_entries[cursor]
                if (entry.phase, entry.block, entry.column, entry.pair) != (phase, block, column, pair):
                    raise RuntimeError(f"N-block order mismatch at patch {entry.patch_index}: {entry}")
                if entry.bd_slot != expected_slots[pair]:
                    raise RuntimeError(f"BD slot mismatch at patch {entry.patch_index}: {entry.bd_slot}")
                cursor += 1


def main() -> None:
    check_manifest()
    entries = generate_patch_manifest()

    print("=" * 78)
    print("Experiment 57: MyLM Exact Patch Manifest")
    print("=" * 78)
    print(f"  phases: {PHASE_NAMES}")
    print(f"  main columns: {MAIN_COLUMNS}")
    print(f"  patch pairs per column: {PATCH_PAIRS_PER_COLUMN}")
    print(f"  total patches: {len(entries)}")
    print(f"  total weight bytes: {expected_total_weight_bytes()} (0x{expected_total_weight_bytes():x})")
    print()

    print("  phase ranges:")
    for phase, patch_count in zip(PHASE_NAMES, PHASE_PATCH_COUNTS, strict=True):
        start, end = phase_patch_range(entries, phase)
        input_dim = entries[start].input_dim
        size = entries[start].size_bytes
        print(f"    {phase:<4} patches {start:3d}..{end - 1:3d} count={patch_count:3d} "
              f"input_dim={input_dim:5d} patch_size=0x{size:x}")

    print()
    print("  first two N-blocks:")
    for entry in entries[:16]:
        print(
            f"    patch {entry.patch_index:3d}: phase={entry.phase:<4} block={entry.block:2d} "
            f"col=c{entry.column} pair={entry.pair} bank={entry.bank} "
            f"slot={entry.bd_slot} offset=0x{entry.offset_bytes:x} size=0x{entry.size_bytes:x}"
        )

    print()
    print("  PASS: MyLM exact 608-patch manifest is internally consistent.")


if __name__ == "__main__":
    main()
