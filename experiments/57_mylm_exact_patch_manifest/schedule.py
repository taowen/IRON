"""Generate the MyLM-style exact Qwen3 weight patch manifest."""

from __future__ import annotations

from dataclasses import dataclass

MAIN_COLUMNS = (2, 3, 4, 5)
PATCH_PAIRS_PER_COLUMN = 2
OUTPUT_BLOCK_ROWS = 512
PATCH_OUTPUT_ROWS = 64
Q4_BYTES_PER_WEIGHT_NUMERATOR = 5
Q4_BYTES_PER_WEIGHT_DENOMINATOR = 8

PHASE_NAMES = ("Q", "K", "V", "O", "UP", "GATE", "DOWN")
PHASE_INPUT_DIMS = (4096, 4096, 4096, 4096, 4096, 4096, 12288)
PHASE_OUTPUT_DIMS = (4096, 1024, 1024, 4096, 12288, 12288, 4096)
PHASE_BLOCKS = tuple(output_dim // OUTPUT_BLOCK_ROWS for output_dim in PHASE_OUTPUT_DIMS)
PHASE_PATCH_COUNTS = tuple(blocks * len(MAIN_COLUMNS) * PATCH_PAIRS_PER_COLUMN for blocks in PHASE_BLOCKS)
TOTAL_PATCHES = sum(PHASE_PATCH_COUNTS)
BANK_NAMES = ("A", "B")
BD_SLOTS_BY_BANK = {
    "A": ("slot_0x1d020", "slot_0x1d040"),
    "B": ("slot_0x1d120", "slot_0x1d140"),
}


@dataclass(frozen=True)
class PatchEntry:
    patch_index: int
    phase: str
    phase_index: int
    block: int
    column: int
    pair: int
    bank: str
    bd_slot: str
    input_dim: int
    output_rows: int
    offset_bytes: int
    size_bytes: int


def patch_size_bytes(input_dim: int) -> int:
    weights = PATCH_OUTPUT_ROWS * input_dim
    numerator = weights * Q4_BYTES_PER_WEIGHT_NUMERATOR
    if numerator % Q4_BYTES_PER_WEIGHT_DENOMINATOR != 0:
        raise ValueError(f"non-integral q4 patch size for input_dim={input_dim}")
    return numerator // Q4_BYTES_PER_WEIGHT_DENOMINATOR


def generate_patch_manifest() -> tuple[PatchEntry, ...]:
    entries: list[PatchEntry] = []
    offset = 0
    patch_index = 0

    for phase_index, phase in enumerate(PHASE_NAMES):
        input_dim = PHASE_INPUT_DIMS[phase_index]
        size = patch_size_bytes(input_dim)
        for block in range(PHASE_BLOCKS[phase_index]):
            bank = BANK_NAMES[block % len(BANK_NAMES)]
            bd_slots = BD_SLOTS_BY_BANK[bank]
            for column in MAIN_COLUMNS:
                for pair in range(PATCH_PAIRS_PER_COLUMN):
                    entries.append(
                        PatchEntry(
                            patch_index=patch_index,
                            phase=phase,
                            phase_index=phase_index,
                            block=block,
                            column=column,
                            pair=pair,
                            bank=bank,
                            bd_slot=bd_slots[pair],
                            input_dim=input_dim,
                            output_rows=PATCH_OUTPUT_ROWS,
                            offset_bytes=offset,
                            size_bytes=size,
                        )
                    )
                    patch_index += 1
                    offset += size

    return tuple(entries)


def expected_total_weight_bytes() -> int:
    return sum(PHASE_PATCH_COUNTS[idx] * patch_size_bytes(PHASE_INPUT_DIMS[idx]) for idx in range(len(PHASE_NAMES)))


def phase_patch_range(entries: tuple[PatchEntry, ...], phase: str) -> tuple[int, int]:
    phase_entries = [entry.patch_index for entry in entries if entry.phase == phase]
    if not phase_entries:
        raise ValueError(f"unknown phase {phase}")
    return min(phase_entries), max(phase_entries) + 1
