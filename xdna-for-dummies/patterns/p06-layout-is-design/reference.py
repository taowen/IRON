"""CPU reference for P6: layout is design.

Demonstrates: two tiles need even-indexed and odd-indexed elements respectively.
In natural (interleaved) layout, these are non-contiguous — DMA can't extract
them with a single 1D BD. Solution: pre-pack into [all_evens | all_odds].

This mirrors qwen3-layer's current K/V even/odd scatter pattern.
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p06-layout-is-design"

# 32 elements total, two tiles each get 16 (even or odd indices)
TOTAL_ELEMENTS = 32
ELEMENTS_PER_TILE = TOTAL_ELEMENTS // 2
NUM_TILES = 2
OUTPUT_DWORDS = NUM_TILES


def make_data_interleaved() -> np.ndarray:
    """Natural layout: [0,1,2,3,4,5,...,31]. Evens and odds are interleaved."""
    return np.arange(TOTAL_ELEMENTS, dtype=np.int32)


def pack_even_odd(data: np.ndarray) -> np.ndarray:
    """Hardware-friendly layout: [all_evens | all_odds].
    Each tile gets a contiguous block — no stride needed."""
    evens = data[0::2]  # [0,2,4,6,...,30]
    odds = data[1::2]   # [1,3,5,7,...,31]
    return np.concatenate([evens, odds])


def expected_output(data: np.ndarray) -> np.ndarray:
    """tile0 sums even-indexed elements, tile1 sums odd-indexed."""
    evens = data[0::2]
    odds = data[1::2]
    return np.array([int(np.sum(evens)), int(np.sum(odds))], dtype=np.int32)


def validate_output(got: np.ndarray, data: np.ndarray) -> list[str]:
    expected = expected_output(data)
    errors: list[str] = []
    if got[0] != expected[0]:
        errors.append(f"tile0 (evens sum): got={got[0]} expected={expected[0]}")
    if got[1] != expected[1]:
        errors.append(f"tile1 (odds sum): got={got[1]} expected={expected[1]}")
    return errors
