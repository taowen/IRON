"""CPU reference for P08: append/scan separation.

Two DMA phases separated by npu.sync:
  Phase 1: writer tile → shim S2MM → cache BO at offset (append)
  Phase 2: shim MM2S → scanner tile (scan entire cache including appended data)

The cache BO is pre-filled by host. Append position has poison.
If scan happens before append completes, sum will contain poison → test fails.
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p08-append-scan-separation"
CACHE_DWORDS = 64
APPEND_DWORDS = 4
APPEND_POSITION = 8
OUTPUT_DWORDS = 1
POISON_VALUE = np.int32(0x7EADBEEF)


def make_cache() -> np.ndarray:
    """Pre-fill cache: sequential values with poison at append position."""
    cache = np.arange(CACHE_DWORDS, dtype=np.int32) + 1
    cache[APPEND_POSITION:APPEND_POSITION + APPEND_DWORDS] = POISON_VALUE
    return cache


def append_values() -> np.ndarray:
    """What the writer tile will produce (deterministic from position)."""
    return np.array([9000 + APPEND_POSITION + i for i in range(APPEND_DWORDS)], dtype=np.int32)


def expected_output(cache: np.ndarray) -> np.ndarray:
    """Expected sum after append overwrites poison slots."""
    merged = cache.copy()
    merged[APPEND_POSITION:APPEND_POSITION + APPEND_DWORDS] = append_values()
    return np.array([int(np.sum(merged))], dtype=np.int32)


def validate_output(got: np.ndarray, cache: np.ndarray) -> list[str]:
    expected = expected_output(cache)
    errors: list[str] = []
    if got[0] != expected[0]:
        errors.append(f"sum: got={got[0]} expected={expected[0]}")
    # Check if we got the pre-append sum (means sync didn't work)
    pre_sum = int(np.sum(cache))
    if got[0] == pre_sum:
        errors.append("CRITICAL: sum equals pre-append value — scan read BEFORE append completed!")
    return errors
