"""CPU reference for P4: ping-pong credit lock."""

from __future__ import annotations

import numpy as np

CASE_NAME = "p04-ping-pong-credit-lock"
CHUNK_DWORDS = 64
NUM_BATCHES = 2
CONSTANT = 42
OUTPUT_DWORDS = CHUNK_DWORDS * NUM_BATCHES


def expected_output() -> np.ndarray:
    out = np.zeros(OUTPUT_DWORDS, dtype=np.int32)
    for batch in range(NUM_BATCHES):
        for i in range(CHUNK_DWORDS):
            out[batch * CHUNK_DWORDS + i] = batch * CHUNK_DWORDS + i + CONSTANT
    return out


def validate_output(got: np.ndarray) -> list[str]:
    expected = expected_output()
    errors: list[str] = []
    if got.shape != expected.shape:
        errors.append(f"shape mismatch: {got.shape} vs {expected.shape}")
        return errors
    mismatches = np.where(got != expected)[0]
    for idx in mismatches[:10]:
        errors.append(f"[{idx}] got={got[idx]} expected={expected[idx]}")
    if len(mismatches) > 10:
        errors.append(f"... and {len(mismatches) - 10} more mismatches")
    return errors
