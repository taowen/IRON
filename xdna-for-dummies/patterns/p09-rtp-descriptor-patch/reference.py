"""CPU reference for P09: RTP + descriptor patch.

Demonstrates two layers of dynamic parameters:
1. RTP: tile reads rtp[0] as an additive offset
2. Descriptor patch: host BD buffer_offset selects which INPUT SLICE to send

Both change between runs WITHOUT recompiling xclbin.
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p09-rtp-descriptor-patch"
SLICE_DWORDS = 16
TOTAL_INPUT_DWORDS = 64  # 4 slices of 16


def make_input() -> np.ndarray:
    return np.arange(TOTAL_INPUT_DWORDS, dtype=np.int32) + 1


def expected_output(input_data: np.ndarray, slice_idx: int, rtp_value: int) -> np.ndarray:
    """Output = input[slice_idx*16 : (slice_idx+1)*16] + rtp_value."""
    start = slice_idx * SLICE_DWORDS
    return input_data[start:start + SLICE_DWORDS] + rtp_value


def validate_output(got: np.ndarray, input_data: np.ndarray, slice_idx: int, rtp_value: int) -> list[str]:
    expected = expected_output(input_data, slice_idx, rtp_value)
    errors: list[str] = []
    mismatches = np.where(got != expected)[0]
    for idx in mismatches[:10]:
        errors.append(f"[{idx}] got={got[idx]} expected={expected[idx]}")
    return errors
