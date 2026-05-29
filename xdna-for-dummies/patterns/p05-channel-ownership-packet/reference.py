"""CPU reference for P5: channel ownership + packet ID.

Demonstrates: one physical channel (MM2S ch0 on producer) carries TWO logical
streams distinguished by packet ID. Two consumer tiles each filter their packet.
This is the correct pattern: channel ownership is fixed, packet ID multiplexes.
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p05-channel-ownership-packet"
DATA_DWORDS = 32
WORKER0_CONSTANT = 10
WORKER1_CONSTANT = 20
OUTPUT_DWORDS = DATA_DWORDS * 2
PACKET_ID_0 = 0
PACKET_ID_1 = 1


def make_input() -> np.ndarray:
    return np.arange(OUTPUT_DWORDS, dtype=np.int32) + 1


def expected_output(input_data: np.ndarray) -> np.ndarray:
    out = np.zeros(OUTPUT_DWORDS, dtype=np.int32)
    out[:DATA_DWORDS] = input_data[:DATA_DWORDS] + WORKER0_CONSTANT
    out[DATA_DWORDS:] = input_data[DATA_DWORDS:] + WORKER1_CONSTANT
    return out


def validate_output(got: np.ndarray, input_data: np.ndarray) -> list[str]:
    expected = expected_output(input_data)
    errors: list[str] = []
    mismatches = np.where(got != expected)[0]
    for idx in mismatches[:10]:
        errors.append(f"[{idx}] got={got[idx]} expected={expected[idx]}")
    if len(mismatches) > 10:
        errors.append(f"... and {len(mismatches) - 10} more mismatches")
    return errors
