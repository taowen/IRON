"""CPU reference for P03: record column compact.

On-chip compact: 2 producers emit records → memtile collects via separate
S2MM channels → counting lock drain → compact output strips row1's header.

Compact layout:
  [row0 full record (header + payload)] [row1 payload only (header stripped)]
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p03-record-column-compact"
PAYLOAD_DWORDS = 16
RECORD_DWORDS = 1 + PAYLOAD_DWORDS
NUM_PRODUCERS = 2
# Compact: row0 keeps full record, row1 only payload (header stripped)
COMPACT_DWORDS = RECORD_DWORDS + PAYLOAD_DWORDS


def record_header(tile_id: int) -> int:
    return (tile_id << 16) | PAYLOAD_DWORDS


def expected_output() -> np.ndarray:
    """Compact = [row0_header, row0_payload..., row1_payload...]."""
    out = np.zeros(COMPACT_DWORDS, dtype=np.int32)
    # Row0: full record (header + payload)
    out[0] = record_header(0)
    for i in range(PAYLOAD_DWORDS):
        out[1 + i] = 0 * 100 + i
    # Row1: payload only (header stripped by BD offset)
    for i in range(PAYLOAD_DWORDS):
        out[RECORD_DWORDS + i] = 1 * 100 + i
    return out


def validate_output(got: np.ndarray) -> list[str]:
    expected = expected_output()
    errors: list[str] = []
    if got.shape[0] < COMPACT_DWORDS:
        errors.append(f"output too short: {got.shape[0]} < {COMPACT_DWORDS}")
        return errors
    for idx in range(COMPACT_DWORDS):
        if got[idx] != expected[idx]:
            errors.append(f"[{idx}] got={got[idx]} expected={expected[idx]}")
    return errors
