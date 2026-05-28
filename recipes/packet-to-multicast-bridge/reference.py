"""CPU reference for packet-source to multicast-bridge cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAIN_COLUMNS = (2, 3, 4, 5)
MAIN_ROWS = (2, 3, 4, 5)
ROWS_PER_COLUMN = len(MAIN_ROWS)
SUMMARY_DWORDS = 8
COLUMN_SUMMARY_DWORDS = ROWS_PER_COLUMN * SUMMARY_DWORDS
TOTAL_SUMMARY_DWORDS = len(MAIN_COLUMNS) * COLUMN_SUMMARY_DWORDS
MAIN_CHUNK_DWORDS = 128
BRIDGE_QUANTUM_DWORDS = 256


@dataclass(frozen=True)
class BridgeCase:
    name: str
    packet_id: int
    payload_dwords: int
    payload_base: int

    @property
    def main_chunks(self) -> int:
        return self.payload_dwords // MAIN_CHUNK_DWORDS

    @property
    def bridge_iterations(self) -> int:
        return self.payload_dwords // BRIDGE_QUANTUM_DWORDS


BRIDGE_CASES = (
    BridgeCase("fanout-2048d", 2, 2048, 100_000),
    BridgeCase("fanout-6144d", 0, 6144, 200_000),
)


def bridge_case(name: str) -> BridgeCase:
    for case in BRIDGE_CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown bridge case {name}")


def make_payload(case: BridgeCase) -> np.ndarray:
    return case.payload_base + np.arange(case.payload_dwords, dtype=np.int32)


def _payload_summary(case: BridgeCase) -> tuple[int, int, int, int, int]:
    payload = make_payload(case)
    payload_xor = np.int32(0)
    for value in payload:
        payload_xor = np.int32(payload_xor ^ value)
    return (
        case.main_chunks,
        int(payload[0]),
        int(payload[-1]),
        int(payload.sum(dtype=np.int64)),
        int(payload_xor),
    )


def expected_output(case: BridgeCase) -> np.ndarray:
    chunk_count, first, last, payload_sum, payload_xor = _payload_summary(case)
    output = np.empty(TOTAL_SUMMARY_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            start = group * COLUMN_SUMMARY_DWORDS + row * SUMMARY_DWORDS
            output[start:start + SUMMARY_DWORDS] = (
                group,
                row,
                chunk_count,
                first,
                last,
                payload_sum,
                payload_xor,
                case.payload_dwords,
            )
    return output


def validate_output(case: BridgeCase, got: np.ndarray) -> list[str]:
    expected = expected_output(case)
    errors: list[str] = []
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    mismatch = np.where(got != expected)[0]
    for idx in mismatch[:32]:
        errors.append(f"out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional mismatches")
    return errors
