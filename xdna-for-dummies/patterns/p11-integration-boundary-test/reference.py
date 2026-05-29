"""CPU reference for P11: integration boundary testing."""

from __future__ import annotations

import numpy as np

CASE_NAME = "p11-integration-boundary-test"
DATA_DWORDS = 64
POISON_VALUE = np.int32(0x7EADBEEF)


def make_input() -> np.ndarray:
    return np.arange(DATA_DWORDS, dtype=np.int32) + 1


def expected_output(input_data: np.ndarray) -> np.ndarray:
    return input_data * 3 + 7


def check_poison(got: np.ndarray) -> list[str]:
    poison_indices = np.where(got == POISON_VALUE)[0]
    if len(poison_indices) == 0:
        return []
    return [f"poison 0x{POISON_VALUE:08X} remains at indices: {poison_indices.tolist()[:10]}"]


def check_values(got: np.ndarray, input_data: np.ndarray) -> list[str]:
    expected = expected_output(input_data)
    errors: list[str] = []
    mismatches = np.where(got != expected)[0]
    for idx in mismatches[:10]:
        errors.append(f"[{idx}] got={got[idx]} expected={expected[idx]}")
    if len(mismatches) > 10:
        errors.append(f"... and {len(mismatches) - 10} more mismatches")
    return errors


def validate_output(got: np.ndarray, input_data: np.ndarray) -> list[str]:
    errors: list[str] = []
    errors.extend(check_poison(got))
    errors.extend(check_values(got, input_data))
    return errors
