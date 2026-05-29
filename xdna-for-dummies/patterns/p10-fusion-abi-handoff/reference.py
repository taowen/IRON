"""CPU reference for P10: fusion ABI handoff.

Demonstrates stable ABI between two fused operators:
- Fixed record format: 1 header + N payload
- Header encodes operator_id + payload_count
- Each operator only needs to know the ABI, not the other's internals
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p10-fusion-abi-handoff"
PAYLOAD_DWORDS = 16
RECORD_DWORDS = 1 + PAYLOAD_DWORDS
DATA_DWORDS = PAYLOAD_DWORDS  # input size
ADD_CONSTANT = 10
MUL_CONSTANT = 2
OP_A_ID = 0xA0
OP_B_ID = 0xB0


def make_input() -> np.ndarray:
    return np.arange(DATA_DWORDS, dtype=np.int32) + 1


def op_a_header() -> int:
    return (OP_A_ID << 8) | PAYLOAD_DWORDS


def op_b_header() -> int:
    return (OP_B_ID << 8) | PAYLOAD_DWORDS


def expected_output(input_data: np.ndarray) -> np.ndarray:
    """Output is op_b's record: [header_b | (input+10)*2... | op_a_id]."""
    out = np.zeros(RECORD_DWORDS, dtype=np.int32)
    out[0] = op_b_header()
    intermediate = input_data + ADD_CONSTANT
    for i in range(PAYLOAD_DWORDS - 1):
        out[1 + i] = int(intermediate[i]) * MUL_CONSTANT
    # Last payload slot = source op_id for traceability
    out[PAYLOAD_DWORDS] = OP_A_ID
    return out


def validate_output(got: np.ndarray, input_data: np.ndarray) -> list[str]:
    expected = expected_output(input_data)
    errors: list[str] = []
    if got[0] != expected[0]:
        errors.append(f"header: got=0x{got[0]:08X} expected=0x{expected[0]:08X}")
    for i in range(1, RECORD_DWORDS):
        if got[i] != expected[i]:
            errors.append(f"[{i}]: got={got[i]} expected={expected[i]}")
    return errors
