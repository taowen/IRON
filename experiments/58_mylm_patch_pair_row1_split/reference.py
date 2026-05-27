"""CPU reference for exp58 MyLM patch-pair row1 split."""

from __future__ import annotations

import numpy as np

ROWS_PER_COLUMN = 4
M_PER_TILE = 32
K_CHUNK = 256
GROUP_SIZE = 32
GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2
PATCH_PAIR_BF16 = CHUNK_BF16 * 2
NUM_PATCH_PAIRS = 2
TOTAL_INPUT_BF16 = NUM_PATCH_PAIRS * PATCH_PAIR_BF16
TOTAL_INPUT_I32 = TOTAL_INPUT_BF16 // 2
WORDS_PER_RECORD = 4
TOTAL_OUTPUT_DWORDS = ROWS_PER_COLUMN * WORDS_PER_RECORD


def make_input_u16() -> np.ndarray:
    data = np.empty(TOTAL_INPUT_BF16, dtype=np.uint16)
    for row in range(ROWS_PER_COLUMN):
        patch_pair = row // 2
        row_in_pair = row % 2
        start = patch_pair * PATCH_PAIR_BF16 + row_in_pair * CHUNK_BF16
        data[start:start + CHUNK_BF16] = np.uint16((row + 1) * 4096) + np.arange(CHUNK_BF16, dtype=np.uint16)
    return data


def make_input_i32() -> np.ndarray:
    return make_input_u16().view(np.int32)


def expected_output() -> np.ndarray:
    data = make_input_u16()
    output = np.empty(TOTAL_OUTPUT_DWORDS, dtype=np.int32)
    for row in range(ROWS_PER_COLUMN):
        patch_pair = row // 2
        row_in_pair = row % 2
        start = patch_pair * PATCH_PAIR_BF16 + row_in_pair * CHUNK_BF16
        chunk = data[start:start + CHUNK_BF16].astype(np.int32)
        base = row * WORDS_PER_RECORD
        output[base + 0] = row
        output[base + 1] = chunk[0]
        output[base + 2] = chunk[-1]
        output[base + 3] = np.sum(chunk, dtype=np.int32)
    return output


if __name__ == "__main__":
    print(f"chunk_bf16={CHUNK_BF16}")
    print(f"patch_pair_bf16={PATCH_PAIR_BF16}")
    print(f"input_i32={TOTAL_INPUT_I32}")
    print(expected_output().reshape(ROWS_PER_COLUMN, WORDS_PER_RECORD))
