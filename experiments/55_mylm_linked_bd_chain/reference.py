"""CPU reference for exp55 MyLM-style linked BD chain."""

from __future__ import annotations

import numpy as np

NUM_CHUNKS = 8
CHUNK_DWORDS = 64
WORDS_PER_RECORD = 4
TOTAL_INPUT_DWORDS = NUM_CHUNKS * CHUNK_DWORDS
TOTAL_OUTPUT_DWORDS = NUM_CHUNKS * WORDS_PER_RECORD


def make_input() -> np.ndarray:
    data = np.empty(TOTAL_INPUT_DWORDS, dtype=np.int32)
    for chunk in range(NUM_CHUNKS):
        start = chunk * CHUNK_DWORDS
        data[start:start + CHUNK_DWORDS] = (chunk + 1) * 1000 + np.arange(CHUNK_DWORDS, dtype=np.int32)
    return data


def expected_output() -> np.ndarray:
    data = make_input()
    output = np.empty(TOTAL_OUTPUT_DWORDS, dtype=np.int32)
    for chunk in range(NUM_CHUNKS):
        values = data[chunk * CHUNK_DWORDS:(chunk + 1) * CHUNK_DWORDS]
        base = chunk * WORDS_PER_RECORD
        output[base + 0] = np.int32(0x55000000 | chunk)
        output[base + 1] = values[0]
        output[base + 2] = values[-1]
        output[base + 3] = np.sum(values, dtype=np.int32)
    return output


if __name__ == "__main__":
    print(f"input={make_input().shape[0]} dwords")
    print(f"output={expected_output().shape[0]} dwords")
    print(expected_output().reshape(NUM_CHUNKS, WORDS_PER_RECORD))
