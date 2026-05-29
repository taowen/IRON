"""CPU reference for P7: block carrier + online merge.

Demonstrates online weighted-mean merge with carrier = (max, sum, weighted_value).
This is the minimal sufficient statistic for online softmax-style merge:
you can't merge blocks without knowing their max (for rescaling) and sum (for normalization).
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p07-block-carrier-online-merge"
BLOCK_DWORDS = 16
NUM_BLOCKS = 4
CARRIER_DWORDS = 3  # (block_max, block_sum, block_weighted_value)
STATE_DWORDS = 3
OUTPUT_DWORDS = 3  # (weighted_mean, final_max, final_sum)
SCORES_PER_BLOCK = BLOCK_DWORDS
VALUES_PER_BLOCK = BLOCK_DWORDS
INPUT_DWORDS = (SCORES_PER_BLOCK + VALUES_PER_BLOCK) * NUM_BLOCKS


def make_input() -> tuple[np.ndarray, np.ndarray]:
    """Generate scores and values for 4 blocks."""
    rng = np.random.RandomState(42)
    scores = rng.randint(0, 100, (NUM_BLOCKS, BLOCK_DWORDS)).astype(np.int32)
    values = rng.randint(1, 50, (NUM_BLOCKS, BLOCK_DWORDS)).astype(np.int32)
    # Make block 2 have the highest score to test rescaling
    scores[2, 3] = 200
    return scores, values


def pack_input(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Pack as [scores_block0, values_block0, scores_block1, values_block1, ...]."""
    parts = []
    for block in range(NUM_BLOCKS):
        parts.append(scores[block])
        parts.append(values[block])
    return np.concatenate(parts)


def _exp_approx(delta: int) -> int:
    """Approximate exp(-delta) as max(0, 4096 - delta*64)."""
    w = 4096 - delta * 64
    return max(0, w)


def expected_output(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute expected output using same online merge logic as kernel."""
    running_max = -2147483647
    running_sum = 0
    running_wv = 0

    for block in range(NUM_BLOCKS):
        # Compute block carrier
        block_max = int(np.max(scores[block]))
        block_sum = 0
        block_wv = 0
        for i in range(BLOCK_DWORDS):
            delta = block_max - int(scores[block, i])
            weight = _exp_approx(delta)
            block_sum += weight
            block_wv += weight * int(values[block, i])

        # Online merge
        if running_sum == 0:
            running_max = block_max
            running_sum = block_sum
            running_wv = block_wv
        else:
            if block_max >= running_max:
                new_max = block_max
                delta = block_max - running_max
                old_scale = _exp_approx(delta)
                new_scale = 4096
            else:
                new_max = running_max
                delta = running_max - block_max
                old_scale = 4096
                new_scale = _exp_approx(delta)

            running_max = new_max
            running_sum = (running_sum * old_scale + block_sum * new_scale) // 4096
            running_wv = (running_wv * old_scale + block_wv * new_scale) // 4096

    weighted_mean = running_wv // running_sum if running_sum != 0 else 0
    return np.array([weighted_mean, running_max, running_sum], dtype=np.int32)


def validate_output(got: np.ndarray, scores: np.ndarray, values: np.ndarray) -> list[str]:
    expected = expected_output(scores, values)
    errors: list[str] = []
    if got[0] != expected[0]:
        errors.append(f"weighted_mean: got={got[0]} expected={expected[0]}")
    if got[1] != expected[1]:
        errors.append(f"final_max: got={got[1]} expected={expected[1]}")
    if got[2] != expected[2]:
        errors.append(f"final_sum: got={got[2]} expected={expected[2]}")
    return errors
