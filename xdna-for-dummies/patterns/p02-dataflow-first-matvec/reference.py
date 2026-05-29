"""CPU reference for P2: dataflow-first matvec.

Demonstrates block-major scheduling with weight ping-pong:
- Weights stream through, each chunk used once (ping-pong for DMA/compute overlap)
- Activation replayed from host (block-major: one output block sees all K chunks)
- Weight stream order aligns with compute loop
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p02-dataflow-first-matvec"

ROWS_PER_TILE = 4
NUM_TILES = 2
TOTAL_OUTPUT_ROWS = ROWS_PER_TILE * NUM_TILES
K_DIM = 16
NUM_CHUNKS = 4
CHUNK_COLS = K_DIM // NUM_CHUNKS

RECORD_DWORDS = 1 + ROWS_PER_TILE
WEIGHT_CHUNK_DWORDS = ROWS_PER_TILE * CHUNK_COLS
WEIGHT_DWORDS_PER_TILE = ROWS_PER_TILE * K_DIM
ACT_DWORDS = K_DIM
OUTPUT_DWORDS = RECORD_DWORDS * NUM_TILES
TOTAL_WEIGHT_DWORDS = WEIGHT_DWORDS_PER_TILE * NUM_TILES


def make_weights() -> np.ndarray:
    return (np.arange(TOTAL_OUTPUT_ROWS * K_DIM, dtype=np.int32) + 1).reshape(
        TOTAL_OUTPUT_ROWS, K_DIM
    )


def make_activation() -> np.ndarray:
    return np.arange(K_DIM, dtype=np.int32) + 1


def pack_weights_chunk_major(weights: np.ndarray) -> np.ndarray:
    """Pack weights so each chunk is contiguous: weight stream order = compute loop order."""
    packed = np.zeros(TOTAL_WEIGHT_DWORDS, dtype=np.int32)
    for tile in range(NUM_TILES):
        tile_rows = weights[tile * ROWS_PER_TILE : (tile + 1) * ROWS_PER_TILE]
        for chunk in range(NUM_CHUNKS):
            for row in range(ROWS_PER_TILE):
                src = tile_rows[row, chunk * CHUNK_COLS : (chunk + 1) * CHUNK_COLS]
                offset = (
                    tile * WEIGHT_DWORDS_PER_TILE
                    + chunk * WEIGHT_CHUNK_DWORDS
                    + row * CHUNK_COLS
                )
                packed[offset : offset + CHUNK_COLS] = src
    return packed


def record_header(tile_id: int) -> int:
    return (tile_id << 16) | 0x4D00 | ROWS_PER_TILE


def expected_output(weights: np.ndarray, activation: np.ndarray) -> np.ndarray:
    out = np.zeros(OUTPUT_DWORDS, dtype=np.int32)
    for tile in range(NUM_TILES):
        tile_rows = weights[tile * ROWS_PER_TILE : (tile + 1) * ROWS_PER_TILE]
        matvec = (tile_rows @ activation).astype(np.int32)
        base = tile * RECORD_DWORDS
        out[base] = record_header(tile)
        out[base + 1 : base + 1 + ROWS_PER_TILE] = matvec
    return out


def validate_output(
    got: np.ndarray, weights: np.ndarray, activation: np.ndarray
) -> list[str]:
    expected = expected_output(weights, activation)
    errors: list[str] = []
    for tile in range(NUM_TILES):
        base = tile * RECORD_DWORDS
        if got[base] != expected[base]:
            errors.append(f"tile{tile} header: got=0x{got[base]:08X} expected=0x{expected[base]:08X}")
        for i in range(ROWS_PER_TILE):
            idx = base + 1 + i
            if got[idx] != expected[idx]:
                errors.append(f"tile{tile} row{i}: got={got[idx]} expected={expected[idx]}")
    return errors
