"""CPU reference for P1: tile-local work units.

Demonstrates the real pattern: 2 tiles each own a fixed output slice,
receive weight+activation chunks, accumulate locally, emit a record
with header + payload ABI.
"""

from __future__ import annotations

import numpy as np

CASE_NAME = "p01-tile-local-work-units"

# Tile work unit definition (mirrors main16 structure at small scale)
ROWS_PER_TILE = 4
NUM_TILES = 2
TOTAL_OUTPUT_ROWS = ROWS_PER_TILE * NUM_TILES
K_DIM = 16
NUM_CHUNKS = 4
CHUNK_COLS = K_DIM // NUM_CHUNKS

# Record ABI: 1 header + ROWS_PER_TILE payload
RECORD_DWORDS = 1 + ROWS_PER_TILE

# Weight chunk per tile = ROWS_PER_TILE * CHUNK_COLS
WEIGHT_CHUNK_DWORDS = ROWS_PER_TILE * CHUNK_COLS

# Host I/O
WEIGHT_DWORDS_PER_TILE = ROWS_PER_TILE * K_DIM
TOTAL_WEIGHT_DWORDS = WEIGHT_DWORDS_PER_TILE * NUM_TILES
ACT_DWORDS = K_DIM
OUTPUT_DWORDS = RECORD_DWORDS * NUM_TILES


def make_weights() -> np.ndarray:
    """8x16 weight matrix (2 tiles x 4 rows x 16 cols)."""
    return (np.arange(TOTAL_OUTPUT_ROWS * K_DIM, dtype=np.int32) + 1).reshape(
        TOTAL_OUTPUT_ROWS, K_DIM
    )


def make_activation() -> np.ndarray:
    return np.arange(K_DIM, dtype=np.int32) + 1


def pack_weights_chunk_major(weights: np.ndarray) -> np.ndarray:
    """Pack into chunk-major: for each tile, [chunk0, chunk1, ...] where
    each chunk = ROWS_PER_TILE * CHUNK_COLS contiguous."""
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
    return (tile_id << 16) | 0xAB00 | ROWS_PER_TILE


def expected_output(weights: np.ndarray, activation: np.ndarray) -> np.ndarray:
    """Expected: NUM_TILES records, each = [header, matvec_result...]."""
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
        # Check header
        if got[base] != expected[base]:
            errors.append(
                f"tile{tile} header: got=0x{got[base]:08X} expected=0x{expected[base]:08X}"
            )
        # Check payload
        for i in range(ROWS_PER_TILE):
            idx = base + 1 + i
            if got[idx] != expected[idx]:
                errors.append(f"tile{tile} row{i}: got={got[idx]} expected={expected[idx]}")
    return errors
