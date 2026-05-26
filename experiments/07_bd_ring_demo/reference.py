"""CPU reference for BD ring checksum demo."""

import numpy as np

CHUNK_ELEMS_SMOKE = 256   # int32 = 1024 bytes
CHUNK_ELEMS_REAL = 4096   # int32 = 0x4000 bytes


def make_input(num_chunks: int, chunk_elems: int) -> np.ndarray:
    """Generate test input: chunk[i][j] = i * chunk_elems + j."""
    data = np.zeros(num_chunks * chunk_elems, dtype=np.int32)
    for i in range(num_chunks):
        for j in range(chunk_elems):
            data[i * chunk_elems + j] = i * chunk_elems + j
    return data


def checksums_reference(data: np.ndarray, num_chunks: int, chunk_elems: int) -> np.ndarray:
    """Compute expected per-chunk checksums. Returns array of num_chunks int32."""
    result = np.zeros(num_chunks, dtype=np.int32)
    for i in range(num_chunks):
        chunk = data[i * chunk_elems : (i + 1) * chunk_elems]
        result[i] = np.sum(chunk, dtype=np.int32)
    return result
