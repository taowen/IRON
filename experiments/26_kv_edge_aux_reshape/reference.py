import numpy as np

CURRENT_DWORDS = 512
HISTORY_SOURCE_DWORDS = 4096
HISTORY_DWORDS = 2048
SIDEBAND_DWORDS = 17
SHAPE_OUT_DWORDS = 8
SIDEBAND_OUT_DWORDS = 4
SHAPE_B_OUT_DWORDS = 4
TOTAL_OUT_DWORDS = SHAPE_OUT_DWORDS + SIDEBAND_OUT_DWORDS + SHAPE_B_OUT_DWORDS


def make_current_payload() -> np.ndarray:
    values = np.arange(CURRENT_DWORDS, dtype=np.int32)
    return values * 3 + 11


def make_history_payload() -> np.ndarray:
    values = np.arange(HISTORY_SOURCE_DWORDS, dtype=np.int32)
    return values * 5 + 101


def make_sideband_payload() -> np.ndarray:
    values = np.arange(SIDEBAND_DWORDS, dtype=np.int32)
    return values * 7 + 1009


def checksum_words(values: np.ndarray) -> tuple[np.int32, np.int32, np.int32, np.int32]:
    total = np.int32(0)
    xors = np.int32(0)
    for value in values.astype(np.int32):
        total = np.int32(total + value)
        xors = np.int32(xors ^ value)
    return total, xors, np.int32(values[0]), np.int32(values[-1])


def expected_output() -> np.ndarray:
    current = make_current_payload()
    history = make_history_payload()[:HISTORY_DWORDS]
    sideband = make_sideband_payload()
    shape_b = sideband + np.int32(17)

    out = np.zeros(TOTAL_OUT_DWORDS, dtype=np.int32)
    out[0:4] = np.asarray(checksum_words(current), dtype=np.int32)
    out[4:8] = np.asarray(checksum_words(history), dtype=np.int32)
    out[8:12] = np.asarray(checksum_words(sideband), dtype=np.int32)
    out[12:16] = np.asarray(checksum_words(shape_b), dtype=np.int32)
    return out
