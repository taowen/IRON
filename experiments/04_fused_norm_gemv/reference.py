"""CPU reference for fused RMSNorm → GEMV."""

import numpy as np
from ml_dtypes import bfloat16


def rms_norm(x: np.ndarray) -> np.ndarray:
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2) + 1e-5)
    return (x_f32 / rms).astype(bfloat16)


def fused_norm_gemv(hidden: np.ndarray, W: np.ndarray) -> np.ndarray:
    normed = rms_norm(hidden)
    output = (W.astype(np.float32) @ normed.astype(np.float32)).astype(bfloat16)
    return output, normed
