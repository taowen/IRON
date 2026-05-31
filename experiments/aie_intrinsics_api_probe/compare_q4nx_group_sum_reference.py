#!/usr/bin/env python3
"""Compare the current Q4NX reference with a MyLM-style group-sum form."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_LAYER = REPO_ROOT / "qwen3-layer"
sys.path.insert(0, str(QWEN3_LAYER))

from q4nx_reference import (  # noqa: E402
    ACT_SLICE_BF16,
    CHUNK_BYTES,
    GROUPS_PER_CHUNK,
    GROUP_SIZE,
    M_PER_TILE,
    Q4NX_DATA_BYTES_PER_LANE,
    Q4NX_DATA_OFFSET,
    Q4NX_LANES,
    Q4NX_ROWS_PER_LANE,
    Q4NX_SCALE_BYTES,
    make_q4nx_chunk,
    q4nx_matvec_from_chunk,
)


@dataclass(frozen=True)
class DiffStats:
    name: str
    max_abs: float
    mean_abs: float
    mismatches_1e_3: int
    mismatches_1e_2: int


def bf16_round(values: np.ndarray) -> np.ndarray:
    return values.astype(bfloat16).astype(np.float32)


def make_activation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.75, (ACT_SLICE_BF16,)).astype(np.float32)
    return values.astype(bfloat16)


def unpack_chunk(packed_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if packed_chunk.shape[0] != CHUNK_BYTES:
        raise ValueError(f"bad Q4NX chunk bytes: {packed_chunk.shape[0]} != {CHUNK_BYTES}")

    scales_raw = np.frombuffer(packed_chunk[:Q4NX_SCALE_BYTES], dtype=bfloat16).astype(np.float32)
    zeros_raw = np.frombuffer(
        packed_chunk[Q4NX_SCALE_BYTES:Q4NX_DATA_OFFSET],
        dtype=bfloat16,
    ).astype(np.float32)
    scales = scales_raw.reshape(GROUPS_PER_CHUNK, M_PER_TILE).T
    zeros = zeros_raw.reshape(GROUPS_PER_CHUNK, M_PER_TILE).T

    packed_u8 = packed_chunk[Q4NX_DATA_OFFSET:]
    weights_u4 = np.empty((M_PER_TILE, ACT_SLICE_BF16), dtype=np.float32)
    for lane in range(Q4NX_LANES):
        lane_base = lane * Q4NX_DATA_BYTES_PER_LANE
        row_base = lane * Q4NX_ROWS_PER_LANE
        lane_bytes = packed_u8[
            lane_base : lane_base + Q4NX_DATA_BYTES_PER_LANE
        ].reshape(ACT_SLICE_BF16, Q4NX_ROWS_PER_LANE // 2)
        for byte_idx in range(Q4NX_ROWS_PER_LANE // 2):
            row0 = row_base + byte_idx * 2
            row1 = row0 + 1
            byte_values = lane_bytes[:, byte_idx]
            weights_u4[row0, :] = (byte_values & 0x0F).astype(np.float32)
            weights_u4[row1, :] = (byte_values >> 4).astype(np.float32)
    return scales, zeros, weights_u4


def q4nx_group_sum_matvec(
    packed_chunk: np.ndarray,
    activation_slice: np.ndarray,
    round_group_sum: bool,
) -> np.ndarray:
    scales, zeros, weights_u4 = unpack_chunk(packed_chunk)
    grouped_weights = weights_u4.reshape(M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
    grouped_act = activation_slice.astype(np.float32).reshape(GROUPS_PER_CHUNK, GROUP_SIZE)
    scaled = bf16_round(grouped_weights * scales[..., None])
    output = np.zeros((M_PER_TILE,), dtype=np.float32)
    for group in range(GROUPS_PER_CHUNK):
        for dim in range(GROUP_SIZE):
            np.add(output, scaled[:, group, dim] * grouped_act[group, dim], out=output)
        group_sum = np.sum(grouped_act[group], dtype=np.float32)
        if round_group_sum:
            group_sum = bf16_round(np.array([group_sum], dtype=np.float32))[0]
        np.add(output, zeros[:, group] * group_sum, out=output)
    return output


def diff_stats(name: str, got: np.ndarray, expected: np.ndarray) -> DiffStats:
    diff = np.abs(got - expected)
    return DiffStats(
        name=name,
        max_abs=float(np.max(diff)),
        mean_abs=float(np.mean(diff)),
        mismatches_1e_3=int(np.count_nonzero(diff > 1e-3)),
        mismatches_1e_2=int(np.count_nonzero(diff > 1e-2)),
    )


def render_stats(stats: tuple[DiffStats, ...]) -> str:
    lines = [
        "q4nx_group_sum_reference_compare:",
        "  current_reference=bf16((q * scale) + offset) * activation",
        "  group_sum_reference=sum(bf16(q * scale) * activation) + offset * sum(activation_group)",
        "  stats:",
    ]
    for stat in stats:
        lines.extend(
            [
                f"    {stat.name}:",
                f"      max_abs={stat.max_abs:.8f}",
                f"      mean_abs={stat.mean_abs:.8f}",
                f"      mismatches_1e_3={stat.mismatches_1e_3}",
                f"      mismatches_1e_2={stat.mismatches_1e_2}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    rng = np.random.default_rng(20260531)
    stats: list[DiffStats] = []
    for sample in range(8):
        packed = make_q4nx_chunk(rng)
        activation = make_activation(9000 + sample)
        expected = q4nx_matvec_from_chunk(packed, activation)
        group_sum_fp32 = q4nx_group_sum_matvec(packed, activation, round_group_sum=False)
        group_sum_bf16 = q4nx_group_sum_matvec(packed, activation, round_group_sum=True)
        stats.append(diff_stats(f"sample{sample}_group_sum_fp32", group_sum_fp32, expected))
        stats.append(diff_stats(f"sample{sample}_group_sum_bf16", group_sum_bf16, expected))
    print(render_stats(tuple(stats)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
