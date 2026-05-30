"""Compare two raw bf16 dump files produced by reference/MyLM probes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


@dataclass(frozen=True)
class CompareStats:
    count: int
    max_abs: float
    mean_abs: float
    mismatches: int
    first_mismatch: int


def load_bf16(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=bfloat16).astype(np.float32)


def compare_values(expected: np.ndarray, got: np.ndarray, abs_tol: float, rel_tol: float) -> CompareStats:
    if expected.shape != got.shape:
        raise ValueError(f"shape mismatch: {got.shape} != {expected.shape}")
    abs_err = np.abs(expected - got)
    limit = np.maximum(np.float32(abs_tol), np.float32(rel_tol) * np.abs(expected))
    mismatch_indices = np.flatnonzero(abs_err > limit)
    first = int(mismatch_indices[0]) if mismatch_indices.size else -1
    return CompareStats(
        count=int(expected.size),
        max_abs=float(np.max(abs_err)) if expected.size else 0.0,
        mean_abs=float(np.mean(abs_err)) if expected.size else 0.0,
        mismatches=int(mismatch_indices.size),
        first_mismatch=first,
    )


def top_values(values: np.ndarray, count: int) -> tuple[tuple[int, float], ...]:
    if count <= 0:
        return ()
    if count > values.shape[0]:
        raise ValueError(f"top-k {count} exceeds value count {values.shape[0]}")
    indices = np.argpartition(values, -count)[-count:]
    sorted_indices = indices[np.argsort(values[indices])[::-1]]
    return tuple((int(idx), float(values[idx])) for idx in sorted_indices)


def print_stats(expected: np.ndarray, got: np.ndarray, stats: CompareStats, top_k: int) -> None:
    print(
        f"count={stats.count} max_abs={stats.max_abs:.9f} "
        f"mean_abs={stats.mean_abs:.9f} mismatches={stats.mismatches}"
    )
    if stats.first_mismatch >= 0:
        idx = stats.first_mismatch
        print(
            f"first_mismatch index={idx} "
            f"expected={float(expected[idx]):.9f} got={float(got[idx]):.9f} "
            f"abs={float(abs(expected[idx] - got[idx])):.9f}"
        )
    if top_k > 0:
        print("expected_top=" + ",".join(f"{idx}:{value:.9f}" for idx, value in top_values(expected, top_k)))
        print("got_top=" + ",".join(f"{idx}:{value:.9f}" for idx, value in top_values(got, top_k)))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--got", type=Path, required=True)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--rel-tol", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    expected = load_bf16(args.expected)
    got = load_bf16(args.got)
    stats = compare_values(expected, got, args.abs_tol, args.rel_tol)
    print_stats(expected, got, stats, args.top_k)
    if stats.mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
