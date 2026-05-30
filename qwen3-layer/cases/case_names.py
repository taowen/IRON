"""Runnable qwen3-layer case names shared by registry and static checks."""

from __future__ import annotations

CASE_NAMES = (
    "qwen3-8b-decode-layer",
    "qwen3-8b-c1r2-input-norm-replay",
    "qwen3-8b-qkv-compact-output",
    "qwen3-8b-qkv-cache-write-bridge",
    "full-layer-qkv-prefix",
    "full-layer-attention-o-bf16",
    "row1-weight-stream-perf",
    "main16-q4nx-compute-perf",
)
DEFAULT_CASE_NAME = CASE_NAMES[0]
