"""CPU reference helpers for rounded KV scan windows."""

from __future__ import annotations

import numpy as np

from cases import attention_block_reference as base

CASE_NAME = "kv-scan-reference"
K_CACHE_SIDE_DWORDS = base.K_WINDOW_DWORDS * 2
V_CACHE_SIDE_DWORDS = base.V_WINDOW_DWORDS * 2
KV_CACHE_SIDE_DWORDS = K_CACHE_SIDE_DWORDS + V_CACHE_SIDE_DWORDS

CONTEXT = base.CONTEXT
HEAD_DIM = base.HEAD_DIM
HEADS_PER_WINDOW = base.HEADS_PER_WINDOW
K_WINDOW_DWORDS = base.K_WINDOW_DWORDS
KV_HEADS_PER_WINDOW = base.KV_HEADS_PER_WINDOW
KV_SIDE_DWORDS = base.KV_SIDE_DWORDS
KV_SLOT_DWORDS = base.KV_SLOT_DWORDS
OUTPUT_DWORDS = base.OUTPUT_DWORDS
Q_DWORDS = base.Q_DWORDS
SCALAR_DWORDS = base.SCALAR_DWORDS
TOTAL_SUMMARY_DWORDS = base.TOTAL_SUMMARY_DWORDS
V_WINDOW_DWORDS = base.V_WINDOW_DWORDS
WEIGHT_DWORDS = base.WEIGHT_DWORDS
WINDOW_DWORDS = base.WINDOW_DWORDS

expected_output = base.expected_output
make_q_payload = base.make_q_payload
validate_output = base.validate_output


def make_k_cache_payload(side: int) -> np.ndarray:
    payload = base.make_kv_payload(side)
    return np.concatenate(
        (
            payload[0:K_WINDOW_DWORDS],
            payload[KV_SLOT_DWORDS : KV_SLOT_DWORDS + K_WINDOW_DWORDS],
        )
    ).astype(np.int32)


def make_v_cache_payload(side: int) -> np.ndarray:
    payload = base.make_kv_payload(side)
    return np.concatenate(
        (
            payload[K_WINDOW_DWORDS:KV_SLOT_DWORDS],
            payload[KV_SLOT_DWORDS + K_WINDOW_DWORDS : KV_SIDE_DWORDS],
        )
    ).astype(np.int32)


def make_kv_cache_payload(side: int) -> np.ndarray:
    return np.concatenate((make_k_cache_payload(side), make_v_cache_payload(side))).astype(np.int32)


def route_summary() -> list[str]:
    return [
        f"case={CASE_NAME}",
        "host provides logical KV cache sides instead of prepacked KV side payloads",
        f"KV cache side={KV_CACHE_SIDE_DWORDS} dwords laid out as K0,K1,V0,V1",
        "shim BD slices rebuild K,V,K,V side layout in row1 memtiles",
        base.route_summary()[-1],
    ]
