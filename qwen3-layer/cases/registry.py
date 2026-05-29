"""Dispatch table for runnable qwen3-layer NPU cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import npu_build
from cases import (
    currentkv_full_layer_q4nx_down_runner,
    currentkv_kvscan_attention_kv16_runner,
    q4nx_qkv_body_post_runner,
)

CaseFn = Callable[[int | None, int | None], bool]


@dataclass(frozen=True)
class CaseEntry:
    name: str
    check_only: CaseFn
    build_only: CaseFn
    run: CaseFn


def _without_decode_schedule(fn: Callable[[], bool]) -> CaseFn:
    def run_case(current_token: int | None, patch_from_token: int | None) -> bool:
        if current_token is not None or patch_from_token is not None:
            raise ValueError("decode schedule options are only supported by currentkv cases")
        return fn()

    return run_case


CASES = (
    CaseEntry(
        currentkv_kvscan_attention_kv16_runner.CASE_NAME,
        currentkv_kvscan_attention_kv16_runner.check_only,
        currentkv_kvscan_attention_kv16_runner.build_only,
        currentkv_kvscan_attention_kv16_runner.run,
    ),
    CaseEntry(
        currentkv_full_layer_q4nx_down_runner.CASE_NAME,
        currentkv_full_layer_q4nx_down_runner.check_only,
        currentkv_full_layer_q4nx_down_runner.build_only,
        currentkv_full_layer_q4nx_down_runner.run,
    ),
    CaseEntry(
        q4nx_qkv_body_post_runner.CASE_NAME,
        _without_decode_schedule(q4nx_qkv_body_post_runner.check_only),
        _without_decode_schedule(q4nx_qkv_body_post_runner.build_only),
        _without_decode_schedule(q4nx_qkv_body_post_runner.run),
    ),
)
CASE_NAMES = tuple(case.name for case in CASES)
DEFAULT_CASE_NAME = currentkv_full_layer_q4nx_down_runner.CASE_NAME
_CASE_BY_NAME = {case.name: case for case in CASES}


def _case(case_name: str) -> CaseEntry:
    try:
        return _CASE_BY_NAME[case_name]
    except KeyError as exc:
        raise ValueError(f"unknown case {case_name}") from exc


def check_only(
    case_name: str,
    current_token: int | None = None,
    patch_from_token: int | None = None,
) -> bool:
    return _case(case_name).check_only(current_token, patch_from_token)


def build_only(
    case_name: str,
    current_token: int | None = None,
    patch_from_token: int | None = None,
) -> bool:
    return _case(case_name).build_only(current_token, patch_from_token)


def run(
    case_name: str,
    current_token: int | None = None,
    patch_from_token: int | None = None,
) -> bool:
    print(f"NPU device: {npu_build.device()}")
    return _case(case_name).run(current_token, patch_from_token)
