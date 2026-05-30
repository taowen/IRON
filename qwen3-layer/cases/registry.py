"""Dispatch table for runnable qwen3-layer NPU cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cases import (
    full_layer_attention_o_bf16_runner,
    full_layer_qkv_prefix_runner,
    main16_q4nx_compute_perf_runner,
    qwen3_8b_c1r2_input_norm_runner,
    qwen3_8b_decode_layer_runner,
    qwen3_8b_qkv_cache_write_runner,
    row1_weight_stream_perf_runner,
)

CaseFn = Callable[[int | None, Path | None, int, bool], bool]


@dataclass(frozen=True)
class CaseEntry:
    name: str
    check_only: CaseFn
    build_only: CaseFn
    run: CaseFn


CASES = (
    CaseEntry(
        qwen3_8b_decode_layer_runner.CASE_NAME,
        qwen3_8b_decode_layer_runner.check_only,
        qwen3_8b_decode_layer_runner.build_only,
        qwen3_8b_decode_layer_runner.run,
    ),
    CaseEntry(
        qwen3_8b_c1r2_input_norm_runner.CASE_NAME,
        qwen3_8b_c1r2_input_norm_runner.check_only,
        qwen3_8b_c1r2_input_norm_runner.build_only,
        qwen3_8b_c1r2_input_norm_runner.run,
    ),
    CaseEntry(
        qwen3_8b_qkv_cache_write_runner.CASE_NAME,
        qwen3_8b_qkv_cache_write_runner.check_only,
        qwen3_8b_qkv_cache_write_runner.build_only,
        qwen3_8b_qkv_cache_write_runner.run,
    ),
    CaseEntry(
        full_layer_qkv_prefix_runner.CASE_NAME,
        full_layer_qkv_prefix_runner.check_only,
        full_layer_qkv_prefix_runner.build_only,
        full_layer_qkv_prefix_runner.run,
    ),
    CaseEntry(
        full_layer_attention_o_bf16_runner.CASE_NAME,
        full_layer_attention_o_bf16_runner.check_only,
        full_layer_attention_o_bf16_runner.build_only,
        full_layer_attention_o_bf16_runner.run,
    ),
    CaseEntry(
        row1_weight_stream_perf_runner.CASE_NAME,
        row1_weight_stream_perf_runner.check_only,
        row1_weight_stream_perf_runner.build_only,
        row1_weight_stream_perf_runner.run,
    ),
    CaseEntry(
        main16_q4nx_compute_perf_runner.CASE_NAME,
        main16_q4nx_compute_perf_runner.check_only,
        main16_q4nx_compute_perf_runner.build_only,
        main16_q4nx_compute_perf_runner.run,
    ),
)
CASE_NAMES = tuple(case.name for case in CASES)
DEFAULT_CASE_NAME = qwen3_8b_decode_layer_runner.CASE_NAME
_CASE_BY_NAME = {case.name: case for case in CASES}


def _case(case_name: str) -> CaseEntry:
    try:
        return _CASE_BY_NAME[case_name]
    except KeyError as exc:
        raise ValueError(f"unknown case {case_name}") from exc


def check_only(
    case_name: str,
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    return _case(case_name).check_only(current_token, model_path, layer, download_model)


def build_only(
    case_name: str,
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    return _case(case_name).build_only(current_token, model_path, layer, download_model)


def run(
    case_name: str,
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    return _case(case_name).run(current_token, model_path, layer, download_model)
