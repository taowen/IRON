"""Dispatch table for runnable qwen3-layer NPU cases."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import npu_build
from cases import qwen3_8b_decode_layer_runner, qwen3_8b_qkv_body_post_runner

CaseFn = Callable[[int | None, int | None, Path | None, int, bool], bool]
LegacyCaseFn = Callable[[int | None, int | None], bool]


@dataclass(frozen=True)
class CaseEntry:
    name: str
    check_only: CaseFn
    build_only: CaseFn
    run: CaseFn


def _legacy(module_name: str, fn_name: str) -> CaseFn:
    def run_case(
        current_token: int | None,
        patch_from_token: int | None,
        model_path: Path | None,
        layer: int,
        download_model: bool,
    ) -> bool:
        if model_path is not None or download_model:
            raise ValueError("model options are only supported by real qwen3-8b cases")
        if layer != 0:
            raise ValueError("--layer is only supported by real qwen3-8b cases")
        module = importlib.import_module(f"cases.{module_name}")
        fn: LegacyCaseFn = getattr(module, fn_name)
        return fn(current_token, patch_from_token)

    return run_case


def _without_decode_schedule(module_name: str, fn_name: str) -> CaseFn:
    def run_case(
        current_token: int | None,
        patch_from_token: int | None,
        model_path: Path | None,
        layer: int,
        download_model: bool,
    ) -> bool:
        if current_token is not None or patch_from_token is not None:
            raise ValueError("decode schedule options are only supported by currentkv cases")
        if model_path is not None or download_model:
            raise ValueError("model options are only supported by real qwen3-8b cases")
        if layer != 0:
            raise ValueError("--layer is only supported by real qwen3-8b cases")
        module = importlib.import_module(f"cases.{module_name}")
        fn: Callable[[], bool] = getattr(module, fn_name)
        return fn()

    return run_case


CASES = (
    CaseEntry(
        qwen3_8b_decode_layer_runner.CASE_NAME,
        qwen3_8b_decode_layer_runner.check_only,
        qwen3_8b_decode_layer_runner.build_only,
        qwen3_8b_decode_layer_runner.run,
    ),
    CaseEntry(
        qwen3_8b_qkv_body_post_runner.CASE_NAME,
        qwen3_8b_qkv_body_post_runner.check_only,
        qwen3_8b_qkv_body_post_runner.build_only,
        qwen3_8b_qkv_body_post_runner.run,
    ),
    CaseEntry(
        "currentkv-kvscan-attention-kv16-o-bridge",
        _legacy("currentkv_kvscan_attention_kv16_runner", "check_only"),
        _legacy("currentkv_kvscan_attention_kv16_runner", "build_only"),
        _legacy("currentkv_kvscan_attention_kv16_runner", "run"),
    ),
    CaseEntry(
        "currentkv-full-layer-q4nx-down-bridge",
        _legacy("currentkv_full_layer_q4nx_down_runner", "check_only"),
        _legacy("currentkv_full_layer_q4nx_down_runner", "build_only"),
        _legacy("currentkv_full_layer_q4nx_down_runner", "run"),
    ),
    CaseEntry(
        "q4nx-qkv-body-post-bridge",
        _without_decode_schedule("q4nx_qkv_body_post_runner", "check_only"),
        _without_decode_schedule("q4nx_qkv_body_post_runner", "build_only"),
        _without_decode_schedule("q4nx_qkv_body_post_runner", "run"),
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
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    return _case(case_name).check_only(current_token, patch_from_token, model_path, layer, download_model)


def build_only(
    case_name: str,
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    return _case(case_name).build_only(current_token, patch_from_token, model_path, layer, download_model)


def run(
    case_name: str,
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    if case_name != qwen3_8b_decode_layer_runner.CASE_NAME:
        print(f"NPU device: {npu_build.device()}")
    return _case(case_name).run(current_token, patch_from_token, model_path, layer, download_model)
