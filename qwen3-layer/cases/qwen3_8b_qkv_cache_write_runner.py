"""NPU runner for phase-local Qwen3 Q/K/V current-cache writeback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import qwen3_8b_qkv_cache_write_generate as generate
from cases.full_layer_engine_reference import (
    HIDDEN_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
    aux_as_i32,
    bf16_cache_payload,
    cache_writeback_stats,
    expected_cache_writeback,
    format_stage_stats,
    hidden_input_as_i32,
    input_norm_activation as physical_input_norm_activation,
    packed_as_i32,
    validate_cache_writeback,
)
from cases.decode_cache_reference import (
    DecodeSchedule,
    make_decode_schedule,
    validate_cache_layout_contract,
)
from cases.qwen3_8b_decode_layer_reference import (
    DEFAULT_CURRENT_TOKEN,
    DEFAULT_LAYER,
    make_reference_inputs,
    validate_model_assets,
)
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent


@dataclass(frozen=True)
class QKVCacheWriteFixture:
    k_cache_i32: np.ndarray
    v_cache_i32: np.ndarray
    hidden_i32: np.ndarray
    qkv_activation_bf16: np.ndarray
    k_norm_bf16: np.ndarray
    packed_weights: np.ndarray
    weights_i32: np.ndarray
    rope_theta: float


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _decode_token(current_token: int | None) -> int:
    token = DEFAULT_CURRENT_TOKEN if current_token is None else current_token
    if token < 0:
        raise ValueError("--current-token must be non-negative")
    return token


def _padded_cache(values: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    expected = (schedule.current_token + 1, 8, 128)
    if values.shape != expected:
        raise ValueError(f"reference cache shape mismatch: {values.shape} != {expected}")
    padded = np.zeros((schedule.total_context, 8, 128), dtype=values.dtype)
    padded[: schedule.current_token + 1] = values
    return padded


def _poison_current_cache(values: np.ndarray, schedule: DecodeSchedule, poison: float) -> np.ndarray:
    poisoned = _padded_cache(values, schedule)
    poisoned[schedule.current_token, :, :] = np.array(poison, dtype=values.dtype)
    return poisoned


def _cache_buffer_payload(payload: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    if payload.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"KV cache shape mismatch: {payload.shape} != {(schedule.kv_cache_dwords,)}")
    return payload


def _make_fixture(model: Qwen3Q4NXModel, layer: int, schedule: DecodeSchedule) -> QKVCacheWriteFixture:
    inputs = make_reference_inputs(schedule.current_token)
    input_norm, post_norm, q_norm, k_norm = model.layer_norm_weights(layer)
    qkv_activation = physical_input_norm_activation(inputs.hidden, input_norm)
    packed = model.layer_weight_stream(layer)
    aux = aux_as_i32(
        schedule.current_token,
        input_norm,
        post_norm,
        q_norm,
        k_norm,
        model.config.rope_theta,
    )
    weights = np.concatenate((aux, packed_as_i32(packed))).astype(np.int32)
    initial_k_cache = bf16_cache_payload(schedule, _poison_current_cache(inputs.k_cache, schedule, 19.0))
    initial_v_cache = bf16_cache_payload(schedule, _poison_current_cache(inputs.v_cache, schedule, -19.0))
    if weights.shape != (TOTAL_WEIGHT_AND_AUX_I32,):
        raise RuntimeError(f"aux-prefixed weight i32 mismatch: {weights.shape} != {(TOTAL_WEIGHT_AND_AUX_I32,)}")
    return QKVCacheWriteFixture(
        k_cache_i32=initial_k_cache,
        v_cache_i32=initial_v_cache,
        hidden_i32=hidden_input_as_i32(inputs.hidden),
        qkv_activation_bf16=qkv_activation,
        k_norm_bf16=k_norm,
        packed_weights=packed,
        weights_i32=weights,
        rope_theta=model.config.rope_theta,
    )


def build_kernel(schedule: DecodeSchedule, build_name: str = CASE_NAME) -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir(schedule)
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text, schedule)
    if errors:
        raise RuntimeError("\n".join(f"  QWEN3 QKV CACHE-WRITE STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(_decode_token(current_token))
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    mlir = generate.generate_mlir(schedule)
    errors.extend(generate.validate_generated_mlir(mlir, schedule))
    errors.extend(validate_cache_layout_contract(schedule))
    if errors:
        for error in errors:
            print(f"  QWEN3 QKV CACHE-WRITE FAIL: {error}")
        return False
    fixture = _make_fixture(model, layer, schedule)
    print(f"  PASS: {CASE_NAME} MLIR covers phase-local Q/K/V current cache writeback")
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    print(f"  hidden_dwords={fixture.hidden_i32.shape[0]}")
    print(f"  aux_prefixed_weight_dwords={fixture.weights_i32.shape[0]}")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(_decode_token(current_token))
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3 ASSET FAIL: {error}")
        return False
    _make_fixture(model, layer, schedule)
    xclbin_path, insts_path = build_kernel(schedule)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(_decode_token(current_token))
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3 ASSET FAIL: {error}")
        return False

    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    print("  route=c1r2 input RMSNorm -> phase-local main16 Q/K/V -> c1r3 -> packet8/9 cache write")
    print()

    xclbin_path, insts_path = build_kernel(schedule)
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    print("  Preparing raw hidden, aux-prefixed full weight stream, and poisoned K/V cache...")
    fixture = _make_fixture(model, layer, schedule)
    k_cache = _cache_buffer_payload(fixture.k_cache_i32, schedule)
    v_cache = _cache_buffer_payload(fixture.v_cache_i32, schedule)
    if fixture.hidden_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"hidden i32 mismatch: {fixture.hidden_i32.shape} != {(HIDDEN_DWORDS,)}")

    k_cache_buf = XRTTensor.from_torch(torch.from_numpy(k_cache.copy()).to(torch.int32))
    v_cache_buf = XRTTensor.from_torch(torch.from_numpy(v_cache.copy()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(fixture.weights_i32.copy()).to(torch.int32))
    hidden_buf = XRTTensor.from_torch(torch.from_numpy(fixture.hidden_i32.copy()).to(torch.int32))

    print("  Running on NPU...")
    result = npu_build.run(handle, [k_cache_buf, v_cache_buf, weights_buf, hidden_buf])
    got_k = k_cache_buf.to_torch().numpy().astype(np.int32)
    got_v = v_cache_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    expected_cache = expected_cache_writeback(
        schedule,
        fixture.packed_weights,
        fixture.qkv_activation_bf16,
        fixture.k_norm_bf16,
        fixture.rope_theta,
        k_cache,
        v_cache,
    )
    cache_stats = cache_writeback_stats(
        schedule,
        got_k[: schedule.kv_cache_dwords],
        got_v[: schedule.kv_cache_dwords],
        expected_cache,
    )
    for stats in cache_stats:
        print(f"  stage_budget: {format_stage_stats(stats)}")

    errors = validate_cache_writeback(
        schedule,
        got_k[: schedule.kv_cache_dwords],
        got_v[: schedule.kv_cache_dwords],
        expected_cache,
    )
    if errors:
        print(f"  FAIL: {len(errors)} Q/K/V cache-write mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: phase-local Q/K/V wrote current K/V cache correctly")
    return True
