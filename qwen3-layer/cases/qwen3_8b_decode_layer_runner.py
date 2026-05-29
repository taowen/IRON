"""Runner for the real Qwen3-8B single-layer numerical target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

from cases import currentkv_full_layer_q4nx_down_generate as generate
from cases.currentkv_full_layer_q4nx_down_reference import (
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    TOTAL_WEIGHT_I32,
    expected_output as physical_expected_output,
    packed_as_i32,
    validate_cache_writeback,
    validate_expected_output,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    DecodeSchedule,
    make_decode_schedule,
    make_history_k_cache_payload,
    make_history_v_cache_payload,
    validate_cache_layout_contract,
)
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel
from cases.qwen3_8b_decode_layer_reference import (
    CASE_NAME,
    DEFAULT_CURRENT_TOKEN,
    DEFAULT_LAYER,
    Qwen3LayerReference,
    make_reference_inputs,
    validate_model_assets,
)

EXPERIMENT_DIR = Path(__file__).parent.parent


@dataclass(frozen=True)
class Qwen3PhysicalFixture:
    hidden_i32: np.ndarray
    hidden_bf16: np.ndarray
    packed_weights: np.ndarray
    weights_i32: np.ndarray
    weight_bytes: int
    expected: np.ndarray


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _reject_patch_from_token(patch_from_token: int | None) -> None:
    if patch_from_token is not None:
        raise ValueError("--patch-from-token is not wired for the real Qwen3-8B full-layer case yet")


def _pack_bf16_i32(values: np.ndarray) -> np.ndarray:
    if values.shape != (HIDDEN_DWORDS * 2,):
        raise ValueError(f"hidden bf16 shape mismatch: {values.shape}")
    return np.frombuffer(values.astype(bfloat16).tobytes(), dtype=np.int32).copy()


def _cache_buffer_payload(payload: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    if payload.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"KV cache shape mismatch: {payload.shape} != {(schedule.kv_cache_dwords,)}")
    return payload


def _make_physical_fixture(
    model: Qwen3Q4NXModel,
    layer: int,
    schedule: DecodeSchedule,
) -> Qwen3PhysicalFixture:
    reference = Qwen3LayerReference(model, layer)
    hidden = make_reference_inputs(schedule.current_token).hidden
    normalized_hidden = reference.input_norm_activation(hidden)
    packed = model.layer_weight_stream(layer)
    weights_i32 = packed_as_i32(packed)
    expected = physical_expected_output(schedule, packed, normalized_hidden)
    return Qwen3PhysicalFixture(
        hidden_i32=_pack_bf16_i32(normalized_hidden),
        hidden_bf16=normalized_hidden,
        packed_weights=packed,
        weights_i32=weights_i32,
        weight_bytes=packed.shape[0],
        expected=expected,
    )


def _validate_real_physical_inputs(model: Qwen3Q4NXModel, layer: int, schedule: DecodeSchedule) -> tuple[int, int]:
    reference = Qwen3LayerReference(model, layer)
    hidden = make_reference_inputs(schedule.current_token).hidden
    normalized_hidden = reference.input_norm_activation(hidden)
    hidden_i32 = _pack_bf16_i32(normalized_hidden)
    weight_stream = model.layer_weight_stream(layer)
    weights_i32 = packed_as_i32(weight_stream)
    if hidden_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"hidden i32 shape mismatch: {hidden_i32.shape} != {(HIDDEN_DWORDS,)}")
    if weights_i32.shape != (TOTAL_WEIGHT_I32,):
        raise RuntimeError(f"weight i32 shape mismatch: {weights_i32.shape} != {(TOTAL_WEIGHT_I32,)}")
    return weight_stream.shape[0], hidden_i32.shape[0]


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
        raise RuntimeError("\n".join(f"  REAL QWEN3 FULL-LAYER STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_patch_from_token(patch_from_token)
    schedule = make_decode_schedule(DEFAULT_CURRENT_TOKEN if current_token is None else current_token)
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    mlir_text = generate.generate_mlir(schedule)
    errors.extend(generate.validate_generated_mlir(mlir_text, schedule))
    errors.extend(validate_cache_layout_contract(schedule))
    if errors:
        for error in errors:
            print(f"  QWEN3-8B FULL-LAYER FAIL: {error}")
        return False
    weight_bytes, hidden_dwords = _validate_real_physical_inputs(model, layer, schedule)
    print(f"  PASS: {CASE_NAME} assets valid for layer {layer}")
    print(f"  weight_stream_bytes={weight_bytes}")
    print(f"  input_norm_hidden_dwords={hidden_dwords}")
    print(f"  current_token={schedule.current_token}")
    print("  PASS: real Qwen3 weights fit the current full-layer NPU topology")
    return True


def build_only(
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_patch_from_token(patch_from_token)
    schedule = make_decode_schedule(DEFAULT_CURRENT_TOKEN if current_token is None else current_token)
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3-8B ASSET FAIL: {error}")
        return False
    _validate_real_physical_inputs(model, layer, schedule)
    xclbin_path, insts_path = build_kernel(schedule)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run(
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_patch_from_token(patch_from_token)

    schedule = make_decode_schedule(DEFAULT_CURRENT_TOKEN if current_token is None else current_token)
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3-8B ASSET FAIL: {error}")
        return False

    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    print("  topology=current full-layer NPU physical frontier with real Qwen3 Q4NX weights")
    print("  NOTE: c1r2/c1r3/attention still use the current physical oracle; c6r2 uses bf16 SiLU with the NPU bounded table sigmoid")
    print()

    xclbin_path, insts_path = build_kernel(schedule)
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    print("  Preparing input RMSNorm hidden, K/V cache, real Q4NX weights, and physical CPU oracle...")
    fixture = _make_physical_fixture(model, layer, schedule)
    k_cache = _cache_buffer_payload(make_history_k_cache_payload(schedule), schedule)
    v_cache = _cache_buffer_payload(make_history_v_cache_payload(schedule), schedule)
    if fixture.hidden_i32.shape[0] != HIDDEN_DWORDS:
        raise RuntimeError(f"hidden i32 mismatch: {fixture.hidden_i32.shape[0]} != {HIDDEN_DWORDS}")
    if fixture.weights_i32.shape[0] != TOTAL_WEIGHT_I32:
        raise RuntimeError(f"weight i32 mismatch: {fixture.weights_i32.shape[0]} != {TOTAL_WEIGHT_I32}")

    k_cache_buf = XRTTensor.from_torch(torch.from_numpy(k_cache.copy()).to(torch.int32))
    v_cache_buf = XRTTensor.from_torch(torch.from_numpy(v_cache.copy()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(fixture.weights_i32.copy()).to(torch.int32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)
    hidden_buf = XRTTensor.from_torch(torch.from_numpy(fixture.hidden_i32.copy()).to(torch.int32))

    print("  Running on NPU...")
    result = npu_build.run(handle, [k_cache_buf, v_cache_buf, weights_buf, output_buf, hidden_buf])
    got_k = k_cache_buf.to_torch().numpy().astype(np.int32)
    got_v = v_cache_buf.to_torch().numpy().astype(np.int32)
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {fixture.expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")
    print(f"  expected[-4:]: {fixture.expected[-4:].tolist()}")
    print(f"  got[-4:]:      {got[-4:].tolist()}")

    errors = validate_cache_writeback(
        schedule,
        got_k[: schedule.kv_cache_dwords],
        got_v[: schedule.kv_cache_dwords],
        fixture.packed_weights,
        fixture.hidden_bf16,
    )
    errors.extend(validate_expected_output(fixture.expected, got))
    if errors:
        print(f"  FAIL: {len(errors)} real-qwen3 physical full-layer mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: real Qwen3 Q4NX weights run through the full-layer NPU physical frontier")
    print("  NEXT: replace the remaining physical oracle kernels with production Qwen3 numerics")
    return True
