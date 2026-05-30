"""Runner for the real Qwen3-8B single-layer numerical target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import full_layer_engine_generate as generate
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    FULL_PIPELINE_ABS_TOL,
    FULL_PIPELINE_REL_TOL,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
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
    validate_expected_output,
)
from cases.decode_cache_reference import (
    DecodeSchedule,
    make_decode_schedule,
    validate_cache_layout_contract,
)
from cases.decode_instruction_patch import patch_instruction_stream
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel
from cases.qwen3_8b_decode_layer_reference import (
    CASE_NAME,
    DEFAULT_CURRENT_TOKEN,
    DEFAULT_LAYER,
    Qwen3LayerReference,
    bf16_compare_stats,
    format_bf16_compare_stats,
    make_reference_inputs,
    validate_model_assets,
)
from resource_manifest import write_resource_manifest

EXPERIMENT_DIR = Path(__file__).parent.parent
CAPACITY_TOKEN = 127


@dataclass(frozen=True)
class Qwen3PhysicalFixture:
    k_cache_i32: np.ndarray
    v_cache_i32: np.ndarray
    hidden_i32: np.ndarray
    qkv_activation_bf16: np.ndarray
    k_norm_bf16: np.ndarray
    packed_weights: np.ndarray
    weights_i32: np.ndarray
    weight_bytes: int
    expected: np.ndarray
    rope_theta: float


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _target_schedule(current_token: int | None) -> DecodeSchedule:
    return make_decode_schedule(DEFAULT_CURRENT_TOKEN if current_token is None else current_token)


def _build_schedule(target_schedule: DecodeSchedule) -> DecodeSchedule:
    return make_decode_schedule(max(target_schedule.current_token, CAPACITY_TOKEN))


def _capacity_build_name(schedule: DecodeSchedule) -> str:
    return f"{CASE_NAME}-capacity-token{schedule.current_token}"


def _patch_capacity_instructions(
    insts_path: Path,
    build_schedule: DecodeSchedule,
    target_schedule: DecodeSchedule,
) -> tuple[Path, list[str]]:
    if build_schedule.current_token == target_schedule.current_token:
        return insts_path, []
    target_path = insts_path.with_name(
        f"design-token{build_schedule.current_token}-to-token{target_schedule.current_token}.bin"
    )
    return target_path, patch_instruction_stream(insts_path, target_path, build_schedule, target_schedule)


def _cache_buffer_payload(payload: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    if payload.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"KV cache shape mismatch: {payload.shape} != {(schedule.kv_cache_dwords,)}")
    return payload


def _padded_cache(
    values: np.ndarray,
    target_schedule: DecodeSchedule,
    buffer_schedule: DecodeSchedule,
) -> np.ndarray:
    expected = (target_schedule.current_token + 1, 8, 128)
    if values.shape != expected:
        raise ValueError(f"reference cache shape mismatch: {values.shape} != {expected}")
    if target_schedule.current_token > buffer_schedule.current_token:
        raise ValueError(
            f"target token{target_schedule.current_token} exceeds buffer token{buffer_schedule.current_token}"
        )
    padded = np.zeros((buffer_schedule.total_context, 8, 128), dtype=values.dtype)
    padded[: target_schedule.current_token + 1] = values
    return padded


def _poison_current_cache(
    values: np.ndarray,
    target_schedule: DecodeSchedule,
    buffer_schedule: DecodeSchedule,
    poison: float,
) -> np.ndarray:
    poisoned = _padded_cache(values, target_schedule, buffer_schedule)
    poisoned[target_schedule.current_token, :, :] = np.array(poison, dtype=values.dtype)
    return poisoned


def _make_physical_fixture(
    model: Qwen3Q4NXModel,
    layer: int,
    target_schedule: DecodeSchedule,
    buffer_schedule: DecodeSchedule,
) -> Qwen3PhysicalFixture:
    inputs = make_reference_inputs(target_schedule.current_token)
    input_norm, post_norm, q_norm, k_norm = model.layer_norm_weights(layer)
    qkv_activation = physical_input_norm_activation(inputs.hidden, input_norm)
    packed = model.layer_weight_stream(layer)
    aux_i32 = aux_as_i32(
        target_schedule.current_token,
        input_norm,
        post_norm,
        q_norm,
        k_norm,
        model.config.rope_theta,
    )
    weights_i32 = np.concatenate((aux_i32, packed_as_i32(packed))).astype(np.int32)
    reference = Qwen3LayerReference(model, layer)
    result = reference.forward(inputs)
    initial_k_cache = bf16_cache_payload(
        buffer_schedule,
        _poison_current_cache(inputs.k_cache, target_schedule, buffer_schedule, 19.0),
    )
    initial_v_cache = bf16_cache_payload(
        buffer_schedule,
        _poison_current_cache(inputs.v_cache, target_schedule, buffer_schedule, -19.0),
    )
    return Qwen3PhysicalFixture(
        k_cache_i32=initial_k_cache,
        v_cache_i32=initial_v_cache,
        hidden_i32=hidden_input_as_i32(inputs.hidden),
        qkv_activation_bf16=qkv_activation,
        k_norm_bf16=k_norm,
        packed_weights=packed,
        weights_i32=weights_i32,
        weight_bytes=packed.shape[0],
        expected=result.hidden_out_i32,
        rope_theta=model.config.rope_theta,
    )


def _validate_real_physical_inputs(model: Qwen3Q4NXModel, layer: int, schedule: DecodeSchedule) -> tuple[int, int, int]:
    hidden = make_reference_inputs(schedule.current_token).hidden
    input_norm, post_norm, q_norm, k_norm = model.layer_norm_weights(layer)
    hidden_i32 = hidden_input_as_i32(hidden)
    aux_i32 = aux_as_i32(schedule.current_token, input_norm, post_norm, q_norm, k_norm, model.config.rope_theta)
    weight_stream = model.layer_weight_stream(layer)
    weights_i32 = np.concatenate((aux_i32, packed_as_i32(weight_stream))).astype(np.int32)
    if hidden_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"hidden i32 shape mismatch: {hidden_i32.shape} != {(HIDDEN_DWORDS,)}")
    if aux_i32.shape != (AUX_DWORDS,):
        raise RuntimeError(f"aux i32 shape mismatch: {aux_i32.shape} != {(AUX_DWORDS,)}")
    if weights_i32.shape != (TOTAL_WEIGHT_AND_AUX_I32,):
        raise RuntimeError(
            f"aux-prefixed weight i32 shape mismatch: {weights_i32.shape} != {(TOTAL_WEIGHT_AND_AUX_I32,)}"
        )
    return weight_stream.shape[0], hidden_i32.shape[0], aux_i32.shape[0]


def build_kernel(schedule: DecodeSchedule, build_name: str = CASE_NAME) -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    manifest_path = build_dir / "resource_manifest.json"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir(schedule)
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text, schedule)
    if errors:
        raise RuntimeError("\n".join(f"  REAL QWEN3 FULL-LAYER STRUCTURE FAIL: {error}" for error in errors))
    write_resource_manifest(manifest_path, generate.resource_manifest_for_case(CASE_NAME))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    target_schedule = _target_schedule(current_token)
    build_schedule = _build_schedule(target_schedule)
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    mlir_text = generate.generate_mlir(build_schedule)
    errors.extend(generate.validate_generated_mlir(mlir_text, build_schedule))
    errors.extend(validate_cache_layout_contract(target_schedule))
    errors.extend(validate_cache_layout_contract(build_schedule))
    if target_schedule.kv_blocks > build_schedule.kv_blocks:
        errors.append(
            f"target token{target_schedule.current_token} needs {target_schedule.kv_blocks} blocks, "
            f"but build token{build_schedule.current_token} reserves {build_schedule.kv_blocks}"
        )
    if errors:
        for error in errors:
            print(f"  QWEN3-8B FULL-LAYER FAIL: {error}")
        return False
    weight_bytes, hidden_dwords, aux_dwords = _validate_real_physical_inputs(model, layer, target_schedule)
    print(f"  PASS: {CASE_NAME} assets valid for layer {layer}")
    print(f"  weight_stream_bytes={weight_bytes}")
    print(f"  hidden_dwords={hidden_dwords}")
    print(f"  aux_prefix_dwords={aux_dwords}")
    print(f"  current_token={target_schedule.current_token}")
    print(f"  capacity_token={build_schedule.current_token}")
    print(f"  capacity_kv_cache_dwords={build_schedule.kv_cache_dwords}")
    print("  PASS: real Qwen3 weights fit the current full-layer NPU topology")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    target_schedule = _target_schedule(current_token)
    build_schedule = _build_schedule(target_schedule)
    model = _load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3-8B ASSET FAIL: {error}")
        return False
    _validate_real_physical_inputs(model, layer, target_schedule)
    xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))
    runtime_insts_path, patch_changes = _patch_capacity_instructions(insts_path, build_schedule, target_schedule)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    if patch_changes:
        print(f"  PASS: patched {runtime_insts_path}")
        for change in patch_changes:
            print(f"    {change}")
    return True


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    target_schedule = _target_schedule(current_token)
    build_schedule = _build_schedule(target_schedule)
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
    print(f"  current_token={target_schedule.current_token}")
    print(f"  capacity_token={build_schedule.current_token}")
    print("  topology=current full-layer NPU frontier with real Qwen3 Q4NX weights")
    print("  numerics=c1r2 RMSNorm/residual + c1r3 Q/K norm/RoPE + bf16 KV scan attention")
    print()

    xclbin_path, insts_path = build_kernel(build_schedule, _capacity_build_name(build_schedule))
    runtime_insts_path, patch_changes = _patch_capacity_instructions(insts_path, build_schedule, target_schedule)
    if patch_changes:
        print(f"  Runtime instruction stream: {runtime_insts_path.name}")
        for change in patch_changes:
            print(f"    {change}")
    else:
        print(f"  Runtime instruction stream: {runtime_insts_path.name}")
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, runtime_insts_path)

    print("  Preparing raw hidden, aux weights, K/V cache, real Q4NX weights, and CPU oracle...")
    fixture = _make_physical_fixture(model, layer, target_schedule, build_schedule)
    k_cache = _cache_buffer_payload(fixture.k_cache_i32, build_schedule)
    v_cache = _cache_buffer_payload(fixture.v_cache_i32, build_schedule)
    if fixture.hidden_i32.shape[0] != HIDDEN_DWORDS:
        raise RuntimeError(f"hidden i32 mismatch: {fixture.hidden_i32.shape[0]} != {HIDDEN_DWORDS}")
    if fixture.weights_i32.shape[0] != TOTAL_WEIGHT_AND_AUX_I32:
        raise RuntimeError(
            f"aux-prefixed weight i32 mismatch: {fixture.weights_i32.shape[0]} != {TOTAL_WEIGHT_AND_AUX_I32}"
        )

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
    final_stats = bf16_compare_stats(
        "final_hidden_out",
        fixture.expected,
        got,
        FULL_PIPELINE_ABS_TOL,
        FULL_PIPELINE_REL_TOL,
    )
    expected_cache = expected_cache_writeback(
        target_schedule,
        fixture.packed_weights,
        fixture.qkv_activation_bf16,
        fixture.k_norm_bf16,
        fixture.rope_theta,
        k_cache,
        v_cache,
    )
    cache_stats = cache_writeback_stats(
        target_schedule,
        got_k,
        got_v,
        expected_cache,
    )
    print("  " + format_bf16_compare_stats(final_stats))
    print("  stage_budget: " + format_bf16_compare_stats(final_stats))
    for stats in cache_stats:
        print("  stage_budget: " + format_stage_stats(stats))

    errors = validate_cache_writeback(
        target_schedule,
        got_k,
        got_v,
        expected_cache,
    )
    errors.extend(validate_expected_output(fixture.expected, got))
    if errors:
        print(f"  FAIL: {len(errors)} real-qwen3 full-layer mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: real Qwen3 Q4NX weights run through the full-layer NPU frontier")
    return True
