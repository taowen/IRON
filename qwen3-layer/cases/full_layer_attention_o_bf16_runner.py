"""NPU runner for the full-layer bf16 attention -> O slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import full_layer_attention_o_bf16_generate as generate
from cases.full_layer_engine_reference import (
    AUX_DWORDS,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
    aux_as_i32,
    bf16_cache_payload,
    cache_writeback_stats,
    expected_cache_writeback,
    format_stage_stats,
    hidden_input_as_i32,
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
    O_PROJECTION,
    Qwen3LayerReference,
    attention_bf16_npu,
    bf16_compare_stats,
    format_bf16_compare_stats,
    make_reference_inputs,
    pack_bf16_i32,
    validate_model_assets,
)
from ml_dtypes import bfloat16
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel
from resource_manifest import write_resource_manifest

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent
O_ABS_TOL = 0.01
O_REL_TOL = 0.05


@dataclass(frozen=True)
class AttentionOFixture:
    k_cache_i32: np.ndarray
    v_cache_i32: np.ndarray
    hidden_i32: np.ndarray
    qkv_activation_bf16: np.ndarray
    k_norm_bf16: np.ndarray
    packed_weights: np.ndarray
    weights_i32: np.ndarray
    expected_o_i32: np.ndarray
    rope_theta: float


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _schedule(current_token: int | None) -> DecodeSchedule:
    return make_decode_schedule(DEFAULT_CURRENT_TOKEN if current_token is None else current_token)


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
        raise RuntimeError("\n".join(f"  FULL-LAYER ATTENTION-O BF16 STRUCTURE FAIL: {error}" for error in errors))
    write_resource_manifest(manifest_path, generate.resource_manifest(schedule))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = _schedule(current_token)
    model = _load_model(model_path, download_model)
    errors = generate.validate_generated_mlir(generate.generate_mlir(schedule), schedule)
    errors.extend(validate_cache_layout_contract(schedule))
    errors.extend(validate_model_assets(model, layer))
    if errors:
        for error in errors:
            print(f"  FULL-LAYER ATTENTION-O BF16 FAIL: {error}")
        return False
    print("  PASS: full-layer-attention-o-bf16 uses production qwen3_attention_bf16")
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = _schedule(current_token)
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


def _cache_buffer_payload(payload: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    if payload.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"KV cache shape mismatch: {payload.shape} != {(schedule.kv_cache_dwords,)}")
    return payload


def _padded_cache(values: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    expected = (schedule.current_token + 1, 8, 128)
    if values.shape != expected:
        raise ValueError(f"reference cache shape mismatch: {values.shape} != {expected}")
    padded = np.zeros((schedule.total_context, 8, 128), dtype=bfloat16)
    padded[: schedule.current_token + 1] = values
    return padded


def _poison_current_cache(values: np.ndarray, schedule: DecodeSchedule, poison: float) -> np.ndarray:
    poisoned = _padded_cache(values, schedule)
    poisoned[schedule.current_token, :, :] = np.array(poison, dtype=poisoned.dtype)
    return poisoned


def _make_fixture(
    model: Qwen3Q4NXModel,
    layer: int,
    schedule: DecodeSchedule,
) -> AttentionOFixture:
    inputs = make_reference_inputs(schedule.current_token)
    input_norm, post_norm, q_norm, k_norm = model.layer_norm_weights(layer)
    reference = Qwen3LayerReference(model, layer)
    result = reference.forward(inputs)
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
    if weights.shape != (TOTAL_WEIGHT_AND_AUX_I32,):
        raise RuntimeError(f"aux-prefixed weight i32 mismatch: {weights.shape} != {(TOTAL_WEIGHT_AND_AUX_I32,)}")
    return AttentionOFixture(
        k_cache_i32=bf16_cache_payload(schedule, _poison_current_cache(inputs.k_cache, schedule, 19.0)),
        v_cache_i32=bf16_cache_payload(schedule, _poison_current_cache(inputs.v_cache, schedule, -19.0)),
        hidden_i32=hidden_input_as_i32(inputs.hidden),
        qkv_activation_bf16=result.input_norm,
        k_norm_bf16=k_norm,
        packed_weights=packed,
        weights_i32=weights,
        expected_o_i32=pack_bf16_i32(
            reference.project(
                O_PROJECTION,
                attention_bf16_npu(result.q, result.k_cache, result.v_cache, schedule.current_token),
            )
        ),
        rope_theta=model.config.rope_theta,
    )


def _validate_o_output(expected: np.ndarray, got: np.ndarray) -> list[str]:
    if got.shape != expected.shape:
        return [f"O output shape mismatch: {got.shape} != {expected.shape}"]
    errors: list[str] = []
    expected_values = np.frombuffer(expected.tobytes(), dtype=bfloat16).astype(np.float32)
    got_values = np.frombuffer(got.tobytes(), dtype=bfloat16).astype(np.float32)
    expected_bad = np.flatnonzero(~np.isfinite(expected_values))
    got_bad = np.flatnonzero(~np.isfinite(got_values))
    for lane in expected_bad[:16]:
        errors.append(f"O expected lane {int(lane)} is not finite: {float(expected_values[lane])}")
    for lane in got_bad[:16]:
        errors.append(f"O got lane {int(lane)} is not finite: {float(got_values[lane])}")
    if errors:
        return errors
    abs_err = np.abs(expected_values - got_values)
    rel_err = abs_err / np.maximum(np.abs(expected_values), 1e-6)
    limit = np.maximum(O_ABS_TOL, O_REL_TOL * np.abs(expected_values))
    mismatch = np.flatnonzero(abs_err > limit)
    for lane in mismatch[:32]:
        errors.append(
            f"O lane {int(lane)}: expected={float(expected_values[lane]):.6f} "
            f"got={float(got_values[lane]):.6f} abs={float(abs_err[lane]):.6f} "
            f"rel={float(rel_err[lane]):.6f} limit={float(limit[lane]):.6f}"
        )
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional O lane mismatches")
    return errors


def _print_slice_errors(errors: list[str]) -> None:
    for error in errors:
        if error.startswith("K cache"):
            print(f"    edge=current_k_writeback packet8 post->shim_left phase=QKV-prefix: {error}")
        elif error.startswith("V cache"):
            print(f"    edge=current_v_writeback packet9 post->shim_right phase=QKV-prefix: {error}")
        elif error.startswith("O lane"):
            print(f"    edge=O compact bridge->full_vector->host phase=attention_o: {error}")
        else:
            print(f"    {error}")


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = _schedule(current_token)
    model = _load_model(model_path, download_model)
    asset_errors = validate_model_assets(model, layer)
    if asset_errors:
        for error in asset_errors:
            print(f"  QWEN3 ASSET FAIL: {error}")
        return False
    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  case={CASE_NAME}")
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print("  closed_loop=host hidden -> c1r2 input RMSNorm replay -> main16 Q4NX Q/K/V -> c1r3 Q/K norm+RoPE -> current K/V writeback")
    print("  attention=KV scan -> qwen3_attention_bf16_* -> packet2 -> c1r1 bridge -> main16 Q4NX O")
    print("  output=O compact records -> c1r2 host-tapped O vector vs bf16 attention contract")
    print(f"  decode_token={schedule.current_token}, blocks={schedule.kv_blocks}, tail={schedule.tail_tokens}")
    print(f"  hidden={HIDDEN_DWORDS} dwords, aux={AUX_DWORDS} dwords, host_output={OUTPUT_DWORDS} dwords")
    print("  tail=postnorm/up/gate/SwiGLU/down not started")
    print()

    layout_errors = validate_cache_layout_contract(schedule)
    if layout_errors:
        raise RuntimeError("\n".join(f"  FULL-LAYER ATTENTION-O BF16 CACHE LAYOUT FAIL: {error}" for error in layout_errors))

    xclbin_path, insts_path = build_kernel(schedule)
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    print("  Preparing real Qwen3 hidden, aux-prefixed weights, K/V cache, and CPU oracle...")
    fixture = _make_fixture(model, layer, schedule)
    k_cache = _cache_buffer_payload(fixture.k_cache_i32, schedule)
    v_cache = _cache_buffer_payload(fixture.v_cache_i32, schedule)
    hidden = fixture.hidden_i32
    weights_i32 = fixture.weights_i32
    expected = fixture.expected_o_i32
    if hidden.shape[0] != HIDDEN_DWORDS:
        raise RuntimeError(f"hidden i32 mismatch: {hidden.shape[0]} != {HIDDEN_DWORDS}")
    if weights_i32.shape[0] != TOTAL_WEIGHT_AND_AUX_I32:
        raise RuntimeError(
            f"aux-prefixed weight i32 mismatch: {weights_i32.shape[0]} != {TOTAL_WEIGHT_AND_AUX_I32}"
        )
    if expected.shape[0] != OUTPUT_DWORDS:
        raise RuntimeError(f"expected O output mismatch: {expected.shape[0]} != {OUTPUT_DWORDS}")

    k_cache_buf = XRTTensor.from_torch(torch.from_numpy(k_cache.copy()).to(torch.int32))
    v_cache_buf = XRTTensor.from_torch(torch.from_numpy(v_cache.copy()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights_i32.copy()).to(torch.int32))
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)
    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden.copy()).to(torch.int32))

    print("  Running on NPU...")
    result = npu_build.run(handle, [k_cache_buf, v_cache_buf, weights_buf, output_buf, hidden_buf])
    got_k = k_cache_buf.to_torch().numpy().astype(np.int32)
    got_v = v_cache_buf.to_torch().numpy().astype(np.int32)
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")
    print(f"  expected[-4:]: {expected[-4:].tolist()}")
    print(f"  got[-4:]:      {got[-4:].tolist()}")
    o_stats = bf16_compare_stats("attention_o", expected, got, O_ABS_TOL, O_REL_TOL)
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
    print(f"  {format_bf16_compare_stats(o_stats)}")
    print(f"  stage_budget: {format_bf16_compare_stats(o_stats)}")
    for stats in cache_stats:
        print(f"  stage_budget: {format_stage_stats(stats)}")

    errors = validate_cache_writeback(
        schedule,
        got_k[: schedule.kv_cache_dwords],
        got_v[: schedule.kv_cache_dwords],
        expected_cache,
    )
    errors.extend(_validate_o_output(expected, got))
    if errors:
        print(f"  FAIL: {len(errors)} full-layer attention-o bf16 mismatches")
        _print_slice_errors(errors)
        return False

    print("  PASS: production bf16 attention feeds Q4NX O and host-tapped O vector correctly")
    return True
