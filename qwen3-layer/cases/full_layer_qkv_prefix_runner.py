"""NPU runner for the full-layer physical Q/K/V prefix slice."""

from __future__ import annotations

from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import full_layer_qkv_prefix_generate as generate
from cases.full_layer_engine_reference import (
    HIDDEN_DWORDS,
    TOTAL_WEIGHT_AND_AUX_I32,
    aux_as_i32,
    hidden_input_as_i32,
    input_norm_activation,
    make_hidden_bf16,
    make_history_k_cache_payload_bf16,
    make_history_v_cache_payload_bf16,
    make_packed_weights,
    packed_as_i32,
    route_summary,
    validate_cache_writeback,
)
from cases.currentkv_kvscan_attention_kv16_reference import (
    DecodeSchedule,
    make_decode_schedule,
    validate_cache_layout_contract,
)
from resource_manifest import write_resource_manifest

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent


def _reject_model_options(model_path: Path | None, layer: int, download_model: bool) -> None:
    if model_path is not None or download_model:
        raise ValueError(f"model options are not used by {CASE_NAME}")
    if layer != 0:
        raise ValueError(f"--layer is not used by {CASE_NAME}")


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
        raise RuntimeError("\n".join(f"  FULL-LAYER QKV PREFIX STRUCTURE FAIL: {error}" for error in errors))
    write_resource_manifest(manifest_path, generate.resource_manifest(schedule))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_model_options(model_path, layer, download_model)
    schedule = make_decode_schedule(current_token)
    errors = generate.validate_generated_mlir(generate.generate_mlir(schedule), schedule)
    errors.extend(validate_cache_layout_contract(schedule))
    if errors:
        for error in errors:
            print(f"  FULL-LAYER QKV PREFIX FAIL: {error}")
        return False
    print("  PASS: full-layer-qkv-prefix uses the full-layer physical prefix ABI")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_model_options(model_path, layer, download_model)
    xclbin_path, insts_path = build_kernel(make_decode_schedule(current_token))
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def _cache_buffer_payload(payload: np.ndarray, schedule: DecodeSchedule) -> np.ndarray:
    if payload.shape != (schedule.kv_cache_dwords,):
        raise ValueError(f"KV cache shape mismatch: {payload.shape} != {(schedule.kv_cache_dwords,)}")
    return payload


def _print_prefix_errors(errors: list[str]) -> None:
    for error in errors:
        if error.startswith("K cache"):
            print(f"    edge=current_k_writeback packet8 post->shim_left: {error}")
        elif error.startswith("V cache"):
            print(f"    edge=current_v_writeback packet9 post->shim_right: {error}")
        else:
            print(f"    {error}")


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_model_options(model_path, layer, download_model)
    schedule = make_decode_schedule(current_token)
    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    for line in route_summary(schedule):
        print(f"  {line}")
    print("  slice=full c1r2/full -> bridge -> main16 -> row1 weight -> c1r3 -> packet8/9")
    print("  tail=KV scan/attention/O/up/gate/down not started")
    print()

    layout_errors = validate_cache_layout_contract(schedule)
    if layout_errors:
        raise RuntimeError("\n".join(f"  FULL-LAYER QKV PREFIX CACHE LAYOUT FAIL: {error}" for error in layout_errors))

    xclbin_path, insts_path = build_kernel(schedule)
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    print("  Preparing synthetic hidden, aux-prefixed full weights, and K/V cache...")
    k_cache = _cache_buffer_payload(make_history_k_cache_payload_bf16(schedule), schedule)
    v_cache = _cache_buffer_payload(make_history_v_cache_payload_bf16(schedule), schedule)
    packed = make_packed_weights()
    hidden_bf16 = make_hidden_bf16()
    input_norm_weight = np.ones(hidden_bf16.shape, dtype=hidden_bf16.dtype)
    post_norm_weight = np.ones(hidden_bf16.shape, dtype=hidden_bf16.dtype)
    hidden = hidden_input_as_i32(hidden_bf16)
    aux = aux_as_i32(schedule.current_token, input_norm_weight, post_norm_weight)
    weights_i32 = np.concatenate((aux, packed_as_i32(packed))).astype(np.int32)
    qkv_activation = input_norm_activation(hidden_bf16, input_norm_weight)
    if hidden.shape[0] != HIDDEN_DWORDS:
        raise RuntimeError(f"hidden i32 mismatch: {hidden.shape[0]} != {HIDDEN_DWORDS}")
    if weights_i32.shape[0] != TOTAL_WEIGHT_AND_AUX_I32:
        raise RuntimeError(
            f"aux-prefixed weight i32 mismatch: {weights_i32.shape[0]} != {TOTAL_WEIGHT_AND_AUX_I32}"
        )

    k_cache_buf = XRTTensor.from_torch(torch.from_numpy(k_cache.copy()).to(torch.int32))
    v_cache_buf = XRTTensor.from_torch(torch.from_numpy(v_cache.copy()).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights_i32.copy()).to(torch.int32))
    hidden_buf = XRTTensor.from_torch(torch.from_numpy(hidden.copy()).to(torch.int32))

    print("  Running on NPU...")
    result = npu_build.run(handle, [k_cache_buf, v_cache_buf, weights_buf, hidden_buf])
    got_k = k_cache_buf.to_torch().numpy().astype(np.int32)
    got_v = v_cache_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")

    errors = validate_cache_writeback(
        schedule,
        got_k[: schedule.kv_cache_dwords],
        got_v[: schedule.kv_cache_dwords],
        packed,
        qkv_activation,
        None,
        1_000_000.0,
        k_cache,
        v_cache,
    )
    if errors:
        print(f"  FAIL: {len(errors)} full-layer qkv-prefix cache mismatches")
        _print_prefix_errors(errors)
        return False

    print("  PASS: full-layer physical Q/K/V prefix writes current K/V correctly")
    return True
