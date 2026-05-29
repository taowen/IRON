"""NPU runner for the real Qwen3-8B Q/K/V body handoff boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from cases import q4nx_qkv_body_post_generate as generate
from cases.q4nx_qkv_body_post_reference import (
    CURRENT_DWORDS,
    HIDDEN_DWORDS,
    OUTPUT_DWORDS,
    TOTAL_WEIGHT_I32,
    expected_output,
    k_body_payload,
    q_body_payload,
    validate_output,
    v_body_payload,
)
from cases.qwen3_8b_decode_layer_reference import (
    DEFAULT_LAYER,
    K_PROJECTION,
    Q_PROJECTION,
    Qwen3LayerReference,
    V_PROJECTION,
    make_hidden_bf16,
    validate_model_assets,
)
from q4nx_reference import packed_as_i32
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel

CASE_NAME = "qwen3-8b-qkv-body-post-bridge"
EXPERIMENT_DIR = Path(__file__).parent.parent


@dataclass(frozen=True)
class QKVBoundaryFixture:
    activation_i32: np.ndarray
    weights_i32: np.ndarray
    expected: np.ndarray
    weight_bytes: int


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _reject_decode_schedule(current_token: int | None, patch_from_token: int | None) -> None:
    if current_token is not None:
        raise ValueError("--current-token is not used by qwen3-8b-qkv-body-post-bridge")
    if patch_from_token is not None:
        raise ValueError("--patch-from-token is not used by qwen3-8b-qkv-body-post-bridge")


def _pack_bf16_i32(values: np.ndarray) -> np.ndarray:
    if values.shape != (HIDDEN_DWORDS * 2,):
        raise ValueError(f"activation shape mismatch: {values.shape}")
    return np.frombuffer(values.astype(bfloat16).tobytes(), dtype=np.int32).copy()


def _payload_mismatch_errors(label: str, payload: np.ndarray, direct: np.ndarray) -> list[str]:
    payload_raw = np.frombuffer(payload.tobytes(), dtype=np.uint16)
    direct_raw = np.frombuffer(direct.astype(bfloat16).tobytes(), dtype=np.uint16)
    if payload_raw.shape != direct_raw.shape:
        return [f"{label} shape mismatch: {payload_raw.shape} != {direct_raw.shape}"]
    mismatch = np.where(payload_raw != direct_raw)[0]
    if mismatch.size == 0:
        return []
    payload_values = np.frombuffer(payload.tobytes(), dtype=bfloat16).astype(np.float32)
    direct_values = direct.astype(np.float32)
    errors: list[str] = []
    for idx in mismatch[:8]:
        errors.append(
            f"{label}[{int(idx)}]: payload={float(payload_values[idx]):.6f} "
            f"direct={float(direct_values[idx]):.6f}"
        )
    if mismatch.size > 8:
        errors.append(f"{label}: {mismatch.size - 8} additional payload/direct mismatches")
    return errors


def _validate_stream_semantics(
    reference: Qwen3LayerReference,
    activation: np.ndarray,
    packed: np.ndarray,
) -> tuple[str, ...]:
    errors: list[str] = []
    errors.extend(
        _payload_mismatch_errors(
            Q_PROJECTION,
            q_body_payload(packed, activation),
            reference.project(Q_PROJECTION, activation),
        )
    )
    errors.extend(
        _payload_mismatch_errors(
            K_PROJECTION,
            k_body_payload(packed, activation),
            reference.project(K_PROJECTION, activation),
        )
    )
    errors.extend(
        _payload_mismatch_errors(
            V_PROJECTION,
            v_body_payload(packed, activation),
            reference.project(V_PROJECTION, activation),
        )
    )
    return tuple(errors)


def _make_fixture(model: Qwen3Q4NXModel, layer: int) -> QKVBoundaryFixture:
    reference = Qwen3LayerReference(model, layer)
    activation = reference.input_norm_activation(make_hidden_bf16())
    packed = model.layer_qkv_body_weight_stream(layer)
    expected_weight_bytes = TOTAL_WEIGHT_I32 * 4
    if packed.shape != (expected_weight_bytes,):
        raise ValueError(
            f"Q/K/V weight stream bytes mismatch: {packed.shape[0]} != {expected_weight_bytes}"
        )
    semantic_errors = _validate_stream_semantics(reference, activation, packed)
    if semantic_errors:
        raise RuntimeError(
            "\n".join(f"real Q/K/V stream semantic mismatch: {error}" for error in semantic_errors)
        )
    return QKVBoundaryFixture(
        activation_i32=_pack_bf16_i32(activation),
        weights_i32=packed_as_i32(packed),
        expected=expected_output(packed, activation),
        weight_bytes=packed.shape[0],
    )


def _validate_model_and_fixture(model: Qwen3Q4NXModel, layer: int) -> QKVBoundaryFixture:
    errors = validate_model_assets(model, layer)
    if errors:
        raise RuntimeError("\n".join(f"Qwen3-8B asset mismatch: {error}" for error in errors))
    fixture = _make_fixture(model, layer)
    if fixture.activation_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"bad activation shape: {fixture.activation_i32.shape[0]}")
    if fixture.weights_i32.shape != (TOTAL_WEIGHT_I32,):
        raise RuntimeError(f"bad weight shape: {fixture.weights_i32.shape[0]}")
    if fixture.expected.shape != (OUTPUT_DWORDS,):
        raise RuntimeError(f"bad expected shape: {fixture.expected.shape[0]}")
    return fixture


def build_kernel() -> tuple[Path, Path]:
    import npu_build

    build_dir = EXPERIMENT_DIR / "build" / CASE_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError(
            "\n".join(f"  REAL QWEN3 QKV BODY POST STRUCTURE FAIL: {error}" for error in errors)
        )

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_decode_schedule(current_token, patch_from_token)
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  REAL QWEN3 QKV BODY POST STRUCTURE FAIL: {error}")
        return False
    model = _load_model(model_path, download_model)
    fixture = _validate_model_and_fixture(model, layer)
    print(f"  PASS: {CASE_NAME} MLIR reuses the Q4NX Q/K/V body-post topology")
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  activation_dwords={fixture.activation_i32.shape[0]}")
    print(f"  qkv_weight_stream_bytes={fixture.weight_bytes}")
    print(
        f"  output_dwords={fixture.expected.shape[0]} = "
        f"Q 2048 + K {CURRENT_DWORDS} + V {CURRENT_DWORDS}"
    )
    return True


def build_only(
    current_token: int | None = None,
    patch_from_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_decode_schedule(current_token, patch_from_token)
    model = _load_model(model_path, download_model)
    _validate_model_and_fixture(model, layer)
    xclbin_path, insts_path = build_kernel()
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
    import npu_build
    import torch
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    _reject_decode_schedule(current_token, patch_from_token)
    model = _load_model(model_path, download_model)
    fixture = _validate_model_and_fixture(model, layer)

    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print("  activation=CPU RMSNorm(hidden_in) bf16, then c1r2 full-vector replay")
    print("  weights=real Qwen3 Q/K/V Q4NX chunks on row1 S2MM4/5 -> main16 DMA1")
    print("  output=c1r3 Q body payload + current K/V packed layout")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    activation_buf = XRTTensor.from_torch(
        torch.from_numpy(fixture.activation_i32.copy()).to(torch.int32)
    )
    weight_buf = XRTTensor.from_torch(
        torch.from_numpy(fixture.weights_i32.copy()).to(torch.int32)
    )
    output_buf = XRTTensor((OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [activation_buf, weight_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {fixture.expected[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")
    print(f"  expected[-8:]: {fixture.expected[-8:].tolist()}")
    print(f"  got[-8:]:      {got[-8:].tolist()}")

    errors = validate_output(fixture.expected, got)
    if errors:
        print(f"  FAIL: {len(errors)} real Qwen3 Q/K/V boundary mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: real Qwen3 Q/K/V Q4NX projection reached c1r3 current-layout output on NPU")
    return True
