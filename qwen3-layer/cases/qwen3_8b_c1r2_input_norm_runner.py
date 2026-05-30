"""NPU runner for the real Qwen3 c1r2 input RMSNorm replay boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import npu_build
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

from cases import qwen3_8b_c1r2_input_norm_generate as generate
from cases.qwen3_8b_decode_layer_reference import (
    DEFAULT_LAYER,
    Qwen3LayerReference,
    make_hidden_bf16,
    pack_bf16_i32,
    validate_model_assets,
)
from q4nx_reference import HIDDEN_DWORDS
from qwen3_download import ensure_qwen3_8b_model
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent
MAX_ABS_TOLERANCE = 0.0
MEAN_ABS_TOLERANCE = 0.0


@dataclass(frozen=True)
class C1R2InputNormFixture:
    hidden_i32: np.ndarray
    input_norm_i32: np.ndarray
    expected_i32: np.ndarray


def _model_path(model_path: Path | None) -> Path:
    return DEFAULT_QWEN3_8B_MODEL_PATH if model_path is None else model_path


def _load_model(model_path: Path | None, download_model: bool) -> Qwen3Q4NXModel:
    path = _model_path(model_path)
    ensure_qwen3_8b_model(path, download_model)
    return Qwen3Q4NXModel(path)


def _reject_decode_schedule(current_token: int | None) -> None:
    if current_token is not None:
        raise ValueError(f"--current-token is not used by {CASE_NAME}")


def _validate_model_and_fixture(model: Qwen3Q4NXModel, layer: int) -> C1R2InputNormFixture:
    errors = validate_model_assets(model, layer)
    if errors:
        raise RuntimeError("\n".join(f"Qwen3-8B asset mismatch: {error}" for error in errors))

    reference = Qwen3LayerReference(model, layer)
    hidden = make_hidden_bf16()
    input_norm, _post_norm, _q_norm, _k_norm = model.layer_norm_weights(layer)
    expected = reference.input_norm_activation(hidden)

    hidden_i32 = pack_bf16_i32(hidden)
    input_norm_i32 = pack_bf16_i32(input_norm)
    expected_i32 = pack_bf16_i32(expected)
    if hidden_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"bad hidden dwords: {hidden_i32.shape[0]}")
    if input_norm_i32.shape != (HIDDEN_DWORDS,):
        raise RuntimeError(f"bad input_norm dwords: {input_norm_i32.shape[0]}")
    if expected_i32.shape != (generate.REPLAY_PAYLOAD_DWORDS,):
        raise RuntimeError(f"bad replay payload dwords: {expected_i32.shape[0]}")
    return C1R2InputNormFixture(
        hidden_i32=hidden_i32,
        input_norm_i32=input_norm_i32,
        expected_i32=expected_i32,
    )


def _bf16_values(words: np.ndarray) -> np.ndarray:
    return np.frombuffer(words.astype(np.int32).tobytes(), dtype=bfloat16).astype(np.float32)


def _validate_output(expected: np.ndarray, got: np.ndarray) -> list[str]:
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    expected_values = _bf16_values(expected)
    got_values = _bf16_values(got)
    abs_err = np.abs(expected_values - got_values)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    errors: list[str] = []
    if max_abs > MAX_ABS_TOLERANCE:
        errors.append(f"max_abs={max_abs:.9f} > {MAX_ABS_TOLERANCE:.9f}")
    if mean_abs > MEAN_ABS_TOLERANCE:
        errors.append(f"mean_abs={mean_abs:.9f} > {MEAN_ABS_TOLERANCE:.9f}")
    mismatch = np.where(abs_err > MAX_ABS_TOLERANCE)[0]
    for idx in mismatch[:16]:
        errors.append(
            f"lane {int(idx)} expected={float(expected_values[idx]):.9f} "
            f"got={float(got_values[idx]):.9f} abs={float(abs_err[idx]):.9f}"
        )
    if mismatch.size > 16:
        errors.append(f"{int(mismatch.size - 16)} additional c1r2 replay mismatches")
    return errors


def build_kernel() -> tuple[Path, Path]:
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
            "\n".join(f"  C1R2 INPUT NORM STRUCTURE FAIL: {error}" for error in errors)
        )

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_decode_schedule(current_token)
    mlir_text = generate.generate_mlir()
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        for error in errors:
            print(f"  C1R2 INPUT NORM STRUCTURE FAIL: {error}")
        return False
    model = _load_model(model_path, download_model)
    fixture = _validate_model_and_fixture(model, layer)
    print(f"  PASS: {CASE_NAME} MLIR isolates c1r2 input RMSNorm replay")
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  hidden_dwords={fixture.hidden_i32.shape[0]}")
    print(f"  input_norm_dwords={fixture.input_norm_i32.shape[0]}")
    print(f"  output_dwords={fixture.expected_i32.shape[0]}")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_decode_schedule(current_token)
    model = _load_model(model_path, download_model)
    _validate_model_and_fixture(model, layer)
    xclbin_path, insts_path = build_kernel()
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    _reject_decode_schedule(current_token)
    model = _load_model(model_path, download_model)
    fixture = _validate_model_and_fixture(model, layer)

    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  model={_model_path(model_path)}")
    print(f"  layer={layer}")
    print("  input=host hidden + real input_layernorm.weight")
    print("  output=c1r2 full_c1r2_make_input_norm_replay payload")
    print()

    xclbin_path, insts_path = build_kernel()

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    hidden_buf = XRTTensor.from_torch(torch.from_numpy(fixture.hidden_i32.copy()).to(torch.int32))
    norm_buf = XRTTensor.from_torch(torch.from_numpy(fixture.input_norm_i32.copy()).to(torch.int32))
    output_buf = XRTTensor((generate.REPLAY_PAYLOAD_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [hidden_buf, norm_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    expected_values = _bf16_values(fixture.expected_i32)
    got_values = _bf16_values(got)
    abs_err = np.abs(expected_values - got_values)
    mismatch_count = int(np.flatnonzero(abs_err > MAX_ABS_TOLERANCE).size)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {fixture.expected_i32[:8].tolist()}")
    print(f"  got[0:8]:      {got[:8].tolist()}")
    print(f"  max_abs={float(np.max(abs_err)):.9f}")
    print(f"  mean_abs={float(np.mean(abs_err)):.9f}")
    print(
        "  stage_budget: "
        f"c1r2_input_norm: max_abs={float(np.max(abs_err)):.9f} "
        f"mean_abs={float(np.mean(abs_err)):.9f} mismatches={mismatch_count} "
        f"abs_tol={MAX_ABS_TOLERANCE:.9f} rel_tol=0.000000"
    )

    errors = _validate_output(fixture.expected_i32, got)
    if errors:
        print(f"  FAIL: {len(errors)} c1r2 input RMSNorm replay mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: c1r2 input RMSNorm replay matches the real Qwen3 CPU reference")
    return True
