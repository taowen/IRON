"""NPU runner for the Qwen3 QKV compact-output integration slice."""

from __future__ import annotations

from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

from cases import qwen3_8b_qkv_cache_write_runner as qkv_runner
from cases import qwen3_8b_qkv_compact_output_generate as generate
from cases.full_layer_engine_reference import k_body_compact, q_body_compact, v_body_compact
from cases.decode_cache_reference import make_decode_schedule
from cases.qwen3_8b_decode_layer_reference import DEFAULT_LAYER, validate_model_assets

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent
MAX_PAYLOAD_BF16_ULP = 1


def build_kernel(schedule, build_name: str = CASE_NAME) -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir(schedule)
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text, schedule)
    if errors:
        raise RuntimeError("\n".join(f"  QWEN3 QKV COMPACT-OUTPUT STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(qkv_runner._decode_token(current_token))
    model = qkv_runner._load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    errors.extend(generate.validate_generated_mlir(generate.generate_mlir(schedule), schedule))
    if errors:
        for error in errors:
            print(f"  QWEN3 QKV COMPACT-OUTPUT FAIL: {error}")
        return False
    print(f"  PASS: {CASE_NAME} MLIR isolates c1r2->main16->row1->c1r1 compact output")
    print(f"  model={qkv_runner._model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(qkv_runner._decode_token(current_token))
    model = qkv_runner._load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3 ASSET FAIL: {error}")
        return False
    qkv_runner._make_fixture(model, layer, schedule)
    xclbin_path, insts_path = build_kernel(schedule)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def _bf16_ordered(bits: np.ndarray) -> np.ndarray:
    values = bits.astype(np.int32)
    sign = (values & 0x8000) != 0
    return np.where(sign, 0x8000 - (values & 0x7fff), values + 0x8000)


def _payload_words(compact: np.ndarray) -> np.ndarray:
    packets = compact.reshape(generate.QKV_COMPACT_OUT_RECORDS, generate.COMPACT_PACKET_DWORDS)
    return packets[:, 1:].reshape(-1).copy()


def _compact_errors(got: np.ndarray, expected: np.ndarray) -> tuple[list[str], str]:
    errors: list[str] = []
    got_headers = got.reshape(generate.QKV_COMPACT_OUT_RECORDS, generate.COMPACT_PACKET_DWORDS)[:, 0]
    expected_headers = expected.reshape(generate.QKV_COMPACT_OUT_RECORDS, generate.COMPACT_PACKET_DWORDS)[:, 0]
    header_mismatches = np.flatnonzero(got_headers != expected_headers)
    for packet in header_mismatches:
        errors.append(
            f"record[{int(packet)}] header expected=0x{int(expected_headers[packet]) & 0xffffffff:08x} "
            f"got=0x{int(got_headers[packet]) & 0xffffffff:08x}"
        )

    got_payload = _payload_words(got)
    expected_payload = _payload_words(expected)
    got_bits = np.frombuffer(got_payload.tobytes(), dtype=np.uint16)
    expected_bits = np.frombuffer(expected_payload.tobytes(), dtype=np.uint16)
    got_values = np.frombuffer(got_payload.tobytes(), dtype=bfloat16).astype(np.float32)
    expected_values = np.frombuffer(expected_payload.tobytes(), dtype=bfloat16).astype(np.float32)
    finite = np.isfinite(got_values) & np.isfinite(expected_values)
    ulp = np.abs(_bf16_ordered(got_bits) - _bf16_ordered(expected_bits))
    abs_err = np.abs(got_values - expected_values)
    payload_mismatches = np.flatnonzero((ulp > MAX_PAYLOAD_BF16_ULP) | ~finite)
    for lane in payload_mismatches[:16]:
        word = int(lane // 2)
        packet = word // (generate.COMPACT_PACKET_DWORDS - 1)
        packet_word = 1 + word % (generate.COMPACT_PACKET_DWORDS - 1)
        errors.append(
            f"payload packet={packet} word={packet_word} lane={int(lane)} "
            f"expected_bits=0x{int(expected_bits[lane]):04x} got_bits=0x{int(got_bits[lane]):04x} "
            f"expected={float(expected_values[lane]):.9f} got={float(got_values[lane]):.9f} "
            f"abs={float(abs_err[lane]):.9f} ulp={int(ulp[lane])}"
        )
    if payload_mismatches.size > 16:
        errors.append(f"{int(payload_mismatches.size - 16)} additional payload mismatches")

    summary = (
        f"payload_exact_word_mismatches={int(np.flatnonzero(got_payload != expected_payload).size)} "
        f"payload_max_abs={float(np.max(abs_err)):.9f} "
        f"payload_max_ulp={int(np.max(ulp))}"
    )
    return errors, summary


def _compact_failure_summary(got: np.ndarray, expected: np.ndarray) -> list[str]:
    got_packets = got.reshape(generate.QKV_COMPACT_OUT_RECORDS, generate.COMPACT_PACKET_DWORDS)
    expected_packets = expected.reshape(generate.QKV_COMPACT_OUT_RECORDS, generate.COMPACT_PACKET_DWORDS)
    lines = ["header_trace:"]
    for packet in range(generate.QKV_COMPACT_OUT_RECORDS):
        lines.append(
            f"record[{packet}] "
            f"expected=0x{int(expected_packets[packet, 0]) & 0xffffffff:08x} "
            f"got=0x{int(got_packets[packet, 0]) & 0xffffffff:08x} "
            f"payload0_expected=0x{int(expected_packets[packet, 1]) & 0xffffffff:08x} "
            f"payload0_got=0x{int(got_packets[packet, 1]) & 0xffffffff:08x}"
        )
    return lines


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = DEFAULT_LAYER,
    download_model: bool = False,
) -> bool:
    schedule = make_decode_schedule(qkv_runner._decode_token(current_token))
    model = qkv_runner._load_model(model_path, download_model)
    errors = validate_model_assets(model, layer)
    if errors:
        for error in errors:
            print(f"  QWEN3 ASSET FAIL: {error}")
        return False

    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print(f"  model={qkv_runner._model_path(model_path)}")
    print(f"  layer={layer}")
    print(f"  current_token={schedule.current_token}")
    print("  route=c1r2 input RMSNorm -> main16 Q/K/V records -> row1 65/64 -> c1r1 257 -> host")
    print()

    xclbin_path, insts_path = build_kernel(schedule)
    print("  Loading NPU kernel...")
    npu_build.cleanup()
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    try:
        print("  Preparing raw hidden and aux-prefixed full weight stream...")
        fixture = qkv_runner._make_fixture(model, layer, schedule)
        compact_out = np.zeros((generate.COMPACT_OUT_DWORDS,), dtype=np.int32)
        compact_buf = XRTTensor.from_torch(torch.from_numpy(compact_out).to(torch.int32))
        weights_buf = XRTTensor.from_torch(torch.from_numpy(fixture.weights_i32.copy()).to(torch.int32))
        hidden_buf = XRTTensor.from_torch(torch.from_numpy(fixture.hidden_i32.copy()).to(torch.int32))

        print("  Running on NPU...")
        result = npu_build.run(handle, [compact_buf, weights_buf, hidden_buf])
        got = compact_buf.to_torch().numpy().astype(np.int32)
        expected = np.concatenate(
            (
                q_body_compact(fixture.packed_weights, fixture.qkv_activation_bf16),
                k_body_compact(fixture.packed_weights, fixture.qkv_activation_bf16),
                v_body_compact(fixture.packed_weights, fixture.qkv_activation_bf16),
            )
        ).astype(np.int32)
        print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
        print(
            f"  compact_header: expected=0x{int(expected[0]) & 0xffffffff:08x} "
            f"got=0x{int(got[0]) & 0xffffffff:08x}"
        )
        print(f"  compact_payload[0:8]: {got[1:9].tolist()}")

        errors, stats = _compact_errors(got, expected)
        print(f"  {stats}")
        if errors:
            print(f"  FAIL: {len(errors)} Q compact-output mismatches")
            for line in _compact_failure_summary(got, expected):
                print(f"    {line}")
            for error in errors:
                print(f"    {error}")
            return False
        print("  PASS: all Q/K/V global compact records reached host with correct headers and payloads")
        return True
    finally:
        npu_build.cleanup()
