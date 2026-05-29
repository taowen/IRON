"""NPU runner for currentkv-kvscan-attention-kv16-o-bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cases import currentkv_kvscan_attention_kv16_generate as generate
from cases.currentkv_instruction_patch import patch_instruction_stream
from cases.currentkv_kvscan_attention_kv16_reference import (
    CASE_NAME,
    HOST_OUTPUT_DWORDS,
    DecodeSchedule,
    expected_output,
    make_history_k_cache_payload,
    make_history_v_cache_payload,
    make_decode_schedule,
    route_summary,
    validate_cache_layout_contract,
    validate_cache_writeback,
    validate_expected_output,
)

EXPERIMENT_DIR = Path(__file__).parent.parent


def build_kernel(schedule: DecodeSchedule, build_name: str = CASE_NAME) -> tuple[Path, Path]:
    import npu_build

    build_dir = EXPERIMENT_DIR / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir(schedule)
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text, schedule)
    if errors:
        raise RuntimeError("\n".join(f"  CURRENTKV KVSCAN STRUCTURE FAIL: {error}" for error in errors))

    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def build_patched_kernel(base_schedule: DecodeSchedule, target_schedule: DecodeSchedule) -> tuple[Path, Path]:
    build_name = f"{CASE_NAME}-patch-base-token{base_schedule.current_token}"
    xclbin_path, base_insts_path = build_kernel(base_schedule, build_name)
    patched_insts_path = base_insts_path.with_name(
        f"design-token{base_schedule.current_token}-to-token{target_schedule.current_token}.bin"
    )
    changes = patch_instruction_stream(base_insts_path, patched_insts_path, base_schedule, target_schedule)
    print(f"  PASS: patched {base_insts_path.name} -> {patched_insts_path.name}")
    for change in changes:
        print(f"    {change}")
    return xclbin_path, patched_insts_path


def check_only(current_token: int | None = None, patch_from_token: int | None = None) -> bool:
    schedule = make_decode_schedule(current_token)
    schedules = [schedule]
    if patch_from_token is not None:
        schedules.append(make_decode_schedule(patch_from_token))
    errors: list[str] = []
    for item in schedules:
        mlir_text = generate.generate_mlir(item)
        errors.extend(generate.validate_generated_mlir(mlir_text, item))
        errors.extend(validate_cache_layout_contract(item))
    if errors:
        for error in errors:
            print(f"  CURRENTKV KVSCAN STRUCTURE FAIL: {error}")
        return False
    print("  PASS: currentkv-kvscan-attention-kv16-o-bridge MLIR has packet8/9 cache writeback")
    return True


def build_only(current_token: int | None = None, patch_from_token: int | None = None) -> bool:
    schedule = make_decode_schedule(current_token)
    if patch_from_token is None:
        xclbin_path, insts_path = build_kernel(schedule)
    else:
        xclbin_path, insts_path = build_patched_kernel(make_decode_schedule(patch_from_token), schedule)
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def _cache_buffer_payload(
    payload: np.ndarray,
    schedule: DecodeSchedule,
    storage_schedule: DecodeSchedule | None,
) -> np.ndarray:
    if storage_schedule is None:
        return payload
    if storage_schedule.kv_cache_dwords < schedule.kv_cache_dwords:
        raise ValueError(
            f"base token{storage_schedule.current_token} cache has {storage_schedule.kv_cache_dwords} dwords, "
            f"target token{schedule.current_token} needs {schedule.kv_cache_dwords}"
        )
    padded = np.zeros(storage_schedule.kv_cache_dwords, dtype=np.int32)
    padded[: schedule.kv_cache_dwords] = payload
    return padded


def run(current_token: int | None = None, patch_from_token: int | None = None) -> bool:
    import torch
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    import npu_build

    schedule = make_decode_schedule(current_token)
    base_schedule = make_decode_schedule(patch_from_token) if patch_from_token is not None else None
    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    for line in route_summary(schedule):
        print(f"  {line}")
    print(
        "  route: main16 Q/K/V compact -> c1r3 Q + packet8/9 current K/V -> "
        "K/V cache writeback -> sync -> shim KV scan -> kv16 attention -> O compact"
    )
    if base_schedule is not None:
        print(
            f"  patch: compile token{base_schedule.current_token} cache-capacity xclbin once, "
            f"patch design.bin to token{schedule.current_token}"
        )
    print()

    layout_errors = validate_cache_layout_contract(schedule)
    if layout_errors:
        raise RuntimeError("\n".join(f"  CURRENTKV CACHE LAYOUT FAIL: {error}" for error in layout_errors))

    if base_schedule is None:
        xclbin_path, insts_path = build_kernel(schedule)
    else:
        xclbin_path, insts_path = build_patched_kernel(base_schedule, schedule)

    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)

    k_cache = _cache_buffer_payload(make_history_k_cache_payload(schedule), schedule, base_schedule)
    v_cache = _cache_buffer_payload(make_history_v_cache_payload(schedule), schedule, base_schedule)
    expected_cache_dwords = base_schedule.kv_cache_dwords if base_schedule is not None else schedule.kv_cache_dwords
    if k_cache.shape != (expected_cache_dwords,) or v_cache.shape != (expected_cache_dwords,):
        raise ValueError(f"kv cache payload shape mismatch: {k_cache.shape}/{v_cache.shape}")

    expected = expected_output(schedule)
    k_cache_buf = XRTTensor.from_torch(torch.from_numpy(k_cache.copy()).to(torch.int32))
    v_cache_buf = XRTTensor.from_torch(torch.from_numpy(v_cache.copy()).to(torch.int32))
    output_buf = XRTTensor((HOST_OUTPUT_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [k_cache_buf, v_cache_buf, output_buf])
    got_k = k_cache_buf.to_torch().numpy().astype(np.int32)
    got_v = v_cache_buf.to_torch().numpy().astype(np.int32)
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected: {expected.tolist()}")
    print(f"  got:      {got.tolist()}")

    errors = validate_cache_writeback(schedule, got_k[: schedule.kv_cache_dwords], got_v[: schedule.kv_cache_dwords])
    errors.extend(validate_expected_output(expected, got))
    if errors:
        print(f"  FAIL: {len(errors)} currentkv kvscan mismatches")
        for error in errors:
            print(f"    {error}")
        return False

    print("  PASS: packet8/9 wrote K/V cache, synced, scanned, and reached O compact")
    return True
