#!/usr/bin/env python3
"""Integration check for the qwen3-dataflow implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from contract import (
    ATTENTION_PACKET_DWORDS,
    C1R2_PACKET_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    DOWN_PACKET_DWORDS,
    O_CHUNKS,
    SWIGLU_SLICES,
    TOTAL_PATCHES,
    summary_lines,
    validate_contract,
)
from dataflow import dataflow_lines, validate_all_dataflow_slices, validate_dataflow
from generate import generate_mlir, validate_generated_mlir
from cases import full_layer_engine_generate
from cases import full_layer_attention_o_bf16_generate
from cases import full_layer_qkv_prefix_generate
from cases.registry import CASE_NAMES
from resource_manifest import compare_main16_qkv_contracts, validate_manifest_matches_mlir, validate_resource_manifest

EXPECTED_CASE_NAMES = (
    "qwen3-8b-decode-layer",
    "qwen3-8b-c1r2-input-norm-replay",
    "qwen3-8b-qkv-cache-write-bridge",
    "full-layer-qkv-prefix",
    "full-layer-attention-o-bf16",
)
RETIRED_FILES = (
    "debug_contract.cc",
    "qkv_compact_dataflow.py",
    "cases/currentkv_full_layer_q4nx_down_generate.py",
    "cases/currentkv_full_layer_q4nx_down_reference.py",
    "cases/currentkv_full_layer_q4nx_down_runner.py",
    "cases/currentkv_instruction_patch.py",
    "cases/currentkv_kvscan_attention_kv16_generate.py",
    "cases/currentkv_kvscan_attention_kv16_runner.py",
    "cases/q4nx_qkv_body_post_generate.py",
    "cases/q4nx_qkv_body_post_reference.py",
    "cases/q4nx_qkv_body_post_runner.py",
    "cases/qwen3_8b_c1r2_qkv_body_post_generate.py",
    "cases/qwen3_8b_c1r2_qkv_body_post_runner.py",
    "cases/qwen3_8b_postprocess_qkv_rope_generate.py",
    "cases/qwen3_8b_postprocess_qkv_rope_runner.py",
    "cases/qwen3_8b_qkv_body_post_runner.py",
    "cases/qwen3_8b_qkv_full_trace_post_generate.py",
    "cases/qwen3_8b_qkv_full_trace_post_runner.py",
    "cases/qwen3_8b_qkv_rope_post_generate.py",
    "cases/qwen3_8b_qkv_rope_post_runner.py",
)
ACTIVE_CODE_FILES = (
    "run_npu.py",
    "cases/registry.py",
    "cases/full_layer_engine_generate.py",
    "cases/full_layer_engine_reference.py",
    "cases/full_layer_qkv_prefix_generate.py",
    "cases/full_layer_qkv_prefix_runner.py",
    "cases/full_layer_attention_o_bf16_generate.py",
    "cases/full_layer_attention_o_bf16_runner.py",
    "cases/qwen3_8b_decode_layer_runner.py",
    "cases/qwen3_8b_qkv_cache_write_generate.py",
    "cases/qwen3_8b_qkv_cache_write_runner.py",
    "cases/qwen3_8b_c1r2_input_norm_generate.py",
    "cases/qwen3_8b_c1r2_input_norm_runner.py",
)
RETIRED_MARKERS = (
    "currentkv_full_layer_q4nx_down",
    "currentkv-full-layer-q4nx-down-bridge",
    "q4nx_qkv_body_post",
    "q4nx-qkv-body-post-bridge",
    "qwen3_8b_c1r2_qkv_body_post",
    "qwen3_8b_postprocess_qkv_rope",
    "qwen3_8b_qkv_full_trace_post",
    "qwen3_8b_qkv_rope_post",
)


def validate_single_mode_registry() -> list[str]:
    errors: list[str] = []
    if CASE_NAMES != EXPECTED_CASE_NAMES:
        errors.append(f"active registry drifted: {CASE_NAMES} != {EXPECTED_CASE_NAMES}")

    base = Path(__file__).parent
    for retired in RETIRED_FILES:
        if (base / retired).exists():
            errors.append(f"retired qwen3-layer file still exists: {retired}")

    for active in ACTIVE_CODE_FILES:
        path = base / active
        text = path.read_text()
        for marker in RETIRED_MARKERS:
            if marker in text:
                errors.append(f"active qwen3-layer code {active} still references retired marker {marker}")
    return errors


def validate_runnable_boundaries() -> list[str]:
    errors: list[str] = []
    full_layer_mlir = full_layer_engine_generate.generate_mlir()
    errors.extend(full_layer_engine_generate.validate_generated_mlir(full_layer_mlir))
    full_layer_qkv_prefix_mlir = full_layer_qkv_prefix_generate.generate_mlir()
    errors.extend(full_layer_qkv_prefix_generate.validate_generated_mlir(full_layer_qkv_prefix_mlir))
    attention_o_mlir = full_layer_attention_o_bf16_generate.generate_mlir()
    errors.extend(full_layer_attention_o_bf16_generate.validate_generated_mlir(attention_o_mlir))
    qkv_prefix_manifest = full_layer_qkv_prefix_generate.resource_manifest()
    full_layer_manifest = full_layer_engine_generate.resource_manifest()
    attention_o_manifest = full_layer_attention_o_bf16_generate.resource_manifest()
    errors.extend(
        compare_main16_qkv_contracts(
            "qkv-prefix/full-layer",
            qkv_prefix_manifest,
            full_layer_manifest,
        )
    )
    errors.extend(
        compare_main16_qkv_contracts(
            "qkv-prefix/attention-o",
            qkv_prefix_manifest,
            attention_o_manifest,
        )
    )
    return errors


def validate_resource_manifest_negative_checks() -> list[str]:
    errors: list[str] = []
    baseline = full_layer_qkv_prefix_generate.resource_manifest()
    main16 = next(tile for tile in baseline.tiles if tile.role == "main16")
    q_records = next(buffer for buffer in main16.buffers if buffer.name == "q_records")
    shifted_buffers = tuple(
        replace(buffer, address=buffer.address + 4) if buffer.name == q_records.name else buffer
        for buffer in main16.buffers
    )
    shifted_tile = replace(main16, buffers=shifted_buffers)
    shifted_manifest = replace(
        baseline,
        case="negative-qkv-residency-drift",
        tiles=tuple(shifted_tile if tile.name == main16.name else tile for tile in baseline.tiles),
    )
    drift_errors = compare_main16_qkv_contracts(
        "resource-manifest-negative-qkv",
        baseline,
        shifted_manifest,
    )
    if not drift_errors:
        errors.append("resource manifest negative check failed to catch QKV record address drift")

    activation = next(buffer for buffer in main16.buffers if buffer.name == "chunk_ping")
    overlap_buffers = tuple(
        replace(
            buffer,
            address=activation.address,
            size_bytes=activation.size_bytes,
            phases=activation.phases,
        )
        if buffer.name == q_records.name
        else buffer
        for buffer in main16.buffers
    )
    overlap_tile = replace(main16, buffers=overlap_buffers)
    overlap_manifest = replace(
        baseline,
        case="negative-buffer-overlap",
        tiles=tuple(overlap_tile if tile.name == main16.name else tile for tile in baseline.tiles),
    )
    overlap_errors = validate_resource_manifest("resource-manifest-negative-overlap", overlap_manifest)
    if not overlap_errors:
        errors.append("resource manifest negative check failed to catch overlapping QKV buffers")

    full_layer_mlir = full_layer_engine_generate.generate_mlir()
    wt_pong = "%m0_0_wt_pong = aie.buffer(%m0_0)"
    k_records = "%m0_0_k_records = aie.buffer(%m0_0)"
    wt_line = next(line for line in full_layer_mlir.splitlines() if wt_pong in line)
    k_line = next(line for line in full_layer_mlir.splitlines() if k_records in line)
    bad_mlir = full_layer_mlir.replace(f"{wt_line}\n{k_line}", f"{k_line}\n{wt_line}", 1)
    cursor_errors = validate_manifest_matches_mlir(
        "resource-manifest-negative-bank-cursor",
        full_layer_engine_generate.resource_manifest(),
        bad_mlir,
    )
    if not any("declared after a later address" in error for error in cursor_errors):
        errors.append("resource manifest negative check failed to catch AIECC bank cursor regression")
    return errors


def main() -> int:
    mlir = generate_mlir()
    errors = validate_contract()
    errors.extend(validate_dataflow())
    errors.extend(validate_all_dataflow_slices())
    errors.extend(validate_generated_mlir(mlir))
    errors.extend(validate_single_mode_registry())
    errors.extend(validate_runnable_boundaries())
    errors.extend(validate_resource_manifest_negative_checks())

    print("\n".join(summary_lines()))
    print("\n".join(dataflow_lines()))
    print(f"  generated_mlir_bytes={len(mlir.encode())}")
    print(
        "  constants: "
        f"c1r2={C1R2_PACKET_DWORDS} compact={COMPACT_PACKET_DWORDS} "
        f"c6r2={C6R2_INPUT_DWORDS} attn={ATTENTION_PACKET_DWORDS} "
        f"down={DOWN_PACKET_DWORDS}"
    )
    print(f"  chunks: O={O_CHUNKS} swiglu={SWIGLU_SLICES} patches={TOTAL_PATCHES}")
    print("  frontier=qwen3-8b-decode-layer")
    print(f"  active_cases={','.join(CASE_NAMES)}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"  {error}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
