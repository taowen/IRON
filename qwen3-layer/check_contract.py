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
from cases import full_layer_engine_generate
from cases import full_layer_attention_o_bf16_generate
from cases import full_layer_qkv_prefix_generate
from cases import main16_q4nx_compute_perf_generate
from cases import qwen3_8b_c1r2_input_norm_generate
from cases import qwen3_8b_qkv_cache_write_generate
from cases import qwen3_8b_qkv_compact_output_generate
from cases import row1_weight_stream_perf_generate
from cases.decode_cache_reference import (
    make_decode_schedule,
    validate_cache_layout_contract,
)
from cases.case_names import CASE_NAMES
from resource_manifest import compare_main16_qkv_contracts, validate_manifest_matches_mlir, validate_resource_manifest

TOKEN_GATE_TOKENS = (0, 1, 31, 91, 127)
DECODE_CAPACITY_TOKEN = 127
FORBIDDEN_ACTIVE_MLIR_MARKERS = (
    "dataflow slice",
    "qwen3-contract",
    "qwen3-edge",
    "legacy exp67",
    "qwen3_layer.o",
    "qwen3_bridge.o",
    "debug_contract.o",
    "main_projection_q4nx_scheduled",
    "q4nx_emit_",
)
EXPECTED_CASE_NAMES = (
    "qwen3-8b-decode-layer",
    "qwen3-8b-c1r2-input-norm-replay",
    "qwen3-8b-qkv-compact-output",
    "qwen3-8b-qkv-cache-write-bridge",
    "full-layer-qkv-prefix",
    "full-layer-attention-o-bf16",
    "row1-weight-stream-perf",
    "main16-q4nx-compute-perf",
)
RETIRED_FILES = (
    "dataflow.py",
    "emit_mlir.py",
    "generate.py",
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
    "run_stage_budget.py",
    "cases/case_names.py",
    "cases/registry.py",
    "cases/full_layer_engine_generate.py",
    "cases/full_layer_engine_reference.py",
    "cases/full_layer_qkv_prefix_generate.py",
    "cases/full_layer_qkv_prefix_runner.py",
    "cases/full_layer_attention_o_bf16_generate.py",
    "cases/full_layer_attention_o_bf16_runner.py",
    "cases/decode_instruction_patch.py",
    "cases/qwen3_8b_decode_layer_runner.py",
    "cases/qwen3_8b_qkv_cache_write_generate.py",
    "cases/qwen3_8b_qkv_cache_write_runner.py",
    "cases/qwen3_8b_qkv_compact_output_generate.py",
    "cases/qwen3_8b_qkv_compact_output_runner.py",
    "cases/qwen3_8b_c1r2_input_norm_generate.py",
    "cases/qwen3_8b_c1r2_input_norm_runner.py",
    "cases/row1_weight_stream_perf_generate.py",
    "cases/row1_weight_stream_perf_runner.py",
    "cases/main16_q4nx_compute_perf_generate.py",
    "cases/main16_q4nx_compute_perf_runner.py",
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


def validate_active_mlir(case_name: str, mlir: str) -> list[str]:
    errors: list[str] = []
    case_marker = f"case marker {case_name}"
    marker_count = mlir.count(case_marker)
    if marker_count != 1:
        errors.append(f"{case_name}: expected one case marker, found {marker_count}")
    for marker in FORBIDDEN_ACTIVE_MLIR_MARKERS:
        if marker in mlir:
            errors.append(f"{case_name}: forbidden active MLIR marker found: {marker}")
    return errors


def validate_runnable_boundaries() -> list[str]:
    errors: list[str] = []
    full_layer_mlir = full_layer_engine_generate.generate_mlir()
    errors.extend(full_layer_engine_generate.validate_generated_mlir(full_layer_mlir))
    errors.extend(validate_active_mlir(full_layer_engine_generate.CASE_NAME, full_layer_mlir))
    full_layer_qkv_prefix_mlir = full_layer_qkv_prefix_generate.generate_mlir()
    errors.extend(full_layer_qkv_prefix_generate.validate_generated_mlir(full_layer_qkv_prefix_mlir))
    errors.extend(validate_active_mlir(full_layer_qkv_prefix_generate.CASE_NAME, full_layer_qkv_prefix_mlir))
    attention_o_mlir = full_layer_attention_o_bf16_generate.generate_mlir()
    errors.extend(full_layer_attention_o_bf16_generate.validate_generated_mlir(attention_o_mlir))
    errors.extend(validate_active_mlir(full_layer_attention_o_bf16_generate.CASE_NAME, attention_o_mlir))
    qkv_cache_write_mlir = qwen3_8b_qkv_cache_write_generate.generate_mlir()
    errors.extend(qwen3_8b_qkv_cache_write_generate.validate_generated_mlir(qkv_cache_write_mlir))
    errors.extend(validate_active_mlir(qwen3_8b_qkv_cache_write_generate.CASE_NAME, qkv_cache_write_mlir))
    qkv_compact_output_mlir = qwen3_8b_qkv_compact_output_generate.generate_mlir()
    errors.extend(qwen3_8b_qkv_compact_output_generate.validate_generated_mlir(qkv_compact_output_mlir))
    errors.extend(validate_active_mlir(qwen3_8b_qkv_compact_output_generate.CASE_NAME, qkv_compact_output_mlir))
    c1r2_replay_mlir = qwen3_8b_c1r2_input_norm_generate.generate_mlir()
    errors.extend(qwen3_8b_c1r2_input_norm_generate.validate_generated_mlir(c1r2_replay_mlir))
    errors.extend(validate_active_mlir(qwen3_8b_c1r2_input_norm_generate.CASE_NAME, c1r2_replay_mlir))
    row1_weight_stream_mlir = row1_weight_stream_perf_generate.generate_mlir()
    errors.extend(row1_weight_stream_perf_generate.validate_generated_mlir(row1_weight_stream_mlir))
    errors.extend(validate_active_mlir(row1_weight_stream_perf_generate.CASE_NAME, row1_weight_stream_mlir))
    main16_compute_mlir = main16_q4nx_compute_perf_generate.generate_mlir()
    errors.extend(main16_q4nx_compute_perf_generate.validate_generated_mlir(main16_compute_mlir))
    errors.extend(validate_active_mlir(main16_q4nx_compute_perf_generate.CASE_NAME, main16_compute_mlir))
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


def validate_decode_token_gate() -> list[str]:
    errors: list[str] = []
    for token in TOKEN_GATE_TOKENS:
        target_schedule = make_decode_schedule(token)
        build_schedule = make_decode_schedule(max(token, DECODE_CAPACITY_TOKEN))
        if target_schedule.kv_blocks > build_schedule.kv_blocks:
            errors.append(
                f"token{token}: target blocks {target_schedule.kv_blocks} exceed "
                f"capacity blocks {build_schedule.kv_blocks}"
            )

        full_layer_mlir = full_layer_engine_generate.generate_mlir(build_schedule)
        errors.extend(full_layer_engine_generate.validate_generated_mlir(full_layer_mlir, build_schedule))
        errors.extend(validate_active_mlir(full_layer_engine_generate.CASE_NAME, full_layer_mlir))
        errors.extend(validate_cache_layout_contract(target_schedule))
        errors.extend(validate_cache_layout_contract(build_schedule))

        qkv_prefix_mlir = full_layer_qkv_prefix_generate.generate_mlir(target_schedule)
        errors.extend(full_layer_qkv_prefix_generate.validate_generated_mlir(qkv_prefix_mlir, target_schedule))
        errors.extend(validate_active_mlir(full_layer_qkv_prefix_generate.CASE_NAME, qkv_prefix_mlir))
        attention_o_mlir = full_layer_attention_o_bf16_generate.generate_mlir(target_schedule)
        errors.extend(full_layer_attention_o_bf16_generate.validate_generated_mlir(attention_o_mlir, target_schedule))
        errors.extend(validate_active_mlir(full_layer_attention_o_bf16_generate.CASE_NAME, attention_o_mlir))
        qkv_cache_write_mlir = qwen3_8b_qkv_cache_write_generate.generate_mlir(target_schedule)
        errors.extend(qwen3_8b_qkv_cache_write_generate.validate_generated_mlir(qkv_cache_write_mlir, target_schedule))
        errors.extend(validate_active_mlir(qwen3_8b_qkv_cache_write_generate.CASE_NAME, qkv_cache_write_mlir))
        qkv_compact_output_mlir = qwen3_8b_qkv_compact_output_generate.generate_mlir(target_schedule)
        errors.extend(qwen3_8b_qkv_compact_output_generate.validate_generated_mlir(qkv_compact_output_mlir, target_schedule))
        errors.extend(validate_active_mlir(qwen3_8b_qkv_compact_output_generate.CASE_NAME, qkv_compact_output_mlir))
    return errors


def validate_resource_manifest_negative_checks() -> list[str]:
    errors: list[str] = []
    baseline = full_layer_qkv_prefix_generate.resource_manifest()
    main16 = next(tile for tile in baseline.tiles if tile.role == "main16")
    record_ping = next(buffer for buffer in main16.buffers if buffer.name == "record_ping")
    shifted_buffers = tuple(
        replace(buffer, address=buffer.address + 4) if buffer.name == record_ping.name else buffer
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
        errors.append("resource manifest negative check failed to catch main16 record ping address drift")

    activation = next(buffer for buffer in main16.buffers if buffer.name == "chunk_ping")
    overlap_buffers = tuple(
        replace(
            buffer,
            address=activation.address,
            size_bytes=activation.size_bytes,
            phases=activation.phases,
        )
        if buffer.name == record_ping.name
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
    record_pong = "%m0_0_record_pong = aie.buffer(%m0_0)"
    wt_line = next(line for line in full_layer_mlir.splitlines() if wt_pong in line)
    record_line = next(line for line in full_layer_mlir.splitlines() if record_pong in line)
    bad_mlir = full_layer_mlir.replace(f"{wt_line}\n{record_line}", f"{record_line}\n{wt_line}", 1)
    cursor_errors = validate_manifest_matches_mlir(
        "resource-manifest-negative-bank-cursor",
        full_layer_engine_generate.resource_manifest(),
        bad_mlir,
    )
    if not any("declared after a later address" in error for error in cursor_errors):
        errors.append("resource manifest negative check failed to catch AIECC bank cursor regression")
    return errors


def main() -> int:
    errors = validate_contract()
    errors.extend(validate_single_mode_registry())
    errors.extend(validate_runnable_boundaries())
    errors.extend(validate_decode_token_gate())
    errors.extend(validate_resource_manifest_negative_checks())

    print("\n".join(summary_lines()))
    print(
        "  constants: "
        f"c1r2={C1R2_PACKET_DWORDS} compact={COMPACT_PACKET_DWORDS} "
        f"c6r2={C6R2_INPUT_DWORDS} attn={ATTENTION_PACKET_DWORDS} "
        f"down={DOWN_PACKET_DWORDS}"
    )
    print(f"  chunks: O={O_CHUNKS} swiglu={SWIGLU_SLICES} patches={TOTAL_PATCHES}")
    print("  frontier=qwen3-8b-decode-layer")
    print(f"  token_gate={','.join(str(token) for token in TOKEN_GATE_TOKENS)}")
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
