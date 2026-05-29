#!/usr/bin/env python3
"""Integration check for the qwen3-dataflow implementation."""

from __future__ import annotations

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
from dataflow import dataflow_lines, validate_dataflow
from generate import generate_mlir, validate_generated_mlir
from cases import q4nx_qkv_body_post_generate
from cases import currentkv_kvscan_attention_kv16_generate
from cases import currentkv_kvscan_attention_kv16_reference
from cases import currentkv_full_layer_q4nx_down_generate
from cases.registry import CASE_NAMES


def validate_runnable_boundaries() -> list[str]:
    errors: list[str] = []
    currentkv_kvscan_mlir = currentkv_kvscan_attention_kv16_generate.generate_mlir()
    errors.extend(currentkv_kvscan_attention_kv16_generate.validate_generated_mlir(currentkv_kvscan_mlir))
    errors.extend(currentkv_kvscan_attention_kv16_reference.validate_cache_layout_contract())
    currentkv_q4nx_mlir = currentkv_full_layer_q4nx_down_generate.generate_mlir()
    errors.extend(currentkv_full_layer_q4nx_down_generate.validate_generated_mlir(currentkv_q4nx_mlir))
    q4nx_qkv_body_post_mlir = q4nx_qkv_body_post_generate.generate_mlir()
    errors.extend(q4nx_qkv_body_post_generate.validate_generated_mlir(q4nx_qkv_body_post_mlir))
    return errors


def main() -> int:
    mlir = generate_mlir()
    errors = validate_contract()
    errors.extend(validate_dataflow())
    errors.extend(validate_generated_mlir(mlir))
    errors.extend(validate_runnable_boundaries())

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
    print("  frontier=currentkv-full-layer-q4nx-down-bridge")
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
