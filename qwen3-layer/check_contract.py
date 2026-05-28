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
    PHASE_BLOCKS,
    PHASE_CHUNKS,
    PHASE_INPUT_DIMS,
    PHASE_NAMES,
    PHASE_OUTPUT_DIMS,
    SWIGLU_SLICES,
    TOTAL_PATCHES,
    TOTAL_WEIGHT_I32,
    summary_lines,
    validate_contract,
)
from dataflow import dataflow_lines, validate_dataflow
from generate import generate_mlir, validate_generated_mlir
import bridge_generate
from bridge_reference import BRIDGE_CASES
import c1r2_generate
import npu_generate
import shape_generate
import swiglu_generate
from cases import qkv_shape_o_c1r2_generate
from cases import full_layer_contract_generate
from cases import attention_kv16_generate
from cases import kvscan_attention_kv16_generate
from cases import mainq_kvscan_attention_kv16_generate


def validate_runnable_backend() -> list[str]:
    errors: list[str] = []
    expected = (
        ("PHASE_NAMES", npu_generate.PHASE_NAMES, PHASE_NAMES),
        ("PHASE_INPUT_DIMS", npu_generate.PHASE_INPUT_DIMS, PHASE_INPUT_DIMS),
        ("PHASE_OUTPUT_DIMS", npu_generate.PHASE_OUTPUT_DIMS, PHASE_OUTPUT_DIMS),
        ("PHASE_BLOCKS", npu_generate.PHASE_BLOCKS, PHASE_BLOCKS),
        ("PHASE_CHUNKS", npu_generate.PHASE_CHUNKS, PHASE_CHUNKS),
        ("TOTAL_PATCHES", npu_generate.TOTAL_PATCHES, TOTAL_PATCHES),
        ("TOTAL_WEIGHT_I32", npu_generate.TOTAL_WEIGHT_I32, TOTAL_WEIGHT_I32),
    )
    for name, actual, wanted in expected:
        if actual != wanted:
            errors.append(f"runnable backend {name}: {actual} != {wanted}")

    npu_mlir = npu_generate.generate_mlir()
    required = (
        "aie.runtime_sequence",
        "qwen3_layer.o",
        f"memref<{TOTAL_WEIGHT_I32}xi32>",
        "func.func private @q4nx_chunk_accum_slice",
    )
    for marker in required:
        if marker not in npu_mlir:
            errors.append(f"runnable backend missing marker: {marker}")

    for case in BRIDGE_CASES:
        bridge_mlir = bridge_generate.generate_mlir(case)
        errors.extend(bridge_generate.validate_generated_mlir(bridge_mlir, case))
    swiglu_mlir = swiglu_generate.generate_mlir()
    errors.extend(swiglu_generate.validate_generated_mlir(swiglu_mlir))
    c1r2_mlir = c1r2_generate.generate_mlir()
    errors.extend(c1r2_generate.validate_generated_mlir(c1r2_mlir))
    shape_mlir = shape_generate.generate_mlir()
    errors.extend(shape_generate.validate_generated_mlir(shape_mlir))
    attention_kv16_mlir = attention_kv16_generate.generate_mlir()
    errors.extend(attention_kv16_generate.validate_generated_mlir(attention_kv16_mlir))
    kvscan_attention_mlir = kvscan_attention_kv16_generate.generate_mlir()
    errors.extend(kvscan_attention_kv16_generate.validate_generated_mlir(kvscan_attention_mlir))
    mainq_kvscan_mlir = mainq_kvscan_attention_kv16_generate.generate_mlir()
    errors.extend(mainq_kvscan_attention_kv16_generate.validate_generated_mlir(mainq_kvscan_mlir))
    qkv_mlir = qkv_shape_o_c1r2_generate.generate_mlir()
    errors.extend(qkv_shape_o_c1r2_generate.validate_generated_mlir(qkv_mlir))
    full_mlir = full_layer_contract_generate.generate_mlir()
    errors.extend(full_layer_contract_generate.validate_generated_mlir(full_mlir))
    return errors


def main() -> int:
    mlir = generate_mlir()
    errors = validate_contract()
    errors.extend(validate_dataflow())
    errors.extend(validate_generated_mlir(mlir))
    errors.extend(validate_runnable_backend())

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
    print("  runnable_backend=608-patch NPU integration")
    print(
        "  bridge_cases=c1r1-o-bridge,c1r1-down-bridge,"
        "ffn-upgate-c6r2-bridge,c1r2-o-upgate-bridge,"
        "shape-attention-o-bridge,attention-kv16-o-bridge,"
        "kvscan-attention-kv16-o-bridge,"
        "mainq-kvscan-attention-kv16-o-bridge,"
        "qkv-shape-o-c1r2-bridge,"
        "full-layer-contract-bridge"
    )

    if errors:
        print("FAIL")
        for error in errors:
            print(f"  {error}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
