<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Symptom Lookup

Each entry starts from an observed symptom and records only diagnostics that
were used in this repository. Detailed entries are split by failure class.

## Symptom Index

### Runtime

- [Full-ELF Runtime APIs Are Missing](symptoms-runtime.md#full-elf-runtime-apis-are-missing)
- [Pytest Cannot Import pyxrt](symptoms-runtime.md#pytest-cannot-import-pyxrt)
- [Clean Graph Edits Appear To Do Nothing](symptoms-runtime.md#clean-graph-edits-appear-to-do-nothing)
- [Host Buffer Assignment TypeError](symptoms-runtime.md#host-buffer-assignment-typeerror)
- [Runtime Segfaults In XRT BO Validation](symptoms-runtime.md#runtime-segfaults-in-xrt-bo-validation)
- [CREATE_HWCTX Fails After Many Distinct Xclbins In One Process](symptoms-runtime.md#create_hwctx-fails-after-many-distinct-xclbins-in-one-process)
- [First Iteration Is Zero, Later Iterations Improve](symptoms-runtime.md#first-iteration-is-zero-later-iterations-improve)
- [Diagnostic Bundle Crashes While Serializing A Layer Tensor](symptoms-runtime.md#diagnostic-bundle-crashes-while-serializing-a-layer-tensor)
- [Decode Wall Time Is Much Larger Than NPU Time](symptoms-runtime.md#decode-wall-time-is-much-larger-than-npu-time)
- [Decode Compiles A New Artifact For Each Position](symptoms-runtime.md#decode-compiles-a-new-artifact-for-each-position)
- [Runtime Position Metadata Variant Times Out](symptoms-runtime.md#runtime-position-metadata-variant-times-out)
- [N-Layer Chunk 6/8 Compiles But Times Out At Runtime](symptoms-runtime.md#n-layer-chunk-68-compiles-but-times-out-at-runtime)
- [Diagnostic Chunk Is Rejected Before Compile](symptoms-runtime.md#diagnostic-chunk-is-rejected-before-compile)

### Numeric

- [One-Step Decode Token Is Wrong](symptoms-numeric.md#one-step-decode-token-is-wrong)
- [Layout-Only Changes Move The Token](symptoms-numeric.md#layout-only-changes-move-the-token)
- [QKV Numeric Errors Appear After Runtime Packing](symptoms-numeric.md#qkv-numeric-errors-appear-after-runtime-packing)
- [RoPE Outputs Fail Under GEMV Tolerance](symptoms-numeric.md#rope-outputs-fail-under-gemv-tolerance)
- [BF16 Output Differs By One ULP Until Rounding Mode Is Set](symptoms-numeric.md#bf16-output-differs-by-one-ulp-until-rounding-mode-is-set)
- [Full-Layer Attention Residual Fails Strict Full-Reference Tolerance](symptoms-numeric.md#full-layer-attention-residual-fails-strict-full-reference-tolerance)
- [MLP Gate/Up Fails Full Reference But Passes Local Boundary](symptoms-numeric.md#mlp-gateup-fails-full-reference-but-passes-local-boundary)
- [SiLU Negative Inputs Exceed The Positive-Only Operator Tolerance](symptoms-numeric.md#silu-negative-inputs-exceed-the-positive-only-operator-tolerance)
- [Full-Depth Multi-Layer Hidden Fails After Short Ladder Passes](symptoms-numeric.md#full-depth-multi-layer-hidden-fails-after-short-ladder-passes)
- [N-Layer Final-Only Cache Fails Full Reference But Debug Path Passes Local Boundary](symptoms-numeric.md#n-layer-final-only-cache-fails-full-reference-but-debug-path-passes-local-boundary)
- [Attention2 N-Layer Hidden Fails Strict Full Reference But Boundary Replay Matches](symptoms-numeric.md#attention2-n-layer-hidden-fails-strict-full-reference-but-boundary-replay-matches)
- [Attention Probe Returns NaN After Metadata Stream Change](symptoms-numeric.md#attention-probe-returns-nan-after-metadata-stream-change)

### Resources

- [Persistent QKV Cannot Place On 8 Columns](symptoms-resources.md#persistent-qkv-cannot-place-on-8-columns)
- [Persistent QKV Exceeds Output DMA Channels](symptoms-resources.md#persistent-qkv-exceeds-output-dma-channels)
- [Attention Score Worker Exceeds Input DMA Channels](symptoms-resources.md#attention-score-worker-exceeds-input-dma-channels)
- [Fixed-Chunk Attention Worker Exceeds Input DMA Channels](symptoms-resources.md#fixed-chunk-attention-worker-exceeds-input-dma-channels)
- [K Cache Matrix Does Not Fit In L1](symptoms-resources.md#k-cache-matrix-does-not-fit-in-l1)
- [Debug Pass-Through FIFO Exceeds L1](symptoms-resources.md#debug-pass-through-fifo-exceeds-l1)
- [K Cache Block DMA Exhausts BD IDs](symptoms-resources.md#k-cache-block-dma-exhausts-bd-ids)
- [N-Layer Chunk 8 Exhausts Current KV Writeback BD IDs](symptoms-resources.md#n-layer-chunk-8-exhausts-current-kv-writeback-bd-ids)
- [Full-Layer MLP Worker Exceeds Input DMA Channels](symptoms-resources.md#full-layer-mlp-worker-exceeds-input-dma-channels)
- [Full-Layer K Project Exceeds Output DMA Channels](symptoms-resources.md#full-layer-k-project-exceeds-output-dma-channels)
- [Full-Layer Q/K Project Exceeds Input DMA Channels](symptoms-resources.md#full-layer-qk-project-exceeds-input-dma-channels)
- [Full-Layer MLP Debug Output Exceeds Output DMA Channels](symptoms-resources.md#full-layer-mlp-debug-output-exceeds-output-dma-channels)
- [Full-Layer Down Projection Exceeds L1](symptoms-resources.md#full-layer-down-projection-exceeds-l1)
- [Multidimensional TAP Is Legal But NPU BD Rejects It](symptoms-resources.md#multidimensional-tap-is-legal-but-npu-bd-rejects-it)
- [New Stage Exceeds SequentialPlacer Worker Capacity](symptoms-resources.md#new-stage-exceeds-sequentialplacer-worker-capacity)
- [Real Full-Layer Graph Is Locked To One Column](symptoms-resources.md#real-full-layer-graph-is-locked-to-one-column)
- [Attention2 QKV Worker Exceeds Output Channels](symptoms-resources.md#attention2-qkv-worker-exceeds-output-channels)
- [Attention2 Plus MLP2 Still Exhausts Runtime Output Endpoints](symptoms-resources.md#attention2-plus-mlp2-still-exhausts-runtime-output-endpoints)
- [Cache-Pair TAP Passes Preflight But Fails BD Or Numerics](symptoms-resources.md#cache-pair-tap-passes-preflight-but-fails-bd-or-numerics)

### Dataflow

- [Static MLIR Shows One FIFO Drained Twice](symptoms-dataflow.md#static-mlir-shows-one-fifo-drained-twice)
- [Runtime Start Does Not Create A Phase Barrier](symptoms-dataflow.md#runtime-start-does-not-create-a-phase-barrier)
- [Inactive FIFO Skip Requires A Different Static Graph](symptoms-dataflow.md#inactive-fifo-skip-requires-a-different-static-graph)
- [DMA BD Transfer Length Is Not 4-Byte Aligned](symptoms-dataflow.md#dma-bd-transfer-length-is-not-4-byte-aligned)
- [Worker Closure Calls An Unresolved Kernel](symptoms-dataflow.md#worker-closure-calls-an-unresolved-kernel)
- [Attention Scores Overwrite The First GQA Head](symptoms-dataflow.md#attention-scores-overwrite-the-first-gqa-head)
- [PV/context Fails During L1 Buffer Allocation](symptoms-dataflow.md#pvcontext-fails-during-l1-buffer-allocation)
- [PV/context Inputs Pass But Context Has A Few Large Errors](symptoms-dataflow.md#pvcontext-inputs-pass-but-context-has-a-few-large-errors)
- [New FIFO Has Consumer But No Producer](symptoms-dataflow.md#new-fifo-has-consumer-but-no-producer)
- [Disabled Debug Stream Creates Zero-Length TAP](symptoms-dataflow.md#disabled-debug-stream-creates-zero-length-tap)
- [Producer Order Deadlocks Despite Balanced FIFO Counts](symptoms-dataflow.md#producer-order-deadlocks-despite-balanced-fifo-counts)
- [Attention2 Multi-Layer Pack Order Mismatches Runtime TAP](symptoms-dataflow.md#attention2-multi-layer-pack-order-mismatches-runtime-tap)
- [Hidden/Metadata Split Times Out At Larger Chunks](symptoms-dataflow.md#hiddenmetadata-split-times-out-at-larger-chunks)
