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
- [First Iteration Is Zero, Later Iterations Improve](symptoms-runtime.md#first-iteration-is-zero-later-iterations-improve)
- [Diagnostic Bundle Crashes While Serializing A Layer Tensor](symptoms-runtime.md#diagnostic-bundle-crashes-while-serializing-a-layer-tensor)
- [Decode Wall Time Is Much Larger Than NPU Time](symptoms-runtime.md#decode-wall-time-is-much-larger-than-npu-time)

### Numeric

- [One-Step Decode Token Is Wrong](symptoms-numeric.md#one-step-decode-token-is-wrong)
- [Layout-Only Changes Move The Token](symptoms-numeric.md#layout-only-changes-move-the-token)
- [QKV Numeric Errors Appear After Runtime Packing](symptoms-numeric.md#qkv-numeric-errors-appear-after-runtime-packing)
- [RoPE Outputs Fail Under GEMV Tolerance](symptoms-numeric.md#rope-outputs-fail-under-gemv-tolerance)
- [Full-Layer Attention Residual Fails Strict Full-Reference Tolerance](symptoms-numeric.md#full-layer-attention-residual-fails-strict-full-reference-tolerance)
- [MLP Gate/Up Fails Full Reference But Passes Local Boundary](symptoms-numeric.md#mlp-gateup-fails-full-reference-but-passes-local-boundary)
- [SiLU Negative Inputs Exceed The Positive-Only Operator Tolerance](symptoms-numeric.md#silu-negative-inputs-exceed-the-positive-only-operator-tolerance)
- [Full-Depth Multi-Layer Hidden Fails After Short Ladder Passes](symptoms-numeric.md#full-depth-multi-layer-hidden-fails-after-short-ladder-passes)
- [N-Layer Final-Only Cache Fails Full Reference But Debug Path Passes Local Boundary](symptoms-numeric.md#n-layer-final-only-cache-fails-full-reference-but-debug-path-passes-local-boundary)

### Resources

- [Persistent QKV Cannot Place On 8 Columns](symptoms-resources.md#persistent-qkv-cannot-place-on-8-columns)
- [Persistent QKV Exceeds Output DMA Channels](symptoms-resources.md#persistent-qkv-exceeds-output-dma-channels)
- [Attention Score Worker Exceeds Input DMA Channels](symptoms-resources.md#attention-score-worker-exceeds-input-dma-channels)
- [K Cache Matrix Does Not Fit In L1](symptoms-resources.md#k-cache-matrix-does-not-fit-in-l1)
- [Debug Pass-Through FIFO Exceeds L1](symptoms-resources.md#debug-pass-through-fifo-exceeds-l1)
- [K Cache Block DMA Exhausts BD IDs](symptoms-resources.md#k-cache-block-dma-exhausts-bd-ids)
- [Full-Layer MLP Worker Exceeds Input DMA Channels](symptoms-resources.md#full-layer-mlp-worker-exceeds-input-dma-channels)
- [Full-Layer K Project Exceeds Output DMA Channels](symptoms-resources.md#full-layer-k-project-exceeds-output-dma-channels)
- [Full-Layer Q/K Project Exceeds Input DMA Channels](symptoms-resources.md#full-layer-qk-project-exceeds-input-dma-channels)
- [Full-Layer MLP Debug Output Exceeds Output DMA Channels](symptoms-resources.md#full-layer-mlp-debug-output-exceeds-output-dma-channels)
- [Full-Layer Down Projection Exceeds L1](symptoms-resources.md#full-layer-down-projection-exceeds-l1)
- [Multidimensional TAP Is Legal But NPU BD Rejects It](symptoms-resources.md#multidimensional-tap-is-legal-but-npu-bd-rejects-it)
- [New Stage Exceeds SequentialPlacer Worker Capacity](symptoms-resources.md#new-stage-exceeds-sequentialplacer-worker-capacity)

### Dataflow

- [Static MLIR Shows One FIFO Drained Twice](symptoms-dataflow.md#static-mlir-shows-one-fifo-drained-twice)
- [Runtime Start Does Not Create A Phase Barrier](symptoms-dataflow.md#runtime-start-does-not-create-a-phase-barrier)
- [Attention Scores Overwrite The First GQA Head](symptoms-dataflow.md#attention-scores-overwrite-the-first-gqa-head)
- [PV/context Fails During L1 Buffer Allocation](symptoms-dataflow.md#pvcontext-fails-during-l1-buffer-allocation)
- [PV/context Inputs Pass But Context Has A Few Large Errors](symptoms-dataflow.md#pvcontext-inputs-pass-but-context-has-a-few-large-errors)
- [New FIFO Has Consumer But No Producer](symptoms-dataflow.md#new-fifo-has-consumer-but-no-producer)
- [Disabled Debug Stream Creates Zero-Length TAP](symptoms-dataflow.md#disabled-debug-stream-creates-zero-length-tap)
