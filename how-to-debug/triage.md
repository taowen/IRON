<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Debug Triage

Use this page before changing code. Pick the first row that matches the
observed failure, run the listed check, then continue in the linked note.

| Observed symptom | First diagnostic | Notes |
| --- | --- | --- |
| Full-ELF compile works but runtime cannot execute | Probe `pyxrt` for `elf` and `ext` APIs | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#full-elf-runtime-apis-are-missing), [method 2](qwen3-megakernel/diagnostic-methods.md#2-probe-pyxrt-capabilities) |
| Graph edit appears to do nothing | Inspect build artifacts and clean the build directory | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#clean-graph-edits-appear-to-do-nothing), [method 3](qwen3-megakernel/diagnostic-methods.md#3-inspect-artifacts-before-rerunning) |
| MLIR generation fails on `scf.if` or a Python value | Decide whether the condition is compile-time Python or runtime MLIR | [MLIR generation failures](mlir-generation.md) |
| `aiecc` reports input/output DMA channel exhaustion | Inspect generated ObjectFIFO graph around the named tile | [AIECC resource failures](aiecc-resource-allocation.md), [method 13](qwen3-megakernel/diagnostic-methods.md#13-count-tile-fifo-inputs-before-changing-kernels) |
| `aiecc` reports buffer allocation or L1 memory failure | Read the MemoryMap literally and compute ObjectFIFO object bytes | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#k-cache-matrix-does-not-fit-in-l1), [method 14](qwen3-megakernel/diagnostic-methods.md#14-read-l1-memorymap-literally) |
| `aiecc` reports BD exhaustion or BD dimension legality | Count DMA tasks and inspect generated `aie.dma_bd` dimensions | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#k-cache-block-dma-exhausts-bd-ids), [method 15](qwen3-megakernel/diagnostic-methods.md#15-inspect-dma-task-count-not-just-tap-correctness) |
| Peano/clang rejects AIE vector code | Check the local AIE API type returned by the expression | [AIE kernel compile failures](kernel-compile.md) |
| Host segfaults in XRT BO validation | Compare MLIR runtime memrefs with `main_kernels.json` BO metadata | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#runtime-segfaults-in-xrt-bo-validation), [method 4](qwen3-megakernel/diagnostic-methods.md#4-compare-runtime_sequence-with-main_kernelsjson) |
| Host buffer assignment fails before NPU execution | Check host buffer ABI and tensor type, not AIE kernels | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#host-buffer-assignment-typeerror) |
| First iteration is zero and later iterations improve | Repeat the same input and inspect the first wrong debug drain | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#first-iteration-is-zero-later-iterations-improve), [method 7](qwen3-megakernel/diagnostic-methods.md#7-repeat-the-same-input) |
| Final token or logits are wrong | Add stage-local debug drains and find the first wrong semantic tensor | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#one-step-decode-token-is-wrong), [method 5](qwen3-megakernel/diagnostic-methods.md#5-add-stage-local-debug-drains) |
| A downstream stage fails but the previous stage is close | Verify against a local reference fed by the actual NPU output | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#qkv-numeric-errors-appear-after-runtime-packing), [method 8](qwen3-megakernel/diagnostic-methods.md#8-use-local-reference-and-full-reference-separately) |
| MLP gate/up fails full reference while `mlp_x_norm` passes | Recompute gate/up from actual NPU `mlp_x_norm` | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#mlp-gateup-fails-full-reference-but-passes-local-boundary), [method 26](qwen3-megakernel/diagnostic-methods.md#26-rebuild-local-references-from-the-actual-fifo-boundary) |
| SiLU fails only on negative model activations | Check the approximation kernel on actual gate values | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#silu-negative-inputs-exceed-the-positive-only-operator-tolerance), [method 27](qwen3-megakernel/diagnostic-methods.md#27-test-approximation-kernels-on-the-models-real-input-distribution) |
| Correct values are in the wrong order | Inspect ObjectFIFO forward/split/join and generated MLIR links | [ObjectFIFO layout mistakes](objectfifo-layout.md), [method 11](qwen3-megakernel/diagnostic-methods.md#11-check-objectfifo-consumers-in-generated-mlir) |
| `resolve_program()` reports an ObjectFIFO endpoint missing | Check the active Program variant has a producer/fill for that FIFO | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#new-fifo-has-consumer-but-no-producer), [method 23](qwen3-megakernel/diagnostic-methods.md#23-check-producer-endpoints-before-reading-placer-errors-as-resource-errors) |
| `SequentialPlacer` cannot find a tile after adding a phase | Count Workers and drop/fuse debug-only stages before changing kernels | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#new-stage-exceeds-sequentialplacer-worker-capacity), [method 24](qwen3-megakernel/diagnostic-methods.md#24-count-workers-against-the-actual-placer-budget) |
| TAP creation fails with size 0 | Guard size/FIFO/Worker/fill/drain/TAP with the same optional-debug flag | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#disabled-debug-stream-creates-zero-length-tap), [method 25](qwen3-megakernel/diagnostic-methods.md#25-optional-debug-streams-need-one-boolean) |
| A layout-only edit changes correctness | Rebuild cleanly and treat scratch layout or patch sites as semantic | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#layout-only-changes-move-the-token), [method 10](qwen3-megakernel/diagnostic-methods.md#10-treat-layout-only-changes-as-correctness-changes) |
| Small numeric mismatch appears in a known operator | Compare with the operator's own reference and tolerance | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#rope-outputs-fail-under-gemv-tolerance), [method 12](qwen3-megakernel/diagnostic-methods.md#12-use-operator-specific-tolerance) |
| Attention scores fail with a structured count | Use mismatch cardinality to localize layout or head mapping | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#attention-scores-run-but-head-layout-is-wrong), [method 18](qwen3-megakernel/diagnostic-methods.md#18-use-error-cardinality-to-find-layout-bugs) |
| Attention context fails after weights and V streams pass | Treat the context kernel as the first unproven boundary | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#pvcontext-inputs-pass-but-context-has-a-few-large-errors), [method 21](qwen3-megakernel/diagnostic-methods.md#21-prove-pv-inputs-before-changing-the-context-kernel), [method 22](qwen3-megakernel/diagnostic-methods.md#22-match-the-accumulation-boundary) |
| A later `rt.start()` or task group causes timeout | Inspect generated cores; persistent workers are not phase barriers | [Qwen3 symptoms](qwen3-megakernel/symptoms.md#runtime-start-does-not-create-a-phase-barrier), [method 17](qwen3-megakernel/diagnostic-methods.md#17-treat-runtime-phase-assumptions-as-suspect) |
| Correct implementation is slower than baseline | Inspect scalar loops, layout conversion placement, and sample stability | [Performance comparison failures](performance-comparison.md) |
| Performance timing has outliers | Clean up runtime state before assertions and increase timed samples | [Runtime isolation](runtime-isolation.md) |

## Boundary Order

The Qwen3 bring-up repeatedly found bugs by checking boundaries in this order:

```text
environment/API
generated artifacts and cache
MLIR generation
aiecc resource allocation
host runtime BO metadata
host buffer ABI and dirty state
ObjectFIFO graph and data layout
local numeric reference
full-model numeric reference
performance
```

Skipping earlier boundaries usually led to changing the wrong layer of the
system.
