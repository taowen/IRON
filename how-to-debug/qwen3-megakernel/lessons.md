<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Lessons

These lessons are inferred from failures already diagnosed in this repository.
They are not a list of untried debugging ideas.

## 1. Accepted Checkpoints Are The Unit Of Progress

The useful checkpoints so far are:

```text
input-rmsnorm
input-rmsnorm-qkv
input-rmsnorm-qkv-rope-cache
```

The score/softmax checkpoint is not accepted yet. The next change should prove
whether `qk_pair` is correct by drain or checksum before changing score math.

## 2. Graph And Resource Bugs Dominate Earlier Than Math Bugs

The real blockers were not external-kernel multiply-add mistakes. They were:

```text
too many tile input/output FIFOs
ObjectFIFO object too large for L1
too many DMA tasks / exhausted BD IDs
runtime BO metadata mismatch
one FIFO endpoint consumed as if it were two consumers
phase ordering assumed from Runtime.start()
```

The practical consequence is that a persistent stage needs preflight graph
checks before running numerical tests.

## 3. TAP Correctness Has Two Levels

A `TensorAccessPattern` can describe the intended tensor region and still lower
to an illegal NPU BD. The K-cache bring-up proved both checks are needed:

```text
semantic access map: coverage, order, no unexpected overlap
hardware legality: BD dimension count, size range, positive strides
```

For attention, large cache reads should be expressed as legal sequence blocks,
not as one full `[seq, head_dim]` L1 object.

## 4. Full-ELF Scratch Layout Is Semantic

The full-ELF scaffold showed that changes described as "layout-only" can move
the output token. Scratch buffer class, patch locations, and in-place operator
lifetimes are part of correctness.

The accepted pattern is to use clean A/B builds and stage-local debug copies
before keeping a scratch-layout refactor.

## 5. Runtime State Bugs Can Masquerade As Math Bugs

The first-iteration-zero failure looked like QKV was broken. Stage drains showed
the first wrong tensor was `x_norm`, and the root cause was parent BO dirty-state
propagation from `XRTSubBuffer.torch_view()`.

Any "iteration 0 bad, iteration 1 better" symptom should be handled as host/NPU
state synchronization until proven otherwise.

## 6. Local Reference Beats Final-Token Guessing

The false QKV failure after runtime packing was resolved by comparing Q/K/V to:

```text
F.linear(actual_npu_x_norm, W)
```

not to the full PyTorch reference. For megakernel development, each stage needs
both a local reference and a full-reference drift metric.

## 7. Persistent Workers Need Explicit Data Dependencies

Generated workers are persistent `aie.core` loops. A later `rt.start()` or a
second task group did not create a phase barrier for Q/score after K-cache
write. Dataflow ordering must be expressed with FIFOs, phase tokens, or a graph
shape that can safely block from the beginning.

## 8. Preflight Checks Worth Building Next

These are worth turning into code because each comes from a real diagnosed
failure. The persistent artifact preflight now covers the checks marked
`implemented`:

```text
pyxrt capability probe for full-ELF APIs
clean-build or artifact freshness assertion after graph edits
runtime_sequence memref count vs main_kernels.json BO metadata [implemented]
ObjectFIFO producer/consumer endpoint count from generated MLIR
tile input/output FIFO count before aiecc [implemented]
ObjectFIFO object bytes * depth vs L1 budget [implemented]
DMA task count per FIFO and BD dimension legality [partly implemented]
runtime patch-site count and uniqueness
local-reference verifier per accepted checkpoint
```

The current score/softmax work should get the `qk_pair` debug drain/checksum
first, then the FIFO/resource preflight checks should be factored out so future
megakernel stages fail earlier than `aiecc` or final-token verification.
