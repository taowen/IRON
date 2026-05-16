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
input-rmsnorm-qkv-rope-cache-scores-softmax
input-rmsnorm-qkv-rope-cache-scores-softmax-context
input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj
post-attn-rmsnorm-mlp-gate-up
post-attn-mlp-down-residual
post-attn-rmsnorm-full-mlp
```

The score/softmax checkpoint was accepted only after proving that `qk_pair` and
the K-cache stream were correct, then fixing the score output FIFO slots.

## 2. Graph And Resource Bugs Dominate Earlier Than Math Bugs

The real blockers were not external-kernel multiply-add mistakes. They were:

```text
too many tile input/output FIFOs
ObjectFIFO object too large for L1
too many DMA tasks / exhausted BD IDs
runtime BO metadata mismatch
one FIFO endpoint consumed as if it were two consumers
phase ordering assumed from Runtime.start()
repeated acquire(1) on the same FIFO did not allocate two live output objects
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
non-advancing ObjectFIFO acquire before release [implemented]
runtime patch-site count and uniqueness
local-reference verifier per accepted checkpoint
```

The next persistent stages should keep expanding FIFO/resource preflight checks
so failures show up before `aiecc` or final-token verification.

## 9. ObjectFIFO Acquire Is A Held-Set Request

The score worker failure proved that this pattern is wrong:

```python
score0 = score_fifo.acquire(1)
score1 = score_fifo.acquire(1)
```

The second call does not mean "give me the next object" if the process already
holds one object. Generated LLVM showed both pointers could name the same
buffer, and the second score kernel overwrote the first GQA head's output.

Use one of these forms instead:

```text
score_pair = score_fifo.acquire(2), then index both subviews
separate FIFO handles for q_select 0 and q_select 1
a packed object with both logical score rows inside one FIFO token
```

This is now implemented in preflight as the non-advancing acquire check.

## 10. Local Softmax Reference Must Match The FIFO Boundary

After score output was fixed, `attn_scores_errors` went to zero but six
`attn_weights` elements still failed. The root was the verifier: it compared
the NPU softmax output against `softmax(float32_matmul_scores)`, while the NPU
softmax Worker consumes the bf16 `attn_scores` FIFO.

The accepted local reference is:

```text
scores = matmul(Q, K^T) / sqrt(head_dim)
padded_scores = scores cast to bf16 at the FIFO boundary
attn_weights_ref = softmax(padded_scores[:valid].to(float32)).to(bf16)
```

This keeps the reference at the same boundary as the Worker input instead of
mixing in extra precision that the NPU path no longer has.

## 11. PV Needs A Merge Stage, Not A Three-Input Context Worker

The context calculation needs softmax weights, historical V-cache blocks, and
the current token V. Putting all three into one Worker would recreate the same
input-channel/resource class that broke the score stage.

The accepted context checkpoint uses two Workers:

```text
V merge Worker: current V + V-cache block -> merged V block
context Worker: weights + merged V block -> attention context
```

The merge Worker also exposed a real L1 lesson: a legal 64x128 bf16 object can
still fail if several depth-2 block FIFOs land on one tile. For block streams,
depth=1 is a valid correctness-first bring-up choice.

## 12. Reduction Kernels Should Accumulate At The Intended Precision Boundary

The first context kernel updated bf16 output memory on every row. All input
streams passed, but six context elements had large errors because the reduction
boundary was wrong.

The accepted kernel accumulates each block in local float and writes bf16 once
per block. The host verifier now follows that block-level boundary for the
local context reference.

## 13. A Deeper Checkpoint May Need To Drop Older Debug Copies

The O-projection checkpoint proved that debug visibility has a tile cost. The
accepted context checkpoint used 15 Workers, so adding three more Workers for
context flatten, O projection, and residual could not fit under the current
SequentialPlacer budget.

The accepted pattern was:

```text
keep the new boundary drains: attn_context_flat, attn_o_proj, attn_residual
drop an older debug-only K-cache copy Worker in this deeper checkpoint
fuse context flatten into the existing context Worker
```

Earlier checkpoints still cover K-cache stream layout. Deeper checkpoints
should spend Workers on the next unproven boundary.

## 14. DIM_K-Specific GEMV Objects Need Distinct Symbols

QKV projection uses a GEMV object compiled with `DIM_K=1024`. Attention
O projection uses `DIM_K=2048`. They cannot safely share the same external
symbol name even if the Python `Kernel(...)` declarations look reasonable.

The accepted implementation compiles the O projection object with renamed
symbols:

```text
qwen3_o_proj_matvec_vectorized_bf16_bf16
qwen3_o_proj_matvec_scalar_bf16_bf16
```

Preflight now rejects a `memref<4x2048xbf16>` O-projection declaration that
still uses the generic `matvec_vectorized_bf16_bf16` symbol.

## 15. Full Reference And Local Boundary Reference Answer Different Questions

The MLP gate/up checkpoint showed why both references are needed:

```text
full reference:  measures accumulated model drift from PyTorch
local reference: proves whether the current Worker consumes and computes correctly
```

When `mlp_x_norm` passed but `ffn_gate` and `ffn_up` failed full reference, the
local reference built from actual `mlp_x_norm` showed zero GEMV error. That
kept the diagnosis focused on the true boundary instead of rewriting the GEMV
dataflow.

## 16. Operator Tests Must Cover The Model Distribution

The existing SiLU operator test used positive random inputs. Qwen3 gate
projection feeds negative values into SiLU, exposing the tanh-approximation
error that the standalone test did not exercise.

For model bring-up, keep the operator's standalone tolerance in mind but also
record the model-distribution evidence:

```text
ffn_gate_silu_max_abs: 0.019531
ffn_hidden_errors: 0 when checked against actual_silu * actual_up
```

This is enough to accept the MLP front-half checkpoint, but a future
full-accuracy run should decide whether to keep the fast SiLU approximation or
replace it with a more accurate negative-input path.

## 17. Full MLP Composition Fits As An Isolated Program

The composed post-attention full MLP checkpoint proved that the MLP half can be
expressed as one IRON Program with one input BO, one packed weight BO, and one
packed debug/output BO:

```text
attn_residual -> post RMSNorm -> gate/up -> SiLU -> multiply -> down -> residual
```

Accepted preflight evidence:

```text
runtime_memrefs: 3
arg_specs: 3
max_fifo_buffered_bytes: 49152
non_advancing_acquires: 0
```

This does not prove that the whole attention+MLP layer fits in one naive
SequentialPlacer graph. The attention O-projection path already hit the worker
budget once. The next composition step should use the same checkpoint pattern:
combine only the minimum adjacent boundary, keep local-reference verification,
and let preflight/placer identify the actual resource limit.
