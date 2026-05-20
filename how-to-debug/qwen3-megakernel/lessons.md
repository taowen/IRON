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

## 4. Scratch Layout Is Semantic

The deleted fused scaffold showed that changes described as "layout-only" can
move the output token. Scratch buffer class, runtime metadata, and in-place
operator lifetimes are part of correctness.

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
pyxrt/XRT environment probe
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

## 18. Large Persistent Graphs Must Compress Repetition

The synthetic graph probe isolated a Qwen3-like current KV writeback without
model weights or external kernels. It proved:

```text
per-layer separate fill/drain is safe through 8 layers and fails at 9
one repeated TAP stays at 1 DMA task per FIFO through 64 layers
grouped-by-4 stays under the measured FIFO BD limit for Qwen3's 28 layers
real Qwen3 n-layer final-only chunk=8 compiles after segment-major weight packing
real Qwen3 n-layer final-only chunk=8 runs after grouping cache fill/writeback by 4
real Qwen3 n-layer final-only chunk=28 compiles after grouping cache
fill/writeback as one full-depth TAP
```

This changes the next megakernel direction. A large persistent decode graph
should not be built by copying the single-layer runtime sequence N times.
Whenever the tensor layout allows it, express the layer dimension as a TAP
dimension or a small number of grouped TAP dimensions. Then use preflight to
fail before `aiecc` if any FIFO exceeds eight DMA tasks.

The current performance path is therefore the full 28-layer chunk. The earlier
8 + 8 + 8 + 4 path remains useful as a debug ladder, but it is no longer the
fast-generate target. The important lesson is not the number 8 or 28 itself.
The real fix was changing the layout and data movement expression:
segment-major weights removed per-layer weight DMA replication, grouped K/V
cache fill/writeback avoided runtime backpressure on shallow cache FIFOs, and
the full-depth TAP reduced the final Qwen3 layer pass to one NPU dispatch per
decoded token.

This is still not a descriptor-driven layer state machine. `Runtime.fill()` and
`Runtime.drain()` emit static DMA tasks for the compiled runtime sequence. The
full-depth path works by making those static tasks describe the whole 28-layer
stream. A true reusable state machine would need either patchable instruction
offsets or a lower-level runtime/descriptor mechanism that advances layer
offsets inside one persistent invocation.

## 19. Real Performance Work Must Unlock Columns In The Full Graph

The latest packed-weight chunk=4 generate run matched the reference token, but
the timing split showed the dominant cost is inside the NPU call:

```text
npu_layer_time_us_total: 201411.213
decode_s: 0.203401
fast_op_call_s: 0.201889
fast_output_drain_s: 0.000382
cpu_final_lm_head_s: 0.011655
token_match: True
```

This means the previous host-side work succeeded: weight packing, XRT buffer
reuse, cache residency, RoPE sync, and output drain are no longer the main
bottleneck for this checkpoint.

The real graph column probe then showed:

```text
QKV can preflight at 4 columns.
MLP gate/up can preflight at 2 columns.
The full attention+MLP layer is accepted only at 1 column.
```

So the next high-leverage performance direction is not another chunk-size
sweep. It is to make the attention/full-layer closure multi-column while
preserving the resource lessons already learned: no three-input score/context
workers, no oversized K/V cache objects, no debug-only third outputs, and no
unbounded per-layer DMA task replication.

## 20. Column Speedups Need Both Correct Shards And Warm Timing

The first accepted full-MLP column-scaling patch did not shard gate/up yet. It
only proved the second half of the MLP:

```text
full ffn_hidden FIFO
  -> broadcast to per-column down workers
  -> per-column down weight FIFO/TAP
  -> per-column residual add
  -> sharded output drain
```

This was still useful because it proved the join/broadcast side of the MLP
closure before changing the larger n-layer graph.

Accepted evidence:

```text
full-mlp cols=1,2,4 preflight: ok
max_dma_tasks_per_fifo: 1
max_fifo_buffered_bytes: 49152
verify: errors=0 for all full-MLP debug buffers
```

The performance lesson was separate: a single clean run made cols=4 look
slower, but repeated warm runs showed the expected scaling:

```text
cols=1 late iterations: about 2.33-2.37 ms
cols=2 late iterations: about 1.85-1.88 ms
cols=4 late iterations: about 1.61-1.69 ms
```

Do not classify a column layout as slow from one iteration. First prove the
shard layout numerically, then compare warm timing.

## 21. Decouple A Shared Scaling Knob Before Opening The Full Graph

In `n-layer-final-only`, the original `num_columns` parameter controlled the
attention path and the MLP path together. Removing the guard directly would
have changed Q/K/V head ownership, O-projection row ownership, MLP down
ownership, and final output layout in one edit.

The accepted diagnostic step was to keep attention single-column and route the
public column knob only to MLP down sharding for `layer_iterations=1`:

```text
attention Q/K/V/score/PV/O: num_columns=1
MLP gate/up: single-column full vector
MLP down: 1/2/4 output-row shards
final drain: compact per-column residual tiles into host slices
```

Accepted evidence:

```text
n-layer chunk=1 cols=2 verify: chunk_hidden_errors=0
n-layer chunk=1 cols=4 verify: chunk_hidden_errors=0
preflight cols=4: compute_cores=25, max_dma_tasks_per_fifo=1
```

This made the next blocker precise: full-depth multi-column MLP now needs a
cross-column `layer_residual` join before feedback to the next layer. Without
that join, `layer_iterations>1` should fail early instead of producing a
partially assembled hidden state.

## 22. A Residual Join Can Unlock Multi-Layer Column Scaling

The two-column MLP down experiment became useful only after the compact output
tiles were joined back into a full hidden vector before the chunk feedback
router:

```text
col0 residual tiles + col1 residual tiles
  -> two-input join Worker
  -> full hidden FIFO
  -> existing chunk feedback/router
```

Accepted evidence:

```text
cols=2 layers=2: chunk_hidden_errors=0
cols=2 layers=4: chunk_hidden_errors=0
layers=4 warm timing:
  cols=1 about 29.6-30.0 ms
  cols=2 about 24.5-25.3 ms
```

The join did not violate the known tile input/output budget:

```text
max_tile_inputs=2
max_tile_outputs=2
max_fifo_buffered_bytes=32768
```

The remaining scaling limit came from data movement, not the join itself. The
multi-column down-weight shards are currently filled once per layer, so
`max_dma_tasks_per_fifo` reaches 8 at `layer_iterations=8`. Full-depth
multi-column generate needs a better packed layout/TAP for sharded down
weights, or the next speedup should come from sharding gate/up before pushing
the chunk length further.

## 23. Weight Layout Must Match The Column Plan

The packed MLP2 experiment replaced two failed graph-only approaches:

```text
per-layer gate/up shard fills:
  failed placement by exhausting shim Runtime.fill endpoints

NPU split-copy of one full gate/up stream:
  verified numerically but took about 153 ms for two layers
```

The accepted layout moved the split to host-side packing:

```text
post_norm for all layers
gate0 rows then up0 rows for all layers
gate1 rows then up1 rows for all layers
down0 rows for all layers
down1 rows for all layers
```

This made the graph smaller and faster:

```text
cols=2 layers=2:
  compute_cores=26
  max_dma_tasks_per_fifo=1
  warm time about 9.8-9.9 ms

cols=2 layers=4:
  max_dma_tasks_per_fifo=1
  warm time about 19.0-19.8 ms
```

The practical rule is that column scaling is not just adding Workers. The
weight artifact must be arranged so each Worker receives a legal contiguous
stream with no third tile input and no per-layer DMA replication.

## 24. Verify Deep-Chunk Cache Errors Against A Control

The packed MLP2 layers=8 run had `chunk_hidden_errors=0`, but a few current-K
cache elements exceeded the existing tolerance in layers 2, 4, and 5. A
single-column layers=8 control showed the same class of current-K failures.

That changes the diagnosis:

```text
not enough evidence for packed MLP2 layout corruption
enough evidence for a deep-chunk verifier/tolerance follow-up
```

For performance work, do not reject a graph solely on a cache-current tolerance
failure if:

```text
the final hidden chunk passes
the same failure appears in the single-column control
token-level generation still needs to be checked
```

The next accepted check should be token-level generate on several prompts with
the packed MLP2 path, plus a decision on whether current-K tolerance should be
layer-dependent or compared at a more local boundary. The first token-level
checks on the default prompt passed for chunk=4 and chunk=8:

```text
chunk=4 cols=2: token_match=True, new_text='Paris'
chunk=8 cols=2: token_match=True, new_text='Paris'
```

The follow-up full-depth run made chunk=28 the fastest validated packed MLP2
path:

```text
chunk=28 cols=2 default prompt:
  token_match=True
  new_text='Paris'
  npu_layer_time_us_total about 137079

chunk=28 cols=2 raw Fibonacci prompt:
  token_match=True at positions 17, 18, 19, and 20
  new_text=' 5, 8'
  npu_layer_time_us_total about 128429-131611 per position
```

This is now the accepted performance target before attention head sharding. The
remaining cost is still inside the NPU call, so the next speed work should
either parallelize attention/O projection or reduce per-position compile cost.
