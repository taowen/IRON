<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments

This file tracks the active Qwen3 persistent megakernel performance experiment.
It is not a symptom database. Use it to decide what to try next and what result
would prove that the experiment moved the design forward.

## Current Baseline

Accepted fast-generate path:

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
runtime shape: one full-depth static stream per decode token
final norm / LM head: CPU
```

Evidence:

```text
default prompt:
  token_match=True
  new_text='Paris'

raw prompt "Fibonacci numbers: 1, 1, 2, 3,":
  token_match=True for decode positions 17, 18, 19, and 20
  new_text=' 5, 8'

preflight:
  compute_cores=21
  max_fifo_buffered_bytes=32768
  max_dma_tasks_per_fifo=1

timing:
  npu_layer_time_us_total: about 190-200ms per decoded token
```

The current path solved the `8 + 8 + 8 + 4` host-dispatch pattern by using one
full-depth cache TAP for `layer_iterations=28`. It did not implement a
descriptor-driven layer state machine; the compiled Runtime sequence is still
static for a decode position.

## Bottleneck Hypothesis

After packed weights, cache residency, prefix K/V blocks, output-only drains,
and full-depth chunking, host overhead is no longer the dominant cost. The
remaining latency is mainly single-column NPU compute and dataflow.

The next speedup must therefore increase useful column parallelism without
reintroducing the resource failures already diagnosed:

```text
no per-layer DMA task replication
no oversized ObjectFIFO objects
no three-input score/context worker pattern
no debug-only third output on hot tiles
no unvalidated intermediate chunk sizes
```

## Known Column Status

Use `persistent/real_graph_probe.py` to recheck this table after every graph
change.

```text
QKV checkpoint: can preflight at multiple columns.
MLP gate/up checkpoint: can preflight beyond one column.
post-attn full-MLP checkpoint: verifies at columns=1,2,4 with down projection
  sharded by column.
n-layer chunk=1 path: verifies at columns=1,2,4 with attention still
  single-column and MLP down projection sharded by column.
full-depth n-layer path: currently accepted only at one column.
```

The full-depth n-layer path is still single-column. Removing the old guard was
not sufficient because `num_columns` originally controlled both attention and
MLP. The current experiment deliberately keeps attention single-column and uses
the public column knob only for MLP down sharding in `layer_iterations=1`.

## Experiment 1: Two-Column MLP Closure

Goal:

```text
Make the MLP half consume and produce column shards inside the full-layer graph.
```

Planned shape:

```text
residual full vector
  -> broadcast xnorm/residual to 2 gate/up columns
  -> each column computes intermediate_size / 2 gate and up rows
  -> local SiLU * up
  -> join ffn_hidden
  -> broadcast ffn_hidden to 2 down columns
  -> each column computes hidden_size / 2 output rows
  -> join layer residual for next layer
```

Acceptance criteria:

```text
preflight passes for columns=2
compute_cores <= NPU2 budget
max_dma_tasks_per_fifo <= 8
max_fifo_buffered_bytes <= 64KB
n-layer chunk=1 stage verify passes before trying chunk=28
chunk=28 fast-generate token_match=True on the default and Fibonacci prompts
```

Expected risk:

```text
Join/broadcast FIFOs may increase tile input/output count.
Down projection may need a different weight layout so each column receives
only its row shard.
```

Progress on 2026-05-20:

```text
Implemented a partial full-MLP column closure:
  gate/up remains one full-vector worker pair
  ffn_hidden is broadcast to down workers
  down projection uses one weight FIFO/TAP per column
  residual add and output drain are hidden-row sharded

preflight full-mlp:
  cols=1 compute_cores=7  total_dma_tasks=13 max_dma_tasks_per_fifo=1
  cols=2 compute_cores=9  total_dma_tasks=17 max_dma_tasks_per_fifo=1
  cols=4 compute_cores=13 total_dma_tasks=25 max_dma_tasks_per_fifo=1
  max_fifo_buffered_bytes=49152 for all three

verify full-mlp:
  cols=1 errors=0 for all debug buffers
  cols=2 errors=0 for all debug buffers
  cols=4 errors=0 for all debug buffers
```

Timing note:

```text
A single clean-build run was misleading because the first runtime call is cold.
Using --verify-repeat 5 on cached builds, late iterations were approximately:
  cols=1: 2.33-2.37 ms
  cols=2: 1.85-1.88 ms after warmup
  cols=4: 1.61-1.69 ms after warmup
```

Conclusion:

```text
The ffn_hidden broadcast + per-column down weight shard is correct and has real
speedup after warmup. This is not the full Experiment 1 goal yet because
gate/up is still single-column. The next useful patch is gate/up sharding with
an explicit ffn_hidden join that can later be ported into the n-layer graph.
```

Progress on 2026-05-20, n-layer chunk=1:

```text
Implemented the same partial down-shard inside `n-layer-final-only` while
keeping attention single-column:
  attention Q/K/V/score/PV/O remains num_columns=1
  public --num-aie-columns controls only MLP down sharding for layer_iterations=1
  each down column drains compact 128-element residual tiles into the final
  host output slice

preflight n-layer-final-only, layer_iterations=1:
  cols=1 compute_cores=19 total_dma_tasks=15 max_dma_tasks_per_fifo=1
  cols=2 compute_cores=21 total_dma_tasks=17 max_dma_tasks_per_fifo=1
  cols=4 compute_cores=25 total_dma_tasks=21 max_dma_tasks_per_fifo=1
  max_fifo_buffered_bytes=32768 for all three

verify n-layer-final-only, layer_iterations=1:
  cols=2 chunk_hidden_errors=0, layer0 K/V current errors=0
  cols=4 chunk_hidden_errors=0, layer0 K/V current errors=0
```

Timing note:

```text
Using --verify-repeat 5:
  cols=1 late iterations: about 6.73-7.81 ms
  cols=2 late iterations: about 5.92-6.32 ms
  cols=4 late iterations: about 5.57-7.05 ms
```

Conclusion:

```text
The down-only shard survives inside the true full-layer graph and gives a
modest n-layer chunk=1 speedup. It is not enough to unlock full-depth
performance because layer_iterations>1 still needs a cross-column
layer-residual join before feedback to the next layer. The next patch should
build that join or shard gate/up plus join; continuing to tune down-only
columns has limited leverage.
```

Progress on 2026-05-20, n-layer residual join:

```text
Implemented a two-column residual join for layer_iterations>1:
  per-column down workers emit compact 128-element residual tiles
  a two-input join Worker copies left/right tiles into one full hidden vector
  the existing chunk feedback/router consumes that full joined residual

preflight n-layer-final-only:
  cols=2 layers=2 compute_cores=24 total_dma_tasks=18
    max_dma_tasks_per_fifo=2 max_fifo_buffered_bytes=32768
  cols=2 layers=4 compute_cores=24 total_dma_tasks=22
    max_dma_tasks_per_fifo=4 max_fifo_buffered_bytes=32768
  cols=2 layers=8 compute_cores=24 total_dma_tasks=34
    max_dma_tasks_per_fifo=8 max_fifo_buffered_bytes=32768
```

Diagnosed issue:

```text
The first cross-layer sharded down-weight TAP used a layer stride of 3145728,
which aiecc rejected:
  'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range

The fix keeps the single-column path as one contiguous full-depth TAP and emits
one linear down-weight shard fill per layer only for multi-column down.
```

Verification:

```text
cols=2 layers=2 verify:
  chunk_hidden_errors=0
  layer0/layer1 K/V current errors=0

cols=2 layers=4 verify:
  chunk_hidden_errors=0
  layer0..layer3 K/V current errors=0
```

Timing:

```text
layers=2, --verify-repeat 5:
  cols=1 late iterations: about 14.21-14.51 ms
  cols=2 late iterations: about 12.39-12.77 ms

layers=4, --verify-repeat 3:
  cols=1 late iterations: about 29.58-30.00 ms
  cols=2 late iterations: about 24.47-25.33 ms
```

Conclusion:

```text
The two-column down-shard plus residual join is now valid beyond one layer and
has a consistent net speedup. The current limit is not correctness but scaling:
because multi-column down weights are filled per layer, cols=2 reaches
max_dma_tasks_per_fifo=8 at layers=8. Full-depth chunk=28 needs a better
multi-column down weight layout/TAP, or gate/up/attention parallelism must be
added before attempting full-depth multi-column generate.
```

Progress on 2026-05-20, packed two-column MLP:

```text
Tried direct gate/up sharding after the residual join.

Rejected intermediate shape:
  per-layer runtime fills for post_norm/gate/up shards
  failed resolve_program() because shim/runtime output endpoints were exhausted

Rejected second shape:
  one full gate/up Runtime.fill
  NPU split-copy Worker produced post_norm + two shard streams
  first failed preflight with 3 output FIFOs on one tile
  after removing post_norm output it verified but took about 153 ms for layers=2

Accepted shape:
  host packs MLP weights in two-column order
  post_norm for all layers is one contiguous segment
  gate0+up0 and gate1+up1 are two contiguous per-column segments
  down0 and down1 are two contiguous per-column segments
  each gate/up shard Worker consumes only xnorm + one packed gate/up weight FIFO
```

Preflight:

```text
cols=2 layers=2:
  compute_cores=26 total_dma_tasks=18 max_dma_tasks_per_fifo=1
  max_tile_inputs=2 max_tile_outputs=2

cols=2 layers=4:
  compute_cores=26 total_dma_tasks=18 max_dma_tasks_per_fifo=1
  max_tile_inputs=2 max_tile_outputs=2

cols=2 layers=8:
  compute_cores=26 total_dma_tasks=22 max_dma_tasks_per_fifo=2
  max_tile_inputs=2 max_tile_outputs=2
```

Verification and timing:

```text
cols=2 layers=2, --verify-repeat 3:
  chunk_hidden_errors=0
  layer0/layer1 K/V current errors=0
  late iterations: about 9.84-9.92 ms

cols=2 layers=4, --verify-repeat 3:
  chunk_hidden_errors=0
  layer0..layer3 K/V current errors=0
  late iterations: about 19.02-19.81 ms

cols=2 layers=8, --verify-repeat 2:
  chunk_hidden_errors=0
  warm iteration: about 39.93 ms
  current K cache has a few tolerance errors in layers 2/4/5

cols=1 layers=8 comparison:
  chunk_hidden_errors=0
  current K cache shows the same layers 2/4/5 tolerance class

token-level generate:
  chunk=2 cols=2 default prompt token_match=True
  new_text='Paris'
  npu_layer_time_us_total about 140995 us

  chunk=4 cols=2 default prompt token_match=True
  new_text='Paris'
  npu_layer_time_us_total about 140480 us

  chunk=8 cols=2 default prompt token_match=True
  new_text='Paris'
  npu_layer_time_us_total about 145024 us

  chunk=28 cols=2 default prompt token_match=True
  new_text='Paris'
  npu_layer_time_us_total about 137079 us

  chunk=28 cols=2 raw Fibonacci prompt token_match=True for positions 17-20
  new_text=' 5, 8'
  npu_layer_time_us_total per position about 128429-131611 us
```

Conclusion:

```text
Packed MLP2 is the accepted next performance path. It improves layers=2 from
about 12.4-12.8 ms to about 9.8-9.9 ms, and layers=4 from about 24.5-25.3 ms
to about 19.0-19.8 ms. It also removes the per-layer down-weight DMA scaling
limit for the two-column path. Full-depth chunk=28 now preflights and runs:
compute_cores=26, total_dma_tasks=18, max_dma_tasks_per_fifo=1, and the default
prompt NPU layer time is about 137 ms, materially faster than the earlier
single-column 190-200 ms baseline.

The layers=8 cache verifier failure is not specific to packed MLP2 because the
single-column layers=8 comparison has the same current-K tolerance class while
final hidden still passes. Token-level generate also matches for chunk=8, so
treat this as a verifier/tolerance follow-up before using cache-current errors
alone to reject the path.
```

## Experiment 2: Two-Column Attention Head Shard

Goal:

```text
Shard Q/K/V, score, softmax, PV, context packing, and O projection by attention
head groups.
```

Planned shape for Qwen3-0.6B:

```text
16 Q heads / 8 KV heads / GQA repeat 2
2 columns -> each column owns 4 KV heads and 8 Q heads
```

Data movement requirements:

```text
K/V cache TAP adds a column head offset.
Each column reads only its KV heads.
Q/K/V weights are segment-major and column-sharded.
Context shards must be joined or placed into the correct slice before O-proj.
```

Acceptance criteria:

```text
attention-only boundary verifies before full MLP integration
softmax row sums stay valid per owned Q head
no future KV reads
token_match=True after reconnecting O-proj
```

Expected risk:

```text
The current context/O-proj worker packs two Q heads per KV head into one flat
context object. The flat context object may become the next join bottleneck.
```

## Experiment 3: Two-Column Full-Depth Generate

Goal:

```text
Run layer_chunk_size=28 with the two-column full-layer graph.
```

Acceptance criteria:

```text
chunk=28 compile/preflight passes
max_dma_tasks_per_fifo remains bounded by full-depth TAPs
default prompt token_match=True
Fibonacci prompt token_match=True for at least four decode steps
npu_layer_time_us_total improves materially from the 190-200ms baseline
```

Stop condition:

```text
If NPU time does not improve, split timing/trace by phase before changing more
graph structure. Possible causes are extra join/broadcast overhead, L3 DMA
contention, or only a small fraction of GEMV work actually moving to column 1.
```

## Experiment 4: Instruction Offset Patching

Goal:

```text
Reduce per-position compile cost without changing math.
```

Current observation:

```text
chunk=28 first-position compile is about 66s.
Following position compiles are about 3.8-4.0s.
The runtime sequence still bakes position-dependent TAP and mask constants.
```

Candidate method:

```text
Reuse the strided_copy / Llama pattern: emit magic offset values, locate them
in the instruction stream or ELF, patch position/cache offsets at runtime, and
assert exact patch-site counts.
```

Do not attempt this until the two-column compute path is understood. It reduces
compile overhead, not the 190-200ms NPU execution time.

## Current Next Step

Continue Experiment 1 from the accepted packed MLP2 path. Next targets:

```text
1. Treat chunk=28 cols=2 packed MLP2 as the current fastest validated path.
2. Re-run a same-build warm timing suite if timing variance becomes a decision
   point; current single-token timings are chunk=2 ~141 ms, chunk=4 ~140 ms,
   chunk=8 ~145 ms, chunk=28 ~137 ms.
3. Decide whether the layers=8 current-K verifier should use looser per-layer
   tolerance or a local reference at the actual feedback boundary.
4. Move to Experiment 2 attention head sharding or reduce per-position compile
   overhead with instruction offset patching.
```
