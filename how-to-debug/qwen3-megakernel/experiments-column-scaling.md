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
full attention+MLP n-layer path: currently accepted only at one column.
```

The full n-layer path still has an explicit single-column guard. Removing it is
not sufficient because the downstream dataflow still routes the MLP from
`residual_out_fifos[0]` and does not yet join/broadcast column shards.

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

Start with Experiment 1. The smallest useful patch is a two-column MLP closure
inside the full-layer graph while attention remains single-column. This targets
the dominant GEMV work first and should expose the real join/broadcast resource
cost before attention head sharding adds more moving parts.
