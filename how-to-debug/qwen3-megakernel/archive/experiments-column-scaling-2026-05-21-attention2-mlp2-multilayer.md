<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments

This is the active decision page for Qwen3 persistent megakernel column-scaling
work. Keep it short: accepted baseline, current diagnosis, rejected branches,
and the next executable experiments.

Detailed historical logs are archived in:

```text
how-to-debug/qwen3-megakernel/archive/experiments-column-scaling-2026-05-21.md
how-to-debug/qwen3-megakernel/archive/experiments-column-scaling-2026-05-21-attention2-mlp2.md
```

## Current Baseline

Accepted fast-generate path:

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: single-column
MLP layout: packed two-column segment-major weights
final norm / LM head: CPU
```

Correctness evidence:

```text
default prompt:          token_match=1/1, new_text='Paris'
Fibonacci prompt:        token_match=5/5, new_text=' 5, 8,'
weekdays prompt:         token_match=5/5, new_text=' Thursday, Friday, Saturday,'
numeric sequence prompt: token_match=5/5, new_text=' 10, 1'
```

Performance evidence from the 2026-05-21 same-build suite:

```text
default prompt:     138.077 ms, one NPU decode step before EOS
Fibonacci prompt:   mean 131.179 ms, min 130.028 ms, max 133.291 ms
weekdays prompt:    mean 121.899 ms, min 120.276 ms, max 123.286 ms
numeric sequence:   mean 125.676 ms, min 124.116 ms, max 127.984 ms
```

Resource evidence:

```text
compute_cores=26
total_dma_tasks=18
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
```

Accepted improvements already banked:

```text
8 + 8 + 8 + 4 host dispatch -> one layer_chunk_size=28 dispatch
single-column MLP -> packed two-column gate/up/down MLP
```

Still not implemented:

```text
descriptor-driven reusable layer state machine
runtime-position reusable artifact
multi-column attention/O-proj in the accepted full-depth path
```

## Current Diagnosis

The next useful speed target is still `attention_columns=2` while keeping full
two-column MLP. Attention-only scaling is real, but it is not enough if it
forces MLP back to one column.

Measured layer-1 timing decision, all with `--verify-repeat 5` and
`decode_position=26`:

```text
attention-only, attention1:      warm mean 2163.765 us
attention-only, attention2:      warm mean 1640.395 us
full layer, attention1 + MLP1:   warm mean 6913.680 us
full layer, attention2 + MLP1:   warm mean 6522.273 us
full layer, attention1 + MLP2:   warm mean 4473.282 us
full layer, attention2 + down-only MLP2:
                                  warm mean 5684.301 us
full layer, attention2 + full MLP2, hidden+metadata fused:
                                  warm mean 4393.854 us
```

Interpretation:

```text
attention2 is about 24% faster at the attention-only boundary.
attention2 + MLP1 is still slower than attention1 + full MLP2.
attention2 + down-only MLP2 is a useful diagnostic ladder, not a speed path.
attention2 + full MLP2 now fits and is slightly faster at layer_iterations=1.
The next performance path must generalize that winning one-layer shape to
full-depth chunks without reintroducing the extra post-norm Runtime.fill.
```

Resolved one-layer blocker:

```text
attention2 + full MLP2, layer_iterations=1:
  after QK merge and post-norm/gate-up fusion, placement failed on one more
  host->NPU Runtime.fill output endpoint.

accepted fix:
  pack hidden[1024] and QK/RoPE runtime metadata[384] into one runtime input
  object, then split it into the existing hidden and qk_rope_metadata FIFOs.

accepted evidence:
  preflight ok
  placement_trace_counts runtime_output=16, runtime_input=6, other_input=1
  chunk_hidden_errors=0
  current K/V cache errors=0
  warm iterations 1-4:
    3982.903, 5018.653, 4034.890, 4538.970 us
  warm mean: 4393.854 us
```

This remains a resource-expression problem in the graph. The evidence supports
small-stream fusion; it does not support changing external math kernels.

## Rejected Or Deferred Branches

Do not spend more time on these unless a new measurement changes the diagnosis:

```text
chunk-size tuning:
  chunk=28 is already the accepted baseline. Smaller chunks are host-dispatch
  regressions unless needed for diagnosis.

attention2 + MLP1:
  validated and correct, but slower than attention1 + MLP2.

attention2 + down-only MLP2:
  preflight and verify pass, but the warm mean is 5684.301 us versus
  4473.282 us for attention1 + full MLP2.

cache-pair K/V fill:
  rejected for now. It made placement fit, but the first TAP generated an
  illegal 6D BD, and the corrected single-active-block variant was slightly
  slower than hidden+metadata fusion.

bucket compilation:
  defer until one artifact can handle multiple decode positions. Otherwise
  buckets multiply position-specific artifacts and do not improve NPU runtime.

columns=4:
  defer until columns=2 attention/O-proj has a measured speedup and known
  resource pattern.
```

## Next N Steps

Update this queue whenever an experiment completes. Each step must end with a
diagnosis, not just a failed command.

### 1. Generalize The Winning One-Layer Shape To Full-Depth Chunks

Goal:

```text
Make attention2 + full MLP2 work for layer_chunk_size=28, starting from the
one-layer hidden+metadata fused shape.
```

Current known state:

```text
layer_iterations=1:
  preflight ok
  verify ok
  warm mean 4393.854 us

layer_iterations>1:
  currently blocked by the one-layer-only post-norm/gate-up fusion. The col0
  gate/up stream needs post_norm_i followed by gate/up shard0_i for each layer
  without a separate post_norm Runtime.fill.
```

Candidate fix:

```text
Change the attention2+MLP2 packed weight layout so gate/up column 0 is stored
per layer as:

  post_norm_i || gate0_i || up0_i

and gate/up column 1 remains:

  gate1_i || up1_i

Then use one segment-major fill for each column across the chunk.
```

Acceptance:

```text
real_graph_probe.py --columns 2 --attention-columns 2 --layer-iterations 2
  --preflight-only passes first
then layer_iterations=4,8,28 pass or fail with a diagnosed resource reason
preflight still reports max_tile_inputs<=2 and max_tile_outputs<=2
```

Archive on completion:

```text
exact command
placement_trace_counts
resource summary
root cause and fix
```

### 2. Verify The Multi-Layer Shape Numerically

Run after step 1 passes preflight for a given chunk size.

Command shape:

```text
main.py --stage n-layer-final-only --verify --verify-repeat 5 \
  --layer-chunk-size <N> --num-aie-columns 2 --attention-columns 2
```

Acceptance:

```text
chunk_hidden_errors=0
current K/V cache errors=0
for N=1, warm mean remains near or below 4393.854 us
for N=28, NPU layer time improves from the current 120-138 ms range
```

If correctness fails, update symptom docs by first-failing boundary:

```text
attention context
O partial sum
post-attention residual
post-norm/gate-up fusion
MLP down residual
```

### 3. Recheck Generate Tokens On The Winning Full-Depth Shape

Run after step 2 verifies at `layer_chunk_size=28`.

Command shape:

```text
main.py --stage n-layer-final-only --verify \
  --layer-chunk-size 28 --num-aie-columns 2 --attention-columns 2
```

Acceptance:

```text
preflight passes
default prompt token_match=True
Fibonacci prompt token_match=True for multiple new tokens
NPU layer time improves from the current 120-138 ms range
```

Do not port slower diagnostic layouts to chunk=28.

### 4. Add Runtime-Position Reuse Only After Runtime Improves

First target:

```text
one max_seq_len=256 artifact handles multiple decode positions
```

Acceptance:

```text
no recompile between decode positions
token_match remains true across at least 5 generated tokens
wall-clock improves, not just NPU kernel time
```

Later optional branch:

```text
64 / 128 / 256 / 512 bucket artifacts selected at runtime
```

Do this only after position reuse works for one artifact.

### 5. Revisit Descriptor-Driven Layer Reuse

This is the structural megakernel direction, but it should not block the
attention2 resource fix.

Goal:

```text
reuse one layer graph with layer descriptor / weight offset / cache offset
instead of statically expanding the full 28-layer runtime sequence
```

Acceptance for a first prototype:

```text
2 layers execute through one reused layer body
layer_id advances through explicit state
output matches the static two-layer reference
resource use does not grow linearly with layer count
```
