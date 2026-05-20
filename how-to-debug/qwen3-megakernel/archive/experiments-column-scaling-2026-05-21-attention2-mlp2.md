<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments Archive, 2026-05-21 Attention2 + MLP2

This is the detailed evidence log that was moved out of the active
`experiments-column-scaling.md` decision page after the attention2/MLP2
resource diagnosis. Keep it as historical proof for timings, placement traces,
and rejected branches; do not use it as the current next-step queue.

The text below is preserved from the active page before it was compacted. Its
section names and "current" wording reflect the state at archive time.

## Current Validated Baseline

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
default prompt:
  token_match=1/1
  new_text='Paris'

Fibonacci prompt:
  token_match=5/5
  new_text=' 5, 8,'

weekdays prompt:
  token_match=5/5
  new_text=' Thursday, Friday, Saturday,'

numeric sequence prompt:
  token_match=5/5
  new_text=' 10, 1'
```

Performance evidence from the same-build suite on 2026-05-21:

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

This baseline has two accepted improvements:

```text
8 + 8 + 8 + 4 host dispatch -> one layer_chunk_size=28 dispatch
single-column MLP -> packed two-column gate/up/down MLP
```

It is not yet a descriptor-driven reusable layer state machine. The runtime
sequence is still static for a decode position, and attention/O-proj still
runs effectively single-column.

## Current Active Experiment

The next real speed target is `attention_columns=2`: split attention by head
groups and shard O projection over the context dimension.

Qwen3-0.6B ownership:

```text
16 Q heads / 8 KV heads / head_dim=128 / GQA repeat=2

attention_columns=2:
  col0 owns KV heads 0-3 and Q heads 0-7
  col1 owns KV heads 4-7 and Q heads 8-15

per-column shapes:
  q_shard_size=1024
  kv_shard_size=512
  context_shard_size=1024
```

O projection shape:

```text
context = concat(context_col0, context_col1)
O = W_o[:, 0:1024] @ context_col0 + W_o[:, 1024:2048] @ context_col1

Each attention column computes a full hidden partial O[1024].
A two-input sum worker reduces the two partials before residual add.
```

Already implemented or scaffolded:

```text
Qwen3PersistentNLayerFinalOnly.attention_columns, default 1
real_graph_probe.py --attention-columns
main.py --attention-columns
main.py --attention-probe-only
packed attention2 Q/K/V/O shard layout
KV cache head-offset TAPs
context_shard[1024] FIFOs
O-proj kernel object with DIM_K=1024
O partial-sum worker
preflight linter accepts O-proj DIM_K=1024 based on object name
prepacked QK/RoPE runtime metadata for attention2
interleaved QK weight stream per attention column
```

Validated isolated attention2 result after QK interleaving:

```text
command:
  main.py --stage n-layer-final-only --verify --layer-chunk-size 1
    --num-aie-columns 1 --attention-columns 2 --attention-probe-only

result:
  attention_probe_residual_errors=0
  layer0_keys_cache_current_errors=0
  layer0_values_cache_current_errors=0
  npu_time_us=3938.720
  compute_cores=24
  total_dma_tasks=18
```

Validated full-layer attention2 + MLP1 result:

```text
command:
  main.py --stage n-layer-final-only --verify --layer-chunk-size 1
    --num-aie-columns 1 --attention-columns 2

result:
  chunk_hidden_errors=0
  layer0_keys_cache_current_errors=0
  layer0_values_cache_current_errors=0
  npu_time_us=9431.758
  compute_cores=28
  total_dma_tasks=20
```

Timing decision on 2026-05-21:

```text
All runs used:
  --stage n-layer-final-only
  --verify
  --verify-repeat 5
  --layer-chunk-size 1
  decode_position=26

attention-only, attention1:
  columns=1 attention_columns=1 attention_probe_only=True
  warm iterations 1-4:
    2179.962, 2167.940, 2162.770, 2144.387 us
  warm mean: 2163.765 us

attention-only, attention2:
  columns=1 attention_columns=2 attention_probe_only=True
  warm iterations 1-4:
    1662.398, 1630.689, 1637.332, 1631.160 us
  warm mean: 1640.395 us

full layer, attention1 + MLP1:
  columns=1 attention_columns=1
  warm iterations 1-4:
    6789.233, 6929.184, 6766.290, 7170.012 us
  warm mean: 6913.680 us

full layer, attention2 + MLP1:
  columns=1 attention_columns=2
  warm iterations 1-4:
    6295.703, 6278.290, 6704.285, 6810.813 us
  warm mean: 6522.273 us

full layer, attention1 + MLP2:
  columns=2 attention_columns=1
  warm iterations 1-4:
    4534.040, 4440.305, 4475.060, 4443.722 us
  warm mean: 4473.282 us
```

Interpretation:

```text
attention2 is useful in isolation:
  2163.765 -> 1640.395 us, about 24% faster for the attention-only boundary.

attention2 still helps when MLP stays single-column:
  6913.680 -> 6522.273 us, about 6% faster for a full single-layer graph.

But attention2 + MLP1 is not a viable performance path:
  attention2 + MLP1 warm mean 6522.273 us
  accepted attention1 + MLP2 warm mean 4473.282 us

So the next performance target must keep MLP2 while adding attention2. Do not
port attention2 + MLP1 to chunk=28 as a speed experiment; it would be slower
than the current baseline unless it is used only as a diagnostic ladder.
```

## Current Blockers

Resolved blockers in this round:

```text
attention2 + MLP1 RuntimeEndpoint output exhaustion:
  fixed by:
    prepacking QK/RoPE runtime metadata
    merging Q and K projection weight streams per attention column
  recheck:
    preflight passes with compute_cores=28, total_dma_tasks=20
    full-layer verify passes with chunk_hidden_errors=0

QKV merged worker deadlock:
  failed as:
    Qwen3PreflightError: Compute tile has 3 output ObjectFIFOs
  then as:
    ERT_CMD_STATE_TIMEOUT when Q was produced before K
  fixed by:
    QK-only Worker, V Worker separate
    QK weights interleaved by KV-head group so production matches qk_pair
```

Current open blocker:

```text
attention2 + MLP2 full layer:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 2
      --attention-columns 2 --layer-iterations 1 --preflight-only
  current failure:
    ValueError: Failed to find a tile matching column 3: tried until column 8.
  trace command:
    real_graph_probe.py --stages n-layer-final-only --columns 2
      --attention-columns 2 --layer-iterations 1 --preflight-only
      --allow-failures --trace-placement
  trace evidence:
    placement_trace_fail_key=runtime_output
    placement_trace_fail_type=RuntimeEndpoint
    placement_trace_fail_output=True
    placement_trace_fail_common_col=3
    placement_trace_fail_counts runtime_output=16, runtime_input=6
    placement_trace_fail_remaining_tiles=[]
  interpretation:
    after QK merge and post-norm/gate-up fusion this is no longer the old
    direct compute-tile exhaustion. The graph is one host->NPU runtime output
    stream over the current shim-output capacity.
```

Down-only MLP2 probe on 2026-05-21:

```text
command:
  real_graph_probe.py --stages n-layer-final-only --columns 2
    --attention-columns 2 --mlp-gate-up-columns 1 --layer-iterations 1
    --preflight-only --trace-placement

preflight:
  ok
  compute_cores=30
  total_dma_tasks=22
  max_dma_tasks_per_fifo=1
  max_tile_inputs=2
  max_tile_outputs=2
  placement_trace_counts runtime_output=16, runtime_input=6

verify/timing:
  command:
    main.py --stage n-layer-final-only --verify --verify-repeat 5
      --layer-chunk-size 1 --num-aie-columns 2 --attention-columns 2
      --mlp-gate-up-columns 1
  chunk_hidden_errors=0
  current K/V cache errors=0
  warm iterations 1-4:
    5420.301, 6004.149, 5946.693, 5366.060 us
  warm mean: 5684.301 us
```

Interpretation:

```text
Down-only MLP2 is a valid diagnostic ladder, but it is still slower than the
accepted attention1 + full MLP2 baseline:
  attention2 + down-only MLP2: 5684.301 us
  attention1 + full MLP2:      4473.282 us

Do not port down-only MLP2 to chunk=28 as the main performance path.
```

Full MLP2 post-norm/gate-up fusion attempt:

```text
Change:
  for attention2 + full MLP2, the post-norm Worker was fused into gate/up
  column 0. This removes the independent mlp_post_norm_weight Runtime.fill and
  the mlp_postnorm_worker.

Result:
  full MLP2 still fails preflight, but much later:
    placement_trace_fail_counts runtime_output=16, runtime_input=6
    placement_trace_fail_key=runtime_output
    placement_trace_fail_common_col=3
    placement_trace_fail_remaining_tiles=[]

Interpretation:
  the graph is now one host->NPU runtime output stream over the current NPU2
  shim-output capacity. The next fix must remove or combine one more
  Runtime.fill stream; TAP tuning is not supported by the evidence.
```

## Decision Rules

Do not spend more time on these unless new evidence changes the diagnosis:

```text
chunk-size tuning:
  chunk=28 is already the accepted baseline. Smaller chunks are host-dispatch
  regressions unless needed for diagnosis.

MLP2 layout:
  packed two-column MLP is accepted. Only revisit it if attention2 integration
  creates a concrete resource or correctness conflict.

bucket compilation:
  defer until one artifact can handle multiple decode positions. Otherwise
  buckets multiply position-specific artifacts and do not improve NPU runtime.

columns=4:
  defer until columns=2 attention/O-proj has a measured speedup and known
  resource pattern.
```

## Next N Steps

Keep this queue executable. When one step completes, archive the result and
move the next unresolved item to the top.

```text
1. Make attention2 + full MLP2 fit by reducing one more host->NPU Runtime.fill
   stream, not by guessing TAPs.
   Current evidence:
     after post-norm/gate-up fusion, placement fails after runtime_output=16
     with the 17th runtime output endpoint
   First candidates:
     combine the two gate/up shard fills with an ObjectFifo split/link design
     if it does not add an expensive compute-copy worker
     combine another small metadata stream with an existing stream only if the
     consumer order is proven
     inspect whether manual placement can use a remaining shim output channel;
     if remaining_tiles=[], assume a stream must be removed
   Acceptance:
     real_graph_probe.py --columns 2 --attention-columns 2
       --layer-iterations 1 --preflight-only passes

2. Use down-only MLP2 only as a diagnostic ladder.
   Acceptance:
     do not port it to chunk=28 for speed unless a later timing result changes
     the conclusion above

3. Port the accepted faster attention2 + full MLP2 shape to layer_chunk_size=28.
   Acceptance:
     preflight passes
     default/Fibonacci token_match=True
     NPU layer time improves from the current 120-138 ms range

4. Only after compute runtime improves, start runtime-position reuse.
   First target:
     one max_seq_len=256 artifact that handles multiple decode positions
   Later:
     optional 64/128/256/512 bucket artifacts selected at runtime
```
