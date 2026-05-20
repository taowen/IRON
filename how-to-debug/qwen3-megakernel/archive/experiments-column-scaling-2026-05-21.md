<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments Archive, 2026-05-21

This is the detailed historical log that was moved out of the active
`experiments-column-scaling.md` decision page. Keep it as evidence for accepted
and rejected column-scaling experiments; do not use it as the current next-step
queue.

The text below is preserved from the previous long-form experiment log. Its
section names reflect the state at archive time, not necessarily the current
active queue.

## Current Baseline

Accepted fast-generate path:

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
MLP layout: packed two-column segment-major weights
attention layout: still single-column
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
  compute_cores=26
  total_dma_tasks=18
  max_dma_tasks_per_fifo=1
  max_fifo_buffered_bytes=32768
  max_tile_inputs=2
  max_tile_outputs=2

timing:
  same-build baseline suite on 2026-05-21:
    default prompt: 138.077ms, 1 NPU decode step before EOS
    Fibonacci prompt: mean 131.179ms, min 130.028ms, max 133.291ms
    weekdays prompt: mean 121.899ms, min 120.276ms, max 123.286ms
    numeric sequence prompt: mean 125.676ms, min 124.116ms, max 127.984ms
```

The current path solved two separate bottlenecks:

```text
8 + 8 + 8 + 4 host dispatch -> one layer_chunk_size=28 dispatch
single-column MLP -> two-column packed gate/up/down MLP
```

It did not implement a descriptor-driven layer state machine. The compiled
Runtime sequence is still static for a decode position, and attention/O-proj
still leaves useful column parallelism on the table.

## Bottleneck Hypothesis

After packed weights, cache residency, prefix K/V blocks, output-only drains,
full-depth chunking, and packed two-column MLP, the next execution-time win is
not another chunk-size change. The remaining latency is mainly the part of the
full-depth graph that still runs single-column or serializes around joins:

```text
attention Q/K/V, score, softmax, PV
context packing
O projection
residual feedback between layers
```

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
QKV checkpoint:
  can preflight at multiple columns.

post-attn full-MLP checkpoint:
  verifies at columns=1,2,4 with down projection sharded by column.

n-layer chunk=1 path:
  verifies at columns=1,2,4 with attention still single-column and MLP down
  projection sharded by column.

n-layer chunk=28 path:
  accepted at columns=2 with packed two-column MLP.
  token-level generate matches the reference on default and Fibonacci prompts.

columns=4 beyond layer_iterations=1:
  not accepted yet.
```

The full-depth accepted path is not a general multi-column graph. It is a
specific, proven shape: attention single-column plus packed two-column MLP.
Removing guards or increasing the public column knob is not enough; each phase
needs an explicit data layout, FIFO shape, and resource check.

## Archived Experiment 1: Two-Column MLP Closure

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

Archived status:

```text
Accepted. Do not keep iterating on MLP2 layout unless a later attention/O-proj
change exposes a concrete resource or correctness failure.
```

## Active Experiment 2: Two-Column Attention Head And O-Projection Shard

Goal:

```text
Shard Q/K/V, score, softmax, PV, context packing, and O projection by attention
head groups so the current two-column path uses column 1 outside the MLP too.
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

Design note before implementation:

```text
Do not just pass num_columns=2 into the current attention code.

The public column knob currently reaches MLP only. Inside
_qwen3_persistent_n_layer_final_only_impl(), attention is still called with
num_columns=1, and q/k/v/o weight fills use one contiguous full segment. If
that guard is removed without a new layout, both attention columns would read
the same leading rows and compute duplicate heads.
```

Proposed two-column ownership:

```text
model constants:
  hidden_size=1024
  q_size=2048
  kv_size=1024
  head_dim=128
  q_heads=16
  kv_heads=8
  GQA repeat=2

attention_columns=2:
  col0 owns KV heads 0-3 and Q heads 0-7
  col1 owns KV heads 4-7 and Q heads 8-15

per-column shapes:
  q_shard_size=1024
  kv_shard_size=512
  context_shard_size=1024
```

Required weight layout:

```text
Use a packed segment-major attention2 layout, not strided per-layer runtime
fills:

  input_norm: full, all layers
  W_q_col0, W_q_col1:
    row shards [0:1024] and [1024:2048], each layer contiguous
  W_k_col0, W_k_col1:
    row shards [0:512] and [512:1024], each layer contiguous
  W_v_col0, W_v_col1:
    row shards [0:512] and [512:1024], each layer contiguous
  W_qk_norm:
    full 2*head_dim metadata, all layers
  W_o_col0, W_o_col1:
    context-column shards, not output-row shards.
    col0 owns W_o[:, 0:1024], col1 owns W_o[:, 1024:2048].
    Each column computes a full hidden partial output.
  MLP2:
    keep the accepted post_norm / gate_up_col0 / gate_up_col1 /
    down_col0 / down_col1 layout.
```

The O projection should shard over context dimension, not hidden rows:

```text
context = concat(context_col0, context_col1)
O = W_o[:, 0:1024] @ context_col0 + W_o[:, 1024:2048] @ context_col1

Therefore:
  each attention column produces context_shard[1024]
  each O worker produces o_partial[1024]
  a two-input o_partial_sum worker adds partial0 + partial1 into full O[1024]
  residual_add consumes hidden[1024] + O[1024] and produces one full residual
```

This avoids broadcasting full context to both columns. It does add one full
hidden reduce after O-proj, but that reduce is cheaper and clearer than a
full-context broadcast plus output-row sharding.

New or changed FIFO names:

```text
qwen3_rc_q_weight_0/1:
  now carry W_q row shards, one stream per attention column
qwen3_rc_k_weight_0/1:
  now carry W_k row shards
qwen3_rc_v_weight_0/1:
  now carry W_v row shards
qwen3_rc_q_raw_0/1:
  8 Q head tokens per layer per column
qwen3_rc_k_raw_0/1:
  4 K head tokens per layer per column
qwen3_rc_v_0/1:
  4 V head tokens per layer per column
qwen3_rc_k_cache_0/1:
  cache reads for KV head ranges 0-3 and 4-7
qwen3_rc_v_cache_0/1:
  cache reads for KV head ranges 0-3 and 4-7
qwen3_rc_attn_context_flat_0/1:
  should become context_shard[1024], not full context[2048]
qwen3_rc_o_weight_0/1:
  carry W_o context-column shards with DIM_K=1024
qwen3_rc_attn_o_partial_0/1:
  new full hidden partial outputs from O-proj
qwen3_rc_attn_o_sum:
  new full hidden sum of both partial O projections
qwen3_rc_attn_residual:
  single full residual after hidden + o_sum
```

Required TAP / pack changes:

```text
segment_major_q_weight_shard_taps()[col]:
  offset = segment_q_weight_base + col * q_shard_size * hidden_size
  length = layer_iterations * q_shard_size * hidden_size

segment_major_k_weight_shard_taps()[col]:
  offset = segment_k_weight_base + col * kv_shard_size * hidden_size
  length = layer_iterations * kv_shard_size * hidden_size

segment_major_v_weight_shard_taps()[col]:
  offset = segment_v_weight_base + col * kv_shard_size * hidden_size
  length = layer_iterations * kv_shard_size * hidden_size

segment_major_o_weight_context_shard_taps()[col]:
  prefer host-packed contiguous W_o[:, col*q_shard_size:(col+1)*q_shard_size].
  A raw row-major strided TAP is possible but should not be the first attempt
  because this project already solved MLP2 scaling by host-packing shards.

cache key/value block taps:
  add kv_head_start = col * kv_heads_per_col
  base += kv_head_start * max_seq_len * head_dim
  dims = [layer_count, kv_heads_per_col, active_seq, head_dim]
  strides = [cache_size, max_seq_len * head_dim, head_dim, 1]

cache current write taps:
  add the same kv_head_start offset for key and value current-position writes.
```

Kernel artifact changes:

```text
The current O-proj mv object is compiled with DIM_K=q_size=2048.
Context-column sharding needs an O-proj shard object compiled with
DIM_K=q_shard_size=1024, using distinct symbol/object names so the old path can
remain available while the probe is developed.
```

Probe order:

```text
1. Add an attention_columns variable separate from mlp_down_columns. [done]
2. Keep attention_columns=1 as the default accepted path. [done]
3. Add a probe path with attention_columns=2 and layer_iterations=1. [done]
4. Implement packed attention2 Q/K/V/O shard layout and TAPs. [done enough
   for attention_probe_only preflight]
5. Validate Q/K/V row ownership with a monotonic pattern or reference boundary.
6. Validate cache read/write TAPs with current-position K/V checks.
7. Validate attention context shards before O-proj.
8. Validate O partial sum + residual before reconnecting MLP.
9. Only then port to layer_iterations=28.
```

Probe scaffolding result on 2026-05-21:

```text
Implemented:
  Qwen3PersistentNLayerFinalOnly.attention_columns, default 1
  operator name alias: attncol
  real_graph_probe.py --attention-columns
  design wrapper support for attention_columns=2
  attention_probe_only single-layer resource probe

Accepted baseline probe:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 2
      --attention-columns 1 --layer-iterations 28 --preflight-only
  result:
    ok
    compute_cores=26
    total_dma_tasks=18
    max_dma_tasks_per_fifo=1
    max_fifo_buffered_bytes=32768
    max_tile_inputs=2
    max_tile_outputs=2

Initial attention2 probe before graph support:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 2
      --attention-columns 2 --layer-iterations 1 --preflight-only
      --allow-failures
  result:
    fails before MLIR generation with a precise ValueError:
      attention_columns=2 needs the attention2 layout before it can generate
      MLIR: packed Q/K/V row shards, cache head-offset TAPs,
      context-shard FIFOs, W_o context-column shards, and an O partial
      sum worker.
```

Interpretation:

```text
This early failure was intentional at the time. It prevented a misleading
"two-column" graph that duplicated the leading attention heads on both columns.
The guard has since been replaced by real attention2 graph scaffolding and the
attention_probe_only preflight.
```

Packed attention2 weight-layout progress on 2026-05-21:

```text
Implemented host-side pack helpers:
  pack_segment_major_full_layer_weights(..., attention_columns=2, mlp_columns=2)
  pack_segment_major_full_layer_weights(..., attention_columns=2, mlp_columns=1)
  pack_segment_major_weights_for_layers(..., attention_columns=2, mlp_columns=2)
  pack_segment_major_weights_for_layers(..., attention_columns=2, mlp_columns=1)
  pack_segment_major_weight_chunk_from_layer_major(
    ..., attention_columns=2, mlp_columns=2
  )
  pack_segment_major_weight_chunk_from_layer_major(
    ..., attention_columns=2, mlp_columns=1
  )

The new layout packs, by segment and then by column:
  input_norm
  W_q_col0, W_q_col1
  W_k_col0, W_k_col1
  W_v_col0, W_v_col1
  W_qk_norm
  W_o_context_col0, W_o_context_col1
  post_norm
  MLP gate/up col0, MLP gate/up col1
  MLP down col0, MLP down col1

Validation:
  synthetic bf16 test compared direct full-layer-input packing against
  layer-major-artifact repacking for:
    attention_columns=1, mlp_columns=1
    attention_columns=1, mlp_columns=2
    attention_columns=2, mlp_columns=1
    attention_columns=2, mlp_columns=2
  all four produced identical bf16 bit patterns and unchanged total numel.

Graph-side progress:
  attention_columns=2 now has Q/K/V/O packed-shard TAPs, KV cache head-offset
  TAPs, context_shard[1024] FIFOs, an O-proj object with DIM_K=1024, and an O
  partial-sum worker.

Preflight linter update:
  qwen3_preflight.py now derives the expected O-proj DIM_K from the linked
  qwen3_persistent_gemv_<DIM_K>k_*_o_proj.o object, so DIM_K=2048 remains
  checked for the default path and DIM_K=1024 is accepted for attention2.
```

Attention2 resource probes on 2026-05-21:

```text
Accepted baseline:
  columns=2 attention_columns=1 layer_iterations=28
  result: ok
  compute_cores=26 total_dma_tasks=18 max_dma_tasks_per_fifo=1
  max_fifo_buffered_bytes=32768 max_tile_inputs=2 max_tile_outputs=2

Attention-only probe:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 1
      --attention-columns 2 --attention-probe-only --layer-iterations 1
      --preflight-only
  result: ok
  compute_cores=27 total_dma_tasks=21 max_dma_tasks_per_fifo=1
  max_fifo_buffered_bytes=32768 max_tile_inputs=2 max_tile_outputs=1

Attention2 + MLP1 full layer:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 1
      --attention-columns 2 --layer-iterations 1 --preflight-only
  result:
    fails placement on a RuntimeEndpoint output:
      Failed to find a tile matching column 1: tried until column 8
  diagnosis:
    shim/runtime output endpoints are exhausted. A monkey-patched
    SequentialPlacer showed the failing endpoint is RuntimeEndpoint
    output=True. The attention-only probe fits because it skips the MLP
    weight fill endpoints.

Attention2 + MLP2 full layer:
  command:
    real_graph_probe.py --stages n-layer-final-only --columns 2
      --attention-columns 2 --layer-iterations 1 --preflight-only
  result:
    fails placement with:
      Ran out of compute tiles for placement!
  diagnosis:
    this is not a TAP or ABI issue. The graph now has too many compute workers
    when both attention and MLP are expanded to two columns.
```

Interpretation:

```text
The attention2 dataflow can now be expressed and statically preflighted in
isolation. The next bottleneck is reconnecting it to the full layer without
exceeding either shim runtime output endpoints or compute tiles.
```

Attention-only numerical verification on 2026-05-21:

```text
Command:
  main.py --stage n-layer-final-only --verify --layer-chunk-size 1
    --num-aie-columns 1 --attention-columns 2 --attention-probe-only
    --build-dir build_qwen3_attncol_probe_attention_only_verify --clean-build

Compile/preflight:
  compile_s=65.948
  compute_cores=27
  max_fifo_buffered_bytes=32768
  max_dma_tasks_per_fifo=1
  max_tile_inputs=2
  max_tile_outputs=1
  non_advancing_acquires=0

Runtime:
  decode_position=26
  npu_time_us=4440.436

Checks:
  attention_probe_residual_errors=0
  attention_probe_residual_max_abs=0.015625
  attention_probe_residual_mean_abs=0.002417
  layer0_keys_cache_current_errors=0
  layer0_values_cache_current_errors=0
```

Interpretation:

```text
The two-column attention/O-proj shard is numerically correct at the
attention-residual boundary for the tested decode position. Q/K/V head
ownership, cache head-offset writeback, context_shard[1024], O-proj
DIM_K=1024, and O partial sum are now validated in the isolated probe.
```

Baseline timing suite on 2026-05-21:

```text
Command shape:
  --stage generate
  --fast-generate
  --verify-generate
  --require-packed-weights
  --layer-chunk-size 28
  --num-aie-columns 2
  --build-dir build_qwen3_column_baseline_suite

default prompt:
  decode positions: 26
  token_match: 1/1
  new_text='Paris'
  npu_layer_time_ms: min 138.077, mean 138.077, max 138.077
  note: stops after EOS, so this prompt has only one NPU decode step

raw Fibonacci prompt "Fibonacci numbers: 1, 1, 2, 3,":
  decode positions: 17-21
  token_match: 5/5
  new_text=' 5, 8,'
  npu_layer_time_ms: min 130.028, mean 131.179, max 133.291

raw weekdays prompt "Monday, Tuesday, Wednesday,":
  decode positions: 6-10
  token_match: 5/5
  new_text=' Thursday, Friday, Saturday,'
  npu_layer_time_ms: min 120.276, mean 121.899, max 123.286

raw numeric sequence prompt "2, 4, 6, 8,":
  decode positions: 11-15
  token_match: 5/5
  new_text=' 10, 1'
  npu_layer_time_ms: min 124.116, mean 125.676, max 127.984
```

Interpretation:

```text
The baseline is stable enough to optimize against. Shorter decode positions are
faster because the single-column attention path reads fewer active cache
blocks; this reinforces that attention/O-proj is now the right next target.
All tested NPU decode steps matched the CPU reference, so the next experiment
does not need to revisit MLP2 correctness before changing attention layout.
```

## Archived Experiment 3: Two-Column Full-Depth Generate

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

Archived status:

```text
Accepted. chunk=28 cols=2 packed MLP2 compiles, preflights, and matches token
generation on default and Fibonacci prompts. The old 8+8+8+4 chunk-dispatch
pattern is no longer the active baseline.
```

## Deferred Experiment 4: Runtime Position And Bucket Compilation

Goal:

```text
Make one compiled artifact reusable across multiple decode positions, then
optionally compile max_seq_len/cache buckets such as 64/128/256/512.
```

Current observation:

```text
chunk=28 first-position compile is about 66s.
Following position compiles in recent runs are about 5s.
The runtime sequence still bakes position-dependent TAP and mask constants.
```

Candidate method:

```text
First remove or patch position-specific values:
  cache read/write offset
  active cache block count
  causal mask length
  RoPE position
  KV read length

Then compile a small number of max_seq_len buckets:
  64 / 128 / 256 / 512

At runtime, choose the smallest bucket that can hold prompt_len + generated_len.
```

Do not start with buckets while `position` is still baked into the graph. That
would multiply artifacts by `bucket * position` and would not solve the main
reuse problem.

## Next N Steps

Keep this list short and executable. When a step completes, archive the result
above and move the next unresolved item to the top.

```text
1. Reconnect sharded attention to the full layer without exceeding resources.
   Current resource blockers:
     attention2 + MLP1: shim/runtime output endpoint exhaustion
     attention2 + MLP2: compute tile exhaustion
   Candidate directions:
     reduce runtime output endpoints by combining or staging attention weight
     fills
     fuse adjacent lightweight workers before re-enabling MLP2
     keep MLP1 for the first correctness probe if endpoint count is reduced

2. Reconnect sharded attention to O-proj and full-layer output.
   Acceptance:
     layer_iterations=1 cols=2 verifies
     no debug-only third output remains on hot tiles
     timing improves or trace explains why it does not

3. Port the attention/O-proj shard into chunk=28.
   Acceptance:
     preflight passes
     default and Fibonacci token_match=True
     NPU layer time improves from the current 120-138ms range

4. If step 3 does not improve speed, collect trace before changing structure.
   Required evidence:
     per-phase latency
     worker idle/active ratio if available
     DMA wait or FIFO wait symptom
     exact first bottlenecked FIFO/tile

5. Only after the compute path is faster, start runtime-position reuse.
   First target one max_seq_len=256 artifact that can handle multiple positions.
   Bucket variants come after this, not before.

6. Revisit columns=4 only after columns=2 attention/O-proj has a verified
   speedup and the resource pattern is understood.
```
