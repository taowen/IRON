<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Archived Column-Scaling Evidence: Softmax Fusion And Projection Boundary

This archive records the detailed evidence behind the current active baseline in
`../experiments-column-scaling.md`. Keep future raw logs here or in another
dated archive; keep the active page focused on decisions and next steps.

## Accepted Baseline

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: two-column attention2
attention score/softmax: fused per attention column
runtime input: hidden[1024] || QK/RoPE metadata[384] per layer
MLP layout: packed two-column segment-major gate/up/down weights
final norm / LM head: CPU F.linear path
weights: prepacked bf16 weights on disk
```

Correctness evidence:

```text
default prompt:
  verified NPU decode steps: 1/1
  new_text='Paris'

raw Fibonacci prompt "Fibonacci numbers: 1, 1, 2, 3,":
  verified NPU decode steps: 5/5
  new_text=' 5, 8,'

raw weekdays prompt "Monday, Tuesday, Wednesday,":
  verified NPU decode steps: 5/5
  new_text=' Thursday, Friday, Saturday,'

raw numeric sequence prompt "2, 4, 6, 8,":
  verified NPU decode steps: 5/5
  new_text=' 10, 1'

raw free-form prompt "The opposite of hot is":
  verified NPU decode steps: 5/5
  new_text=' cold, and the opposite of'
```

Performance evidence for the promoted path:

```text
default prompt:
  position 26
  npu_layer_time_ms: 110.613

Fibonacci prompt:
  positions 17-21
  npu_layer_time_ms: min 101.891, mean 104.794, max 109.631

weekdays prompt:
  positions 6-10
  npu_layer_time_ms: min 101.033, mean 104.133, max 105.890

numeric sequence prompt:
  positions 11-15
  npu_layer_time_ms: min 102.099, mean 106.163, max 108.368

free-form prompt:
  positions 5-9
  npu_layer_time_ms: min 96.085, mean 102.981, max 106.142
```

Previous attention2/MLP2 pre-fusion prompt suite:

```text
default prompt:
  position 26
  npu_layer_time_ms: 113.954

Fibonacci prompt:
  positions 17-21
  npu_layer_time_ms: min 108.527, mean 109.881, max 111.412

weekdays prompt:
  positions 6-10
  npu_layer_time_ms: min 102.766, mean 103.597, max 104.199

numeric sequence prompt:
  positions 11-15
  npu_layer_time_ms: min 100.315, mean 104.018, max 106.382

free-form prompt:
  positions 5-9
  npu_layer_time_ms: min 95.839, mean 100.789, max 105.332
```

Speedup versus the previous accepted single-column-attention baseline:

```text
default prompt:    138.077 ms -> 110.613 ms
Fibonacci prompt:  mean 131.179 ms -> mean 104.794 ms
weekdays prompt:   mean 121.899 ms -> mean 104.133 ms
numeric sequence:  mean 125.676 ms -> mean 106.163 ms
free-form prompt:  measured after attention2 baseline, mean 102.981 ms
```

Comparison against pre-fusion attention2/MLP2:

```text
default prompt:    113.954 ms -> 110.613 ms
Fibonacci prompt:  mean 109.881 ms -> mean 104.794 ms
weekdays prompt:   mean 103.597 ms -> mean 104.133 ms
numeric sequence:  mean 104.018 ms -> mean 106.163 ms
free-form prompt:  mean 100.789 ms -> mean 102.981 ms

suite average over default plus four prompt means:
  106.448 ms -> 105.737 ms
```

## Static Cost Map

Evidence source:

```text
build_qwen3_score_softmax_fused_generate_default/
  Qwen3PersistentNLayerFinalOnly_h1024_q2048_kv1024_hd128_msl256_pos26_
  ffn3072_col2_attncol2_mlpgatecol2_attnprobe0_tsi4_tso128_
  epsilon1en06_layers28_npu2.mlir
```

Compute-tile allocation:

```text
total compute cores: 30

input norm / chunk state:
  2 cores
  tile_0_2: hidden copy to chunk state + unweighted input RMSNorm
  tile_0_3: multiply input RMSNorm by layernorm weight

QKV / RoPE / current-KV preparation:
  8 cores
  two QK matvec workers, two V matvec workers, four Q/K RoPE workers

attention QK / softmax:
  4 cores
  two QK-pair pack workers, two fused score+softmax workers

attention PV / context:
  4 cores
  two current-V merge workers, two context/pack workers

O-proj / attention residual:
  4 cores
  two O-proj workers, one O-proj sum worker, one residual-add worker

MLP gate/up + SiLU:
  3 cores
  two gate/up+SiLU shard workers, one ffn-hidden join worker

MLP down / residual:
  4 cores
  two down-proj workers, two residual slice-add workers

chunk feedback / final drain:
  1 core
  joins residual shards into feedback or final output
```

ObjectFIFO and runtime shape:

```text
ObjectFIFOs: 53 total
  attention/input: 38
  MLP:             12
  chunk:            3

ObjectFIFO depths:
  depth=2: 41 FIFOs
  depth=1: 10 FIFOs
  depth=4:  2 FIFOs

runtime memrefs: 5
runtime DMA tasks: 21
```

Largest FIFO objects by allocated bytes:

```text
qwen3_rc_k_cache_0/1:
  64x128xbf16, object=16384 bytes, depth=2, total=32768 bytes each

qwen3_full_layer_mlp_down_weight_0/1:
  4x3072xbf16, object=24576 bytes, depth=1, total=24576 bytes each

qwen3_rc_v_cache_0/1 and qwen3_rc_v_context_block_0/1:
  64x128xbf16, object=16384 bytes, depth=1, total=16384 bytes each

qwen3_rc_qk_weight_0/1, qwen3_rc_v_weight_0/1, qwen3_rc_o_weight_0/1:
  4x1024xbf16, object=8192 bytes, depth=2, total=16384 bytes each
```

## Softmax Fusion Decision

Accepted change:

```text
Fused each attention column's softmax Worker into its score Worker.

Before:
  score -> softmax Worker -> attn_weights FIFO -> context Worker

After:
  fused score+softmax Worker -> context Worker
```

Why legal:

```text
The fused score Worker still has two input ObjectFIFOs:
  QK pair
  K-cache block

and one output ObjectFIFO:
  softmax weights
```

Resource result:

```text
compute_cores: 32 -> 30
ObjectFIFOs:   55 -> 53
removed qwen3_rc_attn_weights_0/1 FIFOs
max_tile_inputs remains 2
max_tile_outputs remains 2
```

Correctness and timing recheck:

```text
default prompt position 26 token_match=True, new_text='Paris',
  npu_layer_time_ms=110.613

full prompt timing suite token_match=True for all verified NPU decode steps

suite average over default plus four prompt means:
  pre-fusion attention2/MLP2: 106.448 ms
  context-side fusion:       106.054 ms
  score-side fusion:         105.737 ms
```

Interpretation:

```text
This is primarily a resource-enabling optimization, but score-side fusion is
also the fastest tested softmax-fusion placement so far. It releases two
compute cores and gives a small suite-average improvement; individual prompts
are still mixed. The next performance step must spend the freed cores.
```

Rejected intermediate placement:

```text
Fusing softmax into the context Worker also preflighted at compute_cores=30
and matched tokens, but its full prompt suite averaged about 106.054 ms. The
score-side placement averaged about 105.737 ms, so keep score-side fusion.
```

Rejected first attempt:

```text
Tried to fuse tile_0_2 and tile_0_3 by replacing:
  unweighted RMSNorm Worker + norm-weight multiply Worker
with:
  weighted RMSNorm Worker that also copies chunk_hidden

Preflight failed immediately:
  Compute tile %tile_0_2 has 3 input ObjectFIFOs; limit=2.

Root cause:
  The fused full-depth Worker would need runtime hidden, hidden feedback, and
  input norm weight on the same compute tile. The current graph-level limit is
  max_tile_inputs=2, so this is not a legal way to free a core.
```

## Projection Widening Constraint

Static assessment after freeing two cores:

```text
O-proj currently uses:
  two O-proj Workers, each computing a full 1024-element partial output
  one full-vector partial-sum Worker
  one full-vector residual-add Worker

Splitting O-proj by output rows would require:
  four O-proj Workers for two attention columns x two output-row shards
  at least two partial-sum Workers
  residual-add and full-vector join behavior before MLP post-norm
```

Root constraint:

```text
The two freed cores are not enough for that direct row-sharded shape. Directly
fusing partial-sum with residual-add would need three input streams
(left partial, right partial, hidden residual), which violates the current
max_tile_inputs=2 invariant.
```

Direct full-depth cols=4 probe:

```text
real_graph_probe --columns 4 --attention-columns 2 --layer-iterations 28

failed before MLIR/preflight with:
  attention_columns=2 currently supports num_aie_columns=1 for the attention
  probe or num_aie_columns=2 for the packed MLP2 path
```

Implication:

```text
Do not start O-proj widening by simply adding output-row shards. A viable
projection-widening branch needs either a different residual/MLP boundary or an
external kernel that consumes a packed two-part input without adding a third
ObjectFIFO input.

The current operator guard also means a real cols=4 branch must explicitly
define the multi-stage joins before the guard is relaxed.
```
