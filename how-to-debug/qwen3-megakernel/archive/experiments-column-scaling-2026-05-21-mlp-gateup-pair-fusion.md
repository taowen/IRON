<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Archived Column-Scaling Evidence: MLP Gate/Up Pair Fusion

This archive records the evidence that promoted the paired-row MLP gate/up
matvec path to the active baseline. Keep the active
`../experiments-column-scaling.md` page focused on current decisions and next
steps.

## Starting Point

Accepted baseline before this branch:

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: two-column attention2
attention score/softmax: fused per attention column
MLP layout: packed two-column segment-major gate/up/down weights
MLP gate/up compute: separate gate matvec, up matvec, then SiLU/mul
final norm / LM head: CPU F.linear path
weights: prepacked bf16 weights on disk
```

Static work estimate for that graph at position 26:

```text
kernel_calls_per_token: 97,720
estimated_macs_per_token: 443.498M
rough_bf16_element_visits_per_token: 560,504,448
MLP gate/up matvec calls/token: 43,008
MLP gate/up est MACs/token: 176.161M
```

Decision:

```text
MLP gate/up is the largest estimated MAC block and also has the highest
external-kernel call count. Test a gate/up pair boundary before spending the two
free cores on a wider graph.
```

## Paired-Row Boundary

Experiment switch:

```text
--mlp-gate-up-pair-rows
```

Graph change:

```text
Default MLP2 gate/up shard layout:
  [all gate rows for shard][all up rows for shard]

Paired-row layout:
  [4 gate rows][matching 4 up rows] repeated for each shard

Worker behavior during boundary proof:
  acquire 8 weight rows from the same gate/up FIFO
  call the old 4-row matvec twice
  keep the old SiLU/mul path
```

Preflight command:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --layer-iterations 28 \
  --preflight-only \
  --build-dir build_qwen3_mlp_gateup_pair_preflight \
  --clean-build
```

Preflight evidence:

```text
real_graph_probe: ok
compute_cores=30
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
```

MLIR evidence:

```text
qwen3_full_layer_mlp_gate_up_weight_0 depth: 8
qwen3_full_layer_mlp_gate_up_weight_1 depth: 8
gate/up Workers acquire Consume, 8 and release Consume, 8
```

Boundary correctness check:

```text
operator_name includes mlpgatepair1
default prompt token_match=True
new_text='Paris'
npu_layer_time_ms=110.878
```

Interpretation:

```text
The data boundary was legal and numerically correct, but it was not expected to
speed up because the Worker still called the old 4-row matvec twice and then
called the old SiLU/mul kernel.
```

## Paired Matvec Kernel

Implemented external kernel:

```text
qwen3_mlp_gate_up_pair_matvec4_rows_shard_bf16(
  4 gate rows,
  4 matching up rows,
  xnorm[1024],
  gate_output shard,
  up_output shard)
```

The kernel halves the gate/up matvec call count by computing four gate rows and
the matching four up rows in one call. It does not yet fuse SiLU/mul directly
into the hidden shard.

Preflight command:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --layer-iterations 28 \
  --preflight-only \
  --build-dir build_qwen3_mlp_gateup_pair_fused_preflight \
  --clean-build
```

Accepted preflight evidence:

```text
real_graph_probe: ok
compute_cores=30
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
non_advancing_acquires=0
```

Static work estimate at position 26:

```text
kernel_calls_per_token: 76,216
estimated_macs_per_token: 443.498M
rough_bf16_element_visits_per_token: 538,484,352
MLP gate/up matvec calls/token: 21,504
```

Call-count comparison:

```text
total external kernel calls:
  97,720 -> 76,216

MLP gate/up matvec calls:
  43,008 -> 21,504

MAC count:
  unchanged at 443.498M estimated MACs/token

rough bf16 element visits:
  560,504,448 -> 538,484,352
```

## Prompt Suite

Default prompt:

```text
position 26
token_match=True
new_text='Paris'
npu_layer_time_ms=106.914
```

Raw Fibonacci prompt, `"Fibonacci numbers: 1, 1, 2, 3,"`:

```text
positions 17-21
verified NPU decode steps: 5/5
new_text=' 5, 8,'
npu_layer_time_ms: min 99.701, mean 102.262, max 104.871
```

Raw weekdays prompt, `"Monday, Tuesday, Wednesday,"`:

```text
positions 6-10
verified NPU decode steps: 5/5
new_text=' Thursday, Friday, Saturday,'
npu_layer_time_ms: min 97.140, mean 98.682, max 100.235
```

Raw numeric prompt, `"2, 4, 6, 8,"`:

```text
positions 11-15
verified NPU decode steps: 5/5
new_text=' 10, 1'
npu_layer_time_ms: min 99.449, mean 101.138, max 102.889
```

Raw free-form prompt, `"The opposite of hot is"`:

```text
positions 5-9
verified NPU decode steps: 5/5
new_text=' cold, and the opposite of'
npu_layer_time_ms: min 96.750, mean 99.421, max 100.738
```

Suite comparison:

```text
score-side fused-softmax baseline average:
  105.737 ms

paired gate/up matvec average:
  101.683 ms

delta:
  -4.054 ms

relative improvement:
  3.83%
```

## Decision

Status:

```text
accepted
```

Reason:

```text
The branch preserved token correctness across the prompt suite, did not
increase resource pressure, and reduced measured NPU graph time by about 4.1 ms
on the suite average.
```

Remaining limitation:

```text
This is not yet a fully fused MLP gate/up activation path. The graph still calls
qwen3_silu_mul_shard_bf16 after the paired matvec kernel, and MLP gate/up
remains the largest estimated MAC category.
```
