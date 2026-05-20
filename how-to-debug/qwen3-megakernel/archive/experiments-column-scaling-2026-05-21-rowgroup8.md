<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gate/Up Row-Group 8 Branch

This archive records the no-new-runtime-output row-group experiment after the
direct gate/up+SiLU baseline.

## Hypothesis

The accepted graph was blocked from adding a third independent gate/up stream
by the runtime-output endpoint limit. A safer candidate was to keep the same two
gate/up runtime streams but reduce external-kernel call count:

```text
accepted:
  col0: qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16
  col1: qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16

candidate:
  col0: keep 4-row fused postnorm+gate/up stream
  col1: qwen3_mlp_gate_up_pair_silu8_rows_shard_bf16
```

The final implementation only widens the non-fused gate/up stream. The fused
col0 stream contains one post-norm row followed by gate/up rows, so changing it
to 4-row block objects would require padding, a separate post-norm endpoint, or
a more invasive packed object layout.

## First Failure

The first implementation made both gate/up shard FIFOs use row-group 8 by
acquiring 16 single-row FIFO objects per call:

```text
rows = gate_up_weight_fifo.acquire(16)
```

Static preflight passed:

```text
compute_cores=30
total_dma_tasks=21
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
placement_trace_counts:
  {'runtime_output': 16, 'runtime_input': 5,
   'other_output': 0, 'other_input': 1}
```

Full `aiecc` failed later:

```text
error: 'aie.mem' op has more than 16 blocks
note: no space for this BD
Pipeline failed while executing AIEObjectFifoStatefulTransform
```

Root cause:

```text
This was not a placement endpoint problem and not external-kernel math. The
row-group 8 FIFO depth became 16 single-row objects on a busy memory tile, and
resource allocation could not assign the extra ObjectFIFO blocks/BDs. The
current Python preflight did not catch this class.
```

## Root-Cause Fix

The second implementation kept col0 at row-group 4 and packed only non-fused
gate/up streams as 4-row FIFO objects:

```text
col0 fused stream:
  hidden_weight_ty objects, depth 8
  post_norm row + [4 gate rows][4 up rows]

col1 widened stream:
  mlp_gemv_a_ty objects, depth 4
  [4 gate rows][4 gate rows][4 up rows][4 up rows]
```

The 8-row kernel accepts four block objects instead of 16 row objects. This
keeps the runtime endpoint count unchanged and avoids the 16-block FIFO depth.

## Result

Preflight:

```text
real_graph_probe: ok stage=n-layer-final-only cols=2 layers=28 phase=preflight
compute_cores=30
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
placement_trace_counts:
  {'runtime_output': 16, 'runtime_input': 5,
   'other_output': 0, 'other_input': 1}
```

Default prompt generate:

```text
token_match=True
new_text='Paris'
row-group 8 NPU time: 109.638 ms
same-environment accepted baseline NPU time: 106.233 ms
```

Static estimator at position 26:

```text
accepted direct baseline calls/token: 76,160
row-group 8 calls/token:              70,784

MLP gate/up matvec:
  qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16: 10,752 calls/token
  qwen3_mlp_gate_up_pair_silu8_rows_shard_bf16:  5,376 calls/token
  total MLP gate/up calls/token:                16,128
```

## Decision

Rejected as a performance branch.

The branch is correct and reduces static call count, but it is slower on the
real full-depth decode graph. The likely reason is that the wider kernel and
larger FIFO objects trade away more scheduling/local-memory efficiency than the
saved call overhead is worth. The accepted baseline remains direct 4-row
gate/up+SiLU.

Next implication:

```text
Do not keep pushing row-group size as the primary speed path. More MLP
parallelism needs a packed/split gate/up stream behind an existing endpoint, or
a broader state-machine/resource reuse redesign.
```
