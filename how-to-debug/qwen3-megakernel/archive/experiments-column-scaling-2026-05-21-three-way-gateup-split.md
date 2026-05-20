<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Three-Way Gate/Up With Split Runtime Stream

This archive records the packed/split follow-up to the rejected independent
three-way gate/up branch.

## Hypothesis

The independent three-way branch failed because it needed a 17th host->NPU
runtime output endpoint. The follow-up packed gate/up shard 1 and shard 2
behind one runtime stream, then used `ObjectFifo.split()` inside the graph:

```text
runtime gate/up stream 0:
  post_norm + shard0

runtime gate/up stream 12_pair:
  [shard1 row object][shard2 row object]
  [shard1 row object][shard2 row object]
  ...

ObjectFifo.split offsets:
  shard1 offset 0
  shard2 offset hidden_size
```

This keeps three compute workers but avoids a new runtime output endpoint and
does not route multi-megabyte weights through a compute copy Worker.

## Resource Result

Preflight command:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 3 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations 28 \
  --preflight-only \
  --trace-placement \
  --build-dir build_qwen3_mlp_gateup3_split_preflight \
  --clean-build
```

Result:

```text
real_graph_probe: ok stage=n-layer-final-only cols=2 layers=28 phase=preflight
compute_cores=32
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
placement_trace_counts:
  {'runtime_output': 16, 'runtime_input': 5,
   'other_output': 0, 'other_input': 2}
```

Interpretation:

```text
The endpoint problem is solved. Runtime output count stayed at 16 instead of
requiring the failing 17th endpoint. The graph now uses all 32 NPU2 compute
cores, so there is no spare core for another routing worker.
```

## Correctness

Default prompt full compile/generate:

```text
token_match=True
new_text='Paris'
compute_cores=32
```

Fibonacci prompt positions 17-21:

```text
token_match=True for all checked NPU steps
new_text=' 5, 8,'
```

## Timing

Default prompt:

```text
three-way split first run: 106.367 ms
three-way split warm run:  104.377 ms
same-environment accepted two-way baseline: 105.774 ms
```

Fibonacci prompt:

```text
three-way split checked NPU steps:
  103.281, 103.631, 103.758, 106.344, 105.198 ms
  mean: 104.442 ms

same-environment accepted two-way baseline:
  102.577, 104.654, 101.902, 98.724, 103.220 ms
  mean: 102.215 ms
```

Static estimator at position 26:

```text
kernel_calls_per_token: 76,216
estimated_macs_per_token: 443.498M
MLP gate/up matvec: 21,504 calls/token
```

The three-way graph did not reduce MLP gate/up work. It only redistributed the
same 768 gate/up row groups per layer across three workers and added a two-level
join:

```text
shard0 + shard1 -> pair01
pair01 + shard2 -> full ffn_hidden
```

## Decision

Rejected as the next accepted performance path.

The split parent stream is valuable because it proves how to remove the
runtime-output endpoint blocker without a weight copy Worker, but the current
three-way gate/up shape does not beat the accepted two-way direct-SiLU baseline
on a multi-token prompt. It also consumes all 32 compute cores, leaving no local
headroom for another corrective Worker.

Next implication:

```text
Endpoint reduction is now understood. The next performance work should either
re-run phase timing on the accepted two-way graph or move to a reusable
descriptor/state-machine experiment. More static MLP shard count alone is not a
proven speed lever.
```
