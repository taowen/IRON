<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Three-Way MLP Gate/Up Branch

This archive records the first Step 4 MLP-side widening branch after accepting
direct gate/up+SiLU.

## Hypothesis

The four-way gate/up idea does not fit the current graph shape:

```text
single four-input join:
  likely fits worker count, but violates max_tile_inputs<=2

two-level binary join for four shards:
  preserves max_tile_inputs<=2, but adds four workers total and would exceed
  the 32-compute-core budget from the current 30-core baseline
```

The three-way branch was chosen because it uses exactly the two apparent spare
compute cores:

```text
current two-way:
  gate/up workers: 2
  ffn hidden join workers: 1
  total: 3

three-way:
  gate/up workers: 3
  ffn hidden join workers: 2
  total: 5

delta: +2 workers
```

The graph keeps the two-input/two-output invariant by using a two-level join:

```text
shard0 + shard1 -> pair01
pair01 + shard2 -> full ffn_hidden
```

## Implementation Shape

Code changes:

```text
mlp_gate_up_columns accepts 3
ffn_shard_ty becomes intermediate_size / mlp_gate_up_columns
three-way uses ffn_shard_size=1024 and ffn_pair_size=2048
col0 still fuses post-norm + gate/up and broadcasts xnorm
col1/col2 consume xnorm and their own packed gate/up weight streams
down projection remains two-column and consumes the joined full ffn_hidden
```

Static local checks:

```text
layout pack/unpack test passes for a valid 96-row fake intermediate
Qwen3PersistentNLayerFinalOnly accepts mlp_gate_up_columns=3 with paired rows
accepted two-way direct baseline still preflights after the change
```

## Preflight Failure

Command:

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
  --build-dir build_qwen3_mlp_gateup3_direct_silu_preflight_trace \
  --clean-build
```

Result:

```text
real_graph_probe: fail stage=n-layer-final-only cols=2 layers=28 phase=preflight
ValueError: Failed to find a tile matching column 3: tried until column 8.
Try using a device with more columns.
```

Trace:

```text
placement_trace_fail_key: runtime_output
placement_trace_fail_type: RuntimeEndpoint
placement_trace_fail_output: True
placement_trace_fail_common_col: 3
placement_trace_fail_remaining_tiles: []
placement_trace_fail_counts:
  {'runtime_output': 16, 'runtime_input': 6,
   'other_output': 0, 'other_input': 1}
```

Layer count was not the root cause:

```text
layer_iterations=1 fails with the same RuntimeEndpoint output placement class.
```

Comparison with accepted baseline:

```text
two-way direct baseline:
  layer_iterations=28 preflight ok
  compute_cores=30
  total_dma_tasks=21
  max_dma_tasks_per_fifo=1
  max_tile_inputs=2
  max_tile_outputs=2
  placement_trace_counts:
    {'runtime_output': 16, 'runtime_input': 5,
     'other_output': 0, 'other_input': 1}
```

## Decision

Rejected as a direct performance branch.

Root cause:

```text
The extra gate/up shard adds another independent runtime weight stream. The
current accepted graph is already at the runtime-output endpoint boundary:
16 host->NPU runtime outputs are placed, then the next output endpoint fails.
```

Not root causes:

```text
not token/numeric correctness
not C++ external kernel ABI
not layer_iterations=28 static depth
not compute_cores alone
```

Next implication:

```text
More gate/up parallelism needs endpoint reduction first. The likely next
experiment is not another independent gate/up Runtime.fill. It should pack two
gate/up shard streams behind one runtime endpoint and split them inside the
graph, or use a no-new-endpoint MLP optimization such as larger per-call row
groups.
```
