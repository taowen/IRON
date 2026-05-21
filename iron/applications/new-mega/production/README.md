<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# New Mega Production

This directory now keeps one production direction only: a phase-owned decode
topology. The previous static single-layer fusion path has been removed from
production because it proved correctness but used 25 compute cores for one
layer and therefore could not scale by appending more layer stages.

## Current Stage

```text
phase-owned
```

This stage is the current resource-bounded phase-owned megakernel body:

```text
fixed lane Workers
packed lane-local phase streams
shared phase stream broadcast in 4-lane fabric groups
row-sharded phase outputs joined in the same groups
tile-local hidden_state[1024] carried across layer iterations
loop over layers inside each Worker
two input ObjectFIFOs and one output ObjectFIFO per lane in the proven fabric
resource use bounded by num_lanes, not by num_layers * num_phases
```

Phase 0 broadcasts shared hidden/norm data, computes lane-local row-sharded Q
projection outputs, and joins the shard outputs. The `o_proj`, `gate_up`, and
`down_proj` phases now run real row-shard computation from Qwen3 weights. Layer
0 initializes a tile-local hidden buffer from the shared stream; each
`next_layer_token` phase updates that buffer for the next layer. `state[0]` is
only a checksum for diagnostics, not the activation handoff. The runner uses
the local Qwen3-0.6B safetensors by default and fills phase packets with real
layer hidden inputs, attention context, RMSNorm weights, projection row shards,
FFN hidden vectors, residual shards, and next-layer hidden values. The
remaining K/V projection and chunked attention kernels should be added inside
this topology instead of reintroducing standalone op stages.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py
```

Compile/preflight only:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py --compile-only
```

Stress the lane-local packet size toward a real attention chunk:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py \
  --packet-elements 16896 --compile-only
```

## Files

```text
main.py                  CLI entry point
runner.py                packet generation, reference, compile/run/print
ops.py                   phase-owned MLIROperator definition
design.py                stable DesignGenerator entry point
phase_owned_stages.py    IRON Program with fixed lane Workers
phase_owned_kernels.cc   AIE external kernels for the phase-owned body
```

## Acceptance

```text
preflight passes
full aiecc passes
NPU run completes
phase-owned output matches CPU reference
runtime_memrefs <= 5
compute_cores == num_lanes
max compute tile inputs <= 2
max compute tile outputs <= 2
```

Validated on NPU2:

```text
default production body:
  num_lanes=8
  num_layers=28
  phase_packets_per_layer=11
  hidden_size=1024
  attention_size=2048
  q_rows_per_packet=4
  fabric_group_size=4
  shared_packet_elements=2048
  tile_local_hidden_elements=1024
  q_output_values_per_lane=8
  attention_output_values_per_lane=8
  gate_up_output_values_per_lane=8
  residual_output_values_per_lane=8
  output_values_per_lane=32
  output_values_per_layer=256
  packet_elements=15368
  phase 0=tile-local hidden RMSNorm + lane-local Q row shard + grouped join
  o_proj=real attention context + lane-local O row shards + residual add
  gate_up=real post RMSNorm + lane-local gate/up row shards
  down_proj=real ffn_hidden + lane-local down row shards + residual add
  next_layer_token=updates tile-local hidden_state for the next layer
  inputs=real Qwen3 hidden/norm/q_proj/o_proj/post_norm/gate/up/down/next-hidden packets
  preflight_compute_cores=8
  preflight_total_dma_tasks=12
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
  npu_time_us=569918.562
  phase_owned_errors=0
  qwen3_phase_output_max_abs=0.000488
  qwen3_phase_output_mean_abs=0.000000
  qwen3_phase_output_errors=0 at abs_tol=0.5

large packet compile/preflight:
  q_rows_per_packet=4
  fabric_group_size=4
  packet_elements=16896
  packet_bytes=33792
  lane_stream_bytes=10407936
  input_elements=41631744
  output_elements=7168
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
```

## Next Work

```text
1. Expand Q/K/V shard coverage so attention consumes NPU-produced vectors.
2. Replace the attention chunk placeholder phases with real chunked online attention.
3. Feed gate_up from the NPU-produced attention residual once full residual coverage exists.
4. Feed down_proj from the NPU-produced FFN hidden once full gate/up coverage exists.
5. Keep grouped broadcast/join fan-in at 4 lanes unless a measured placer result proves wider groups.
6. Do not add another production standalone op or static single-layer graph.
```
