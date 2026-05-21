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
row-sharded phase outputs joined in the same groups
tile-local hidden_state[1024] carried across layer iterations
tile-local input_norm_weight[1024] cached from the phase 0 packet
loop over layers inside each Worker
two input ObjectFIFOs and two output ObjectFIFOs per lane in the proven fabric
two fixed group reducer Workers for O partial projection reduce
resource use bounded by num_lanes, not by num_layers * num_phases
```

Phase 0 reads hidden/norm data from the lane-local packet, caches the norm
weight in tile-local memory, computes lane-local row-sharded Q projection
outputs, and joins the shard outputs. The first two attention chunk phases
compute real K/V projection row shards by reusing the cached norm weight.
Attention chunk phases 2 and 3 compute real first-head Q/K RMSNorm+RoPE row
shards. The four
primary `attention_score_pv_*` phases and the four secondary score/PV phases
run real fixed-cache chunked QK, online softmax, and PV for all 16 attention
heads. The `o_proj` phase now computes same-fabric-group partial O projection
contributions on the lane Workers, sends those partials to one reducer Worker
per 4-lane group, receives the reduced same-group rows back, and finalizes the
local attention residual rows. Host-packed values still provide the other
fabric group's context contribution and the non-local residual/FFN values until
a second reduce/gather step removes them. The `gate_up` phase consumes the
local attention residual rows produced by `o_proj`; the `down_proj` phase
consumes the local FFN hidden rows produced by `gate_up`. Layer 0 initializes a
tile-local hidden buffer from the phase 0 packet; each `next_layer_token` phase
updates that buffer for the next layer. `state[0]` is only a checksum for
diagnostics, not the activation handoff. The runner uses the local Qwen3-0.6B
safetensors by default and fills phase packets with real layer hidden inputs,
input norm weights, first-head raw Q/K vectors, Q/K norm weights, RoPE cos/sin
values, fixed-cache GQA K/V/mask chunks, projection row shards, partial reduce
metadata, FFN hidden vectors, residual shards, and next-layer hidden values.
The runtime arg spec still contains a legacy shared input buffer so the runner
ABI stays stable, but production no longer creates or fills a shared
ObjectFifo.

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
  phase_packets_per_layer=17
  hidden_size=1024
  attention_size=2048
  attention_head_count=16
  npu_context_heads_per_layer=16
  head_dim=128
  max_seq_len=256
  attention_chunk_size=64
  attention_chunk_count=4
  q_rows_per_packet=4
  fabric_group_size=4
  tile_local_hidden_elements=1024
  tile_local_input_norm_weight_elements=1024
  q_output_values_per_lane=8
  k_output_values_per_lane=8
  v_output_values_per_lane=8
  q_rope_output_values_per_lane=8
  k_rope_output_values_per_lane=8
  context_output_values_per_lane=256
  attention_output_values_per_lane=8
  gate_up_output_values_per_lane=8
  residual_output_values_per_lane=8
  output_values_per_lane=320
  output_values_per_layer=2560
  packet_elements=16576
  phase 0=tile-local hidden RMSNorm + lane-local Q row shard + grouped join
  attention_chunk_0=tile-local hidden RMSNorm + lane-local K row shard
  attention_chunk_1=tile-local hidden RMSNorm + lane-local V row shard
  attention_chunk_2=first-head Q RMSNorm + RoPE row shard
  attention_chunk_3=first-head K RMSNorm + RoPE row shard
  attention_score_pv_0..3=lane-mapped heads 0..7 fixed-cache QK + online softmax + PV
  attention_score_pv_4..7=lane-mapped heads 8..15 fixed-cache QK + online softmax + PV
  o_proj=NPU-produced same-fabric-group context heads + group partial reduce + host other-fabric-group contribution + residual add
  gate_up=NPU-produced local attention residual rows + host remaining residual + real post RMSNorm + lane-local gate/up row shards
  down_proj=NPU-produced local FFN hidden rows + host remaining FFN hidden + lane-local down row shards + residual add
  next_layer_token=updates tile-local hidden_state for the next layer
  inputs=real Qwen3 hidden/norm/qkv_proj/qk_norm_rope/kv_cache_mask/o_proj/post_norm/gate/up/down/next-hidden packets
  preflight_compute_cores=10
  preflight_total_dma_tasks=10
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=2
  npu_time_us=311612.850
  phase_owned_max_abs=1.000000
  phase_owned_mean_abs=0.007770
  phase_owned_errors=0
  qwen3_phase_output_max_abs=0.500000
  qwen3_phase_output_mean_abs=0.007813
  qwen3_phase_output_errors=0 at abs_tol=0.5
  phase_owned_reference_errors=0 at abs_tol=1.0

  O partial-reduce resource diagnosis:
    first design failed aiecc with tile input DMA channel exceeded
    failing lane tile had three input FIFOs: shared, lane packet, o_reduced
    fix removed shared ObjectFifo and packed hidden/norm into phase 0 packet
    reducer workers added two compute cores but kept each tile at <=2 inputs

  phase-owned reference tolerance:
    one phase-level O reduce slot differed from qwen semantic reference by
    exactly one BF16 ULP (181 vs 182 at value scale ~182)
    qwen semantic reference passed at abs_tol=0.5
    phase-owned packet reference is checked at abs_tol=1.0 for this reduce
    boundary while qwen reference remains at abs_tol=0.5

  O handoff poison test, num_layers=1:
    each lane's two host O-packet context heads overwritten with 123.0
    poison_o_host_two_lane_heads_npu_time_us=11171.860
    poison_o_host_two_lane_heads_max_abs=0.017578
    poison_o_host_two_lane_heads_mean_abs=0.000309
    poison_o_host_two_lane_heads_errors_gt_0_5=0

  gate_up handoff poison test, num_layers=1:
    each lane's host gate_up attn_residual rows overwritten with 123.0
    poison_gate_host_local_residual_npu_time_us=11293.627
    poison_gate_host_local_residual_max_abs=0.017578
    poison_gate_host_local_residual_mean_abs=0.000312
    poison_gate_host_local_residual_errors_gt_0_5=0

  down_proj handoff poison test, num_layers=28:
    each lane's host down_proj local FFN hidden rows overwritten with 123.0
    poison_down_host_local_ffn_npu_time_us=320777.251
    poison_down_host_local_ffn_max_abs=0.437500
    poison_down_host_local_ffn_mean_abs=0.007681
    poison_down_host_local_ffn_errors_gt_0_5=0

large packet compile/preflight:
  q_rows_per_packet=4
  fabric_group_size=4
  packet_elements=16896
  packet_bytes=33792
  lane_stream_bytes=12300288
  input_elements=49201152
  output_elements=43008
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
```

## Next Work

```text
1. Extend O partial reduce across both fabric groups so O projection no longer needs host context contribution.
2. Decide whether gate_up should use residual gather/broadcast or the same partial projection reduce pattern.
3. Decide whether down_proj should use FFN gather/broadcast or partial projection reduce.
4. Add a cheap poison/diagnostic harness for the O partial-reduce path that avoids a full recompile.
5. Keep grouped fan-in at 4 lanes unless a measured placer result proves wider groups.
6. Do not add another production standalone op or static single-layer graph.
```
