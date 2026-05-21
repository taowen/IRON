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
projection outputs, and joins the shard outputs. The first two attention chunk
phases compute real K/V projection row shards. Attention chunk phases 2 and 3
compute real first-head Q/K RMSNorm+RoPE row shards. The four
primary `attention_score_pv_*` phases and the four secondary score/PV phases
run real fixed-cache chunked QK, online softmax, and PV for all 16 attention
heads. The `o_proj` phase now consumes each lane's two NPU-produced context
heads from the previous phases and uses host-packed context only for heads
owned by other lanes. The `gate_up` phase consumes the local attention residual
rows produced by `o_proj`; the `down_proj` phase consumes the local FFN hidden
rows produced by `gate_up`. The remaining non-local context/residual/FFN values
are still host-packed until an on-chip gather/broadcast or partial-reduce design
replaces them. Layer 0 initializes a tile-local hidden buffer from the shared
stream; each `next_layer_token` phase updates that buffer for the next layer.
`state[0]` is only a checksum for diagnostics, not the
activation handoff. The runner uses the local Qwen3-0.6B safetensors by default
and fills phase packets with real layer hidden inputs, first-head raw Q/K
vectors, Q/K norm weights, RoPE cos/sin values, fixed-cache GQA K/V/mask chunks,
remaining attention context, projection row shards, FFN hidden vectors,
residual shards, and next-layer hidden values.

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
  shared_packet_elements=2048
  tile_local_hidden_elements=1024
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
  o_proj=NPU-produced two lane-mapped context heads + host other-lane context + lane-local O row shards + residual add
  gate_up=NPU-produced local attention residual rows + host remaining residual + real post RMSNorm + lane-local gate/up row shards
  down_proj=NPU-produced local FFN hidden rows + host remaining FFN hidden + lane-local down row shards + residual add
  next_layer_token=updates tile-local hidden_state for the next layer
  inputs=real Qwen3 hidden/norm/qkv_proj/qk_norm_rope/kv_cache_mask/o_proj/post_norm/gate/up/down/next-hidden packets
  preflight_compute_cores=8
  preflight_total_dma_tasks=12
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
  npu_time_us=319631.346
  phase_owned_max_abs=0.437500
  phase_owned_mean_abs=0.007681
  phase_owned_errors=0
  qwen3_phase_output_max_abs=0.500000
  qwen3_phase_output_mean_abs=0.007685
  qwen3_phase_output_errors=0 at abs_tol=0.5

  segment max_abs against qwen3_reference:
    q=0.000000
    k=0.003906
    v=0.000122
    q_rope=0.000000
    k_rope=0.000000
    context=0.437500
    attention_residual=0.218750
    gate_up=0.031250
    residual=0.500000

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
1. Add on-chip context gather/reduce so every O row can see all NPU-produced heads.
2. Remove the remaining host-packed other-lane attention context from O projection.
3. Expand gate_up handoff beyond local residual rows once full residual coverage exists.
4. Expand down_proj handoff beyond local FFN rows once full gate/up coverage exists.
5. Keep grouped broadcast/join fan-in at 4 lanes unless a measured placer result proves wider groups.
6. Do not add another production standalone op or static single-layer graph.
```
