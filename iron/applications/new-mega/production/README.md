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
four fixed reducer Workers reused by O projection and FFN partial reduce
resource use bounded by num_lanes, not by num_layers * num_phases
```

Phase 0 reads hidden/norm data from the lane-local packet, caches the norm
weight in tile-local memory, computes lane-local row-sharded Q projection
outputs, and joins the shard outputs. The first two attention chunk phases
compute real K/V projection row shards by reusing the cached norm weight.
Attention chunk phases 2 and 3 are currently drained diagnostic packet slots;
the real score/PV packets carry host-packed RoPE Q and fixed-cache K/V chunks.
The four primary `attention_score_pv_*` phases and the four secondary score/PV
phases run real fixed-cache chunked QK, online softmax, and PV for all 16
attention heads. The `o_proj_chunk_*` phases now compute full cross-fabric
partial O projection: lane Workers produce partials for one 32-row O chunk at a
time, one source reducer per 4-lane group sums producer lanes, and one target
reducer per group sums the two source groups before broadcasting each 32-row
chunk back to owner lanes. O projection no longer needs host-packed context
contribution for these rows, and the 32 chunks materialize the full 1024-row
attention residual in `lane_output`. The `gate_up` chunk phases consume the
full attention residual produced by `o_proj_chunk_0..31` and emit four
32-row FFN partial groups through the same fixed source/target reducer fabric.
The `down_partial_chunk_*` phases accumulate those first 128 NPU-produced FFN
hidden rows in tile-local float state, then add host-packed rows 128..3071.
Layer 0 initializes a tile-local hidden buffer from the phase 0 packet; each
`next_layer_token` phase updates that buffer for the next layer. `state[0]` is
only a checksum for diagnostics, not the activation handoff. The runner uses
the local Qwen3-0.6B safetensors by default and fills phase packets with real
layer hidden inputs, input norm weights, first-head raw Q/K vectors, Q/K norm
weights, RoPE cos/sin values, fixed-cache GQA K/V/mask chunks, projection row
shards, partial reduce metadata, FFN hidden vectors, residual shards, and
next-layer hidden values.
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
compute_cores is bounded by lane Workers plus fixed reducer Workers
max compute tile inputs <= 2
max compute tile outputs <= 2
```

Validated on NPU2:

```text
default production body:
  num_lanes=8
  num_layers=28
  phase_packets_per_layer=54
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
  attention_output_values_per_lane=1024
  gate_up_output_values_per_lane=8
  residual_output_values_per_lane=8
  output_values_per_lane=1336
  output_values_per_layer=10688
  packet_elements=16576
  phase 0=tile-local hidden RMSNorm + lane-local Q row shard + grouped join
  attention_chunk_0=tile-local hidden RMSNorm + lane-local K row shard
  attention_chunk_1=tile-local hidden RMSNorm + lane-local V row shard
  attention_chunk_2=drained diagnostic packet slot
  attention_chunk_3=drained diagnostic packet slot
  attention_score_pv_0..3=lane-mapped heads 0..7 fixed-cache QK + online softmax + PV
  attention_score_pv_4..7=lane-mapped heads 8..15 fixed-cache QK + online softmax + PV
  o_proj_chunk_0..31=NPU-produced all context heads + chunked source/target partial reduce + full residual materialization
  ffn_gate_chunk_0..3=NPU-produced full attention residual + real post RMSNorm + 4x32 FFN partial rows
  down_partial_chunk_0..3=NPU-produced first 128 FFN hidden rows + host rows 128..3071 + lane-local down row shards + residual add
  next_layer_token=updates tile-local hidden_state for the next layer
  inputs=real Qwen3 hidden/norm/qkv_proj/qk_norm_rope/kv_cache_mask/o_proj/post_norm/gate/up/down/next-hidden packets
  preflight_compute_cores=12
  preflight_total_dma_tasks=10
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=2
  lane_core_text_bytes=15968
  npu_time_us=748574.161
  phase_owned_max_abs=1.000000
  phase_owned_mean_abs=0.011803
  phase_owned_errors=0
  qwen3_phase_output_max_abs=1.000000
  qwen3_phase_output_mean_abs=0.011796
  qwen3_phase_output_errors=0

  O partial-reduce resource diagnosis:
    first design failed aiecc with tile input DMA channel exceeded
    failing lane tile had three input FIFOs: shared, lane packet, o_reduced
    fix removed shared ObjectFifo and packed hidden/norm into phase 0 packet
    reducer workers added two compute cores but kept each tile at <=2 inputs
    second cross-group design failed routing because a source-reduced vector
    was split through one mem tile
    fix made each source reducer produce two target-group outputs directly

  Historical O handoff poison test before cross-fabric reduce, num_layers=1:
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

  full cross-fabric O reduce compile diagnostic:
    rejected topology:
      source reducer -> full source_reduced[32] -> memtile split -> target reducers
      result: routing pipeline failed after resource allocation
    accepted topology:
      source reducer -> target0 half and target1 half directly
      target reducers sum two source halves
      result: full aiecc passes

  gate_up same-fabric-group residual visibility:
    O finalize writes 16 attention residual rows per lane output object
    gate_up reads those 16 rows from lane_output before post RMSNorm
    fixed ABI drift found during bring-up:
      O packet residual header grew from 4 rows to 16 rows
      O partial C++ kernel and packet reference still used the old weight offset
      Kernel declaration for q_shard was accidentally widened while adding the
      O partial argument; resolve_program caught the arity mismatch before aiecc

  gate_up full residual visibility:
    O projection is split into 32 row chunks of 32 hidden rows
    each chunk reuses the same source/target reducer fabric
    each lane output materializes the full 1024-row attention residual segment
    gate_up reads all 1024 residual values from lane_output instead of the host
    packet
    compile-only proved this is temporal reuse, not resource replication:
      preflight_compute_cores=12
      preflight_max_compute_tile_inputs=2
      preflight_max_compute_tile_outputs=2
      preflight_max_dma_tasks_per_fifo=1
    raw qwen semantic diff had 16 attention_residual values at max_abs=1.0
    while packet-level phase reference had 0 errors; segment diagnostics showed
    the difference was O accumulation-order tolerance, not gate_up/down dataflow

  FFN partial handoff through existing reducer fabric:
    gate_up writes a sparse 32-row FFN partial vector into the O partial FIFO
    source/target reducers consume one extra token per layer and broadcast the
    reduced 32 rows back to every lane
    down_proj consumes ffn_reduced[0:32] instead of host ffn_hidden[0:32]
    compile diagnostic:
      adding a standalone ffn_partial external kernel made lane core .text
      16880 bytes and failed CDO generation with program memory overflow
      fusing partial generation into gate_up reduced the lane core but still
      left old down local_ffn fallback code; deleting the now-unreachable
      fallback brought lane core .text to 16080 bytes
    numeric diagnostic:
      first run failed only in down_residual with 144 errors, max_abs=33
      cause was using gate packet[0] as FFN row base; packet[0] is residual
      replacement base and is 0 for all lanes
      fix stores FFN row base in the final gate packet metadata slot
    accepted result:
      npu_time_us=578549.967
      phase_owned_max_abs=1.000000
      phase_owned_mean_abs=0.011823
      phase_owned_errors=0
      qwen3_phase_output_max_abs=1.000000
      qwen3_phase_output_mean_abs=0.011824
      qwen3_phase_output_errors=0

  FFN 4-group handoff:
    gate_up/down are split into four phase pairs:
      ffn_gate_chunk_0, down_partial_chunk_0, ... chunk_3
    each gate chunk emits 32 FFN rows through the existing reducer fabric
    down keeps a tile-local float accumulator and finalizes only after chunk_3
    first attempt failed CDO generation with lane core .text=17904 bytes
    removing unused q/k RoPE and gate_up visual diagnostic kernels reduced
    lane core .text to 15968 bytes and compile succeeded
    qwen3 reference had late-layer down_residual differences until the first
    128 NPU-produced FFN rows were modeled as float reducer outputs, not BF16
    host hidden rows
    accepted result:
      npu_time_us=748574.161
      phase_owned_errors=0
      qwen3_phase_output_errors=0
```

## Next Work

```text
1. Extend the FFN partial handoff beyond the first 128 rows or switch down_proj
   to a true partial-projection reduce so the full 3072-wide host FFN vector
   can be removed.
2. Add a cheap poison/diagnostic harness for the reducer-handoff paths that
   avoids a full recompile.
3. Keep grouped fan-in at 4 lanes unless a measured placer result proves wider
   groups.
4. Do not add another production standalone op or static single-layer graph.
```
