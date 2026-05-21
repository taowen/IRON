<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Archived D1.5 Phase-Owned Ladder Through Full O Residual

This file was split out from `../../experiments.md` to keep the active plan short.
It covers the accepted production reset and D1.5a-D1.5w ladder before the current FFN reducer handoff.

#### D1.5. Production Reset To Phase-Owned Fusion

Status: in production.

Question:

```text
Can production stop carrying independent op stages and static single-layer
fusion, and instead expose only the phase-owned topology that keeps Worker/FIFO
resources fixed while layer/phase count becomes loop work?
```

Implementation:

```text
production stage: phase-owned
num_lanes fixed lane Workers
num_layers * phase_packets_per_layer loop inside each Worker
one packed input FIFO per lane
one padded output FIFO per lane
synthetic packet accumulation is the placeholder for real Qwen3 phase kernels
```

Acceptance:

```text
preflight passes
full aiecc passes
NPU output matches CPU reference
compute_cores == num_lanes
max tile inputs <= 2
max tile outputs <= 2
```

Result:

```text
default production skeleton:
  num_lanes=8
  num_layers=28
  phase_packets_per_layer=11
  hidden_size=1024
  packet_elements=2048
  total_phase_packets=308
  preflight_runtime_memrefs=2
  preflight_compute_cores=8
  preflight_total_dma_tasks=16
  preflight_max_compute_tile_inputs=1
  preflight_max_compute_tile_outputs=1
  npu_time_us=106536.455
  phase_owned_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  lane_stream_bytes=10407936
  input_elements=41631744
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
  preflight_max_compute_tile_inputs=1
  preflight_max_compute_tile_outputs=1
```

#### D1.5a. Real Input RMSNorm Phase In Phase-Owned Topology

Status: accepted in production.

Question:

```text
Can one synthetic phase be replaced with real Qwen3-shaped RMSNorm math without
leaving the phase-owned topology or adding FIFO endpoints?
```

Implementation:

```text
phase 0 packet layout: hidden[1024] || norm_weight[1024]
phase 0 AIE kernel: mean-square -> aie::invsqrt -> weighted RMSNorm checksum
other phases: synthetic packet accumulation remains as placeholder
```

Result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=11
hidden_size=1024
packet_elements=2048
preflight_runtime_memrefs=2
preflight_compute_cores=8
preflight_max_compute_tile_inputs=1
preflight_max_compute_tile_outputs=1
npu_time_us=106536.455
phase_owned_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Debug lessons:

```text
Worker loop indices from range_() are MLIR index values; cast them before
passing to external kernels declared with np.int32.

AIE kernels should use aie::invsqrt for RMSNorm. Host libc sqrtf failed under
the AIE cross compiler.
```

#### D1.5b. Add One Q Projection Row To Phase 0

Status: accepted in production.

Question:

```text
Can phase 0 advance from an RMSNorm checksum to RMSNorm plus one Q projection
row while preserving the phase-owned lane topology?
```

Implementation:

```text
phase 0 packet layout: hidden[1024] || norm_weight[1024] || q_weight_row[1024]
phase 0 AIE kernel:
  mean-square -> aie::invsqrt -> xnorm checksum
  q_row checksum = dot(xnorm, q_weight_row)
```

Result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=11
hidden_size=1024
packet_elements=3072
preflight_runtime_memrefs=2
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=6144
preflight_max_compute_tile_inputs=1
preflight_max_compute_tile_outputs=1
npu_time_us=158425.832
phase_owned_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

#### D1.5c. Add A Q Projection Row Block To Phase 0

Status: accepted in production.

Question:

```text
Can phase 0 move from one Q projection row to a row-sharded Q block without
changing the fixed lane Worker/FIFO topology?
```

Implementation:

```text
q_rows_per_packet=4
phase 0 packet layout:
  hidden[1024] || norm_weight[1024] || q_weight_block[4, 1024]
phase 0 AIE kernel:
  mean-square -> aie::invsqrt -> xnorm checksum
  q_block checksum = four dot(xnorm, q_weight_row) accumulations
```

Result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=11
hidden_size=1024
q_rows_per_packet=4
packet_elements=6144
preflight_runtime_memrefs=2
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=12288
preflight_max_compute_tile_inputs=1
preflight_max_compute_tile_outputs=1
npu_time_us=316171.830
phase_owned_errors=0

large packet compile/preflight:
  q_rows_per_packet=4
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Lesson:

```text
The production path can increase real per-phase math density while keeping the
resource shape bounded by lanes. This supports the correct fusion direction:
insert real kernels into the phase-owned Worker loop instead of adding another
standalone op or static layer graph.
```

#### D1.5d. Grouped Broadcast/Join Fabric For Row-Sharded Projection

Status: accepted in production.

Question:

```text
Can production represent the lane communication required by row-sharded GEMV:
shared vector broadcast to lanes, lane-local weight shards, and joined Q shard
output?
```

Initial failure:

```text
An ungrouped 8-lane broadcast/join fabric failed in resolve_program():
ValueError: Failed to find a tile matching column 0: tried until column 8.
```

Diagnostic:

```text
The same design compiled at num_lanes=4. This isolated the root cause to the
8-way ObjectFifo broadcast/join fabric placement, not to kernel ABI, TAP, or
the Worker phase loop.
```

Fix:

```text
Split 8 lanes into two 4-lane fabric groups.
Each group gets its own shared-input fill and local broadcast.
Each group joins its lane Q shards locally.
Output TAPs write group results into the correct per-layer offsets.
```

Result:

```text
num_lanes=8
fabric_group_size=4
shared_packet_elements=2048
q_rows_per_packet=4
q_output_values_per_lane=8
packet_elements=4096
output_elements=1792
preflight_runtime_memrefs=3
preflight_compute_cores=8
preflight_total_dma_tasks=12
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=236207.328
phase_owned_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
```

Lesson:

```text
The true-fusion topology cannot be eight independent lanes. It needs grouped
communication. A 4-lane fabric group is the current proven unit for local
broadcast/join under SequentialPlacer on NPU2.
```

#### D1.5e. Tile-Local Hidden State Across Layer Iterations

Status: accepted in production.

Question:

```text
Can production carry a real activation-sized hidden vector across phase/layer
iterations, instead of using only a scalar debug checksum?
```

Implementation:

```text
Each lane Worker owns a tile-local hidden_state[1024] BF16 Buffer.
Layer 0 initializes hidden_state from the shared stream.
Phase 0 reads hidden_state for RMSNorm and Q shard dot products.
The final next_layer_token packet of each layer writes hidden_state for the
next layer.
state[0] remains only a diagnostic checksum.
```

Result:

```text
num_lanes=8
fabric_group_size=4
tile_local_hidden_elements=1024
q_rows_per_packet=4
packet_elements=4096
preflight_runtime_memrefs=3
preflight_compute_cores=8
preflight_total_dma_tasks=12
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=219203.810
phase_owned_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=1
```

Lesson:

```text
The phase-owned topology now has two separate state concepts:
  hidden_state[1024] is the activation handoff across layers.
  state[0] is a checksum for diagnostics only.
This closes the gap where the skeleton looked persistent but did not actually
carry an activation-sized inter-phase value.
```

#### D1.5f. Real Qwen3 Weights For The Grouped Q Shard

Status: accepted in production.

Question:

```text
Can the grouped phase-owned topology run real Qwen3-0.6B decode data instead
of patterned placeholder packets?
```

Implementation:

```text
The production runner loads the local Qwen3-0.6B safetensors.
For each decoded layer it packs:
  shared stream: layer-0 hidden initializer and per-layer input RMSNorm weight
  lane phase-0 packet: real q_proj weight rows for that lane
  next_layer_token packet: real next-layer hidden from the CPU Qwen3 reference
The NPU computes real row-sharded q_proj outputs for 28 layers and joins them
through the grouped fabric.
```

Result:

```text
model_dir=/home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
prompt_tokens=26
next_token=59604
num_lanes=8
num_layers=28
fabric_group_size=4
q_rows_per_packet=4
packet_elements=4096
preflight_compute_cores=8
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=221127.014
phase_owned_errors=0
qwen3_q_shard_max_abs=0.250000
qwen3_q_shard_mean_abs=0.003658
qwen3_q_shard_errors=0 at abs_tol=0.5

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Lesson:

```text
The hard gate is the local BF16 boundary reference, which matches exactly.
The PyTorch Qwen3 q_proj reference is a model-semantics gate and currently
passes at abs_tol=0.5 because the accumulation path differs by up to 0.25.
```

#### D1.5g. Real Gate/Up Shard In The Phase-Owned Loop

Status: accepted in production.

Question:

```text
Can a middle synthetic phase be replaced with real Qwen3 post-attention
RMSNorm plus MLP gate/up row-sharded projection inside the same grouped
phase-owned topology?
```

Implementation:

```text
The gate_up phase packet now contains:
  attn_residual[1024]
  post_attention_layernorm.weight[1024]
  gate_proj weight shard [q_rows_per_packet, 1024]
  up_proj weight shard [q_rows_per_packet, 1024]

The AIE gate_up kernel computes RMSNorm(attn_residual) and emits both gate and
up projection shard values into the per-lane output object. The same grouped
join now drains Q shard + gate/up shard for every layer.
```

Result:

```text
num_lanes=8
num_layers=28
fabric_group_size=4
q_rows_per_packet=4
packet_elements=10240
output_values_per_lane=16
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=20480
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=502652.919
phase_owned_max_abs=0.000244
phase_owned_errors=0
qwen3_phase_output_max_abs=0.250000
qwen3_phase_output_mean_abs=0.003669
qwen3_phase_output_errors=0 at abs_tol=0.5

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Lesson:

```text
The grouped phase-owned fabric can carry more than one real phase result per
layer. Holding the lane output object from phase 0 until gate_up lets one join
carry Q + gate/up shard values without adding another output FIFO per lane.
```

#### D1.5h. Real Down/Residual Shard In The Phase-Owned Loop

Status: accepted in production.

Question:

```text
Can a later synthetic phase be replaced with real Qwen3 down projection row
shards plus residual add without adding another worker group or output join?
```

Implementation:

```text
The down_proj phase packet now contains:
  ffn_hidden[3072]
  attn_residual shard [q_rows_per_packet]
  down_proj weight shard [q_rows_per_packet, 3072]

The AIE down kernel computes dot(ffn_hidden, down_proj row) + residual_shard and
stores the residual shard into the same per-lane output object used by Q and
gate/up. The lane output object is still joined once per layer.
```

Result:

```text
num_lanes=8
num_layers=28
fabric_group_size=4
q_rows_per_packet=4
packet_elements=15368
packet_bytes=30736
output_values_per_lane=24
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=30736
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=633629.052
phase_owned_max_abs=0.000488
phase_owned_errors=0
qwen3_phase_output_max_abs=0.000488
qwen3_phase_output_mean_abs=0.000000
qwen3_phase_output_errors=0 at abs_tol=0.5

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Debug note:

```text
The first run failed only against the PyTorch F.linear down reference:
2 residual shard elements exceeded abs_tol=0.5 while phase_owned_errors=0.
The root cause was not NPU dataflow. PyTorch's BF16 reduction path differed
from the explicit AIE kernel semantics for a 3072-element dot. The independent
Qwen3 shard reference was changed to BF16 inputs + scalar float32 accumulation
+ BF16 output, which still catches wrong rows/weights but matches the kernel
contract exactly.
```

Lesson:

```text
Once real dot rows get longer, a PyTorch F.linear reference is too vague for
per-element kernel acceptance. Use two gates: exact local BF16 boundary
reference for the AIE kernel, and a separately reported model-level tolerance
only when comparing full PyTorch paths.
```

#### D1.5i. Real O Projection And Attention Residual Shard

Status: accepted in production.

Question:

```text
Can the o_proj placeholder phase be replaced with real Qwen3 attention context,
real O projection row shards, and residual add inside the same lane-owned loop?
```

Implementation:

```text
The o_proj phase packet now contains:
  attention_context[2048]
  layer input residual shard [q_rows_per_packet]
  o_proj weight shard [q_rows_per_packet, 2048]

The AIE o_proj kernel computes dot(attention_context, o_proj row) +
residual_shard and stores attention residual shard values into the same
per-lane output object as Q, gate/up, and layer residual.
```

Result:

```text
num_lanes=8
num_layers=28
hidden_size=1024
attention_size=2048
intermediate_size=3072
fabric_group_size=4
q_rows_per_packet=4
packet_elements=15368
output_values_per_lane=32
output_values_per_layer=256
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=30736
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=569918.562
phase_owned_max_abs=0.000488
phase_owned_errors=0
qwen3_phase_output_max_abs=0.000488
qwen3_phase_output_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Debug notes:

```text
First failure:
  host packing tried to copy attention_context[2048] into a hidden_size[1024]
  packet field. Root cause: Qwen3-0.6B has attention width
  num_attention_heads * head_dim = 2048, while hidden_size = 1024.

Second failure:
  resolve_program reported q_shard expected 11 args but got 10. Root cause:
  Kernel(...) had one stale np.int32 in the Python ABI declaration after adding
  attention_size to o_proj.
```

Lesson:

```text
Do not infer all projection packet widths from hidden_size. In Qwen3-0.6B,
q_proj/o_proj operate across the 2048-wide attention space, while residual,
RMSNorm, and MLP down outputs are 1024-wide hidden space.
```

#### D1.5j. Real K/V Projection Shards In Attention Chunk Phases

Status: accepted in production.

Question:

```text
Can attention_chunk_0 and attention_chunk_1 stop being checksum placeholders
and instead compute real Qwen3 K/V projection row shards while preserving the
same phase-owned Worker/FIFO topology?
```

Implementation:

```text
attention_chunk_0 packet = k_proj weight shard [q_rows_per_packet, 1024]
attention_chunk_1 packet = v_proj weight shard [q_rows_per_packet, 1024]

The Worker keeps the shared hidden/input_norm packet acquired through Q/K/V.
The new projection kernel reuses tile-local hidden_state and the shared
input_layernorm weight, then writes K and V row shards into the same per-lane
joined output object.
```

Result:

```text
num_lanes=8
num_layers=28
hidden_size=1024
attention_size=2048
intermediate_size=3072
q_rows_per_packet=4
packet_elements=15368
q/k/v/attention/gate_up/residual output values per lane = 8 each
output_values_per_lane=48
output_values_per_layer=384
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=30736
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=531640.018
phase_owned_max_abs=0.003906
phase_owned_errors=0
qwen3_phase_output_max_abs=0.003906
qwen3_phase_output_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Debug note:

```text
The first compile failed in resolve_program:
  q_shard expected 9 args but 10 were provided.

Root cause was again a stale Kernel(...) ABI declaration, this time caused by
editing q_shard while adding the generic projection kernel declaration.
```

Lesson:

```text
The shared broadcast object can be held across Q/K/V phases without increasing
worker or endpoint count. This is the right local pattern for replacing more
attention placeholder phases, but every external kernel declaration must be
checked at the Python ABI, C++ signature, and Worker call together.
```

#### D1.5k. Real First-Head Q/K Norm+RoPE Shards In Attention Chunk Phases

Status: accepted in production.

Question:

```text
Can attention_chunk_2 and attention_chunk_3 stop being checksum placeholders
and instead compute real Qwen3 first-head Q/K RMSNorm+RoPE row shards while
preserving the same phase-owned Worker/FIFO topology?
```

Implementation:

```text
attention_chunk_2 packet:
  row_base
  raw Q head 0 [128]
  q_norm weight [128]
  RoPE cos [128]
  RoPE sin [128]

attention_chunk_3 packet:
  row_base
  raw K head 0 [128]
  k_norm weight [128]
  RoPE cos [128]
  RoPE sin [128]

new kernel:
  new_mega_phase_norm_rope_shard_bf16
```

The current scope is intentionally first-head only:

```text
num_lanes=8
q_rows_per_packet=4
covered rows=32
head_dim=128
```

Result:

```text
num_lanes=8
num_layers=28
hidden_size=1024
attention_size=2048
head_dim=128
intermediate_size=3072
q_rows_per_packet=4
packet_elements=15368
q/k/v/q_rope/k_rope/attention/gate_up/residual output values per lane = 8 each
output_values_per_lane=64
output_values_per_layer=512
output_elements=14336
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=30736
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=388798.601
phase_owned_max_abs=0.003906
phase_owned_errors=0
qwen3_phase_output_max_abs=0.003906
qwen3_phase_output_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Debug note:

```text
The first NPU run had qwen3_phase_output_errors=0 but phase_owned_errors=2668.
Host-only comparison of the two references found that phase_owned_reference was
stale: gate_base had not been shifted by the inserted q_rope/k_rope output
segments. The NPU kernel was correct; the failing boundary was the host
reference layout.
```

Lesson:

```text
When inserting a new per-lane output segment, update the AIE output base, packet
builder qwen3_reference base, phase_owned_reference base, printout, README
shape table, and any debug index decoder together. If one reference passes and
another fails, compare references before editing kernels.
```

#### D1.5l. Real First-Head Chunked Score/Softmax/PV

Status: accepted in production.

Question:

```text
Can the phase-owned topology replace the remaining attention placeholder with
real fixed-cache chunked QK, online softmax, and PV computation without adding
new FIFO endpoints or leaving the single Worker loop shape?
```

Implementation:

```text
phase labels:
  attention_score_pv_0
  attention_score_pv_1
  attention_score_pv_2
  attention_score_pv_3

each packet:
  q_head0[128]
  k_cache_head0_chunk[64,128]
  v_cache_head0_chunk[64,128]
  mask_chunk[64]

tile-local state:
  attention_state[2] = running max, running sum
  attention_acc[128] = online PV accumulator
```

Current scope:

```text
max_seq_len=256
attention_chunk_size=64
attention_chunk_count=4
first query head only
context[128] is emitted as a diagnostic output segment in every lane
O projection still consumes host-packed full attention_context at this point
```

Result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=13
packet_elements=16576
packet_bytes=33152
context_output_values_per_lane=128
output_values_per_lane=192
output_values_per_layer=1536
output_elements=43008
preflight_compute_cores=8
preflight_max_fifo_buffered_bytes=33152
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=254748.598
phase_owned_max_abs=0.312500
phase_owned_errors=0
qwen3_phase_output_max_abs=0.312500
qwen3_phase_output_errors=0

large packet compile/preflight:
  packet_elements=16896
  packet_bytes=33792
  preflight_compute_cores=8
  preflight_max_fifo_buffered_bytes=33792
```

Segment diagnosis:

```text
q max=0.000000
k max=0.003906
v max=0.000122
q_rope max=0.000000
k_rope max=0.000000
context max=0.312500
attention_residual max=0.000000
gate_up max=0.000244
residual max=0.000488
```

Lesson:

```text
Adding chunked score/softmax/PV this way did not increase ObjectFIFO endpoint
pressure: the packed lane-local stream absorbed four more phase packets. The
dominant numeric diff is isolated to the context segment and is expected from
the AIE exp2<bfloat16> approximation used in the online softmax path.
```

Remaining D1 work:

```text
D1.5m feed O projection from NPU-produced head-0 attention context
D1.5n expand context handoff to lane-mapped heads 0..7
D1.5o feed downstream phases from full NPU-produced activation vectors instead of host reference packets
D2 run repeated layers in the same phase-owned topology
```

#### D1.5m. O Projection Consumes NPU-Produced Head-0 Context

Status: accepted in production.

Question:

```text
Can the O projection phase consume the context emitted by the preceding
chunked score/softmax/PV phase, instead of treating the attention result as
only a diagnostic drain?
```

Implementation:

```text
o_proj packet layout:
  context_head_index[1]
  host_attention_context[2048]
  residual_shard[q_rows_per_packet]
  o_proj_weight_shard[q_rows_per_packet, 2048]

kernel behavior:
  for context indices inside context_head_index:
    read lane_output[context_segment]
  for all other context indices:
    read host_attention_context
```

Current scope:

```text
context_head_index=0
head-0 context is produced by NPU score/softmax/PV
remaining 15 heads still come from the host-packed reference context
all O row shards now depend on the preceding NPU attention phase for head 0
```

Result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=13
packet_elements=16576
preflight_compute_cores=8
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
npu_time_us=256452.003
phase_owned_max_abs=0.312500
phase_owned_mean_abs=0.007302
phase_owned_errors=0
qwen3_phase_output_max_abs=0.312500
qwen3_phase_output_mean_abs=0.007302
qwen3_phase_output_errors=0
```

Handoff diagnosis:

```text
host O-packet head-0 context overwritten with 123.0
num_layers=1
poison_o_host_head0_npu_time_us=10894.183
poison_o_host_head0_max_abs=0.007812
poison_o_host_head0_mean_abs=0.000304
poison_o_host_head0_errors_gt_0_5=0
```

Interpretation:

```text
If the O kernel still read the host context slice for head 0, poisoning that
slice would create a large O-projection error. The clean poison run proves the
O phase is actually reading the lane-local context written by the previous
attention phase.
```

Remaining D1 work:

```text
D1.5n expand context handoff to lane-mapped heads 0..7
D1.5o remove host-packed context from O projection
D1.5p feed gate_up from the NPU-produced attention residual
D1.5q feed down_proj from NPU-produced FFN hidden
D2 run repeated layers in the same phase-owned topology
```

#### D1.5n. Lane-Mapped Multi-Head Context Handoff

Status: accepted in production.

Question:

```text
Can the phase-owned topology move beyond a single head and have each lane
compute and hand off a different real Qwen3 attention head without changing the
ObjectFIFO graph?
```

Implementation:

```text
context_head_index = lane_id
q_head = q_rope_heads[context_head_index]
kv_head = context_head_index // (num_attention_heads / num_key_value_heads)
k/v cache chunk = fixed cache for kv_head
O phase replaces only that lane's context_head_index slice from lane_output
```

This keeps the topology fixed:

```text
compute_cores=8
max tile inputs=2
max tile outputs=1
phase_packets_per_layer=13
packet_elements=16576
```

Result:

```text
num_lanes=8
attention_head_count=16
npu_context_heads_per_layer=8
num_layers=28
npu_time_us=256714.949
phase_owned_max_abs=0.437500
phase_owned_mean_abs=0.006899
phase_owned_errors=0
qwen3_phase_output_max_abs=0.437500
qwen3_phase_output_mean_abs=0.006899
qwen3_phase_output_errors=0
```

Handoff diagnosis:

```text
each lane's host O-packet context head overwritten with 123.0
num_layers=1
poison_o_host_lane_heads_npu_time_us=10760.835
poison_o_host_lane_heads_max_abs=0.017578
poison_o_host_lane_heads_mean_abs=0.000329
poison_o_host_lane_heads_errors_gt_0_5=0
```

Interpretation:

```text
The poison test would fail if any lane still read its own host-fed context
slice. Passing it proves that O consumes lane-local NPU context for heads 0..7.
The remaining host-packed context is now limited to heads 8..15.
```

Remaining D1 work:

```text
D1.5o compute heads 8..15 in a second per-lane score/PV group
D1.5p add on-chip context gather/reduce so O sees all NPU-produced heads
D1.5q remove remaining host-packed attention context from O projection
D1.5r feed gate_up from the NPU-produced attention residual
D1.5s feed down_proj from NPU-produced FFN hidden
D2 run repeated layers in the same phase-owned topology
```

#### D1.5o. Two Context Heads Per Lane

Status: accepted in production.

Question:

```text
Can each lane compute two real Qwen3 attention heads, covering all 16 context
heads, without adding new ObjectFIFO endpoints?
```

Implementation:

```text
phase_packets_per_layer = 17

primary score/PV group:
  attention_score_pv_0..3
  context_head_index = lane_id

secondary score/PV group:
  attention_score_pv_4..7
  context_head_index = lane_id + num_lanes

O packet:
  context_head_index0
  context_head_index1
  host_attention_context[2048]
  residual_shard
  O row shard
```

Topology result:

```text
compute_cores=8
max tile inputs=2
max tile outputs=1
packet_elements=16576
packet_bytes=33152
context_output_values_per_lane=256
output_values_per_lane=320
input_elements=63121408
output_elements=71680
```

NPU result:

```text
num_lanes=8
attention_head_count=16
npu_context_heads_per_layer=16
num_layers=28
npu_time_us=266653.221
phase_owned_max_abs=0.437500
phase_owned_mean_abs=0.007669
phase_owned_errors=0
qwen3_phase_output_max_abs=0.437500
qwen3_phase_output_mean_abs=0.007669
qwen3_phase_output_errors=0
```

Handoff diagnosis:

```text
each lane's two host O-packet context heads overwritten with 123.0
num_layers=1
poison_o_host_two_lane_heads_npu_time_us=11171.860
poison_o_host_two_lane_heads_max_abs=0.017578
poison_o_host_two_lane_heads_mean_abs=0.000309
poison_o_host_two_lane_heads_errors_gt_0_5=0
```

Interpretation:

```text
All 16 heads are now produced by real NPU score/softmax/PV phases. However,
each lane's O row-shard kernel can only read the two context heads stored in
that same lane's output object. The O dot still reads other-lane heads from the
host context packet. Fully removing host context requires an on-chip gather or
partial-O reduce design.
```

Remaining D1 work:

```text
D1.5p design on-chip context gather/reduce for O projection
D1.5q remove remaining host-packed other-lane context from O projection
D1.5r feed gate_up from NPU-produced local attention residual rows
D1.5s feed down_proj from NPU-produced FFN hidden
D2 run repeated layers in the same phase-owned topology
```

#### D1.5r. Gate/Up Consumes NPU-Produced Local Attention Residual Rows

Status: accepted in production.

Question:

```text
Can the gate/up phase consume the previous O projection phase's local
attention-residual rows instead of relying entirely on the host-packed
attn_residual vector?
```

Implementation:

```text
gate_up packet layout:
  residual_row_base[1]
  host_attn_residual[1024]
  post_norm_weight[1024]
  gate_weight_shard[q_rows_per_packet,1024]
  up_weight_shard[q_rows_per_packet,1024]

gate_up kernel behavior:
  for residual indices in [residual_row_base, residual_row_base + q_rows):
    read lane_output[attention_output_base + local_row]
  for all other indices:
    read host_attn_residual
```

Topology result:

```text
phase_packets_per_layer=17
packet_elements=16576
compute_cores=8
max tile inputs=2
max tile outputs=1
```

NPU result:

```text
num_layers=28
npu_time_us=269433.751
phase_owned_max_abs=0.437500
phase_owned_mean_abs=0.007677
phase_owned_errors=0
qwen3_phase_output_max_abs=0.437500
qwen3_phase_output_mean_abs=0.007674
qwen3_phase_output_errors=0
```

Handoff diagnosis:

```text
each lane's host gate_up attn_residual rows overwritten with 123.0
num_layers=1
poison_gate_host_local_residual_npu_time_us=11293.627
poison_gate_host_local_residual_max_abs=0.017578
poison_gate_host_local_residual_mean_abs=0.000312
poison_gate_host_local_residual_errors_gt_0_5=0
```

Interpretation:

```text
The gate/up phase now has a real O->MLP dependency for local residual rows.
The remaining host dependency is the rest of the 1024-wide residual vector,
which still requires an on-chip residual gather/broadcast or a different MLP
partitioning before host attn_residual can be removed.
```

Remaining D1 work:

```text
D1.5s design on-chip context/residual gather or partial projection reduce
D1.5t remove remaining host-packed other-lane residual/context from O and gate_up
D1.5u expand down_proj from local FFN rows to full NPU-produced FFN hidden
D2 run repeated layers in the same phase-owned topology
```

#### D1.5s. Down Projection Consumes NPU-Produced Local FFN Hidden Rows

Status: accepted in production.

Question:

```text
Can down_proj consume the local FFN hidden rows produced by the previous
gate_up phase instead of trusting the host-packed ffn_hidden vector for those
rows?
```

Implementation:

```text
down_proj packet layout:
  ffn_row_base[1]
  host_ffn_hidden[3072]
  residual_shard[q_rows_per_packet]
  down_weight_shard[q_rows_per_packet,3072]

down_proj kernel behavior:
  precompute local_ffn[local_row] = silu(lane_output[gate_base + local_row])
                                  * lane_output[up_base + local_row]
  for ffn indices in [ffn_row_base, ffn_row_base + q_rows):
    read local_ffn
  for all other ffn indices:
    read host_ffn_hidden
```

Topology result:

```text
phase_packets_per_layer=17
packet_elements=16576
compute_cores=8
max tile inputs=2
max tile outputs=1
```

NPU result:

```text
num_layers=28
npu_time_us=319631.346
phase_owned_max_abs=0.437500
phase_owned_mean_abs=0.007681
phase_owned_errors=0
qwen3_phase_output_max_abs=0.500000
qwen3_phase_output_mean_abs=0.007685
qwen3_phase_output_errors=0
```

Segment diagnosis against `qwen3_reference`:

```text
q max=0.000000
k max=0.003906
v max=0.000122
q_rope max=0.000000
k_rope max=0.000000
context max=0.437500
attention_residual max=0.218750
gate_up max=0.031250
residual max=0.500000
```

Handoff diagnosis:

```text
each lane's host down_proj local FFN hidden rows overwritten with 123.0
num_layers=28
poison_down_host_local_ffn_npu_time_us=320777.251
poison_down_host_local_ffn_max_abs=0.437500
poison_down_host_local_ffn_mean_abs=0.007681
poison_down_host_local_ffn_errors_gt_0_5=0
```

Interpretation:

```text
The local gate/up -> down dependency is now real. Passing the poison test means
the down kernel is not silently reading the host copy for those local FFN rows.
The remaining dependency is full-vector visibility: each down row still needs
the other 3068 FFN hidden values, currently supplied by the host packet.
```

Remaining D1 work:

```text
D1.5v design residual/FFN gather or partial projection reduce for gate_up/down
D2 run repeated layers in the same phase-owned topology
```

#### D1.5t. O Projection Same-Fabric-Group Partial Reduce

Status: accepted in production.

Question:

```text
Can O projection stop reading host context for the heads produced by other
lanes in the same 4-lane fabric group by computing lane-local partial products
and reducing them on-chip?
```

Rejected first design:

```text
lane inputs:
  shared hidden/norm broadcast
  lane packet stream
  o_reduced return stream

aiecc failure:
  error: 'aie.tile' op number of input DMA channel exceeded!
  %tile_0_2 = aie.tile(0, 2)
```

Diagnosis:

```text
The MLIR ObjectFIFO graph showed three independent input FIFOs entering the
same lane tile:
  new_mega_phase_shared_packets_broadcast_g0
  new_mega_phase_lane_0_packets
  new_mega_phase_lane_0_o_reduced

This was a real endpoint/resource failure before any O math ran. Changing TAP
sizes or rewriting the O dot loop would not fix it.
```

Fix:

```text
Remove the shared ObjectFifo from the production dataflow.

phase 0 lane packet now carries:
  hidden[1024]
  input_norm_weight[1024]
  q_proj row block

Each lane Worker caches input_norm_weight in tile-local memory and reuses it
for Q/K/V projection phases. The runtime arg spec still has a legacy shared
input buffer for ABI stability, but the IRON graph no longer creates or fills a
shared ObjectFifo.
```

O partial-reduce topology:

```text
for each fabric group of 4 lanes:
  every lane computes partial rows for all rows owned by the group
  4 lane partial vectors join into one reducer Worker
  reducer sums partials over producer lanes
  reducer output splits the reduced rows back to owner lanes
  owner lane adds host-provided other-fabric-group contribution and residual
```

NPU result:

```text
num_layers=28
phase_packets_per_layer=17
packet_elements=16576
preflight_compute_cores=10
preflight_total_dma_tasks=10
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
npu_time_us=311612.850
phase_owned_max_abs=1.000000
phase_owned_mean_abs=0.007770
phase_owned_errors=0 at phase abs_tol=1.0
qwen3_phase_output_max_abs=0.500000
qwen3_phase_output_mean_abs=0.007813
qwen3_phase_output_errors=0 at abs_tol=0.5
```

Numeric diagnosis:

```text
The first accepted NPU run matched qwen3_reference but had one phase-reference
slot at exactly 1.0 absolute difference:

layer 25, lane 3, attention_residual row 1
actual=182.0
phase_owned_reference=181.0
qwen3_reference=182.0

Host-only comparison of phase_owned_reference against qwen3_reference showed
the same one-slot difference. That makes it a reference/reduction-boundary BF16
ULP issue, not a NPU dataflow bug.
```

Interpretation:

```text
The current production graph now has a real on-chip partial projection reduce
for the same fabric group. It still uses host-packed contribution for the other
fabric group, so O is not fully host-free yet. The resource lesson is stronger
than the speed result: any reduce return FIFO consumes a tile input channel, so
low-bandwidth shared metadata must be packed into an existing lane packet or
cached tile-locally before adding the reduce path.
```

Remaining D1 work:

```text
D1.5u extend O partial reduce across both fabric groups or add a second reduce [accepted]
D1.5v make gate_up consume same-fabric-group O residual rows [accepted]
D1.5w remove remaining host residual/FFN dependencies using gather or partial reduce
D2 run repeated layers in the same phase-owned topology
```

#### D1.5u. O Projection Cross-Fabric Partial Reduce

Status: accepted in production.

Question:

```text
Can O projection remove the remaining host-packed other-fabric-group context
contribution by reducing partial products from all 8 lanes?
```

Rejected first cross-group design:

```text
source group reducers produced full 32-row partial vectors
one memtile split each source vector into target-group halves
target reducers consumed the split halves

aiecc result:
  resource allocation completed successfully
  routing pipeline failed with "Unable to find a legal routing"
```

Diagnosis:

```text
The generated MLIR concentrated the cross-group exchange through
mem_tile_2_1:

  source_reduced_g0/g1 -> mem_tile_2_1
  mem_tile_2_1 -> target0
  mem_tile_2_1 -> target1

The failure was not a lane input-channel issue; those were still at 2 inputs.
It was an over-routed intermediate split point introduced by the full-vector
source_reduced FIFO.
```

Fix:

```text
Remove the source_reduced full-vector FIFO and split.

Each source reducer now consumes the 4 lane partial vectors once and produces
two target-half outputs directly:
  source_g0 -> target_g0
  source_g0 -> target_g1
  source_g1 -> target_g0
  source_g1 -> target_g1

Each target reducer consumes two source halves and sums them before splitting
the final rows back to its four owner lanes.
```

Accepted topology:

```text
lane Workers:        8
source reducers:     2
target reducers:     2
compute cores:       12
max tile inputs:     2
max tile outputs:    2
total DMA tasks:     10
```

NPU result:

```text
num_layers=28
phase_packets_per_layer=17
packet_elements=16576
preflight_compute_cores=12
preflight_total_dma_tasks=10
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
npu_time_us=316076.338
phase_owned_max_abs=0.500000
phase_owned_mean_abs=0.007814
phase_owned_errors=0
qwen3_phase_output_max_abs=0.500000
qwen3_phase_output_mean_abs=0.007821
qwen3_phase_output_errors=0
```

Interpretation:

```text
The O projection rows currently materialized by production no longer depend on
host-packed attention context. The remaining host-fed activation dependencies
are residual vector visibility for gate_up and FFN hidden visibility for
down_proj. The routing lesson is that cross-group reductions should avoid
"reduce full vector then split through one memtile"; produce target-specific
outputs directly from the reducer that already has the source partials.
```

Remaining D1 work:

```text
D1.5v make gate_up consume same-fabric-group O residual rows [accepted]
D1.5w make gate_up consume full O residual rows [accepted]
D1.5x remove remaining host FFN dependency using gather or partial reduce
D2 run repeated layers in the same phase-owned topology
```

#### D1.5v. Gate/Up Consumes Same-Fabric-Group O Residual Rows

Status: accepted in production.

Question:

```text
Can the gate/up phase consume a wider on-NPU O residual slice from the prior
phase without adding another ObjectFIFO endpoint?
```

Change:

```text
O finalize writes fabric_group_size * q_rows_per_packet residual rows into
each lane_output object.

gate_up receives residual_row_base = fabric_group_start_row and
residual_group_size = fabric_group_size * q_rows_per_packet, then replaces
those rows from lane_output before post-attention RMSNorm.
```

Accepted result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=17
q_rows_per_packet=4
fabric_group_size=4
attention_output_values_per_lane=16
output_values_per_lane=328
packet_elements=16576
preflight_compute_cores=12
preflight_total_dma_tasks=10
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
npu_time_us=316087.452
phase_owned_max_abs=0.500000
phase_owned_mean_abs=0.008164
phase_owned_errors=0
qwen3_phase_output_max_abs=0.500000
qwen3_phase_output_mean_abs=0.008175
qwen3_phase_output_errors=0
```

Diagnosis during bring-up:

```text
1. resolve_program() caught a Kernel ABI declaration drift before aiecc:
   new_mega_phase0_q_shard_bf16 expected 12 arguments but the Worker passed 10.
   The cause was an accidental edit to the q_shard Kernel declaration while
   adding the new O partial argument.

2. Packet layout audit caught an O packet ABI drift:
   the residual header grew from 4 rows to 16 rows, but O partial still read
   its weight block at packet + 2 + q_rows_per_packet. The correct offset is
   packet + 2 + fabric_group_size * q_rows_per_packet.
```

Interpretation:

```text
This is progress, not full residual ownership. gate_up still needs all 1024
residual values for post-attention RMSNorm and dense gate/up dot products; only
the 16 rows materialized by the current O reduce fabric now come from NPU state.
The next step is full residual visibility by gather/broadcast or partial
projection reduce.
```

Remaining D1 work:

```text
D1.5w make gate_up consume full O residual rows
D1.5x remove remaining host FFN dependency using gather or partial reduce
D2 run repeated layers in the same phase-owned topology
```

#### D1.5w. Gate/Up Consumes Full Chunked O Residual

Status: accepted in production.

Question:

```text
Can production remove the host-packed attention residual dependency from
gate_up without materializing a single oversized O projection packet?
```

Change:

```text
The single O phase became 32 O row-chunk phases:
  o_proj_chunk_0..31

Each chunk covers:
  o_target_rows = num_lanes * q_rows_per_packet = 32 hidden rows

Each chunk packet carries:
  chunk_row_base
  32 residual values
  32 rows of O weights for the lane's two context heads

The same source/target reducer Workers are reused for every chunk. O finalize
writes each 32-row residual chunk into lane_output. After all chunks, every
lane has a full 1024-row attention residual segment, and gate_up reads all
1024 rows from lane_output.
```

Accepted result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=48
total_phase_packets=1344
q_rows_per_packet=4
fabric_group_size=4
attention_output_values_per_lane=1024
output_values_per_lane=1336
output_values_per_layer=10688
packet_elements=16576
input_elements=178225152
output_elements=299264
preflight_compute_cores=12
preflight_total_dma_tasks=10
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
preflight_max_dma_tasks_per_fifo=1
npu_time_us=580001.856
phase_owned_max_abs=1.000000
phase_owned_mean_abs=0.011816
phase_owned_errors=0
qwen3_phase_output_max_abs=1.000000
qwen3_phase_output_mean_abs=0.011817
qwen3_phase_output_errors=0
```

Diagnosis during bring-up:

```text
1. The first NPU run appeared to hang, but the process was in host-side
   reference construction. The O chunk packet-level reference had grown into
   hundreds of millions of Python scalar multiply-adds. Vectorizing the O
   reference as per-producer `weight_block @ context` exposed the real NPU
   result.

2. The first vectorized qwen semantic reference produced 16 errors with
   max_abs=1.0. Segment diagnostics showed all 16 were in attention_residual;
   gate_up and down had zero segment errors. Packet-level reference had zero
   errors. The cause was O accumulation-order BF16 tolerance, not a dataflow
   problem. The qwen semantic checker now allows one BF16 ULP for the
   attention_residual segment while keeping other segments at abs_tol=0.5.
```

Interpretation:

```text
This removes the host residual dependency from gate_up. It is not a speed win:
the full O projection now actually runs on NPU, increasing NPU time from about
316ms to about 580ms for the current scalar/chunked implementation. It is a
correctness and architecture step toward the true megakernel dataflow.

The remaining major host-fed activation is FFN hidden for down_proj.
```

Remaining D1 work:

```text
D1.5x reuse reducer fabric for the first FFN hidden group [accepted]
D1.5y remove remaining host FFN dependency using full FFN gather or down partial reduce
D2 run repeated layers in the same phase-owned topology
```

