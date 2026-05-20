<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Resources Symptoms

[Back to symptom index](symptoms.md).

## Persistent QKV Cannot Place On 8 Columns

Symptom:

```text
ValueError: Failed to find a tile matching column 0: tried until column 8.
Try using a device with more columns.
```

Diagnostic:

```text
This is a placement/resource failure from resolve_program(), not a runtime
numeric issue. Reduce the persistent stage first, then scale placement
explicitly.
```

Root cause established for bring-up:

```text
The first hand-authored QKV persistent stage was too optimistic as an 8-column
layout before resource ownership was explicit.
```

Fix used for current checkpoint:

```text
Set num_aie_columns=1 for the semantic checkpoint.
```

Status:

```text
This is not a final performance placement. It is the accepted correctness
bring-up fallback.
```

## Persistent QKV Exceeds Output DMA Channels

Symptom:

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_0_3
```

Diagnostic:

```text
Inspect the ObjectFIFO graph around the tile named in the aiecc error. Count
producer outputs from that tile before editing kernels.
```

Evidence found:

```text
weight_worker produced four separate outputs: one debug x_norm drain plus
Q, K, and V copies.
```

Root cause:

```text
The design duplicated the same logical x_norm token as multiple producer
outputs, exhausting tile DMA resources.
```

Fix:

```text
Use one ObjectFIFO with multiple consumers:

hidden_fifo -> rmsnorm_worker -> weight_worker ->
single xnorm broadcast FIFO -> Q/K/V matvec workers and optional debug drain
```

## Attention Score Worker Exceeds Input DMA Channels

Symptom:

```text
error: 'aie.tile' op number of input DMA channel exceeded! %tile_2_3
```

Diagnostic:

```bash
rg -n "tile_2_3|objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
qwen3_rc_q_rope_0  -> tile_2_3
qwen3_rc_k_rope_0  -> tile_2_3
qwen3_rc_k_cache_0 -> tile_2_3
```

Root cause:

```text
The score worker had three input ObjectFIFOs: Q RoPE, current K RoPE, and
K-cache block. The tile input DMA channels were exhausted before runtime.
```

Fix direction:

```text
Do not build score as a three-input worker. Pack Q and current K in a prior
two-input worker, then run score as qk_pair + k_cache_block.
```

## K Cache Matrix Does Not Fit In L1

Symptom:

```text
Failed to allocate buffer: "qwen3_rc_k_cache_0_cons_buff_0" with size: 65536 bytes
allocated buffers exceeded available memory
```

Diagnostic:

```text
Read the aiecc MemoryMap. It printed two 65536-byte K-cache consumer buffers on
one compute tile, before score and Q buffers were counted.
```

Root cause:

```text
The first score design tried to move a full [256, 128] bf16 K-cache head as one
ObjectFIFO object. With depth=2 that alone needs 128KB, above the tile L1 budget.
```

Fix:

```text
Stream K cache as [64, 128] blocks. This makes each K-cache object 16KB and
keeps depth=2 within L1.
```

## Debug Pass-Through FIFO Exceeds L1

Symptom:

```text
Adding an independent K-cache debug stream makes aiecc fail allocation on the
debug worker tile, even though the production K-cache stream already fits.
```

Evidence found:

```text
The debug worker carried both input and output ObjectFIFOs with [64, 128] bf16
objects. With depth=2 on both edges, the tile needs 16KB * 4 = 64KB for debug
FIFO objects before stack and other allocations.
```

Root cause:

```text
The debug copy used normal depth=2 FIFOs on both sides. For a pass-through
diagnostic worker that does no compute overlap, double-buffering both input and
output consumed the whole tile L1 budget.
```

Fix:

```text
Set both K-cache debug pass-through FIFOs to depth=1. The diagnostic still
proves the runtime TAP/stream order, while production K-cache keeps depth=2 for
the score worker.
```

## K Cache Block DMA Exhausts BD IDs

Symptom:

```text
Allocator exhausted available buffer descriptor IDs
```

Diagnostic:

```bash
rg -n "dma_configure_task_for @qwen3_rc_k_cache_0" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
One Python rt.fill per q-head/cache-block generated 64 DMA tasks for the same
K-cache FIFO.
```

Root cause:

```text
The logical tiling was correct, but the runtime expression used too many
separate DMA tasks. BD exhaustion is a runtime/tap expression bug, not a math
kernel bug.
```

Fix direction:

```text
Prefer one multidimensional TensorAccessPattern for the repeated block stream.
If repeated GQA access would require illegal stride=0, reuse K blocks inside a
worker instead of asking DMA to reread the same block.
```

## RuntimeEndpoint Output Placement Exhaustion

Symptom:

```text
ValueError: Failed to find a tile matching column 1: tried until column 8.
Try using a device with more columns.
```

Diagnostic:

```text
This message is ambiguous by itself. Monkey-patch
SequentialPlacer._place_endpoint and print the failing endpoint plus the
`output` flag. If the failing object is RuntimeEndpoint with output=True, the
problem is shim/runtime output endpoint pressure from too many Runtime.fill
streams, not compute-tile placement and not a bad TAP stride.
```

Evidence found during attention2 column scaling:

```text
attention_columns=2 + MLP1 full-layer preflight failed before MLIR was emitted.
The patched placer printed:

  PLACE_FAIL_ENDPOINT RuntimeEndpoint
  PLACE_FAIL_KW {'output': True}
  tile AnyShimTile

The attention-only probe with the same attention2 graph passed preflight once
MLP weight fill endpoints were skipped:

  compute_cores=27
  total_dma_tasks=21
  max_dma_tasks_per_fifo=1
  max_tile_inputs=2
  max_tile_outputs=1
```

Root cause:

```text
The graph had too many independent host -> NPU Runtime.fill output endpoints:
hidden, RoPE, norm metadata, Q/K/V/O attention shards for two columns, cache
K/V shards, and MLP weights. NPU2 shim placement ran out of compatible output
endpoint slots even though the attention-only compute graph itself fit.
```

Fix direction:

```text
Separate attention-only resource probes from full-layer probes. Once attention
is numerically correct, reduce Runtime.fill endpoint count by combining or
staging weight streams before reconnecting MLP.
```

## Compute Tile Exhaustion After Parallelizing Multiple Phases

Symptom:

```text
ValueError: Ran out of compute tiles for placement!
```

Evidence found during attention2 + MLP2:

```text
attention_columns=2 + num_aie_columns=2 + layer_iterations=1 failed placement
with compute tile exhaustion, while attention_probe_only passed and the current
attention_columns=1 + MLP2 full-depth baseline still passed.
```

Root cause:

```text
Expanding both attention and MLP to two columns creates too many persistent
Workers for NPU2. This is a worker-count/resource problem, not an external
kernel ABI problem and not a cache TAP problem.
```

Fix direction:

```text
Fuse adjacent lightweight workers or keep only one major phase expanded while
validating the other. Do not keep adding columns to independent phases and hope
SequentialPlacer will find capacity.
```

## Attention2 QKV Worker Exceeds Output Channels

Symptom:

```text
real_graph_probe: fail stage=n-layer-final-only cols=1 layers=1 phase=preflight
Qwen3PreflightError: Compute tile %tile_0_4 has 3 output ObjectFIFOs; limit=2.
```

Diagnostic:

```text
This happened after trying to reduce Runtime.fill endpoints by merging Q, K,
and V projection weights into one per-column stream. The static preflight
failed before NPU execution, so the external GEMV kernel was not the suspect.
Count producer outputs on the new Worker before changing placement or TAPs.
```

Root cause:

```text
The merged qkv_matvec_worker consumed one xnorm stream and produced three
independent output streams: q_raw, k_raw, and v. That violates the established
two-output practical limit for this full-layer graph.
```

Fix used:

```text
Do not make one QKV worker. Merge only Q and K into one qk_matvec_worker with
two outputs, and keep V as a separate matvec worker. This still removes one
Runtime.fill endpoint per attention column and one compute Worker per column,
while staying within max_tile_outputs=2.
```

Recheck:

```text
attention2 attention-only preflight after QK merge:
  compute_cores=24
  total_dma_tasks=18
  max_tile_inputs=2
  max_tile_outputs=2
```

## Attention2 Plus MLP2 Still Exhausts Runtime Output Endpoints

Symptom:

```text
real_graph_probe: fail stage=n-layer-final-only cols=2 layers=1 phase=preflight
ValueError: Failed to find a tile matching column 3: tried until column 8.
```

Diagnostic:

```text
Run real_graph_probe.py with --trace-placement. Do not assume the old
compute-tile exhaustion diagnosis still applies after QK merge or after
post-norm/gate-up fusion.

Command used:
  real_graph_probe.py --stages n-layer-final-only --columns 2
    --attention-columns 2 --layer-iterations 1 --preflight-only
    --allow-failures --trace-placement
```

Evidence found:

```text
after QK merge:
  placement_trace_fail_counts:
    {'runtime_output': 4, 'runtime_input': 0,
     'other_output': 0, 'other_input': 0}
  placement_trace_fail_key: runtime_output
  placement_trace_fail_type: RuntimeEndpoint
  placement_trace_fail_output: True
  placement_trace_fail_common_col: 6

after fusing post-norm into gate/up column 0:
placement_trace_fail_counts:
  {'runtime_output': 16, 'runtime_input': 6,
   'other_output': 0, 'other_input': 0}
  placement_trace_fail_key: runtime_output
  placement_trace_fail_type: RuntimeEndpoint
  placement_trace_fail_output: True
  placement_trace_fail_common_col: 3
  placement_trace_fail_remaining_tiles: []
```

Interpretation:

```text
The failure moved. Before QK merge, attention2 + MLP2 failed with direct
compute tile exhaustion. After QK merge, the graph had fewer Workers but still
placed too many host->NPU fill endpoints near busy columns. After fusing
post-norm into gate/up column 0, the graph reaches the global shim-output
capacity and then fails on the 17th runtime output stream.
```

Fix used for the one-layer performance probe:

```text
Do not combine the hot K/V cache streams as the accepted fix. A cache-pair
ObjectFifo removed one Runtime.fill, but first generated an illegal 6D BD and
then produced only a neutral/slower warm timing after reducing the TAP to 4D.

The accepted one-layer probe combines the small hidden input and QK/RoPE
metadata into one runtime input buffer:

  hidden_qk_rope_runtime = hidden[1024] || q_norm[128] || k_norm[128]
                           || rope_angles[128]

The graph then splits that object into the existing hidden FIFO and
qk_rope_metadata FIFO. K/V cache reads stay as separate proven streams.
```

Related diagnostic result:

```text
attention2 + down-only MLP2:
  --columns 2 --attention-columns 2 --mlp-gate-up-columns 1
  preflight ok
  placement_trace_counts runtime_output=16, runtime_input=6
  verify: chunk_hidden_errors=0
  warm mean: 5684.301 us

attention2 + full MLP2, hidden+metadata fused:
  preflight ok
  placement_trace_counts runtime_output=16, runtime_input=6, other_input=1
  verify: chunk_hidden_errors=0
  warm iterations 1-4:
    3982.903, 5018.653, 4034.890, 4538.970 us
  warm mean: 4393.854 us
```

Remaining limitation:

```text
This fix is currently a one-layer probe. Full-depth chunk=28 still needs the
post-norm/gate-up fused weight stream to be expressed for multiple layers
without reintroducing a separate post-norm Runtime.fill.
```

Later three-way gate/up experiment:

```text
accepted direct gate/up+SiLU baseline:
  --columns 2 --attention-columns 2 --mlp-gate-up-columns 2
  --mlp-gate-up-pair-rows --mlp-gate-up-direct-silu
  layer_iterations=28 preflight ok
  placement_trace_counts: {'runtime_output': 16, 'runtime_input': 5,
                           'other_output': 0, 'other_input': 1}

three-way gate/up branch:
  --columns 2 --attention-columns 2 --mlp-gate-up-columns 3
  --mlp-gate-up-pair-rows --mlp-gate-up-direct-silu
  layer_iterations=1 and layer_iterations=28 both fail before MLIR preflight
  placement_trace_fail_key: runtime_output
  placement_trace_fail_type: RuntimeEndpoint
  placement_trace_fail_output: True
  placement_trace_fail_common_col: 3
  placement_trace_fail_counts: {'runtime_output': 16, 'runtime_input': 6,
                                'other_output': 0, 'other_input': 1}
```

Interpretation:

```text
The three-way branch did not fail because of token correctness, C++ kernel ABI,
or layer depth. It failed because the extra gate/up weight stream required a
17th runtime output endpoint after the graph had already reached the accepted
runtime-output endpoint boundary. More gate/up parallelism needs a packed/split
weight stream or another endpoint reduction before it can be a performance
candidate.
```

Follow-up packed/split experiment:

```text
The follow-up packed gate/up shard1 and shard2 behind one runtime stream and
used ObjectFifo.split inside the graph.

preflight:
  compute_cores=32
  max_tile_inputs=2
  max_tile_outputs=2
  placement_trace_counts:
    {'runtime_output': 16, 'runtime_input': 5,
     'other_output': 0, 'other_input': 2}

full generate:
  token_match=True
```

Interpretation:

```text
ObjectFifo.split solved the 17th runtime output endpoint blocker. This was a
resource fix, but not a performance acceptance: the three-way split graph was
slower than the accepted two-way baseline on the Fibonacci multi-token prompt
and used all 32 compute cores.
```

## Cache-Pair TAP Passes Preflight But Fails BD Or Numerics

Symptom:

```text
Combining K-cache and V-cache reads into one ObjectFifo split makes
attention2 + full MLP2 pass placement, but full aiecc or attention verification
fails.
```

Diagnostics:

```text
1. Run preflight with --trace-placement to prove the RuntimeEndpoint failure is
   gone.
2. Run full compile, because preflight does not validate every BD lowering
   constraint.
3. If compile passes but attention residual is wrong, isolate with
   --attention-probe-only before inspecting MLP.
```

Evidence found:

```text
first cache-pair TAP:
  dimensions [layer, head, block, K/V, row, dim]
  aiecc error:
    At most four data layout transformation dimensions may be provided.

second cache-pair TAP:
  collapsed head*block into one dimension
  compile passed, but attention_probe_residual_errors=610
```

Root cause:

```text
The collapsed head*block dimension used cache_block_seq * head_dim as the
stride. That reads the next active block correctly only when all blocks for a
head are included. At position 26 only one block is active, so the next logical
head was read from the current head's future cache block.
```

Status:

```text
Rejected as the performance path for now. The corrected single-active-block
variant verifies, but the warm mean was about 4484.962 us, slightly slower
than the small-stream hidden+metadata fusion and not general for later decode
positions.
```

## ObjectFIFO Depth Passes Preflight But Fails AIECC Block/BD Allocation

Symptom:

```text
real_graph_probe.py --preflight-only passes:
  max_fifo_buffered_bytes=32768
  max_dma_tasks_per_fifo=1
  max_tile_inputs=2
  max_tile_outputs=2

full aiecc later fails:
  error: 'aie.mem' op has more than 16 blocks
  note: no space for this BD
  Pipeline failed while executing AIEObjectFifoStatefulTransform
```

Diagnostic:

```text
Do not stop at Python preflight for a new ObjectFIFO shape. Re-run full compile
and read the `aie.mem` dump. If one FIFO acquired many small objects, count the
number of FIFO blocks created on that memory tile, not only object bytes.
```

Evidence found during gate/up row-group 8:

```text
The first row-group 8 implementation used a hidden_weight_ty FIFO with depth 16
so one worker could acquire:
  8 gate rows + 8 up rows

Preflight saw the same 32KB max buffered bytes as other accepted streams, but
aiecc resource allocation rejected the memory tile because the FIFO created too
many individual blocks/BD entries.
```

Root cause:

```text
The resource limit was block/BD count on the generated `aie.mem`, not endpoint
placement, not token correctness, and not the 8-row external kernel math.
```

Fix used for diagnosis:

```text
Pack the widened non-fused gate/up stream as 4-row FIFO objects:
  [4 gate rows][4 gate rows][4 up rows][4 up rows]

This reduced the widened stream from 16 single-row FIFO objects to 4 block
objects. The graph then passed full aiecc and matched the default prompt token.
```

Decision:

```text
The fixed row-group 8 branch was rejected for performance because it was slower
than the accepted 4-row direct-SiLU baseline. Keep this symptom entry because
it exposes a preflight blind spot: ObjectFIFO block count can fail later even
when byte, task, and tile-port limits look safe.
```

## Sharded Multi-Layer Weight TAP Exceeds BD Stride

Symptom:

```text
'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.
aie.dma_bd(... [<size = 2, stride = 3145728>, ...])
```

Diagnostic:

```text
Find the failing `aie.dma_bd` and map its offset/length back to the packed
weight segment. In the column-scaling experiment the offset was inside the
MLP down segment, not Q/K/V or cache movement.
```

Evidence found:

```text
The first two-column n-layer down-shard used a TAP shaped like:
  [layer_iterations, 1, 1, shard_rows * intermediate_size]
with layer stride:
  hidden_size * intermediate_size = 3145728 bf16 elements

That stride is legal as a tensor concept but illegal for this NPU BD lowering.
```

Root cause:

```text
The host packed weights are segment-major across layers. A column shard of
down_proj is not contiguous across layers; it jumps by a full down matrix per
layer. Encoding that jump as one repeated TAP exceeded the BD stride limit.
```

Fix used:

```text
Keep the single-column path as one contiguous full-depth down-weight TAP.
For multi-column down, emit one linear shard fill per layer and per column.
This raises max_dma_tasks_per_fifo to the layer count, so it is acceptable for
small chunks but not a final full-depth solution.
```

Accepted recheck:

```text
cols=2 layers=2: max_dma_tasks_per_fifo=2, verify errors=0
cols=2 layers=4: max_dma_tasks_per_fifo=4, verify errors=0
cols=2 layers=8: preflight max_dma_tasks_per_fifo=8
```

## Sharded Gate/Up Runtime Fills Exhaust Shim Endpoints

Symptom:

```text
real_graph_probe: fail stage=n-layer-final-only cols=2 layers=2
ValueError: Failed to find a tile matching column 2: tried until column 8.
```

Diagnostic:

```text
Monkey-patching SequentialPlacer._place_endpoint showed the failure happened
while placing a RuntimeEndpoint with output=True and no shim tiles left. This
was before aiecc and before any external kernel ran.
```

Evidence found:

```text
The first gate/up-sharded n-layer graph emitted per-layer runtime fills for:
  post_norm
  gate shard 0
  gate shard 1
  up shard 0
  up shard 1

This was in addition to the existing Q/K/V/O/cache/down traffic.
```

Root cause:

```text
The graph expressed a better compute partition but a worse runtime DMA graph.
Too many host->NPU Runtime.fill endpoints exhausted shim placement.
```

Fix used:

```text
Do not use per-layer gate/up shard fills. Pack MLP weights on the host into
contiguous two-column segments so each shard FIFO receives one full chunk fill.
```

## Gate/Up Shard Worker Has Too Many Input FIFOs

Symptom:

```text
Qwen3PreflightError: Compute tile %tile_4_4 has 3 input ObjectFIFOs; limit=2.
```

Diagnostic:

```bash
rg -n "tile_4_4|qwen3_full_layer_mlp_" \
  build_qwen3_column_probe_gateup_split/*.mlir
```

Evidence found:

```text
tile_4_4 consumed:
  qwen3_full_layer_mlp_xnorm
  qwen3_full_layer_mlp_gate_weight_0
  qwen3_full_layer_mlp_up_weight_0
```

Root cause:

```text
Splitting gate and up into separate weight FIFOs made the compute worker a
three-input tile even though the math itself was unchanged.
```

Fix used:

```text
Pack each column's gate rows followed by up rows into one ObjectFIFO. The
gate/up shard Worker then consumes only:
  xnorm
  packed gate_up shard weight rows
```

## NPU Weight Split Copy Is Correct But Too Slow

Symptom:

```text
cols=2 layers=2 verify passes, but npu_time_us is about 152935 us.
```

Diagnostic:

```text
Compare against the previous down-only two-column path and the packed MLP2
path. Both use the same attention and residual-join boundaries, so the new
cost is isolated to the NPU-side weight split-copy phase.
```

Evidence found:

```text
down-only two-column layers=2 warm time: about 12.4-12.8 ms
NPU split-copy gate/up layers=2 time: about 153 ms
packed MLP2 layers=2 warm time: about 9.8-9.9 ms
```

Root cause:

```text
The split-copy Worker copied millions of bf16 weight elements on the NPU just
to route gate/up shards. That routing work dominated the GEMV speedup.
```

Fix used:

```text
Move the split to host-side/preprocessed weight packing. Runtime should move
already-contiguous shard streams; NPU Workers should spend cycles on GEMV and
activation, not weight repacking.
```

## N-Layer Chunk 8 Exhausts Current KV Writeback BD IDs

Symptom:

```text
n-layer-final-only --layer-chunk-size 8 --compile-only
Allocator exhausted available buffer descriptor IDs
Failed to allocate buffer: "qwen3_full_layer_mlp_down_weight_cons_buff_0"
```

Diagnostic:

```bash
rg -n "dma_configure_task_for @qwen3_rc_k_rope_0|dma_configure_task_for @qwen3_rc_v_0" \
  build_qwen3_persistent_prefix_chunk8/*.mlir
```

Evidence found:

```text
The original chunk=8 graph emitted one current-K and one current-V drain task
per layer for the same FIFO. After aggregating the chunk writeback into one
drain per FIFO, aiecc still lowered the TAP to repeat_count=7 with dimensions:

[<size = 8, stride = 524288>, <size = 1, stride = 0>,
 <size = 8, stride = 32768>, <size = 128, stride = 1>]

The failure remained on @qwen3_rc_k_rope_0 and @qwen3_rc_v_0.
```

Root cause:

```text
The first diagnosis was incomplete. Chunk=8 is not blocked by attention math,
but the original layer-major weight chunk made every weight FIFO receive one
DMA task per layer. That static task replication combined with current-KV
writeback and cache movement exhausted BD/L1 resources during lowering.
```

Fix used now:

```text
Repack n-layer chunk weights segment-major:

input_norm for all layers
Q weights for all layers
K weights for all layers
V weights for all layers
Q/K norm weights for all layers
O weights for all layers
post_norm + gate + up weights for all layers
down weights for all layers

Then each weight FIFO receives one linear DMA stream for the whole chunk.
```

Follow-up optimization evidence:

```text
segment-major chunk=8 compile/preflight:
compute_cores=21 total_dma_tasks=29 max_dma_tasks_per_fifo=8

after grouping cache fill and writeback by 4 layers:
chunk=8 preflight: max_dma_tasks_per_fifo=2
chunk=8 generate: token_match=True
```

Full-depth follow-up:

```text
Bypassing the 8-layer guard and compiling chunk=28 with cache grouped by 4
failed at NPU lowering on @qwen3_rc_k_rope_0 and @qwen3_rc_v_0. The generated
MLIR had seven K writeback tasks and seven V writeback tasks on the same FIFO,
each with repeat_count=3. aiecc reported:

Allocator exhausted available buffer descriptor IDs
Free called on BD chain with unassigned IDs
```

Root cause of the chunk=28 compile failure:

```text
grouped-by-4 was enough for chunk=8, but still replicated too many writeback
DMA tasks for a 28-layer graph. The failure was not caused by the full-depth
worker loop itself; it was caused by static DMA task replication at the cache
writeback boundary.
```

Fix used for the performance path:

```text
For layer_iterations > 8, group historical cache fill and current K/V writeback
as one full-depth TAP. The chunk=28 graph then compiles with:

compute_cores=21
max_fifo_buffered_bytes=32768
max_dma_tasks_per_fifo=1
```

This still is not a general descriptor-driven state machine. It is a legal
static full-depth stream. Keep chunk sizes 1..8 for bisection; use chunk=28
when the goal is reducing 8+8+8+4 host dispatches to one NPU dispatch per
decoded token.

## Synthetic Persistent Graph Exhausts BD IDs At 9 DMA Tasks Per FIFO

Symptom:

```text
Allocator exhausted available buffer descriptor IDs
Free called on BD chain with unassigned IDs
```

Diagnostic command used:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/graph_probe.py \
  --patterns separate repeat grouped \
  --layers 8 9 28 64 \
  --group-layers 4 \
  --preflight-only \
  --clean-build \
  --build-dir build_qwen3_persistent_graph_probe_preflight
```

Evidence found:

```text
separate layers=8:  ok, total_dma_tasks=16, max_dma_tasks_per_fifo=8
separate layers=9:  fail, current_kv FIFO has 9 DMA tasks
repeat   layers=64: ok, total_dma_tasks=2,  max_dma_tasks_per_fifo=1
grouped  layers=28: ok, total_dma_tasks=14, max_dma_tasks_per_fifo=7
grouped  layers=64: fail, current_kv FIFO has 16 DMA tasks
```

Root cause:

```text
The failure is not caused by the KV scatter TAP region itself. It is caused by
expressing many logically similar transfers as too many DMA tasks on the same
ObjectFIFO endpoint. In this synthetic persistent graph, aiecc lowering fails
as soon as one FIFO reaches 9 DMA tasks.
```

Fix direction:

```text
For large persistent graphs, compress layer repetition into one legal repeated
TAP when possible. If one repeated TAP is not legal for the real layout, group
layers so each FIFO stays at eight or fewer DMA tasks, then spend remaining
resources on the actual compute workers.
```

## Generic GEMV Tile Input 16 Exceeds L1 On LM Head Probe

Symptom:

```text
Failed to allocate buffer: "A_L3L1_0_cons_buff_0" with size: 32768 bytes.
allocated buffers exceeded available memory
Basic sequential allocation also failed.
```

Diagnostic:

```text
Read the aiecc MemoryMap before changing the math kernel. In the LM-head GEMV
probe, the failure happened on a generic GEMV input tile, not on Qwen3 final
logits math or argmax.
```

Evidence found:

```text
Configuration:
  M = 151936
  K = 1024
  cols = 8
  tile_size_input = 16
  tile_size_output = 16

MemoryMap excerpt:
  A_L3L1_0_cons_buff_0: 32768 bytes
  A_L3L1_0_cons_buff_1: 32768 bytes
  B_L3L1_0_cons_buff_0: 2048 bytes
  C_L1L3_0_buff_0: 32 bytes
  C_L1L3_0_buff_1: 32 bytes
```

Root cause:

```text
The generic GEMV design double-buffers the matrix tile. With
tile_size_input=16 and K=1024, the two matrix input buffers alone consume 64KB.
That leaves no room for stack, vector, output buffers, or anonymous allocations
on the same compute tile.
```

Fix:

```text
For this generic GEMV design at K=1024, keep tile_size_input <= 8. The tested
compiling LM-head variants with tile_size_input in {1, 2, 4, 8} were
numerically correct, but still slower than CPU F.linear for the final LM head.
Do not treat the L1 fix as evidence that generic GEMV is the right performance
branch.
```

## Full-Layer MLP Worker Exceeds Input DMA Channels

Symptom:

```text
error: 'aie.tile' op number of input DMA channel exceeded! %tile_3_2
```

Diagnostic:

```bash
rg -n "tile_3_2|objectfifo" build_qwen3_persistent_full_layer/*.mlir
```

Evidence found:

```text
qwen3_rc_attn_residual_0                  -> tile_3_2
qwen3_full_layer_mlp_post_norm_weight     -> tile_3_2
qwen3_full_layer_mlp_gate_weight          -> tile_3_2
qwen3_full_layer_mlp_up_weight            -> tile_3_2
```

Root cause:

```text
The first full-layer MLP worker tried to consume residual, post-norm weight,
gate weight, and up weight on one compute tile. This failed before kernel
execution; the external GEMV math was not the boundary to inspect.
```

Fix:

```text
Stream post-norm/gate/up weights through one row-wise weight FIFO and use a
4-row matvec kernel. Then split down_proj and residual add so the down tile
also has only two input streams.
```

Recheck:

```text
preflight: ok ... max_tile_inputs=2 max_tile_outputs=2
```

## Direct Input-RMSNorm Fusion Exceeds Tile Input Limit

Symptom:

```text
real_graph_probe: fail stage=n-layer-final-only cols=2 layers=28 phase=preflight
Qwen3PreflightError: Compute tile %tile_0_2 has 3 input ObjectFIFOs; limit=2.
Pack or stage inputs before changing external-kernel math.
```

Attempted change:

```text
Fuse the current input-normalization pair:
  tile_0_2: copy hidden to chunk state + unweighted input RMSNorm
  tile_0_3: multiply unweighted norm by input_layernorm.weight

into one Worker:
  copy hidden to chunk state + weighted input RMSNorm
```

Why it looked attractive:

```text
The promoted full-depth graph uses all 32 compute cores, and this fusion would
remove one Worker and one intermediate FIFO if legal.
```

Root cause:

```text
For layer_iterations=28, the fused Worker needs three logical input streams:

1. runtime hidden from the hidden/metadata split FIFO
2. hidden feedback from the previous layer inside the chunk
3. input_layernorm.weight for the current layer

The current resource invariant is max_tile_inputs=2 for compute tiles. The
fusion failed before MLIR lowering/aiecc, so this is a graph resource problem,
not an external kernel math problem.
```

Decision:

```text
Do not use direct input-RMSNorm Worker fusion as the next core-freeing branch.
Any replacement must first reduce or stage the inputs so the target tile still
has at most two input ObjectFIFOs.
```

## Real Full-Layer Graph Is Locked To One Column

Symptom:

```text
real_graph_probe: fail stage=full-layer cols=2 ...
ValueError: scores+softmax checkpoint is currently single-column only
```

Diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages qkv mlp-gate-up full-layer \
  --columns 1 2 4 8 \
  --preflight-only \
  --allow-failures \
  --clean-build
```

Evidence found:

```text
qkv cols=1: ok compute_cores=5
qkv cols=2: ok compute_cores=8
qkv cols=4: ok compute_cores=14
qkv cols=8: placement failure

mlp-gate-up cols=1: ok compute_cores=5
mlp-gate-up cols=2: ok compute_cores=9
mlp-gate-up cols=4: placement failure

full-layer cols=1: ok compute_cores=19
full-layer cols=2/4/8: rejected by scores+softmax single-column guard
```

Root cause:

```text
The current full decode graph cannot use NPU2's available columns because the
attention scores/softmax/full-layer implementation is intentionally
single-column. Some expensive real subgraphs already scale beyond one column,
so the blocker is graph composition and placement around the attention closure,
not merely missing hardware capacity.
```

Fix direction:

```text
Lift the single-column attention/full-layer boundary. Keep QKV and MLP
front-half multi-column, then make score/softmax/context/O-projection consume
partitioned Q heads and shared K/V blocks without adding illegal three-input
tiles or L1-heavy debug streams.
```

## Full-Layer K Project Exceeds Output DMA Channels

Symptom:

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_0_5
```

Evidence found:

```text
The K project+norm+RoPE worker produced k_raw, k_norm, and k_rope. Only k_rope
was needed by downstream compute in the full-layer checkpoint.
```

Root cause:

```text
k_raw and k_norm were debug-only drains. Keeping them as ObjectFIFOs made the
compute tile spend output DMA resources on values the accepted full-layer
verifier no longer consumed.
```

Fix:

```text
Use tile-local Buffer storage for raw/norm temporaries in the full-layer path
and only send k_rope across ObjectFifo.
```

## Full-Layer Q/K Project Exceeds Input DMA Channels

Symptom:

```text
error: 'aie.tile' op number of input DMA channel exceeded! %tile_0_4
```

Evidence found:

```text
The fused Q project+norm+RoPE worker consumed xnorm, q_weight, qk_norm_weight,
and rope_angles.
```

Root cause:

```text
Fusing math reduced worker count, but it created a four-input compute tile.
On this IRON ObjectFifo lowering, worker count was not the limiting resource;
tile DMA endpoint count was.
```

Fix:

```text
Split Q/K into matvec -> norm+RoPE. Add a metadata worker that packs q/k norm
weights and RoPE angles into one ObjectFIFO, so norm+RoPE consumes raw + metadata
instead of raw + norm_weight + rope_angles.
```

Recheck:

```text
compile_s: 71.184
preflight: ok ... compute_cores=19 max_tile_inputs=2 max_tile_outputs=2
```

## Full-Layer MLP Debug Output Exceeds Output DMA Channels

Symptom:

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_3_5
```

Evidence found:

```text
The postnorm/gate/up worker produced ffn_gate, ffn_up, and debug-only
mlp_xnorm.
```

Root cause:

```text
The debug drain was the third producer output from the tile. It was useful while
building the isolated MLP checkpoint, but it was too expensive in the composed
full-layer graph.
```

Fix:

```text
Keep mlp_xnorm in a tile-local Buffer, continue producing ffn_gate and ffn_up,
and verify downstream ffn_hidden/ffn_out/layer_residual instead.
```

## Full-Layer Down Projection Exceeds L1

Symptom:

```text
Failed to allocate buffer: "qwen3_full_layer_mlp_down_weight_cons_buff_0"
allocated buffers exceeded available memory
```

Diagnostic:

```text
Read the aiecc MemoryMap printed under the failing tile. Do not infer this from
tensor shapes alone.
```

Evidence found:

```text
qwen3_full_layer_mlp_down_weight_cons_buff_0 : 24576 bytes
qwen3_full_layer_mlp_down_weight_cons_buff_1 : 24576 bytes
qwen3_full_layer_ffn_hidden_0_cons_buff_0    :  6144 bytes
qwen3_full_layer_ffn_hidden_0_cons_buff_1    :  6144 bytes
qwen3_full_layer_ffn_out_buff_0              :  2048 bytes
qwen3_full_layer_ffn_out_buff_1              :  2048 bytes
```

Root cause:

```text
The down weight ObjectFIFO was double-buffered. One object is already 4x3072
bf16 = 24576 bytes, so depth=2 plus hidden/output buffers exceeded tile L1.
```

Fix:

```text
Set qwen3_full_layer_mlp_down_weight depth=1. This trades overlap for a legal
checkpoint and keeps the full-layer graph compilable.
```

## Multidimensional TAP Is Legal But NPU BD Rejects It

Symptoms:

```text
Stride 2 must be a positive integer
Size 0 exceeds the [0:1023] range
```

Evidence found:

```text
[size = 2, stride = 0] was rejected.
[size = 8192, stride = 1] was rejected in a multidimensional BD.
```

Root cause:

```text
TAP can describe access maps that are semantically clear but illegal for NPU BD
lowering. Reusing a tensor region with stride=0 across a non-unit dimension is
not accepted, and individual BD dimension sizes must stay in range.
```

Fix:

```text
Express K-cache blocks as [kv_head, block, row, dim] with strides
[max_seq_len * head_dim, block_rows * head_dim, head_dim, 1].
```

## New Stage Exceeds SequentialPlacer Worker Capacity

Symptom:

```text
ValueError: Failed to find a tile matching column 3: tried until column 8.
Try using a device with more columns.
```

Root cause:

```text
The accepted context checkpoint already used 15 compute Workers on the current
NPU2 placement. Adding separate context-flatten, O-projection, and residual
Workers pushed the design past the 16 compute-tile budget used by the
SequentialPlacer in this environment.
```

Fix:

```text
Fuse context flatten into the context Worker and disable the older K-cache
debug copy Worker in the deeper O-projection checkpoint. Keep debug at the
new boundary instead: attn_context, attn_context_flat, attn_o_proj,
attn_residual.
```

Recheck:

```text
preflight: ok ... max_tile_inputs=2 max_tile_outputs=2
```
