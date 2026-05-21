<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lessons For The Next Qwen3 Megakernel

This note records what we have learned from the current
`qwen3_0_6b/persistent` work and from reviewing the earlier external
architecture draft.

Treat that earlier draft as a set of hypotheses, not as an implementation
blueprint. It identified some real pressure points, but it did not yet cover
the whole Qwen3 inference path and it assumed several IRON behaviors that were
not proven.

## Current Ground Truth

The accepted persistent path is a single-token decode-body implementation:

```text
model: Qwen3-0.6B
hidden_size: 1024
head_dim: 128
Q projection width: 2048
K/V projection width: 1024
intermediate_size: 3072
accepted body: n-layer-final-only, layer_chunk_size=28
columns: num_aie_columns=2, attention_columns=2, mlp_gate_up_columns=2
final norm / LM head: CPU tail
position handling: exact-position artifacts
```

The current best verified direction is not a "true dynamic position"
megakernel yet. It is:

```text
exact-position precompile + runtime selection
```

That moves compilation out of the token loop, but artifact count and setup time
still scale with the number of generated positions.

The important artifact-diff result is:

```text
same-bucket pos26 -> pos27:
  runtime .bin has DMA offset patch candidates
  core ELFs also change in .text instruction words
```

So patching only the runtime `.bin` is insufficient. Raw ELF/CDO patching is
not a safe implementation path until there is a stable instruction encoding or
relocation rule.

## What The External Architecture Gets Right

The document is useful because it points at the right class of problems:

```text
position-dependent compilation is still a real blocker
tile utilization is still lower than we want
static worker-per-operation dataflow can leave compute tiles idle
GEMV phases are the main place to look for more parallelism
we need small proof steps before another large rewrite
```

Its phased experiment list is directionally useful. A standalone GEMV proof,
then multi-tile GEMV, then phase sequencing, then attention, then full layer is
the right kind of progression.

The document also correctly emphasizes that every step must end with an
accepted/rejected diagnosis, not with a vague "maybe faster" result.

## What It Does Not Cover

It is not an end-to-end Qwen3 inference architecture. It mostly discusses a
decode-body rewrite.

Missing or under-specified pieces:

```text
tokenizer and prompt handling
embedding lookup
prefill path
KV cache initialization and ownership
multi-token decode state machine
GQA head mapping
exact Qwen3 layer order and residual boundaries
packed weight manifest and ABI
XRT buffer lifetime and device/host sync rules
final norm, LM head, argmax/sampling
full-model correctness gates
failure preflight and artifact diff gates
```

Before calling anything a "new megakernel architecture", it needs a full
boundary map:

```text
HF weights / packed weights
-> tokenizer / prompt
-> embedding
-> prefill or cached state
-> decode token loop
   -> 28 transformer layers
   -> KV cache update
   -> final norm
   -> LM head
   -> argmax or sampling
-> text output
```

Every edge in that map needs an owner: CPU, host runtime, XRT buffer, runtime
DMA, ObjectFifo, AIE Worker, or external kernel.

## Technical Lessons

### 1. Use Real Qwen3 Shapes

Do not design from guessed dimensions. The reviewed document used Q=1536 and
K/V=256, which does not match the current Qwen3-0.6B operator.

Use the actual decode-body shapes:

```text
hidden: 1024
Q: 2048
K: 1024
V: 1024
O: 2048 -> 1024
gate: 1024 -> 3072
up: 1024 -> 3072
down: 3072 -> 1024
```

Wrong dimensions make placement, weight traffic, and performance estimates
meaningless.

### 2. `task_group` Is Not A General Worker Phase Scheduler

`Runtime.task_group()` and `finish_task_group()` coordinate runtime DMA tasks.
They are not a proven mechanism for telling persistent Workers to repeatedly
switch between arbitrary phases.

GEMM proves this narrower fact:

```text
host writes RTPs
WorkerRuntimeBarrier releases Workers
Workers read RTPs and run a fixed loop
```

It does not prove this stronger claim:

```text
host can run 28 * 12 phases and update per-phase RTPs while the same persistent
Workers dynamically change behavior at every phase boundary
```

That must be proven separately.

### 3. RTP Solves Tile-Local Scalars, Not Runtime TAPs

`Buffer(use_write_rtp=True)` is useful when a Worker reads a scalar from tile
memory. It does not automatically make `Runtime.fill()` or `Runtime.drain()`
DMA offsets dynamic.

Our position problem has two surfaces:

```text
runtime .bin:
  current K/V DMA offsets and cache block transfer sizes

core ELFs:
  attention score/context/V-merge/mask position immediates
```

An RTP scalar inside an AIE core can help the second surface only if the kernel
is rewritten to read it correctly. It does not by itself fix shim DMA BD/TAP
offsets.

### 4. Bucket Variants Are Not Proven Until Same-Bucket Artifacts Match

The idea "one xclbin per 64-token bucket" is attractive, but it is not proven
by cache read length alone.

Same-bucket pos26 -> pos27 still changed artifacts in the accepted graph. A
bucket design is valid only after:

```text
same-bucket positions do not change core ELFs
same-bucket positions do not require new runtime .bin DMA offsets, or those
  offsets have a proven safe patch/update path
token IDs match PyTorch across multiple positions
```

Until then, bucket variants are an experiment, not a solution.

### 5. Full-Array GEMV Must Respect Endpoint And BD Limits

"Use all 32 tiles" is not enough. A design with 32 independent L3->L1 weight
streams and 32 independent L1->L3 output streams is very likely to hit runtime
endpoint, DMA BD, or placement limits.

The existing GEMM examples use L3->L2->L1 split/forward/join patterns. A real
full-array GEMV experiment should start from those patterns, not from 32
uncoordinated runtime FIFOs.

Minimum evidence before scaling:

```text
preflight runtime endpoint counts
max_dma_tasks_per_fifo
L1 object bytes
tile input/output count
full aiecc, not only resolve_program()
numeric match against F.linear
```

### 6. FIFO Token Semantics Matter More Than Phase Names

A "universal Worker" that sometimes skips a phase must still have balanced
ObjectFifo acquire/release behavior.

This is unsafe:

```text
acquire B
acquire C
then read n_rows and decide to skip
```

If the host does not fill B/C for skipped workers, the Worker hangs. If the host
does fill them, the phase still consumes DMA and FIFO resources. Phase control
must be represented as a real dataflow protocol, not just an RTP flag.

### 7. Attention Is Not Free

For short decode positions, GEMV dominates, but the attention phase is not a
zero-cost scalar detail. Current phase probes already show attention-only
one-layer time at millisecond scale.

Before assigning attention to one tile, prove:

```text
Q/K RMSNorm and RoPE match reference
GQA head mapping is correct
KV cache write/read layout is correct
softmax row sums and masks are correct
context output matches reference
two different positions run without recompile
```

### 8. Performance Estimates Need Real Traffic And Measured Overheads

The reviewed estimate assumes optimistic weight traffic and bandwidth. A useful
estimate must include:

```text
real Q/K/V/O/gate/up/down dimensions
bf16 weight bytes
activation round trips
KV cache traffic
runtime DMA task overhead
ObjectFifo split/join overhead
output drain and host sync
CPU final norm / LM head if still on CPU
```

Use measured phase probes and work estimators as calibration. Do not claim
10x-class speedup from theoretical DDR bandwidth alone.

### 9. Segment-Major Packing Helps, But Does Not Prove Every Phase

The current segment-major packed weights are proven for the accepted persistent
graph. They do not automatically prove a new phase-major layout, new row
sharding, or new full-array broadcast/join topology.

Every new packed layout needs:

```text
manifest offsets
shape/dtype checks
alignment checks
tap/access coverage checks
monotonic pattern or F.linear validation
```

### 10. Exact-Position Precompile Is A Baseline, Not The End State

The new `--precompile-generate-positions` path is valuable because it separates:

```text
hot-loop token time
from
setup-time artifact generation
```

Use it when comparing graph-body changes. Do not confuse it with solving
dynamic position reuse.

### 11. Fixed-Chunk Attention Avoids Position Artifacts On The Read Side

The A0 fixed-chunk attention experiment proved a more useful position strategy
than RTP for the attention read path:

```text
compile max_seq_len, head_dim, and chunk_size
always process the same fixed number of KV chunks
encode live position only in the runtime mask tensor
use online softmax so chunked processing matches full softmax
```

One artifact ran positions 0, 26, 63, 64, 127, 200, and 255 against CPU
reference. This means the attention read side does not need per-position TAPs,
runtime BD patching, or a position immediate in the Worker.

The resource lesson was equally important:

```text
Q + K + V + mask as four input FIFOs exceeds one tile's input DMA channels.
Q + packed(K,V,mask) fits and preserves the fixed-TAP proof.
```

This does not solve current-token K/V cache writeback by itself. It removes the
attention read side from the exact-position artifact problem.

### 12. Host-Side KV Writeback Avoids Dynamic NPU Cache Offsets

The A0B host-side KV writeback experiment proved the Intel-style state boundary
for current-token K/V:

```text
NPU input:  full fixed-shape past cache
NPU output: fixed-shape present_k/present_v
host:       memcpy present_k/present_v into cache[position]
```

One artifact ran 70 sequential token steps and crossed the 63/64 chunk
boundary. The NPU never wrote the KV cache and never received position as an
argument. The next invocation still read the rows that the host wrote, because
the host updated the packed cache buffer between dispatches.

This changes the position problem:

```text
do not make NPU DMA write to a dynamic row
do not patch BD offsets for current K/V writeback
do not compile per-position cache write TAPs
```

Keep state ownership on the host. The NPU blob stays a fixed-shape pure
function.

The debug lesson was also concrete:

```text
BF16 stores in new AIE kernels should set conv_even rounding explicitly.
Without that, a correct dataflow can fail by one BF16 ULP.
```

### 13. Static Phase Ownership Needs Packed Lane Streams

C1 proved a fixed multi-phase Worker protocol. C2 showed that skipping inactive
FIFO dependencies without dummy DMA requires a different static graph. D0 then
tested the resource-safe version of static phase ownership:

```text
8 lane Workers
1 packed input FIFO per lane
1 padded output FIFO per lane
11 fixed phase packets per lane
packet size = 33792 bytes
```

That graph passed full aiecc, preflight, and NPU execution:

```text
compute_cores=8
max_compute_tile_inputs=1
max_compute_tile_outputs=1
max_fifo_buffered_bytes=33792
max_dma_tasks_per_fifo=1
```

The lesson is not that this skeleton is fast. The lesson is that a real
phase-owned Qwen3 layer should avoid "one FIFO per phase input" and instead
pack each lane's fixed phase payloads into a small number of homogeneous
streams. That is how to keep endpoint and BD pressure bounded while preserving
a fixed phase order.

The debug lesson was also concrete:

```text
Do not create scalar BF16/F16 DMA-visible FIFO objects. memref<1xbf16> failed
aiecc because DMA BD transfer length must be 4-byte aligned; memref<2xbf16>
passed that hardware rule but failed the stricter 16-byte FIFO preflight. Pad
small metadata/output objects to 16 bytes.
```

### 14. Fixed-Cache Attention Works On Real Qwen3 Tensors

D1.0 moved A0 from synthetic data to real Qwen3 layer-0 attention context data:

```text
Q after q_norm + RoPE:             [16, 128]
K cache after host current write:  [8, 256, 128]
V cache after host current write:  [8, 256, 128]
mask:                              [256]
NPU output context:                [16, 128]
```

The same fixed artifact shape passed two prompts with different decode
positions:

```text
position=26 max_abs=0.015625 errors=0
position=22 max_abs=0.019531 errors=0
```

This is a stronger result than A0:

```text
A0 proved fixed-cache attention on synthetic one-head tensors.
D1.0 proves the same mechanism on real Qwen3 GQA tensors for all 16 Q heads.
```

The accepted D1.0 boundary has been promoted from experiment to production:

```text
iron/applications/new-mega/production
  main.py
  runner.py
  ops.py
  design.py
  fixed_attention.cc
```

D1.1a adds a second production boundary:

```text
stage: qkv-rope-present
input object:  q_raw + k_raw + v_raw + q_norm_weight + k_norm_weight + rope_lut
output object: q_rope + present_k + present_v
```

It passed two real prompts:

```text
position=26 max_abs=0.031250 errors=0
position=22 max_abs=0.062500 errors=0
```

The resource lesson matches D0:

```text
Packing the six logical inputs into one FIFO object kept the compute tile at
one input FIFO and one output FIFO. Do not split these into six separate
runtime fills unless a later topology proves the endpoint cost is safe.
```

D1.1b showed that qkv-rope-present can feed fixed-attention through host-owned
K/V writeback, but that was still two dispatches with a host step in the
middle.

D1.1b production result:

```text
stage: qkv-rope-attention
position=26 qkv_errors=0 context_max_abs=0.015625 context_errors=0
position=22 qkv_errors=0 context_max_abs=0.019531 context_errors=0
```

D1.1c fixes that structural gap:

```text
stage: qkv-rope-attention-present
single NPU dispatch for q_norm/k_norm/RoPE -> attention
attention combines past cache chunks with current present K/V
host writes present K/V after dispatch, not between qkv and attention
position=26 current_errors=0 context_errors=0 npu_time_us=4403.635
position=22 current_errors=0 context_errors=0 npu_time_us=4096.564
```

The important design lesson is that host-owned KV update is still viable for a
one-token single-dispatch decode path, as long as current present K/V is also
fed directly to the in-dispatch attention worker. The cache update belongs
after the token/layer dispatch, not between QKV and attention.

This changes the production rule:

```text
qkv-rope-attention-present is the base graph.
Append O projection, residual, RMSNorm, and MLP to that graph.
Use older independent-op stages only to isolate numerical boundaries.
```

The remaining boundary question is whether QKV GEMV projection can move into
the same production path without adding unsafe endpoints, then whether MLP can
be added while preserving the same staged checks.

D1.2 production result:

```text
stage: qkv-rope-attention-o
O projection: existing GEMV operator, M=1024, K=2048, columns=4
position=26 attn_out_errors=0 attn_residual_errors=0
position=22 attn_out_errors=0 attn_residual_errors=0
```

The lesson is practical: use the accepted standalone GEMV topology to prove
the O projection boundary before trying to fuse it into the phase-owned graph.
Do not create an L1-sized O-projection weight object.

D1.2c fused that boundary into the main graph:

```text
stage: qkv-rope-attention-o-fused
single NPU dispatch: qkv/rope -> attention -> O projection -> residual
position=26 current_errors=0 attn_out_errors=0 attn_residual_errors=0
position=22 current_errors=0 attn_out_errors=0 attn_residual_errors=0
preflight_compute_cores=4
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
```

The useful pattern was to pack the large O-projection weight and residual into
one runtime buffer, and to pack current K/V, attn_out, and residual output into
one output buffer. That kept the runtime ABI at four memrefs instead of
reintroducing the BO-metadata mismatch class of failures.

D1.3c extended the same main graph through the MLP:

```text
stage: qkv-rope-attention-o-mlp-fused
single NPU dispatch:
  qkv/rope -> attention -> O projection -> post-attention RMSNorm ->
  gate/up -> SiLU*up -> down projection -> layer residual
position=26 current_errors=0 attn_out_errors=0 ffn_hidden_errors=0
            ffn_out_errors=0 layer_residual_errors=0
position=19 current_errors=0 attn_out_errors=0 ffn_hidden_errors=0
            ffn_out_errors=0 layer_residual_errors=0
preflight_compute_cores=14
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
```

D1.3c later pulled the standalone GEMV row-shard pattern into the single graph
for O projection, gate/up, and down projection:

```text
context/xnorm broadcast through ObjectFifo
per-column row-sharded weights
per-column output shards
join back to a full vector for the next phase
```

The useful lesson was not "always add columns". O-only sharding compiled and
matched numerically, but slowed the graph because the join/copy overhead was
larger than the saved O-projection work. O plus gate/up sharding was large
enough to pay for the overhead:

```text
D1.3c baseline:        about 10.8-11.2ms on the default prompt
O-only row sharding:   11.886ms, rejected as a standalone speed change
O + gate/up sharding:   9.581ms, accepted with zero verification errors
O + gate/up + down:      8.586ms, accepted with zero verification errors
```

Two resource lessons mattered more than MLP math:

```text
1. Debug drains count as output DMA endpoints. A worker producing
   attn_residual_debug, final_residual, ffn_hidden, and ffn_hidden_debug
   exhausted output channels.
2. Gate and up weights must be one paired stream when xnorm is also an input.
   Separate gate_weight and up_weight FIFOs made a three-input tile and failed
   preflight.
```

D1.4 input projection pulled the same row-sharded GEMV pattern into the front
of the layer:

```text
diagnostic stage: input-qkv-rope-present
input RMSNorm -> Q/K/V projection -> Q/K norm+RoPE -> current Q/K/V
Q projection: 2 row shards
K projection: 2 row shards
V projection: 2 row shards
position=26 current_errors=0 q_raw_errors=0 k_raw_errors=0 v_raw_errors=0
position=19 current_errors=0 q_raw_errors=0 k_raw_errors=0 v_raw_errors=0
preflight_compute_cores=7
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
```

The important design change was to fuse Q/K norm+RoPE into the Q/K projection
shard workers. The first attempt projected Q/K/V into full raw tensors and fed
q_raw, k_raw, v_raw, and metadata to one final rope worker; aiecc rejected that
with an input DMA channel failure. Shard-local RoPE keeps every projection tile
at two inputs: `xnorm+metadata` and one weight stream.

The runtime lesson was separate: when a stage drains many independent output
shards, waiting on one unrelated drain is not enough. The Q/K drains returned
zeros until every independent drain used `wait=True`. A later performance graph
should join outputs into dependent streams or keep explicit waits for every
host-visible shard.

D1.4b removed the standalone input projection op and fused the same work into
the `qkv-rope-attention-o-mlp-fused` graph:

```text
stage: qkv-rope-attention-o-mlp-fused
hidden input -> input RMSNorm -> Q/K/V projection -> Q/K norm+RoPE ->
  attention -> O projection -> post-attention RMSNorm -> MLP -> layer residual
position=26 current_errors=0 attn_out_errors=0 ffn_hidden_errors=0
            ffn_out_errors=0 layer_residual_errors=0
position=19 current_errors=0 attn_out_errors=0 ffn_hidden_errors=0
            ffn_out_errors=0 layer_residual_errors=0
preflight_runtime_memrefs=5
preflight_compute_cores=25
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
```

The useful packing rule was learned again: do not add a sixth host memref for
Q/K/V weights. The runtime metadata exposes five HOST BO slots, so the first
buffer now packs input metadata and Q/K/V weights together and TAP offsets
select the subregions.

But this was still the wrong production shape. The graph used 25 compute cores
for one layer. That proves stage math can be wired together, but it does not
solve the megakernel problem because adding more layers would again consume
resources statically. Production has now been reset to a single phase-owned
path:

```text
stage: phase-owned
fixed lane Workers
packed layer/phase packet stream
Worker loop: for layer, for phase
one input FIFO and one output FIFO per lane
```

The first production recheck uses the full 28-layer phase count, not a
one-layer toy:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=11
hidden_size=1024
packet_elements=2048
preflight_compute_cores=8
preflight_max_compute_tile_inputs=1
preflight_max_compute_tile_outputs=1
phase_owned_errors=0

packet_elements=16896 compile/preflight also passed with
max_fifo_buffered_bytes=33792 and compute_cores=8.
```

The rule going forward is stricter:

```text
No production standalone op path.
No production static single-layer fusion path.
Real kernels must be inserted into the phase-owned lane topology.
```

D1.5a made the first real math insertion without changing the topology:

```text
phase 0 packet = hidden[1024] || norm_weight[1024]
phase 0 kernel = input RMSNorm checksum
num_lanes=8
num_layers=28
compute_cores=8
max_tile_inputs=1
max_tile_outputs=1
phase_owned_errors=0
```

The practical kernel lesson was to use `aie::invsqrt`, not `sqrtf`, for AIE
RMSNorm code.

D1.5b extended the same phase without adding FIFOs:

```text
phase 0 packet = hidden[1024] || norm_weight[1024] || q_weight_row[1024]
phase 0 kernel = input RMSNorm checksum + one Q row dot
num_lanes=8
num_layers=28
packet_elements=3072
compute_cores=8
max_tile_inputs=1
max_tile_outputs=1
phase_owned_errors=0
```

D1.5c increased phase-0 compute density again without changing the topology:

```text
phase 0 packet = hidden[1024] || norm_weight[1024] || q_weight_block[4, 1024]
phase 0 kernel = input RMSNorm checksum + four Q row dot accumulations
num_lanes=8
num_layers=28
q_rows_per_packet=4
packet_elements=6144
compute_cores=8
max_tile_inputs=1
max_tile_outputs=1
phase_owned_errors=0
```

The practical architecture lesson is that the next true-fusion steps should
keep adding real work inside the phase-owned loop and only widen the topology
after a measured phase proves that one packet stream per lane is the bottleneck.

D1.5d corrected the missing lane-communication structure:

```text
shared packet = hidden[1024] || norm_weight[1024]
lane packet = q_weight_block[4, 1024]
fabric = two groups of four lanes
group input = shared stream duplicated once per group
group output = local join of lane Q shards
num_lanes=8
fabric_group_size=4
compute_cores=8
max_tile_inputs=2
max_tile_outputs=1
phase_owned_errors=0
```

The failed intermediate attempt was useful: one 8-way broadcast/join fabric
failed during `resolve_program()` placement, while a 4-lane version compiled.
That established a concrete rule for this production path: use 4-lane local
communication groups unless a new placement experiment proves a wider group.

D1.5e fixed another structural gap: scalar state was not activation state.

```text
hidden_state[1024] BF16 Buffer per lane Worker
layer 0 initializes hidden_state from the shared stream
phase 0 reads hidden_state
next_layer_token writes hidden_state for the next layer
state[0] remains a checksum only
num_lanes=8
fabric_group_size=4
compute_cores=8
max_tile_inputs=2
max_tile_outputs=1
phase_owned_errors=0
```

The rule is now explicit: debug checksums are allowed, but they do not count as
phase/layer activation transfer. Every real phase must either pass data through
ObjectFifo fabric or write an activation-sized tile-local buffer that the next
phase actually reads.

D1.5f replaced patterned packets with real Qwen3 data for the proven Q shard:

```text
real hidden initializer from Qwen3 decode
real input_layernorm.weight for each layer
real q_proj row shards for each lane
real next-layer hidden written through next_layer_token packets
phase_owned_errors=0
qwen3_q_shard_max_abs=0.250000
qwen3_q_shard_errors=0 at abs_tol=0.5
```

The local BF16 boundary reference is now the strict gate for the AIE kernel and
the PyTorch Qwen3 q_proj reference is the semantic tolerance gate. Do not
interpret a bounded PyTorch BF16 linear difference as a dataflow bug unless the
local boundary reference also fails.

D1.5g replaced another synthetic phase with real Qwen3 gate/up computation:

```text
gate_up packet = attn_residual + post_norm_weight + gate/up row shards
gate_up kernel = post RMSNorm + gate_proj/up_proj dot rows
output object = Q shard + gate/up shard
phase_owned_errors=0
qwen3_phase_output_max_abs=0.250000
qwen3_phase_output_errors=0 at abs_tol=0.5
```

The important fabric lesson is that one lane output object can stay acquired
across multiple phases. That lets the graph add real phase outputs without
adding another lane output FIFO and another join fabric immediately.

D1.5h replaced the down/residual synthetic phase with real Qwen3 computation:

```text
down_proj packet = ffn_hidden + attention residual shard + down_proj row shard
down kernel = dot(ffn_hidden, down_proj row) + residual shard
output object = Q shard + gate/up shard + residual shard
phase_owned_errors=0
qwen3_phase_output_max_abs=0.000488
qwen3_phase_output_errors=0 at abs_tol=0.5
```

The debug lesson was sharper than the implementation: a PyTorch `F.linear`
reference produced two residual-shard mismatches greater than 0.5 even while
the NPU matched the packet-level BF16 reference. The cause was BF16 reduction
semantics for the 3072-wide down dot, not dataflow. For row-shard kernels, the
strict reference must state the contract exactly: BF16 inputs, scalar float32
accumulation, and BF16 output. Use the full PyTorch model path as a semantic
tolerance check, not as the first-line kernel gate.

D1.5i replaced the O projection placeholder with real Qwen3 computation:

```text
o_proj packet = attention_context[2048] + input residual shard + o_proj row shard
o kernel = dot(attention_context, o_proj row) + residual shard
output object = Q shard + attention residual shard + gate/up shard + layer residual shard
phase_owned_errors=0
qwen3_phase_output_max_abs=0.000488
qwen3_phase_output_errors=0
```

The useful debug finding was a shape-modeling bug caught before NPU execution:
the attention context is 2048 wide in Qwen3-0.6B, not 1024. `hidden_size` owns
residual/norm/down output space; `attention_size = num_attention_heads *
head_dim` owns Q/O attention space. Treat these as separate dimensions in
packet manifests and external-kernel ABI declarations.

## What To Do Next

The next useful `new-mega` work should continue the proof ladder, not jump to a
full rewrite.

Recommended order:

```text
1. Expand Q/K/V shard coverage so attention can consume NPU-produced vectors.
2. Keep broadcast/join fabric grouped at four lanes unless a measured placement
   experiment proves wider groups are legal.
3. Run preflight and full aiecc before executing on NPU.
4. Verify every inserted phase against PyTorch/reference buffers.
5. Measure whether lane Workers are compute-bound or DMA-bound before adding
   more columns.
6. Only then decide whether a GEMM-style L2 topology is worth pulling into the
   phase-owned graph.
```

Acceptance for the first new-mega experiment:

```text
same xclbin
multiple decode positions
no recompilation
attention or F.linear match, depending on the experiment
preflight passes
full aiecc passes
runtime does not hang
documented in how-to-debug if it fails
```

If this first proof fails, the full-array phase-based plan should be rejected
or rewritten before touching the Qwen3 layer graph.
