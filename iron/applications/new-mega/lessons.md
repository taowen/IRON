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

The production phase-owned path now proves the first-head fixed-cache
score/softmax/PV boundary inside the reusable Worker loop. It kept endpoint
pressure flat by using four more lane-local packet phases instead of new FIFOs:

```text
phase_packets_per_layer=13
packet_elements=16576
preflight_compute_cores=8
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=1
context_max_abs=0.312500 at abs_tol=0.5
```

For attention changes, always print segment-wise diffs. In the accepted run,
the total `max_abs=0.312500` came only from the context segment; Q/K/V, RoPE,
O residual, gate/up, and residual segments stayed near exact BF16 row-shard
error.

The next production step connected that context to O projection for head 0.
The important diagnostic was to poison the still-present host copy:

```text
O packet host_attention_context[0:128] = 123.0
poison_o_host_head0_max_abs=0.007812
poison_o_host_head0_errors_gt_0_5=0
```

That proves the downstream O phase reads the lane-local context segment from
the previous attention phase. Without this poison test, a passing O projection
could still be a false pass caused by a correct host-packed intermediate.

The following step generalized the same handoff to lane-mapped heads 0..7:

```text
context_head_index = lane_id
kv_head = context_head_index // 2
poison_o_host_lane_heads_max_abs=0.017578
poison_o_host_lane_heads_errors_gt_0_5=0
```

The lesson is that GQA mapping must be part of the packet manifest, not an
implicit reference assumption. The poison test should poison only the host
slice that the lane claims to replace; poisoning unrelated heads would test a
different dependency.

The next step computed two heads per lane and covered all 16 attention heads:

```text
primary head = lane_id
secondary head = lane_id + num_lanes
phase_packets_per_layer=17
npu_context_heads_per_layer=16
poison_o_host_two_lane_heads_max_abs=0.017578
poison_o_host_two_lane_heads_errors_gt_0_5=0
```

This separated two problems that had been easy to conflate:

```text
head production: solved for all 16 heads in the current packet-stream design
O visibility: still unsolved, because each lane sees only its own two heads
```

The next real architecture problem is context gather/broadcast or partial-O
reduce, not another per-head attention kernel.

After wiring O into gate/up for local residual rows, the same poison technique
proved the O->MLP handoff:

```text
gate_up residual_row_base = lane_id * q_rows_per_packet
host gate_up attn_residual[local rows] = 123.0
poison_gate_host_local_residual_max_abs=0.017578
poison_gate_host_local_residual_errors_gt_0_5=0
```

This is progress, but it also sharpens the remaining problem: dense RMSNorm and
gate/up rows need the full residual vector. Local-row handoff is not enough to
remove host residual without gather/broadcast or a different projection
partition.

The same rule now applies to the gate/up->down handoff:

```text
down ffn_row_base = lane_id * q_rows_per_packet
host down ffn_hidden[local rows] = 123.0
poison_down_host_local_ffn_max_abs=0.437500
poison_down_host_local_ffn_errors_gt_0_5=0
```

This proves down consumes the local FFN hidden rows produced by gate/up. It is
not the end of the problem: every down row is dense over all 3072 FFN hidden
values, so the other 3068 values are still host-fed. The remaining architecture
work is a full-vector gather/broadcast or a partial projection reduce, not more
local poison tests.

### 8. Keep Independent References For The Phase-Owned Body

The production phase-owned path now has two host references:

```text
phase_owned_reference:
  follows the packed packet stream and external kernel ABI

qwen3_reference:
  fills the same output slots from real Qwen3 tensors
```

This caught a real bug after adding first-head Q/K RMSNorm+RoPE shards. The NPU
matched `qwen3_reference`, but `phase_owned_reference` failed because its
`gate_base` offset still ignored the inserted `q_rope` and `k_rope` output
segments. The fix was reference layout alignment, not a kernel rewrite.

Rule:

```text
When adding an output segment, update AIE output_base, packet-builder
qwen3_reference, phase_owned_reference, printout, README dimensions, and debug
slot mapping in the same change.
```

### 9. Performance Estimates Need Real Traffic And Measured Overheads

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

D1.5j replaced two attention chunk placeholders with real K/V projection rows:

```text
attention_chunk_0 packet = k_proj row shard
attention_chunk_1 packet = v_proj row shard
projection kernel = input RMSNorm(hidden_state) dot row shard
output object = Q shard + K shard + V shard + attention residual shard + gate/up shard + layer residual shard
phase_owned_errors=0
qwen3_phase_output_max_abs=0.003906
qwen3_phase_output_errors=0
```

The topology lesson is that the shared broadcast packet does not have to be
released immediately after Q. It can be held through adjacent Q/K/V phases so
all three projections reuse the same input RMSNorm weight without adding FIFO
endpoints. The tradeoff is temporal: every lane in the broadcast group must
advance through those phases in the same order.

D1.5t changed that rule after adding a real reduce return path:

```text
O partial projection reduce:
  each lane computes same-fabric-group O partials from its two context heads
  one reducer Worker per 4-lane group sums those partials
  reduced rows return to the owner lanes

accepted resource shape:
  num_lanes=8
  reducer_workers=2
  compute_cores=10
  max_tile_inputs=2
  max_tile_outputs=2
  phase_owned_errors=0 at phase abs_tol=1.0
  qwen3_phase_output_errors=0 at abs_tol=0.5
```

The first implementation failed before running any math:

```text
error: 'aie.tile' op number of input DMA channel exceeded!
%tile_0_2 = aie.tile(0, 2)
```

MLIR inspection showed the lane tile had three independent inputs:

```text
shared hidden/norm broadcast
lane packet stream
O reduced return stream
```

The fix was to remove the shared ObjectFifo from production. Phase 0 now packs
hidden and input-norm weight into the lane packet, and the Worker caches the
norm weight in tile-local memory for K/V phases. The durable lesson is that a
reduce return path consumes one of the lane tile's scarce input channels. When
adding reduce/gather paths, first move low-bandwidth metadata into existing
packets or tile-local state.

The numeric lesson was also concrete. The NPU matched the Qwen semantic
reference at `abs_tol=0.5`, but one packet-level phase reference slot differed
by exactly one BF16 ULP around value 182. Host-only comparison reproduced the
same slot, so the correct action was to adjust the phase-boundary tolerance to
one BF16 ULP while keeping the Qwen semantic tolerance unchanged.

D1.5u extended O partial reduce across both fabric groups:

```text
lane Workers compute partials for all 32 materialized O rows
source reducers sum producer lanes inside each 4-lane group
target reducers sum the two source groups for each target group
O no longer needs host-packed context contribution for those rows

accepted resource shape:
  num_lanes=8
  source_reducers=2
  target_reducers=2
  compute_cores=12
  max_tile_inputs=2
  max_tile_outputs=2
  phase_owned_errors=0
  qwen3_phase_output_errors=0
```

The first cross-group version failed in routing, not in resource allocation:

```text
resource allocation pipeline completed successfully
routing pipeline failed: Unable to find a legal routing
```

The cause was the intermediate full-vector split:

```text
source reducer -> source_reduced[32] -> one memtile split -> two target reducers
```

Generated MLIR showed the cross-group exchange concentrated through
`mem_tile_2_1`. The fix was to avoid that split entirely:

```text
source reducer -> target0 half
source reducer -> target1 half
```

The durable routing rule is that reducer Workers should emit target-specific
outputs directly when the reducer already has the source partials. A
"reduce full vector then split through a shared memtile" can be legal as a
dataflow graph but still fail NoC routing.

D1.5v made the next residual handoff wider without adding a new lane input
stream:

```text
O finalize writes the full target fabric-group residual rows into each lane's
lane_output object
gate_up reads those 16 rows from lane_output before post-attention RMSNorm
num_lanes=8
fabric_group_size=4
attention_output_values_per_lane=16
compute_cores=12
max_tile_inputs=2
max_tile_outputs=2
phase_owned_errors=0
qwen3_phase_output_errors=0
```

This does not remove the full residual-vector problem for gate/up. Dense
post-attention RMSNorm and gate/up projection still need all 1024 residual
values; this step only replaces the rows that the current O reduce fabric
materializes. The useful lesson is different: when a phase already returns a
group result through `lane_output`, the next phase can consume a wider slice of
that same lane-local object without adding another ObjectFIFO endpoint.

The debug lesson was an ABI/layout lesson, not a math lesson. Expanding the O
packet residual header from 4 rows to 16 rows required updating all of these at
once:

```text
packet builder residual payload
O partial C++ weight-block offset
phase_owned_reference producer weight-block offset
qwen3_reference output slot index
gate_up packet residual row base
gate_up external kernel residual_group_size argument
Kernel declaration arity in phase_owned_stages.py
```

Two concrete checks caught the drift:

```text
resolve_program() rejected a widened q_shard Kernel declaration before aiecc:
  Kernel 'new_mega_phase0_q_shard_bf16' expects 12 argument(s), but 10 were provided

manual packet-layout audit found O partial still reading weights at:
  packet + 2 + q_rows_per_packet
instead of:
  packet + 2 + fabric_group_size * q_rows_per_packet
```

For production phase-owned code, packet header-size changes are ABI changes.
Treat them like C struct layout changes: update the Python Kernel declaration,
C++ signature, packet builder, packet-level reference, semantic reference, and
README dimensions in one patch.

D1.5w removed the host residual dependency from gate/up:

```text
O projection is split into 32 row chunks
each chunk covers 32 hidden rows
the same source/target reducer Workers are reused for every chunk
each lane_output stores the full 1024-row attention residual
gate_up reads all 1024 rows from lane_output

accepted resource shape:
  num_lanes=8
  num_layers=28
  phase_packets_per_layer=48
  compute_cores=12
  max_tile_inputs=2
  max_tile_outputs=2
  max_dma_tasks_per_fifo=1
  phase_owned_errors=0
  qwen3_phase_output_errors=0
```

The architecture lesson is that "full residual visibility" does not require a
single giant O packet. A 1024-row O weight block would not fit the packet/L1
budget, but 32 chunks of 32 rows do fit. The graph pays with time and host
input traffic, while keeping the number of Workers, ObjectFIFOs, endpoints, and
BD tasks bounded.

The debug lesson was about tooling scale. The first run looked like a hang, but
the process was spending minutes in the host packet-level reference: the O
chunk change had inflated a nested Python scalar loop into hundreds of
millions of multiply-adds. The fix was to vectorize the reference at the same
semantic boundary:

```text
for each producer lane:
  group_partials += weight_block @ producer_context
```

After that, NPU execution produced real diagnostics. The raw Qwen semantic
comparison had 16 errors with `max_abs=1.0`, and segment stats showed every one
was in `attention_residual`; `gate_up` and `down_residual` had zero segment
errors. Since the packet-level reference matched exactly under the existing
one-BF16-ULP phase tolerance, the correct interpretation was O accumulation
order, not a gate/up dataflow bug. The semantic checker now allows one BF16 ULP
for `attention_residual` while keeping other segments at the normal tolerance.

This step is not a performance win yet:

```text
D1.5u cross-fabric O for 32 rows: about 316 ms
D1.5w full chunked O for 1024 rows: about 580 ms
```

It is a dataflow correctness step. The remaining large host-fed activation is
the FFN hidden vector used by `down_proj`.

### 13. Program Memory Is Now A First-Class Megakernel Resource

D1.5x reused the existing O reducer fabric for the first FFN hidden group:
`gate_up` produces a sparse 32-row FFN partial vector, reducers sum and
broadcast it, and `down_proj` consumes `ffn_reduced[0:32]` instead of the host
packet for those rows.

The first version failed after ELF link, during CDO generation:

```text
[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory
```

The useful diagnostic was not another placement attempt. `llvm-size` showed
the lane Worker `.text` had crossed the AIE program memory budget:

```text
standalone ffn_partial kernel: 16880 bytes
fused into gate_up only:       16640 bytes
accepted slim lane core:       16080 bytes
```

The accepted fix fused the tiny partial producer into `gate_up` and deleted
the old `down_proj` local FFN fallback that became unreachable once the
reduced 32-row group was available.

The numeric failure after that was also structural: `gate_up` accidentally
used `packet[0]` for two meanings, residual replacement base and FFN row base.
Segment stats localized the error to `down_residual`, and the fix was to put
the FFN row base in a separate metadata slot at the end of the gate packet.

The broader lesson is that phase-owned megakernels are now constrained by at
least four budgets at once:

```text
tile input/output DMA channels
ObjectFIFO/L1 object size
BD/routing resources
AIE program memory
```

Adding a small phase can fail any one of those. Always identify which budget
failed before changing topology or math.

## What To Do Next

The next useful `new-mega` work should continue the proof ladder, not jump to a
full rewrite.

Recommended order:

```text
1. Extend the FFN handoff past the first 32 rows by choosing between bounded
   FFN gather/broadcast and down partial projection reduce.
2. Run preflight and full aiecc before executing on NPU.
3. Verify every inserted phase against both packet-level and Qwen semantic
   references.
4. Measure whether lane Workers or reducer Workers are compute-bound or
   DMA-bound before adding more columns.
5. Keep old standalone/static-op paths out of production.
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
