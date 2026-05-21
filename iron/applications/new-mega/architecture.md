<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# New Qwen3 Megakernel Architecture V0

This is a mechanism-grounded architecture proposal for the next Qwen3-0.6B
decode megakernel on IRON/XDNA.

It is not a claim that the final implementation is already complete. The
accepted experiments prove enough mechanisms to choose a direction; D1 and
later experiments still have to prove real Qwen3 layer math and performance.

## Status

Current accepted baseline:

```text
implementation: iron/applications/qwen3_0_6b/persistent
mode: single-token decode body
layer_chunk_size: 28
position handling: exact-position precompiled artifacts
final norm / LM head: CPU
verified prompts: default and Fibonacci token checks
```

Current architecture target:

```text
one fixed decode artifact per max_seq_len bucket
single dispatch per generated token
host-owned KV cache update between dispatches
fixed max-cache attention read with runtime mask
static phase ownership inside the NPU graph
packed lane-local phase streams
real Qwen3 layer loop after D1/D2 acceptance
```

Current production code:

```text
iron/applications/new-mega/production
  stage: phase-owned
  status: resource skeleton, not Qwen3 math
  boundaries:
    fixed lane Workers
    packed lane-local layer/phase streams
    Worker loop over layers and phases inside one dispatch
    one input FIFO and one output FIFO per lane
```

Production rule after correcting the D1.4b direction:

```text
Do not grow the main path by adding another independent op dispatch or another
static single-layer graph. Production has one path: phase-owned lane Workers
consume packed phase streams. Real Qwen3 kernels must be inserted into that
path.
```

The main shift is:

```text
from:
  exact-position artifact + statically specialized position behavior

to:
  fixed-shape artifact + runtime data values for live-position behavior
```

## Goals

Primary goals:

```text
remove per-position compile/artifact selection from the decode hot path
reduce host dispatches to one decode-body dispatch per generated token
keep layer-internal activations on NPU as much as practical
bound ObjectFIFO endpoint, BD, and L1 pressure before full integration
preserve token correctness against the current PyTorch/reference path
beat the accepted persistent baseline before moving CPU tail work onto NPU
```

Secondary goals:

```text
make failure modes diagnosable with preflight checks
make weight/cache/packet layout explicit in manifests
support max_seq_len bucket variants if fixed full-cache attention becomes too wasteful
keep the first architecture small enough to debug one layer at a time
```

Non-goals for V0:

```text
prefill optimization
sampling on NPU
final norm / LM head on NPU
runtime mutation of DMA BDs with control packets
dynamic placement or dynamic ObjectFIFO graph construction
cross-layer attention/MLP pipeline parallelism
```

Prefill is not required for the first target. A correctness-first path can fill
KV cache by repeated decode-style invocations. A fast prefill path can be added
after the decode architecture is proven.

## Proven Mechanisms

The architecture uses only mechanisms that have been directly tested in this
repository.

### A0: Fixed-Chunk Attention Read

Accepted result:

```text
one artifact ran positions 0, 26, 63, 64, 127, 200, 255
max_seq_len=256
head_dim=128
chunk_size=64
position encoded only by runtime mask values
```

Architecture decision:

```text
attention reads the fixed max-cache shape every dispatch
invalid future positions contribute zero through mask
online softmax handles chunked processing without materializing full scores
no per-position TAP is needed for the read side
```

### A0B: Host-Side KV Writeback

Accepted result:

```text
one artifact ran 70 sequential decode steps
the run crossed the 63/64 chunk boundary
NPU produced fixed present_k/present_v outputs
host copied present K/V into cache[position]
next invocation read the updated cache through the same fixed TAP
```

Architecture decision:

```text
NPU never writes dynamic KV-cache offsets
NPU drains only fixed-shape present K/V tensors
host owns cache[position] memcpy between decode dispatches
```

### B1: Real-Shape GEMV Scaling

Accepted result:

```text
all Qwen3 projection shapes compiled and matched reference at 1/2/4/8 columns
scaling was non-monotonic
uniform 4 columns was best among uniform policies
per-shape best was better than any single fixed column count
```

Architecture decision:

```text
do not assume "more columns is always faster"
choose column policy per phase after measurement
start D1 with a conservative known-good lane count, then scale by evidence
```

### C1/C2: Phase Protocol Boundaries

Accepted C1 result:

```text
one Worker executed two FIFO-guarded phases in one dispatch
phase order and output matched reference
```

Partially accepted C2 result:

```text
inactive FIFO dependencies can be removed only by compiling a different graph
same-artifact dynamic phase skip was not proven
```

Architecture decision:

```text
phase order is fixed in the artifact
do not rely on Worker-side if statements to skip unfilled FIFOs
do not use dummy FIFO tokens as the scalable mechanism
```

### D0: Static Phase Ownership Skeleton

Accepted result:

```text
8 lane Workers
1 packed input FIFO per lane
1 padded output FIFO per lane
11 phase packets per lane
packet size = 33792 bytes
preflight_compute_cores = 8
preflight_max_compute_tile_inputs = 1
preflight_max_compute_tile_outputs = 1
preflight_max_fifo_buffered_bytes = 33792
preflight_max_dma_tasks_per_fifo = 1
NPU output matched reference
```

Architecture decision:

```text
use packed lane-local streams
avoid one FIFO per phase input
pad small DMA-visible objects to 16 bytes
keep endpoint and BD pressure bounded before adding real math
```

## Hardware Model

V0 follows the IRON/XDNA model directly:

```text
host DDR / L3
  -> shim DMA
  -> memory tile / ObjectFIFO stream
  -> compute tile L1
  -> AIE Worker + external kernels
  -> ObjectFIFO / Buffer handoff
  -> final decode-body output drain
```

The graph topology is static. The runtime values are dynamic only as tensor
data:

```text
dynamic as data:
  input hidden
  full KV cache contents
  mask values
  RoPE cos/sin values
  present K/V outputs

static in artifact:
  ObjectFIFO graph
  Worker placement
  external kernel ABI
  packet object shape
  TAP shapes/strides for max-cache and phase streams
```

This is not dynamic hardware reconfiguration. It is:

```text
static graph + reusable Worker loop + streamed layer payloads
```

## Decode Boundary Ownership

V0 keeps the model boundary explicit.

| Boundary | Owner In V0 | Notes |
| --- | --- | --- |
| Tokenizer / prompt | CPU | Existing path. |
| Embedding lookup | CPU initially | Can stay CPU until decode body is faster. |
| Input hidden to NPU | Runtime fill | One fixed hidden vector per decode token. |
| Packed weights | Runtime fill | Prepacked on disk, streamed in fixed order. |
| KV cache read | Runtime fill | Fixed max-cache shape, masked by runtime values. |
| Current K/V writeback | CPU host memcpy | A0B-proven; no dynamic NPU write offset. |
| Layer-internal hidden | NPU ObjectFIFO / Buffer | D1 must prove this for one real layer. |
| Decode body output hidden | Runtime drain | Fixed hidden vector. |
| Final norm / LM head | CPU initially | Move only after decode body bottleneck is solved. |
| Argmax / sampling | CPU initially | Existing path. |

The NPU decode body is a pure function for one token:

```text
(hidden_in, full_kv_cache, mask, rope, packed_weights)
  -> (hidden_out, present_k, present_v)
```

The host updates persistent state:

```text
kv_cache[layer, head, position, :] = present_kv[layer, head, :]
position += 1
mask[position] = 1
prepare rope[position]
```

## Graph Shape

V0 uses fixed Worker roles. Workers do not change identity at runtime.

Conceptual topology:

```text
hidden input
  -> norm/projection lanes
  -> attention lanes
  -> O/residual lanes
  -> MLP lanes
  -> residual/output lanes
```

The first D1 implementation should not attempt to use every tile. It should
start with the smallest topology that preserves the D0 resource shape:

```text
few packed input streams
few output streams
one or two input FIFOs per compute tile
one output FIFO per compute tile where possible
explicit preflight before NPU run
```

The full decode loop is conceptually:

```text
for layer in 0..27:
  input_rmsnorm
  qkv_projection
  rope
  fixed_chunk_attention_read
  online_softmax_and_pv
  o_projection_and_residual
  post_attention_rmsnorm
  gate_up_projection
  silu_gate_mul
  down_projection_and_residual
```

Layer `n + 1` cannot start attention until layer `n` produces its final
residual hidden. There is no cross-layer attention/MLP pipeline requirement in
V0.

## Layer State Machine

The intended final form is not 28 statically copied layer graphs. It is one
static graph that reuses a fixed set of Workers across layers:

```text
worker role is fixed
ObjectFIFO topology is fixed
external kernel ABI is fixed

for layer in 0..27:
  acquire next layer packet(s)
  run fixed phase sequence
  release outputs to next phase
```

This is feasible in principle because the Worker program can loop. The hard
constraints are:

```text
ObjectFIFO object type must be fixed
packet shape must be fixed or padded
external kernel ABI must be fixed
TAP sequence must be static
phase acquire/release counts must be balanced
```

D0 proves the resource skeleton for this style. It does not prove the real
Qwen3 layer state machine yet. D1/D2 must prove that.

## Packet And Manifest Design

V0 should avoid ad hoc offsets. Every packed artifact needs a manifest.

Minimum manifest fields:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "dtype": "bf16",
  "max_seq_len": 256,
  "layers": [
    {
      "id": 0,
      "input_rmsnorm": {"offset": 0, "shape": [1024], "dtype": "bf16"},
      "q_proj": {"offset": 0, "shape": [2048, 1024], "dtype": "bf16"},
      "k_proj": {"offset": 0, "shape": [1024, 1024], "dtype": "bf16"},
      "v_proj": {"offset": 0, "shape": [1024, 1024], "dtype": "bf16"},
      "o_proj": {"offset": 0, "shape": [1024, 2048], "dtype": "bf16"},
      "gate_proj": {"offset": 0, "shape": [3072, 1024], "dtype": "bf16"},
      "up_proj": {"offset": 0, "shape": [3072, 1024], "dtype": "bf16"},
      "down_proj": {"offset": 0, "shape": [1024, 3072], "dtype": "bf16"}
    }
  ]
}
```

The real manifest should also include:

```text
alignment
packet_id
phase_id
lane_id
column policy
kernel symbol
expected object bytes
tap coverage metadata
```

Preflight must reject:

```text
offset + nbytes out of bounds
misaligned offsets
shape/dtype mismatch against kernel ABI
ObjectFIFO object bytes > L1 budget
ObjectFIFO object bytes not 16-byte aligned
too many input/output endpoints on one compute tile
too many DMA tasks on one FIFO
non-advancing acquire patterns
```

## Attention Design

Attention uses the A0/A0B strategy.

For each layer/head group:

```text
read current Q from the projection/RoPE phase
read full fixed K/V cache tensor for max_seq_len
read mask tensor with live positions marked valid
process cache in fixed chunks
maintain online softmax state
produce context
drain fixed present_k/present_v for host writeback
```

The attention Worker must not depend on dynamic TAP offsets. Position affects
attention only through runtime data:

```text
mask values
RoPE cos/sin values
host-updated cache contents
```

Initial chunking policy:

```text
head_dim = 128
chunk_size = 64
max_seq_len = 256 for first integration bucket
```

This can support larger buckets by increasing the fixed number of chunks, at
the cost of more masked work for short contexts. Bucket variants are allowed
only after each bucket passes the same correctness and preflight gates.

## MLP And GEMV Design

The first D1 design should use the measured B1 information:

```text
do not scale every projection to 8 columns by default
start with known-good conservative columns
measure per projection
use per-shape policies only when the added stream complexity is justified
```

The MLP phase sequence is:

```text
post_attention_rmsnorm(hidden)
gate = gate_proj(normed)
up = up_proj(normed)
activated = silu(gate) * up
down = down_proj(activated)
hidden_next = hidden_residual + down
```

The first implementation can keep some reductions simple if it preserves the
resource limits. More aggressive L2 split/forward/join GEMV topology is a later
optimization, not a prerequisite for D1.

Column-parallel GEMV/GEMM does not conflict with static phase ownership.
Static phase ownership fixes the order and state boundary between QKV/RoPE,
attention, O projection, residual/norm, gate/up, down projection, and layer
residual. Column parallelism is an implementation choice inside those phases.

The standalone GEMV/GEMM operators cannot be copied into the production graph
unchanged. Standalone GEMV gets its input vector from host-side fills for each
column and drains row shards directly back to host. In the megakernel the vector
usually comes from an upstream Worker through ObjectFifo, so the useful pattern
to borrow is:

```text
phase input vector
  -> graph-internal broadcast/forward to column workers
  -> row-sharded weight streams
  -> row-sharded partial outputs
  -> join/concat for the next phase
```

The first column-scaling targets should be the projection phases that already
dominate D1.3c runtime: O projection, gate/up, and down projection. Each target
must keep the accepted endpoint rule from D1.3c: no compute tile should grow
past two input FIFOs or two output FIFOs unless a compile/preflight check proves
that the exact graph is legal.

Production D1.3c/D1.4a now uses this pattern for O projection, gate/up, down
projection, and the front Q/K/V projections:

```text
attention context
  -> broadcast to 2 O projection workers
  -> row-sharded O outputs
  -> join to full attn_out

post-attention xnorm
  -> broadcast to 2 gate/up workers
  -> row-sharded ffn_hidden
  -> join to full ffn_hidden

ffn_hidden
  -> broadcast to 2 down projection workers
  -> row-sharded ffn_out
  -> join to full ffn_out

input xnorm + q/k metadata
  -> broadcast to 2 Q workers, 2 K workers, 2 V workers
  -> row-sharded Q/K/V outputs
  -> Q/K norm+RoPE inside each Q/K shard worker
```

The measured result is important: splitting only O projection was correct but
slower, because the projection is not large enough to pay for the extra join
and stream overhead. Splitting O plus gate/up reduced the D1.3c default prompt
from about 10.8-11.2ms to 9.58ms; adding down projection sharding reduced it
to 8.59ms while preserving zero verification errors.

## Runtime Token Loop

Host loop:

```text
load or create packed weights once
allocate full KV cache buffers
for each generated token:
  prepare hidden_in
  prepare mask for current position
  prepare rope cos/sin for current position
  dispatch fixed NPU decode body
  drain hidden_out and present K/V
  host memcpy present K/V into kv_cache[position]
  run CPU final norm / LM head / argmax
  append next token
```

There is one NPU decode-body dispatch per token in V0. The CPU tail remains
outside the architecture until the decode body is faster than the accepted
persistent baseline.

## Correctness Gates

Do not validate only final text. D1 and D2 need staged checks.

D1 single-layer checks:

```text
after input RMSNorm
after Q/K/V projection
after RoPE
after attention context
after O projection + residual
after post-attention RMSNorm
after gate/up
after SiLU gate multiply
after down projection + residual
present K/V output
host-updated KV cache row
```

D2 n-layer checks:

```text
layer 0 hidden
layer 1 hidden
layer 2 hidden
selected middle layer hidden
layer 27 hidden
final token match on multiple prompts
multi-token run across chunk boundaries
```

Every check should report:

```text
max_abs_error
max_rel_error
cosine_similarity
num_nan
num_inf
error_count under tolerance
```

## Performance Gates

The architecture is accepted only if measured token time improves.

Measure:

```text
NPU decode-body time
host wall time per token
CPU final norm / LM head time
host KV writeback time
runtime fill/drain time buckets when available
per-phase NPU trace if available
```

Compare against:

```text
current persistent generate --fast-generate baseline
same prompt
same number of generated tokens
same max_seq_len bucket
warmup separated from measured iterations
```

Expected first-order wins:

```text
no hot-loop compilation
no per-position artifact switching inside the token loop
fewer host/NPU round trips inside the decode body
less DDR traffic for layer-internal activations
bounded endpoint/BD pressure from packed streams
```

Expected costs:

```text
fixed max-cache attention reads masked future positions
packet padding wastes some bandwidth
host writes present K/V between dispatches
CPU final norm / LM head still contribute wall time
```

## Rollout Plan

### D1: Single Real Qwen3 Layer

D1.0 accepted:

```text
real Qwen3 layer-0 Q/K/V tensors
host current K/V writeback into full fixed cache
NPU fixed-cache attention read for 16 Q heads
runtime mask controls live position
context matched reference on positions 26 and 22
```

D1.1a accepted:

```text
production stage: qkv-rope-present
real Qwen3 raw Q/K/V tensors
NPU q_norm/k_norm + RoPE
fixed present K/V output
context-free boundary, before QKV GEMV projection
matched reference on positions 26 and 22
```

D1.1b accepted:

```text
production stage: qkv-rope-attention
NPU q_norm/k_norm + RoPE produced Q and fixed present K/V
host wrote present K/V into the fixed cache
NPU fixed-cache attention consumed that cache and Q
context matched reference on positions 26 and 22
```

D1.1c accepted:

```text
production stage: qkv-rope-attention-present
single NPU dispatch for Q/K norm+RoPE -> attention
attention reads past cache chunks and current present K/V
host writes present K/V to cache only after dispatch, for the next token
context matched reference on positions 26 and 22
```

D1.2 accepted:

```text
production stage: qkv-rope-attention-o
attention context fed existing GEMV O projection, M=1024, K=2048, columns=4
host residual add produced attention residual [1024]
attention residual matched reference on positions 26 and 22
```

D1.2c accepted:

```text
production stage: qkv-rope-attention-o-fused
single NPU dispatch for Q/K norm+RoPE -> attention -> O projection -> residual
current present K/V is drained for host writeback after dispatch
attention residual matched reference on positions 26 and 22
```

D1.3c accepted:

```text
production stage: qkv-rope-attention-o-mlp-fused
single NPU dispatch for Q/K norm+RoPE -> attention -> O projection ->
post-attention RMSNorm -> gate/up -> SiLU*up -> down projection -> layer residual
accepted resource shape after row-sharded O/gate/up/down: 14 compute cores,
max tile inputs=2, max tile outputs=2
layer residual matched reference on positions 26 and 19
```

D1.4a accepted:

```text
diagnostic stage: input-qkv-rope-present
input RMSNorm + row-sharded Q/K/V projection + Q/K norm+RoPE
Q/K norm+RoPE is fused into the Q/K projection shard workers to avoid a
4-input final rope worker
accepted resource shape: 7 compute cores, 3 runtime memrefs,
max tile inputs=2, max tile outputs=2
current Q/K/V and raw Q/K/V debug tensors matched reference on positions 26
and 19
```

D1.4b accepted:

```text
removed production stage: qkv-rope-attention-o-mlp-fused
single NPU dispatch from hidden input through layer residual
input RMSNorm + row-sharded Q/K/V projection + Q/K norm+RoPE are fused into
the main attention/O/MLP graph
standalone input-qkv-rope-present op removed from production CLI
runtime ABI kept at 5 memrefs by packing input metadata and Q/K/V weights into
one first buffer
accepted resource shape: 25 compute cores, 21 DMA tasks,
max tile inputs=2, max tile outputs=2
layer residual matched reference on positions 26 and 19
```

Interpretation after review:

```text
This was useful proof that the math kernels can be connected, but it is the
wrong production shape. One layer using 25 compute cores cannot become a
28-layer megakernel by appending more static graph. The current production code
therefore removed this path and kept only the phase-owned topology.
```

D1.5 production reset accepted:

```text
production stage: phase-owned
fixed lane Workers consume packed layer/phase streams
num_layers changes the Worker loop trip count and packet stream length, not the
number of Workers/FIFOs
current skeleton validates resource shape as real Qwen3 phase kernels are
inserted one at a time
NPU2 default skeleton: num_lanes=8, num_layers=28, phase_packets_per_layer=11,
hidden_size=1024, packet_elements=2048, compute_cores=8, errors=0
NPU2 large-packet compile/preflight: packet_elements=16896,
max_fifo_buffered_bytes=33792, compute_cores=8
```

D1.5a accepted:

```text
phase 0 packet layout: hidden[1024] || norm_weight[1024]
phase 0 AIE kernel: input RMSNorm checksum using aie::invsqrt
synthetic phases remain placeholders
num_lanes=8, num_layers=28, compute_cores=8,
max tile inputs=1, max tile outputs=1, errors=0
```

D1.5b accepted:

```text
phase 0 packet layout:
  hidden[1024] || norm_weight[1024] || q_weight_row[1024]
phase 0 AIE kernel:
  input RMSNorm checksum + dot(xnorm, q_weight_row)
num_lanes=8, num_layers=28, packet_elements=3072, compute_cores=8,
max tile inputs=1, max tile outputs=1, errors=0
large packet compile/preflight still passes at packet_elements=16896
```

D1.5c accepted:

```text
phase 0 packet layout:
  hidden[1024] || norm_weight[1024] || q_weight_block[4, 1024]
phase 0 AIE kernel:
  input RMSNorm checksum + four dot(xnorm, q_weight_row) accumulations
num_lanes=8, num_layers=28, q_rows_per_packet=4, packet_elements=6144,
compute_cores=8, max tile inputs=1, max tile outputs=1, errors=0
large packet compile/preflight still passes at packet_elements=16896
```

D1.5d accepted:

```text
production topology now includes lane communication:
  shared hidden/norm stream -> grouped broadcast
  lane-local Q row shard packets -> lane Workers
  lane Q shard outputs -> grouped join -> per-layer output tensor

An ungrouped 8-way broadcast/join fabric failed in SequentialPlacer endpoint
placement. A 4-lane version compiled, proving the issue was fabric fan-in/out.
The accepted fix is two 4-lane fabric groups for 8 lanes.

num_lanes=8, fabric_group_size=4, q_rows_per_packet=4, packet_elements=4096,
compute_cores=8, total_dma_tasks=12, max tile inputs=2, max tile outputs=1,
errors=0
large packet compile/preflight still passes at packet_elements=16896
```

D1.5e accepted:

```text
production topology now carries activation-sized inter-layer state:
  hidden_state[1024] BF16 Buffer per lane Worker
  layer 0 initializes hidden_state from the shared stream
  phase 0 reads hidden_state for RMSNorm/Q shard
  next_layer_token writes hidden_state for the next layer
  state[0] remains a diagnostic checksum only

num_lanes=8, fabric_group_size=4, hidden_size=1024, packet_elements=4096,
compute_cores=8, total_dma_tasks=12, max tile inputs=2, max tile outputs=1,
errors=0
large packet compile/preflight still passes at packet_elements=16896
```

D1.5f accepted:

```text
production runner now uses real Qwen3-0.6B data:
  real per-layer input RMSNorm weights
  real q_proj row shards
  real layer input hidden for layer 0
  real next-layer hidden values written through next_layer_token packets

num_lanes=8, num_layers=28, fabric_group_size=4, q_rows_per_packet=4,
packet_elements=4096, compute_cores=8, max tile inputs=2, max tile outputs=1
phase_owned_errors=0 against local BF16 boundary reference
qwen3_q_shard_max_abs=0.250000, qwen3_q_shard_errors=0 at abs_tol=0.5
large packet compile/preflight still passes at packet_elements=16896
```

D1.5g accepted:

```text
gate_up phase now uses real Qwen3 data:
  real attention residual
  real post_attention_layernorm.weight
  real gate_proj row shards
  real up_proj row shards

The AIE kernel computes post RMSNorm and emits gate/up shard values into the
same per-lane output object as the Q shard. The grouped join drains the combined
Q + gate/up shard output.

num_lanes=8, num_layers=28, fabric_group_size=4, q_rows_per_packet=4,
packet_elements=10240, output_values_per_lane=16, compute_cores=8,
max tile inputs=2, max tile outputs=1
phase_owned_errors=0 against local BF16 boundary reference
qwen3_phase_output_max_abs=0.250000, qwen3_phase_output_errors=0 at abs_tol=0.5
large packet compile/preflight still passes at packet_elements=16896
```

D1.5h accepted:

```text
down_proj phase now uses real Qwen3 data:
  real ffn_hidden produced by the Qwen3 reference path for this checkpoint
  real attention residual shard
  real down_proj row shards

The AIE kernel computes dot(ffn_hidden, down_proj row) + residual_shard and
writes residual shard values into the same per-lane output object as Q and
gate/up. The grouped join still runs once per layer.

num_lanes=8, num_layers=28, fabric_group_size=4, q_rows_per_packet=4,
packet_elements=15368, output_values_per_lane=24, compute_cores=8,
max tile inputs=2, max tile outputs=1
phase_owned_errors=0 against local BF16 boundary reference
qwen3_phase_output_max_abs=0.000488, qwen3_phase_output_errors=0 at abs_tol=0.5
large packet compile/preflight still passes at packet_elements=16896
```

The reference rule after D1.5h is explicit: row-shard acceptance uses BF16
inputs, scalar float32 accumulation, and BF16 output. PyTorch `F.linear` is not
specific enough to be the strict gate for 3072-wide down rows.

D1.5i accepted:

```text
o_proj phase now uses real Qwen3 data:
  real attention_context[2048] produced by the Qwen3 reference path
  real layer input residual shard
  real o_proj row shards

The AIE kernel computes dot(attention_context, o_proj row) + residual_shard and
writes attention residual shard values into the same per-lane output object as
Q, gate/up, and layer residual.

num_lanes=8, num_layers=28, hidden_size=1024, attention_size=2048,
intermediate_size=3072, fabric_group_size=4, q_rows_per_packet=4,
packet_elements=15368, output_values_per_lane=32, compute_cores=8,
max tile inputs=2, max tile outputs=1
phase_owned_errors=0 against local BF16 boundary reference
qwen3_phase_output_max_abs=0.000488, qwen3_phase_output_errors=0 at abs_tol=0.5
large packet compile/preflight still passes at packet_elements=16896
```

The key shape rule after D1.5i is that `attention_size` is not always
`hidden_size`. For Qwen3-0.6B, `hidden_size=1024` while
`num_attention_heads * head_dim = 2048`, so o_proj packet fields and ABI checks
must use the explicit attention width.

D1.5j accepted:

```text
attention_chunk_0 and attention_chunk_1 now use real Qwen3 data:
  real k_proj row shards
  real v_proj row shards
  same tile-local hidden_state and shared input_layernorm weight as Q

The Worker keeps the shared broadcast packet acquired across Q/K/V phases and
then releases it before the remaining attention placeholder phases.

num_lanes=8, num_layers=28, hidden_size=1024, attention_size=2048,
intermediate_size=3072, fabric_group_size=4, q_rows_per_packet=4,
packet_elements=15368, output_values_per_lane=48, compute_cores=8,
max tile inputs=2, max tile outputs=1
phase_owned_errors=0 against local BF16 boundary reference
qwen3_phase_output_max_abs=0.003906, qwen3_phase_output_errors=0 at abs_tol=0.5
large packet compile/preflight still passes at packet_elements=16896
```

The useful topology rule after D1.5j is that a shared broadcast token may be
held across multiple adjacent phases if all consumers advance in the same
order. That let Q/K/V share input RMSNorm data without adding another FIFO.

Production entry:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py
```

Build one layer using:

```text
fixed max-cache past attention read
current present K/V consumed inside attention before host writeback
packed lane-local streams
real Qwen3 weights and shapes
preflight before execution
stage-by-stage reference comparison
```

Acceptance:

```text
one layer matches reference
present K/V and host cache update match reference
preflight passes
full aiecc passes
runtime does not hang
resource stats are recorded
```

Remaining D1 sequence:

```text
D1.5k expand Q/K/V shard coverage beyond the first 32 rows
D1.5l replace remaining attention packet group with real q/k norm, RoPE, and chunked online attention
D1.5m feed downstream phases from full NPU-produced activation vectors instead of host reference packets
D2 run repeated layers in the same phase-owned topology
```

### D2: Small N-Layer Reuse

Do not jump straight to 28 layers. First prove reuse with a small count:

```text
2 layers
4 layers
then 28 layers
```

Acceptance:

```text
same Worker/FIFO topology reused across layers
layer loop consumes packed layer stream in order
hidden output matches reference at each tested depth
no static duplication of all layer Workers/FIFOs
```

### D3: Full Decode Body

Run the full 28-layer decode body:

```text
multiple prompts
multiple generated tokens
positions crossing attention chunk boundary
token match against reference
measured speedup against current baseline
```

### D4: Performance Scaling

Only after correctness:

```text
adjust column policy per projection
try L2 split/forward/join for selected GEMV phases
increase max_seq_len bucket
consider moving final norm / LM head to NPU
```

## Rejected Or Deferred Ideas

Deferred:

```text
RTP as the primary position mechanism
runtime BD patching with NPU control packets
GEMM-style L2 GEMV topology as a prerequisite
NPU-side dynamic KV writeback
full-array column use before measured bottleneck evidence
```

Rejected for V0:

```text
dummy FIFO tokens to skip inactive phases
one FIFO per phase input for large graphs
raw ELF/runtime .bin patching without relocation proof
claiming performance from skeleton experiments
```

## Open Risks

The biggest remaining risks are:

```text
D1 real layer math may need more endpoints than D0 skeleton
real reductions may force extra FIFOs or drains
packet padding may make weight traffic too high
fixed max-cache attention may become costly at larger max_seq_len
external kernel ABI proliferation may make the layer loop hard to keep generic
same topology may not fit 28-layer stream without new BD pressure
```

These risks are why the next step is D1, not a direct full-model rewrite.

## One-Sentence Summary

V0 is a fixed-shape, single-token decode architecture: the host owns dynamic
state updates, the NPU runs a static phase-owned graph over packed lane-local
streams, attention uses fixed full-cache reads plus masks, and real Qwen3
correctness/performance must be proven one layer before scaling to 28 layers.
