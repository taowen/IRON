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

Build one layer using:

```text
fixed max-cache attention read
host-side present K/V writeback
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
