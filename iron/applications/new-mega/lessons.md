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

## What To Do Next

The next useful `new-mega` work should be a proof ladder, not a full rewrite.

Recommended order:

```text
1. Build D1: one Qwen3 layer using fixed-chunk attention read, host-side
   present K/V writeback, and packed lane-local phase streams.
2. Keep each compute tile to one or two input FIFOs and one output FIFO.
3. Run preflight and full aiecc before executing on NPU.
4. Verify every layer boundary against PyTorch/reference buffers.
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
