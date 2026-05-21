<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# New Megakernel Experiments

This file is the execution plan for the next Qwen3 megakernel experiments.
It is based on the reasoning in `why.md`, `answers.md`, and the lessons from
the current `qwen3_0_6b/persistent` implementation.

It is deliberately not a full architecture document. The point is to answer the
few mechanism questions that currently block a real architecture.

Rule:

```text
Do not rewrite the Qwen3 layer graph until the smaller mechanism experiment
that it depends on has been accepted.
```

## Current Ground Truth

Accepted baseline:

```text
implementation: iron/applications/qwen3_0_6b/persistent
stage: generate --fast-generate
body: n-layer-final-only
layer_chunk_size: 28
columns: num_aie_columns=2, attention_columns=2, mlp_gate_up_columns=2
final norm / LM head: CPU
position handling: exact-position artifacts
helper: --precompile-generate-positions
```

Known-good evidence:

```text
default prompt:
  token_match=True
  new_text='Paris'

Fibonacci prompt:
  full chunk=28 precompiled positions 17 and 18 before the token loop
  token loop matched 2/2 decode steps
  new_text=' 5,'
```

What this proves:

```text
one persistent graph can run the 28-layer decode body for one exact position
exact-position precompile removes compilation from the token loop
segment-major weights work for the accepted 2-column topology
```

What it does not prove:

```text
RTP can update Worker behavior across invocations
RTP can replace attention position immediates safely
runtime DMA/TAP offsets can be changed without recompilation
GEMV can use a GEMM-style L2 topology
task_group can schedule multiple Worker phases
```

Accepted mechanism experiment:

```text
A0 fixed-chunk decode attention:
  one artifact ran positions 0, 26, 63, 64, 127, 200, 255
  max_seq_len=256, head_dim=128, chunk_size=64
  fixed TAPs and fixed four-chunk loop
  live position encoded only in runtime mask values
  packed K/V/mask stream used to stay within input DMA channels

A0B host-side KV writeback:
  one artifact ran 70 sequential decode steps across position 63/64
  NPU output fixed present_k/present_v tensors
  host wrote present K/V into the packed cache at the dynamic row offset
  next NPU invocation read the updated cache through the same fixed full-cache TAP
  NPU never wrote KV cache and never received position as a runtime argument

B1 real-shape GEMV scaling:
  all Qwen3 projection shapes compiled and matched reference at 1/2/4/8 columns
  scaling is non-monotonic
  uniform 4 columns was best among uniform policies for the seven projections
  per-shape best was materially better than any single fixed column count
```

## Dependency Graph

The experiments have dependencies. Do not reorder them casually.

```text
A0. fixed-chunk decode attention with runtime mask [accepted]
    |
    +--> attention read side should use fixed max-cache TAP + mask,
         not RTP position immediates or dynamic BD patching
    |
    +--> A0B. host-side present K/V writeback [accepted]
         current-token KV cache writeback should be host memcpy between
         dispatches, not NPU dynamic-offset drain/fill

A1. RTP scalar cross-invocation proof
    |
    +--> A2. RTP replaces core position immediate in a tiny Worker
    |
    +--> A3. current-K/V offset proof without per-position TAP
              |
              +--> A4. attention same-artifact two-position proof

B1. real-shape standalone GEMV scaling
    [accepted]
    |
    +--> B2. GEMV L2 split/forward/join topology proof [deferred]
              |
              +--> B3. segment-major packing for the new topology

C1. two-phase same-Worker protocol proof
    [accepted]
    |
    +--> C2. phase skip / inactive FIFO protocol proof
         [partially accepted: static pruning only, not same-artifact dynamic skip]
         |
         +--> D0. static phase ownership resource skeleton
              [accepted: packed lane-local streams]

Only after A0/A0B + B1 + C1/C2 + D0 are accepted:
    D1. single Qwen3 layer integration
    D2. 28-layer decode body integration
    D3. end-to-end boundary inventory
```

If A1 fails, do not use RTP as the attention-position mechanism. A0 gives the
accepted replacement for the attention read side: fixed max-cache movement plus
runtime mask data. A0B gives the accepted replacement for current-K/V writeback:
fixed present outputs plus host memcpy. B2 is deferred because B1 already gave
a useful near-term column policy signal and L2 topology is not the current
blocking mechanism. C2 rejected same-artifact dynamic skipping, so D0 tested
the safer static phase ownership direction.

## A. Position Reuse Experiments

This is the most important axis. If one artifact cannot run multiple decode
positions, the new megakernel remains an exact-position precompile system.

### A0. Fixed-Chunk Decode Attention

Status: accepted.

Question:

```text
Can decode attention run multiple positions from one fixed artifact by always
processing the full preallocated KV cache and passing position as mask data?
```

Result:

```text
Accepted. One xclbin/runtime .bin ran positions
0, 26, 63, 64, 127, 200, 255 against CPU reference.
```

Key lesson:

```text
The first four-input Worker exceeded tile input DMA channels. The accepted
graph packs K/V/mask into one chunk stream and keeps Q as the second input.
This makes fixed TAP attention practical without RTP or dynamic BD patching.
```

### A0B. Host-Side KV Writeback

Status: accepted.

Question:

```text
Can current-token K/V writeback be removed from the NPU graph by draining fixed
present_k/present_v outputs and letting the host update the persistent cache?
```

Result:

```text
Accepted. One xclbin/runtime .bin ran 70 sequential token steps, crossed the
63/64 chunk boundary, and matched CPU reference exactly after setting AIE BF16
rounding to conv_even.
```

Key lesson:

```text
The NPU does not need dynamic KV-cache write offsets. It can be a pure function
over current state and full past cache. The host owns the stateful cache update
between dispatches.
```

### A1. RTP Scalar Cross-Invocation Proof

Question:

```text
Can one compiled artifact read a host-written RTP scalar and change Worker
behavior across two separate invocations without recompilation?
```

Why it matters:

```text
GEMM proves RTP at Worker startup.
It does not prove that the same xclbin can be invoked twice with different RTP
values and produce different, correct behavior.
```

Minimal design:

```text
one Worker
one input vector
one output vector
one RTP scalar

run 1: scale output by RTP=3
run 2: scale output by RTP=7
same xclbin, same runtime .bin
```

Acceptance:

```text
same artifact for both runs
outputs match CPU reference for both RTP values
no recompile
preflight passes
full aiecc passes
no runtime hang
```

Reject if:

```text
second invocation reads stale RTP
RTP can only be consumed once after WorkerRuntimeBarrier
dynamic branch/loop from RTP compiles but hangs
```

### A2. RTP Replaces A Core Immediate

Question:

```text
Can a value that used to be compiled as a core-ELF immediate be read from RTP
instead, with the core ELF staying byte-identical across values?
```

Minimal design:

```text
one Worker
one RTP integer "position"
kernel computes out[i] = in[i] + position
compare builds for position=26 and position=27
```

Acceptance:

```text
artifact diff shows core ELFs do not change between positions
runtime output changes as expected when RTP changes
same xclbin runs both positions
```

Reject if:

```text
core ELF still changes when only position changes
RTP value is not visible inside the external kernel path
```

### A3. Dynamic Current-K/V Offset Proof

Question:

```text
Can the current-token K/V write/read offset be handled without compiling a new
TAP/BD offset for each position?
```

Why it matters:

```text
RTP can at best affect tile-local logic.
The current K/V offset also lives in runtime .bin DMA descriptors.
```

Minimal design:

```text
small cache tensor in DDR
one current vector
position p writes current vector into cache[p]
position p+1 writes into cache[p+1]
read both back with a monotonic pattern
```

Acceptance:

```text
same xclbin and runtime .bin for p and p+1
or a documented safe runtime patch/update path with explicit patch sites
cache readback matches the monotonic pattern
no future-token read
no overlapping writes
```

Reject if:

```text
runtime .bin still changes per position and no safe update path exists
the proof requires raw NpuControlPacketOp register writes without quiescence
the ObjectFifo metadata path hangs or returns corrupted data
```

### A4. Attention Same-Artifact Two-Position Proof

Question:

```text
Can the attention subgraph run positions p and p+1 with the same compiled
artifact and different runtime values?
```

Why it matters:

```text
Attention is the densest position-dependent part of decode:
RoPE, KV cache write offset, KV read length, mask length, and softmax all depend
on position.
```

Scope:

```text
single layer or attention-only probe
real Qwen3 head_dim and GQA mapping
positions p and p+1
no full MLP
```

Acceptance:

```text
same artifact runs both positions
attention output matches PyTorch reference at both positions
softmax row sums are valid
current K/V cache slices match reference
artifact diff confirms no position-specific core ELF or runtime .bin changes,
  except for an explicitly proven safe patch path
```

Reject if:

```text
NaN residuals
ERT_CMD_STATE_TIMEOUT
future cache reads
wrong current K/V slice
artifact still changes per position
```

## B. GEMV Parallelism Experiments

This axis answers whether "use more tiles" is actually legal and faster for
Qwen3 GEMV shapes.

### B1. Real-Shape Standalone GEMV Scaling

Status: accepted.

Question:

```text
For real Qwen3 GEMV shapes, does increasing columns improve measured NPU time
without hitting endpoint, BD, L1, or placement limits?
```

Shapes:

```text
Q:    2048 x 1024
K:    1024 x 1024
V:    1024 x 1024
O:    1024 x 2048
gate: 3072 x 1024
up:   3072 x 1024
down: 1024 x 3072
```

Run:

```text
cols=1,2,4,8 where legal
same input vector and weights
compare against torch.nn.functional.linear
```

Acceptance:

```text
resolve_program passes
full aiecc passes
preflight resource checks pass
NPU output matches F.linear
measured time improves for at least one wider topology
```

Reject if:

```text
resolve passes but aiecc fails in aie.dma_bd
tile input/output count exceeds known safe limits
runtime endpoints exceed placement capacity
extra columns slow down or do not improve measured time
```

Result:

```text
All seven real Qwen3 projection shapes compiled, ran, and matched reference at
1/2/4/8 columns. At least one wider topology improved every large projection,
but the best column count depends on shape:

q:    best 4 columns
k:    best 2 columns
v:    best 1 column
o:    best 4 columns
gate: best 4 columns
up:   best 8 columns
down: best 8 columns
```

Conclusion:

```text
Existing GEMV scaling is legal but non-monotonic. Do not use a blanket
8-column policy. B2 should focus on topology and endpoint/dataflow overhead,
not just increasing column count.
```

### B2. GEMV L2 Split/Forward/Join Topology

Status: deferred.

Question:

```text
Can GEMV use an L3->L2->L1 topology so runtime endpoint count scales with
columns instead of compute tiles?
```

Reason for deferral:

```text
B1 already proved existing GEMV is legal for all real Qwen3 shapes and exposed
non-monotonic column scaling. The next architectural blocker is not L2 topology
alone; it is whether a layer/state-machine Worker protocol can run multiple
phases without host dispatch explosion.
```

Why it matters:

```text
GEMM uses L2 split/forward/join successfully, but GEMV has M=1 and a different
data distribution pattern. GEMM legality does not imply GEMV legality.
```

Candidate topology:

```text
input vector:
  L3 -> L2 once per column or group
  L2 broadcasts/forwards to compute tiles

weights:
  row-sharded by tile or tile group

output:
  compute tiles -> L2 join -> L3 contiguous output vector
```

Acceptance:

```text
runtime endpoint count is lower than one L3 stream per tile
L2 placement is legal on NPU2
full aiecc passes
output order matches F.linear
measured time beats the corresponding direct-L3 GEMV baseline
```

Reject if:

```text
L2 object bytes exceed capacity
join order is wrong
BD count grows beyond the accepted limit
L2 topology compiles but is slower
```

### B3. Segment-Major Packing For The New Topology

Question:

```text
Does the existing segment-major weight packing remain correct and aligned after
changing GEMV shard count and topology?
```

Acceptance:

```text
manifest offsets are explicit
each shard has shape/dtype/alignment checks
TAP access maps cover exactly the expected rows
monotonic weight pattern identifies row order correctly
F.linear validation passes for every real Qwen3 GEMV shape
```

Reject if:

```text
offsets depend on hidden assumptions from the 2-column layout
alignment breaks after changing shard count
tail rows require ad hoc special cases
```

## C. Phase Sequencing Experiments

This axis answers whether a "universal Worker" or phase-based megakernel can
exist in current IRON.

### C1. Two-Phase Same-Worker Protocol

Status: accepted.

Question:

```text
Can one Worker execute two host-scheduled phases in one dispatch with balanced
ObjectFifo acquire/release behavior?
```

Why it matters:

```text
task_group synchronizes runtime DMA tasks.
It does not prove host<->Worker phase handshakes while Workers are already
running.
```

Minimal design:

```text
phase 0: copy or scale input A -> temp/output
phase 1: copy or scale input B -> output
same Worker
same dispatch
debug counters or output markers identify both phases
```

Acceptance:

```text
one dispatch
both phases run in the intended order
no unbalanced FIFO acquire/release
output matches CPU reference
runtime does not require a second host dispatch
```

Reject if:

```text
Worker blocks waiting for a FIFO token from the wrong phase
host cannot signal phase progress after Worker start
phase value is stale
```

Result:

```text
Accepted. One Worker executed phase0 then phase1 in a single dispatch:
state = 2 * phase0 + 1
out = state + 3 * phase1 + 7

max_abs=0, errors=0
```

Conclusion:

```text
FIFO token order is sufficient for a fixed two-phase same-Worker protocol when
both phases always run and every acquire/release is balanced.
```

### C2. Inactive FIFO / Phase Skip Protocol

Status: partially accepted.

Question:

```text
Can a Worker skip a phase without requiring dummy DMA on inactive FIFOs?
```

Why it matters:

```text
If skipping a phase still requires filling every possible FIFO, the phase-based
design wastes the same endpoint and BD resources it was supposed to save.
```

Acceptance:

```text
inactive phase uses no fill/drain for its inactive data FIFOs
Worker does not acquire inactive FIFOs
active phase still has balanced acquire/release
runtime completes without hang
```

Reject if:

```text
skip path requires dummy ObjectFifo tokens
dummy tokens consume the scarce endpoints/BDs
Worker-side if statements cannot prevent acquire deadlock
```

Result:

```text
active variant: arg_count=3, max_abs=0, errors=0
skip variant:   arg_count=2, max_abs=0, errors=0

same_artifact=False
runtime_bin size changed 420 -> 300 bytes
xclbin size changed 10570 -> 10138 bytes
```

Conclusion:

```text
Inactive FIFO tokens can be removed by compiling a different static graph and
runtime ABI. This avoids dummy DMA for skipped phases, but it is not a
same-artifact dynamic phase skip mechanism.
```

Implication:

```text
A universal phase-based same-artifact Worker is still not proven. The accepted
safe direction is static phase ownership or separate artifacts unless a later
runtime-control proof shows same-artifact branch-controlled acquire omission.
```

### D0. Static Phase Ownership Resource Skeleton

Status: accepted.

Question:

```text
Can a fixed phase-owner topology compile and run without recreating the
endpoint/BD/L1 pressure that made larger static graphs fragile?
```

Design:

```text
8 lane Workers
1 input ObjectFIFO per lane
1 output ObjectFIFO per lane
11 fixed phase packets per lane
packet size = 16896 BF16 elements = 33792 bytes
```

Result:

```text
preflight_compute_cores: 8
preflight_max_fifo_buffered_bytes: 33792
preflight_total_dma_tasks: 16
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 1
preflight_max_compute_tile_outputs: 1
preflight_non_advancing_acquires: 0
npu_time_us: 5538.108
max_abs: 0.000000
errors: 0
decision: accepted
```

Debug note:

```text
Scalar BF16 output FIFOs are not safe DMA payloads. `memref<1xbf16>` failed
aiecc because DMA BD transfer length must be a multiple of 4 bytes. The
accepted graph pads output objects to `memref<8xbf16>` so they also satisfy
the stricter 16-byte FIFO alignment preflight rule.
```

Conclusion:

```text
Packed lane-local streams are the current resource-safe way to express a fixed
multi-phase layer skeleton. This avoids the endpoint explosion of one FIFO per
phase input. It does not prove high tile utilization or real Qwen3 math.
```

## D. Qwen3 Integration Experiments

Start these only after the dependent mechanism proofs pass.

### D1. Single-Layer Integration

Status: in progress; D1.0 accepted.

Dependencies:

```text
A0/A0B accepted for fixed attention read and host KV writeback
B1 accepted for real-shape GEMV column policy
C1/C2 accepted/partially accepted for phase protocol limits
D0 accepted for static phase ownership resource safety
```

Acceptance:

```text
one Qwen3 layer output matches PyTorch reference
current K/V cache update matches reference
preflight passes
full aiecc passes
same-artifact two-position run if dynamic position is claimed
```

#### D1.0. Real Fixed-Cache Attention Context

Status: accepted.

Question:

```text
Can the A0 fixed-cache attention read path run on real Qwen3 layer-0 tensors
and produce the same [16,128] attention context as the PyTorch/reference path?
```

Result:

```text
preflight_compute_cores: 1
preflight_max_fifo_buffered_bytes: 32896
preflight_total_dma_tasks: 3
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 2
preflight_max_compute_tile_outputs: 1

default prompt position=26:
  npu_time_us=3756.100
  max_abs=0.015625
  errors=0

second prompt position=22:
  npu_time_us=3451.432
  max_abs=0.019531
  errors=0
```

Conclusion:

```text
The fixed max-cache + runtime mask strategy now works on real Qwen3 attention
context data, not only synthetic A0 data. Current K/V can be written by the
host before dispatch and consumed by the NPU through the fixed full-cache
stream.
```

Production promotion:

```text
The accepted D1.0 boundary has been promoted to:

  iron/applications/new-mega/production

Production stage:
  fixed-attention

The production CLI passed the same two-prompt validation:
  position=26 max_abs=0.015625 errors=0
  position=22 max_abs=0.019531 errors=0
```

Remaining D1 work:

```text
D1.1 add NPU-side QKV/RoPE and fixed present K/V outputs
D1.2 add O projection + residual
D1.3 add post-attention RMSNorm + MLP
D1.4 run full single-layer output check
```

### D2. 28-Layer Decode Body

Acceptance:

```text
token_match=True on default and Fibonacci prompts
multi-token run has no hot-loop compile if dynamic position is claimed
measured token time beats the accepted persistent baseline
failure diagnostics are written to how-to-debug before changing strategy
```

### D3. End-To-End Boundary Inventory

This is required before calling the result a full inference architecture.

Fill this table with measured bytes and owners:

| boundary | data | owner | frequency | measured cost |
| --- | --- | --- | --- | --- |
| tokenizer -> embedding | token id | CPU | per token | TBD |
| embedding -> layer0 | hidden[1024] | CPU/XRT/NPU | per token | TBD |
| decode body -> final norm | hidden[1024] | NPU->CPU | per token | TBD |
| final norm -> LM head | hidden[1024] | CPU | per token | TBD |
| LM head -> argmax | logits | CPU | per token | TBD |
| KV cache read | per-layer K/V | NPU DMA | per token/layer | TBD |
| KV cache write | current K/V | NPU DMA | per token/layer | TBD |

Acceptance:

```text
every boundary has an owner
every buffer has shape/dtype/bytes
each transfer is classified as per prompt, per token, or per layer
wall-time buckets are measured
```

## Current Priority

Do the next work in this order:

```text
1. D1: single Qwen3 layer integration using the accepted A0/A0B/B1/C1/C2/D0
   mechanisms.
2. Keep final norm / LM head on CPU until the decode body is faster than the
   accepted persistent baseline.
3. Use packed lane-local streams for the first D1 graph; do not create one
   FIFO per phase input.
4. Measure whether the lane Workers are compute-bound or DMA-bound before
   adding more columns.
5. Only revisit RTP/dynamic BD/control-packet work if fixed max-cache + host
   writeback becomes the measured bottleneck.
```

Reason:

```text
A0 and A0B removed the need for dynamic attention/KV offsets in the first new
architecture. D0 showed the safer static phase ownership graph shape can pass
resource preflight. The next unknown is no longer "can the skeleton compile";
it is whether real Qwen3 layer math fits and improves measured token time.
```
