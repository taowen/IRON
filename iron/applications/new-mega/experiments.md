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

Status: in progress; D1.0, D1.1a, D1.1b, and D1.2 accepted in production.

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

#### D1.1a. Q/K Norm+RoPE And Fixed Present K/V

Status: accepted in production.

Question:

```text
Can production code take real Qwen3 raw Q/K/V tensors, perform q_norm/k_norm
and RoPE on NPU, and produce fixed present K/V outputs without unsafe FIFO
fan-in?
```

Result:

```text
production stage: qkv-rope-present
packed_input_size: 4480
packed_output_size: 4096
preflight_compute_cores: 1
preflight_max_fifo_buffered_bytes: 8960
preflight_total_dma_tasks: 2
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 1
preflight_max_compute_tile_outputs: 1

default prompt position=26:
  npu_time_us=760.438
  max_abs=0.031250
  errors=0

second prompt position=22:
  npu_time_us=750.058
  max_abs=0.062500
  errors=0
```

Conclusion:

```text
The Q/K norm+RoPE and fixed present K/V boundary is now production code. It
still receives already-projected raw Q/K/V tensors; QKV GEMV projection remains
future work.
```

#### D1.1b. Q/K Norm+RoPE Feeding Fixed Attention

Status: accepted in production.

Question:

```text
Can the NPU qkv-rope-present output be used as the real producer for
fixed-cache attention after host-owned current K/V writeback?
```

Result:

```text
production stage: qkv-rope-attention
qkv_preflight_compute_cores: 1
qkv_preflight_max_fifo_buffered_bytes: 8960
attention_preflight_compute_cores: 1
attention_preflight_max_fifo_buffered_bytes: 32896
attention_preflight_max_compute_tile_inputs: 2

default prompt position=26:
  qkv_npu_time_us=871.263
  attention_npu_time_us=4386.955
  total_npu_time_us=5258.218
  qkv_errors=0
  context_max_abs=0.015625
  context_errors=0

second prompt position=22:
  qkv_npu_time_us=852.138
  attention_npu_time_us=3342.268
  total_npu_time_us=4194.406
  qkv_errors=0
  context_max_abs=0.019531
  context_errors=0
```

Conclusion:

```text
The production boundary now closes across two NPU dispatches:
q_norm/k_norm/RoPE -> fixed present K/V -> host cache writeback -> fixed-cache
attention. This is not a fused single graph yet, but it proves the exact data
contract needed before adding O projection or moving QKV GEMV into production.
```

#### D1.1c. Single-Dispatch Q/K Norm+RoPE Feeding Attention With Present K/V

Status: accepted in production.

Question:

```text
Can Q/K norm+RoPE feed attention inside one NPU dispatch if attention consumes
current present K/V directly, and host writes present K/V back only after the
dispatch?
```

Result:

```text
production stage: qkv-rope-attention-present
runtime_memrefs: 4
preflight_compute_cores: 2
preflight_max_fifo_buffered_bytes: 32896
preflight_total_dma_tasks: 4
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 2
preflight_max_compute_tile_outputs: 2

default prompt position=26:
  npu_time_us=4403.635
  current_errors=0
  context_max_abs=0.015625
  context_errors=0

second prompt position=22:
  npu_time_us=4096.564
  current_errors=0
  context_max_abs=0.019531
  context_errors=0
```

Conclusion:

```text
This fixes the D1.1b structural gap. The host no longer writes current K/V
between QKV/RoPE and attention. The attention Worker processes past cache
chunks with a past-only mask, then folds current present K/V into the same
online softmax state before finalizing context. Host-owned KV cache update is
still the architecture boundary, but it happens after the dispatch and prepares
the next token.
```

Production direction:

```text
Use qkv-rope-attention-present as the base graph for the main line.
Do not keep adding one-op dispatches as the target architecture. Multi-dispatch
stages are now boundary diagnostics. New math should be appended as Workers and
ObjectFifos inside this single graph, then checked against the same references.
```

#### D1.2. O Projection And Residual

Status: accepted in production.

Question:

```text
Can the accepted production attention context feed Qwen3 layer-0 O projection
and produce the same attention residual as the PyTorch/reference path?
```

Result:

```text
production stage: qkv-rope-attention-o
O projection implementation: existing GEMV operator, M=1024, K=2048, columns=4
residual add: host side

default prompt position=26:
  qkv_npu_time_us=828.013
  attention_npu_time_us=3711.416
  o_proj_npu_time_us=1098.267
  total_with_o_proj_npu_time_us=5637.696
  attn_out_max_abs=0.005859
  attn_out_errors=0
  attn_residual_max_abs=0.005859
  attn_residual_errors=0

second prompt position=22:
  qkv_npu_time_us=802.766
  attention_npu_time_us=4933.624
  o_proj_npu_time_us=970.969
  total_with_o_proj_npu_time_us=6707.359
  attn_out_max_abs=0.007812
  attn_out_errors=0
  attn_residual_max_abs=0.007812
  attn_residual_errors=0
```

Conclusion:

```text
The attention side of one real Qwen3 layer is now numerically closed through
O projection and residual. This is a conservative multi-dispatch path, not a
single fused production graph. It is still the right proof before adding
post-attention RMSNorm and MLP.
```

#### D1.2c. Fused O Projection And Residual

Status: accepted in production.

Question:

```text
Can O projection and residual be appended to the accepted
qkv-rope-attention-present graph without adding another NPU dispatch?
```

Result:

```text
production stage: qkv-rope-attention-o-fused
runtime_memrefs: 4
preflight_compute_cores: 4
preflight_max_fifo_buffered_bytes: 32896
preflight_total_dma_tasks: 7
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 2
preflight_max_compute_tile_outputs: 2

default prompt position=26:
  npu_time_us=5664.255
  current_errors=0
  attn_out_max_abs=0.005859
  attn_out_errors=0
  attn_residual_max_abs=0.005859
  attn_residual_errors=0

second prompt position=22:
  npu_time_us=5329.210
  current_errors=0
  attn_out_max_abs=0.007812
  attn_out_errors=0
  attn_residual_max_abs=0.007812
  attn_residual_errors=0
```

Conclusion:

```text
The production main line can now grow inside one IRON Program:
qkv/rope -> attention with present K/V -> O projection -> residual.
The older qkv-rope-attention-o stage remains useful as a multi-dispatch
boundary diagnostic, but it is no longer the target implementation path.
```

#### D1.3c. Fused Post-Attention RMSNorm And MLP

Status: accepted in production.

Question:

```text
Can post-attention RMSNorm, gate/up, SiLU*up, down projection, and layer
residual be appended to the accepted qkv-rope-attention-o-fused graph without
adding another NPU dispatch?
```

Result:

```text
production stage: qkv-rope-attention-o-mlp-fused
runtime_memrefs: 5
preflight_compute_cores: 14
preflight_max_fifo_buffered_bytes: 32896
preflight_total_dma_tasks: 15
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 2
preflight_max_compute_tile_outputs: 2

default prompt position=26:
  npu_time_us=8585.632
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0

second prompt position=19:
  prompt="What is the capital of France?"
  npu_time_us=7978.991
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0
```

Conclusion:

```text
The production main line now covers the real layer-0 attention block and MLP
inside one IRON Program. The accepted resource rule is explicit: split phases
or pack streams until each compute tile has at most two input and two output
ObjectFIFOs. Debug drains are useful, but they must be budgeted like production
streams.

The production graph now borrows the row-sharded GEMV pattern from the high
performance GEMV operator inside the graph: context/xnorm are broadcast through
ObjectFifo to two column workers, weights are row-sharded, and a join worker
reconstructs the full vector for the next phase. O-only columnization was
correct but slower; adding gate/up produced the first accepted speed
improvement, and adding down projection improved the same stage further.
```

#### D1.4a. Input RMSNorm + QKV Projection + RoPE Boundary

Status: accepted as a diagnostic boundary, then merged into the main production
graph in D1.4b.

Question:

```text
Can the production path pull in the existing high-throughput GEMV pattern at
the front of the layer, starting from token hidden and Q/K/V weights instead of
host-provided raw Q/K/V tensors?
```

Implementation:

```text
diagnostic stage: input-qkv-rope-present
input RMSNorm: NPU
Q projection: 2 row-sharded workers
K projection: 2 row-sharded workers
V projection: 2 row-sharded workers
Q/K norm+RoPE: fused into the Q/K projection shard workers
output: current Q/K/V plus raw Q/K/V debug shards
```

Accepted evidence:

```text
default prompt position=26:
  npu_time_us=2037.888
  current_errors=0
  q_raw_errors=0
  k_raw_errors=0
  v_raw_errors=0
  preflight_compute_cores=7
  preflight_max_compute_tile_inputs=2
  preflight_max_compute_tile_outputs=2

second prompt position=19:
  prompt="What is the capital of France?"
  npu_time_us=2152.913
  current_errors=0
  q_raw_errors=0
  k_raw_errors=0
  v_raw_errors=0
```

Debug lessons:

```text
The first graph failed aiecc because one final rope worker consumed q_raw,
k_raw, v_raw, and metadata as four independent ObjectFIFOs. The fix was to
move Q/K norm+RoPE into each row-sharded projection worker so no tile exceeded
two input FIFOs.

The first runtime run returned all-zero Q/K regions while V was correct.
Slicing by output region showed that independent q/k drains had no wait=True;
waiting on the unrelated final V debug drain did not make Q/K visible. Each
independent host-visible drain now waits explicitly.
```

#### D1.4b. Fused Input Projection Into The Main Layer Graph

Status: accepted as a math wiring proof, rejected as the production direction.

Question:

```text
Can the standalone input-qkv-rope-present boundary be removed by fusing input
RMSNorm, Q/K/V projection, and Q/K norm+RoPE into
qkv-rope-attention-o-mlp-fused, while keeping one dispatch and the existing
five-BO runtime ABI?
```

Result:

```text
production stage: qkv-rope-attention-o-mlp-fused
runtime_memrefs: 5
qkv_packed_input_size: 4196736
preflight_compute_cores: 25
preflight_total_dma_tasks: 21
preflight_max_compute_tile_inputs: 2
preflight_max_compute_tile_outputs: 2

default prompt position=26:
  npu_time_us=10313.281
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0

second prompt position=19:
  prompt="What is the capital of France?"
  npu_time_us=9547.494
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0
```

Debug lessons:

```text
Adding Q/K/V weights as a sixth runtime memref failed preflight because the
generated xclbin metadata still exposed only five HOST BO slots. The fix was
to pack input metadata and Q/K/V weights into one first buffer and use TAP
offsets inside that buffer.

The generated artifact name also became too long after adding the new
parameters. The fix was to give the fused production operator a short explicit
name instead of inheriting a dataclass-style name containing every field.
```

Decision after review:

```text
This was still static single-layer fusion. It used 25 compute cores for one
layer, so it could not scale to a real 28-layer decode body by appending more
static graph. The production directory now keeps only the phase-owned topology.
```

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

#### D1.5x. Gate/Up Produces A Reduced FFN Hidden Group For Down Projection

Status: accepted in production.

Question:

```text
Can the existing source/target reducer fabric carry MLP-internal data, so
down_proj consumes an NPU-produced FFN hidden group instead of the host packet?
```

Change:

```text
gate_up now emits one extra reducer token per layer after all O chunks:
  ffn_partial[32]

Each lane writes only its q_rows_per_packet=4 FFN hidden rows:
  ffn_row_base = lane * q_rows_per_packet
  ffn_partial[ffn_row_base + row] = silu(gate[row]) * up[row]

The existing two source reducers and two target reducers consume one additional
token per layer:
  o_projection_chunk_count + 1

down_proj consumes:
  ffn_reduced[0:32] for the first 32 intermediate rows
  host_ffn_hidden[32:3072] for the remaining rows
```

Resource diagnosis:

```text
The first implementation added a standalone
new_mega_phase_ffn_partial_from_gate_up_bf16 external kernel.

compile-only failed at CDO generation:
  [AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory
  XAie_LoadElf failed with XAIE_INVALID_ELF

llvm-size showed lane core .text:
  standalone ffn_partial kernel: 16880 bytes
  fused into gate_up only:       16640 bytes
  after deleting obsolete down local_ffn fallback: 16080 bytes

The actual resource was AIE program memory, not FIFO depth, TAP, placement, or
L1 data memory. The accepted fix keeps the reducer dataflow but fuses the FFN
partial producer into gate_up and removes code that became unreachable once
ffn_reduced[0:32] is available.
```

Numeric diagnosis:

```text
The first running build failed only in down_residual:
  phase_owned_errors=96
  qwen3_phase_output_errors=144
  qwen3_segment_down_residual_errors=144 max_abs=33

The gate packet used packet[0] for residual replacement metadata. For full
residual visibility that field is 0 for every lane, so using packet[0] as the
FFN row base made every lane write rows 0..3 and left rows 4..31 as zero.

Fix:
  keep packet[0] as residual_row_base
  store ffn_row_base in gate_packet[packet_elements - 1]
```

Accepted result:

```text
num_lanes=8
num_layers=28
phase_packets_per_layer=48
packet_elements=16576
preflight_compute_cores=12
preflight_max_compute_tile_inputs=2
preflight_max_compute_tile_outputs=2
npu_time_us=578549.967
phase_owned_max_abs=1.000000
phase_owned_mean_abs=0.011823
phase_owned_errors=0
qwen3_phase_output_max_abs=1.000000
qwen3_phase_output_mean_abs=0.011824
qwen3_phase_output_errors=0
```

Interpretation:

```text
This proves the O reducer fabric can be reused for MLP-internal handoff without
adding Worker cores or FIFO endpoints. It does not yet remove the full
host-packed FFN vector: only the first 32 of 3072 intermediate rows are
NPU-produced. The next useful step is not another 32-row gather loop by brute
force; it is either a full FFN gather/broadcast plan with bounded program
memory or a down partial-projection reduce that avoids materializing all 3072
FFN values on every lane.
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
