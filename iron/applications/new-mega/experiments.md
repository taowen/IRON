<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# New Megakernel Experiments

This file is the active experiment plan for the Qwen3 new-mega work. It should
stay short enough to answer:

```text
what is accepted
what is currently being tested
what blocks the next implementation step
where the historical details live
```

Historical experiment logs are archived under `experiments/archive/`.

Rule:

```text
Do not rewrite the Qwen3 layer graph until the smaller mechanism experiment
that it depends on has been accepted.
```

## Archive Index

```text
experiments/archive/mechanism-experiments.md
  A0/A0B fixed-position reuse, B1 GEMV scaling, C1/C2 phase protocol, D0 skeleton

experiments/archive/d1-single-layer-legacy.md
  D1.0-D1.4b standalone and static single-layer fusion path

experiments/archive/d1-phase-owned-ladder.md
  D1.5 through D1.5w phase-owned production ladder before FFN reducer handoff
```

The executable experiment directories remain in `experiments/<experiment>/`.

## Current Ground Truth

Accepted persistent baseline:

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

Known-good persistent evidence:

```text
default prompt:
  token_match=True
  new_text='Paris'

Fibonacci prompt:
  full chunk=28 precompiled positions 17 and 18 before the token loop
  token loop matched 2/2 decode steps
  new_text=' 5,'
```

Accepted production direction:

```text
implementation: iron/applications/new-mega/production
stage: phase-owned
body: one fixed topology with lane Workers looping over 28 layers
num_lanes: 8
compute_cores: 12
phase_packets_per_layer: 62
position/cache policy: fixed max-cache reads + host-owned KV writeback
```

Current accepted production evidence:

```text
D1.5z Eight FFN Reduced Groups Feed Down Projection
  ffn_npu_rows=256
  lane_core_text_bytes=15360
  npu_time_us=1040519.463
  phase_owned_max_abs=1.000000
  phase_owned_mean_abs=0.011808
  phase_owned_errors=0
  qwen3_phase_output_max_abs=1.000000
  qwen3_phase_output_mean_abs=0.011807
  qwen3_phase_output_errors=0
```

What production currently proves:

```text
one dispatch runs the 28-layer phase-owned body
O projection is chunked into 32 row chunks
O partial products are reduced across all 8 lanes through fixed reducers
gate_up consumes the full NPU-produced attention residual
gate_up emits eight 32-row FFN partial groups through the same reducer fabric
down_proj consumes NPU-produced FFN rows 0..255
resource use is bounded by lane/reducer topology, not by num_layers * phases
```

What production does not yet prove:

```text
down_proj is not yet free of host-packed FFN hidden rows 256..3071
final norm / LM head / sampling are not inside the production megakernel
multi-token cache update remains host-owned between dispatches
the production body is not yet faster than the accepted persistent baseline
```

## Accepted Mechanisms

These are the mechanisms the architecture is allowed to depend on.

```text
A0 fixed-chunk decode attention:
  one artifact ran positions 0, 26, 63, 64, 127, 200, 255
  live position encoded as runtime mask data
  no RTP or dynamic BD patching required for attention read side

A0B host-side KV writeback:
  one artifact ran 70 sequential decode steps across position 63/64
  NPU outputs fixed present K/V
  host writes present K/V into the dynamic cache offset between dispatches

B1 real-shape GEMV scaling:
  Qwen3 projection shapes compiled and matched at 1/2/4/8 columns
  uniform 4 columns was the best simple policy among tested uniform policies

C1/C2 phase protocol:
  same Worker can execute ordered phase packets
  same-artifact dynamic FIFO skipping is not selected

D0 phase ownership:
  packed lane-local streams are the accepted resource-safe skeleton
```

Details are in `experiments/archive/mechanism-experiments.md`.

## Active Dependency Graph

```text
A0 + A0B + B1 + C1/C2 + D0 [accepted]
    |
    v
D1 phase-owned Qwen3 integration [in progress]
    |
    +--> D1.5x first 32 FFN rows via reducer [accepted]
    |
    +--> D1.5y first 128 FFN rows via reducer [accepted]
    |
    +--> D1.5z first 256 FFN rows via reducer [accepted]
    |
    +--> D1.5zz remove remaining host FFN hidden dependency [next]
    |
    v
D2 28-layer decode body performance/token validation
    |
    v
D3 end-to-end boundary inventory
```

## Last Accepted Experiment: D1.5y

Status: accepted.

Question:

```text
Can production extend the gate_up -> reducer -> down handoff from one 32-row
FFN group to four 32-row groups without exceeding FIFO endpoint, L1, BD/routing,
or AIE program-memory budgets?
```

Accepted change:

```text
gate_up/down are split into four phase pairs:
  ffn_gate_chunk_0, down_partial_chunk_0, ... chunk_3

Each gate chunk emits one 32-row FFN group through the existing source/target
reducer fabric. down_proj keeps a tile-local float accumulator and finalizes
after chunk_3:
  ffn_reduced[0:128] for intermediate rows 0..127
  host_ffn_hidden[128:3072] for the remaining rows
```

Resource finding:

```text
first attempt:
  lane core .text = 17904 bytes
  failure = CDO program-memory overflow

after removing non-essential diagnostic output from the lane program:
  q/k RoPE diagnostic packet slots are drained but no longer computed
  gate_up visual output segment is no longer materialized
  lane core .text = 15968 bytes
  full aiecc + CDO succeeds
```

Numeric finding:

```text
packet-level NPU-vs-kernel reference:
  phase_owned_errors=0

full Qwen3 semantic comparison:
  qwen3_phase_output_errors=0
  down_residual tolerance is 1.0 because it now consumes NPU-produced
  attention/FFN partial values, not only host-packed exact reference values
```

## Last Accepted Experiment: D1.5z

Status: accepted.

Question:

```text
Can production extend the gate_up -> reducer -> down handoff from four 32-row
FFN groups to eight 32-row groups without increasing lane program text or
exceeding FIFO endpoint, L1, BD/routing, or AIE program-memory budgets?
```

Accepted change:

```text
FFN_REDUCE_GROUP_COUNT=8
phase_packets_per_layer=62
ffn_reduced covers rows 0..255
host_ffn_hidden covers rows 256..3071
```

Resource finding:

```text
compile-only accepted
lane core .text = 15360 bytes

Increasing the loop trip count and phase stream length did not add lane
program text. This supports the phase-owned state-machine direction: phase
count grows runtime/input bytes, but not compute-worker count.
```

Performance finding:

```text
npu_time_us=1040519.463

This is slower than the 4-group result. Extending FFN handoff by adding more
packet phases is correctness progress, not a speed path. To go faster, remove
host packet volume and reduce repeated gate/down packet traffic rather than
blindly increasing group count.
```

## Current Experiment: D1.5zz

Status: next.

Question:

```text
How should production remove host_ffn_hidden[256:3072] from down_proj without
linearly increasing packet stream size and NPU runtime?
```

Known constraint:

```text
lane program memory is no longer the immediate blocker for more groups, but
runtime and host packet volume are now visibly worsening.
```

Candidate A, full FFN gather/broadcast:

```text
gate_up emits 3072 FFN hidden values in groups
reducers/gather fabric broadcasts each group to down_proj lanes
down_proj materializes or streams the full FFN vector
```

Risk:

```text
3072 values is too large to casually materialize in each lane output object
more group tokens increase runtime and may stress program memory
```

Candidate B, down partial-projection reduce:

```text
each lane computes down partial sums for the FFN rows it owns
reducers sum partial down rows across producer lanes
owner lanes add residual and write layer residual shard
```

Why this is preferred as the next experiment:

```text
it avoids broadcasting all 3072 FFN values to every lane
it matches the successful O partial-projection reduce pattern
it converts the remaining host activation dependency into a reduce problem
it keeps the output shape small: q_rows_per_packet residual rows per owner lane
```

Acceptance for D1.5zz:

```text
preflight passes
full aiecc passes
lane core .text stays below program-memory limit
NPU run completes
phase_owned_errors=0
qwen3_phase_output_errors=0
segment stats show down_residual has no new errors
how-to-debug updated with any new failure mode before changing strategy
```

## Previous Accepted Experiment: D1.5x

Question:

```text
Can the existing source/target reducer fabric carry MLP-internal data, so
down_proj consumes an NPU-produced FFN hidden group instead of the host packet?
```

Accepted change:

```text
gate_up emits one extra reducer token per layer after all O chunks:
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

Debug findings:

```text
1. Program memory overflow was the real compile failure.
   It was diagnosed with llvm-size and llvm-nm, not by changing TAPs or FIFO
   depth.

2. The first numeric failure was metadata aliasing.
   packet[0] already meant residual_row_base. Reusing it as ffn_row_base made
   every lane write FFN rows 0..3. The fix stores ffn_row_base in the final
   gate packet metadata slot.
```

Detailed debug notes:

```text
how-to-debug/qwen3-megakernel/symptoms-resources.md
how-to-debug/qwen3-megakernel/symptoms-numeric.md
```

## D2. 28-Layer Decode Body

Acceptance:

```text
token_match=True on default and Fibonacci prompts
multi-token run has no hot-loop compile if dynamic position is claimed
measured token time beats the accepted persistent baseline
failure diagnostics are written to how-to-debug before changing strategy
```

## D3. End-To-End Boundary Inventory

This is required before calling the result a full inference architecture.

| boundary | data | owner | frequency | measured cost |
| --- | --- | --- | --- | --- |
| tokenizer -> embedding | token id | CPU | per token | TBD |
| embedding -> layer0 | hidden[1024] | CPU/XRT/NPU | per token | TBD |
| decode body -> final norm | hidden[1024] | NPU->CPU | per token | TBD |
| final norm -> LM head | hidden[1024] | CPU | per token | TBD |
| LM head -> argmax | logits | CPU | per token | TBD |
| KV cache read | per-layer K/V | NPU DMA | per token/layer | TBD |
| KV cache write | current K/V | host memcpy | per token/layer | TBD |

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
1. D1.5zz: remove host_ffn_hidden[256:3072], preferably by down partial reduce
   or a streaming down accumulation plan that does not add lane program text.
2. Keep final norm / LM head on CPU until the decode body is faster than the
   accepted persistent baseline.
3. Do not add standalone production ops or static single-layer graphs.
4. Check program memory, FIFO endpoints, L1 object sizes, and routing as
   separate resource budgets.
5. Measure whether lane Workers or reducer Workers are compute-bound before
   adding more columns.
```
