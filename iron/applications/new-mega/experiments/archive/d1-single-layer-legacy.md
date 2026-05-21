<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Archived D1 Single-Layer Integration Experiments

This file was split out from `../../experiments.md` to keep the active plan short.
It covers the earlier standalone/fused single-layer direction that production no longer follows.

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

