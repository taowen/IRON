<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# D1: Single Qwen3 Layer Integration

Status: `in progress`

D1 is the first real Qwen3 integration step for the new-mega architecture. It
must be built incrementally; D0 proved only a resource skeleton, not real layer
math.

## D1.0: Real Fixed-Cache Attention Context

Status: `accepted`

Question:

```text
Can the A0 fixed-cache attention mechanism run on real Qwen3 layer-0 tensors
and produce the same attention context as the PyTorch/reference path?
```

This tests the most important new-mega boundary before adding QKV projection,
O projection, and MLP:

```text
host computes/owns current K/V writeback
NPU reads full fixed cache shape
runtime mask marks live positions
NPU runs chunked online softmax + PV for all Q heads
```

It is deliberately not yet a full layer. The D1.0 inputs are real Qwen3
reference tensors:

```text
Q after q_norm + RoPE:             [16, 128]
K cache after host current write:  [8, max_seq_len, 128]
V cache after host current write:  [8, max_seq_len, 128]
mask:                              [max_seq_len]
```

The NPU output is:

```text
attention context: [16, 128]
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/d1_single_layer/run.py
```

## Acceptance

```text
preflight passes
full aiecc passes
NPU run completes
context output matches reference under the documented tolerance
artifact is reused for the prompt-derived decode position through runtime mask data
```

## Result

Accepted on NPU2 for two prompts with the same fixed artifact shape:

```text
artifact:
  max_seq_len=256
  chunk_size=64
  q_heads=16
  kv_heads=8
  head_dim=128
  packed_chunk_bytes=32896
  packed_stream_elements=1052672

preflight:
  runtime_memrefs=3
  arg_specs=3
  compute_cores=1
  max_fifo_buffered_bytes=32896
  total_dma_tasks=3
  max_dma_tasks_per_fifo=1
  max_compute_tile_inputs=2
  max_compute_tile_outputs=1
  non_advancing_acquires=0
```

Default prompt:

```text
prompt_tokens: 26
decode_position: 26
prompt_next_token: 33067
npu_time_us: 3756.100
d1_attention_context_max_abs: 0.015625
d1_attention_context_mean_abs: 0.000455
d1_attention_context_errors: 0
decision: accepted
```

Second prompt:

```text
prompt: Name the largest planet. Answer with one word.
prompt_tokens: 22
decode_position: 22
prompt_next_token: 2677
npu_time_us: 3451.432
d1_attention_context_max_abs: 0.019531
d1_attention_context_mean_abs: 0.000479
d1_attention_context_errors: 0
decision: accepted
```

Conclusion:

```text
The fixed max-cache + runtime mask attention read strategy works on real Qwen3
layer-0 tensors for all 16 Q heads. Host-owned current K/V writeback is
compatible with the NPU attention read path.
```

Production promotion:

```text
This boundary is now implemented in:

  iron/applications/new-mega/production

Run:

  python -X faulthandler iron/applications/new-mega/production/main.py
```

## Next Substeps

D1.1a accepted in production:

```text
stage: qkv-rope-present
input:  q_raw, k_raw, v_raw, q_norm_weight, k_norm_weight, rope_lut
output: q_rope, present_k, present_v

default prompt position=26:
  npu_time_us=760.438
  max_abs=0.031250
  errors=0

second prompt position=22:
  npu_time_us=750.058
  max_abs=0.062500
  errors=0
```

D1.1b accepted in production:

```text
stage: qkv-rope-attention
boundary: qkv-rope-present -> host K/V writeback -> fixed-attention

default prompt position=26:
  qkv_npu_time_us=871.263
  attention_npu_time_us=4386.955
  qkv_errors=0
  context_max_abs=0.015625
  context_errors=0

second prompt position=22:
  qkv_npu_time_us=852.138
  attention_npu_time_us=3342.268
  qkv_errors=0
  context_max_abs=0.019531
  context_errors=0
```

D1.1c accepted in production:

```text
stage: qkv-rope-attention-present
boundary: qkv-rope-present -> in-dispatch attention with present K/V
host cache writeback: after dispatch only

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

D1.2 accepted in production:

```text
stage: qkv-rope-attention-o
boundary: accepted attention context -> O projection GEMV -> host residual add
O projection: GEMV M=1024 K=2048 columns=4

default prompt position=26:
  o_proj_npu_time_us=1098.267
  attn_out_max_abs=0.005859
  attn_out_errors=0
  attn_residual_max_abs=0.005859
  attn_residual_errors=0

second prompt position=22:
  o_proj_npu_time_us=970.969
  attn_out_max_abs=0.007812
  attn_out_errors=0
  attn_residual_max_abs=0.007812
  attn_residual_errors=0
```

D1.2c accepted in production:

```text
stage: qkv-rope-attention-o-fused
boundary: qkv/rope -> attention with present K/V -> O projection -> residual
dispatches: 1

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

D1.3c accepted in production:

```text
stage: qkv-rope-attention-o-mlp-fused
boundary: qkv/rope -> attention with present K/V -> O projection ->
          post-attention RMSNorm -> MLP -> layer residual
dispatches: 1

default prompt position=26:
  npu_time_us=10821.846
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0

second prompt position=19:
  prompt="What is the capital of France?"
  npu_time_us=10284.365
  current_errors=0
  attn_out_errors=0
  ffn_hidden_errors=0
  ffn_out_errors=0
  layer_residual_errors=0
```

D1.4b accepted in production:

```text
stage: qkv-rope-attention-o-mlp-fused
boundary: hidden input -> input RMSNorm -> Q/K/V projection -> Q/K norm+RoPE ->
          attention with present K/V -> O projection -> post-attention RMSNorm ->
          MLP -> layer residual
dispatches: 1
runtime_memrefs=5
preflight_compute_cores=25
preflight_total_dma_tasks=21

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

Implementation note:

```text
The first D1.3c shape exceeded output DMA channels by producing too many debug
and production FIFOs from one MLP Worker. The accepted graph splits post-norm
from gate/up and packs gate/up row groups into one input FIFO so every compute
tile stays within two input and two output ObjectFIFOs.

D1.4b removed the standalone input-qkv-rope-present diagnostic op. Adding a
separate Q/K/V-weight memref would have produced six runtime memrefs, exceeding
the five HOST BO slots exposed by metadata, so input metadata and Q/K/V weights
are packed into one first buffer.
```

Remaining substeps:

```text
D1.4c: add final RMSNorm/logit boundary check for a complete single layer
D2: extend the accepted layer graph toward repeated layers
```
