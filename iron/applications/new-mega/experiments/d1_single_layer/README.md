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

After D1.0:

```text
D1.1: add NPU-side QKV/RoPE and fixed present K/V outputs
D1.2: add O projection + residual
D1.3: add post-attention RMSNorm + MLP
D1.4: run full single-layer output check
```
