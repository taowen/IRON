<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Dataflow Symptoms

[Back to symptom index](symptoms.md).

## Static MLIR Shows One FIFO Drained Twice

Symptom:

```text
The design intends to write k_rope and values both to debug outputs and to KV
cache, but generated ObjectFIFO declarations show only one shim consumer.
```

Diagnostic:

```bash
rg -n "objectfifo @qwen3_rc_(v_0|k_rope_0|q_rope_0|k_raw_0|q_raw_0)" \
  build_qwen3_persistent/*.mlir
```

Evidence found:

```text
qwen3_rc_k_rope_0(%tile_2_2, {%shim_noc_tile_2_0}, ...)
qwen3_rc_v_0(%tile_1_2, {%shim_noc_tile_3_0}, ...)
```

The Python design had two `rt.drain(... k_rope_fifos[col].cons() ...)` calls
and two `rt.drain(... v_fifos[col].cons() ...)` calls. The MLIR showed this was
not a broadcast to two independent consumers; it was one consumer endpoint.

Root cause:

```text
Two runtime drains from the same FIFO consumer would consume the token stream
twice. For k_rope/v there were only 8 produced head tokens, but two drains
would require 16 consumed tokens.
```

Fix:

```text
Make cache the only drain for k_rope and values. Read semantic `keys` and
`values` for verification from the current-position cache slice instead of
adding separate debug drains.
```

Recheck:

```text
runtime_sequence kept five BOs.
k_rope and v ObjectFIFOs each had one shim consumer.
cache current-position errors were zero.
```

## Runtime Start Does Not Create A Phase Barrier

Symptom:

```text
Host runtime returns ERT_CMD_STATE_TIMEOUT after adding a second Runtime task group.
```

Diagnostic:

```bash
sed -n '350,470p' build_qwen3_persistent/*.mlir
rg -n "core|start|dma_await_task" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
All workers are emitted as always-running aie.core loops. A later rt.start()
does not create the intended "start Q path after K cache write" barrier.
```

Root cause:

```text
The attempted phase split relied on runtime start ordering, but the actual
program is a static dataflow graph with persistent cores. Q/score consumers
still participated in ObjectFIFO synchronization from the beginning.
```

Fix direction:

```text
Use explicit data dependencies, phase tokens, or a different dataflow shape.
For attention score, packing Q/current-K before score avoids the phase barrier.
```

## Attention Scores Overwrite The First GQA Head

Symptom:

```text
upstream Q/K/V, RoPE, and KV cache checks pass
attn_scores_errors: nonzero
qk_pair_errors: 0
k_cache_stream_prefix_errors: 0
```

First diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/qwen3_persistent.py \
  --model Qwen/Qwen3-0.6B \
  --stage input-rmsnorm-qkv-rope-cache-scores-softmax \
  --verify \
  --verify-repeat 1 \
  --dump-proof \
  --clean-build
```

Evidence found:

```text
The verifier drained qk_pair and an independent K-cache stream:

qk_pair_errors: 0
k_cache_stream_prefix_errors: 0
attn_scores_errors: 413

So the failing boundary is after qk_pair packing and after K-cache DMA/TAP.
```

Second diagnostic:

```bash
sed -n '145,260p' \
  build_qwen3_persistent/*.mlir.prj/main_core_2_4.peanohack.ll
```

Generated LLVM evidence:

```text
call void @llvm.aie2p.acquire(i32 48, i32 -1)
%15 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...

%25 = phi ptr ... @qwen3_rc_attn_scores_0_buff_0 ...

call void @qwen3_attention_scores_bf16(... ptr %16 ... i32 0)
call void @qwen3_attention_scores_bf16(... ptr %26 ... i32 1)
```

There is a lock acquire before the first `score_debug_fifo.acquire(1)`, but not
before the second `score_debug_fifo.acquire(1)`. Both score pointers can name
the same ObjectFIFO buffer before any release.

Root cause:

```text
The Worker treated two sequential acquire(1) calls on the same producer FIFO as
two different output tokens. IRON/ObjectFIFO acquire is stateful: if enough
objects are already acquired, a later acquire of the same or smaller size does
not acquire additional objects. The second score kernel overwrote the first GQA
head's score buffer.
```

Confirmed API rule:

```text
ObjectFifo acquire only performs new lock acquires if necessary. If one object
is already acquired, another acquire(1) in the same process returns within the
already-acquired set.
```

Fix:

```text
The score Worker now uses one acquire(2), indexes subviews [0] and [1], and
releases two objects after both q_select score rows are produced.
```

Recheck:

```text
Generated MLIR:
  aie.objectfifo.acquire @qwen3_rc_attn_scores_0(Produce, 2)
  aie.objectfifo.subview.access %2[0]
  aie.objectfifo.subview.access %2[1]
  aie.objectfifo.release @qwen3_rc_attn_scores_0(Produce, 2)

Runtime verifier:
  qk_pair_errors: 0
  k_cache_stream_prefix_errors: 0
  attn_scores_errors: 0
  attn_weights_errors: 0
  preflight: ok ... non_advancing_acquires=0
```

Follow-up verifier bug found:

```text
After score passed, six attn_weights elements still failed because the expected
weights used softmax(float32_matmul_scores). The NPU softmax consumes bf16
attn_scores from the FIFO, so the local reference must softmax the bf16-rounded
score tensor converted back to float32.
```

## PV/context Fails During L1 Buffer Allocation

Symptom:

```text
Failed to allocate buffer: "qwen3_rc_v_cache_0_cons_buff_0" with size: 16384 bytes
error: 'aie.tile' op allocated buffers exceeded available memory
tile_3_3 MemoryMap includes:
  qwen3_rc_v_context_debug_0_buff_0
  qwen3_rc_v_context_block_0_buff_0
  qwen3_rc_v_context_block_0_buff_1
  qwen3_rc_v_cache_0_cons_buff_0
  qwen3_rc_v_cache_0_cons_buff_1
```

Root cause:

```text
The V merge tile buffered too many 64x128xbf16 blocks at once. The object size
was legal, but depth=2 on both V-cache and V-context block FIFOs plus a debug
block exceeded L1.
```

Fix:

```text
Keep the 64-token V block shape, but make the V-cache and V-context block
FIFOs depth=1 so the tile streams one block at a time.
```

Recheck:

```text
preflight: ok ... max_fifo_buffered_bytes=32768 max_tile_inputs=2 max_tile_outputs=2
```

## PV/context Inputs Pass But Context Has A Few Large Errors

Symptom:

```text
attn_weights_errors: 0
v_context_stream_prefix_errors: 0
v_context_stream_current_errors: 0
attn_context_errors: 6
```

Root cause:

```text
The context external kernel did per-row bf16 read/modify/write accumulation.
That was the first unproven boundary after all input streams passed.
```

Fix:

```text
Accumulate each 64-row V block in a local float array and write bf16 context
once per block.
```

Recheck:

```text
attn_context_errors: 0
attn_context_max_abs: 0.000000
```

## New FIFO Has Consumer But No Producer

Symptom:

```text
ValueError: Prod endpoint not set for ObjectFifo(... name='qwen3_rc_o_weight_0',
prod=None, cons=[...])
```

Root cause:

```text
The O projection worker consumed qwen3_rc_o_weight_0, but the matching
Runtime.fill() was accidentally added to an earlier QKV-only design function,
not the RoPE/cache/context implementation that owns the worker.
```

Fix:

```text
Search the generated design path, not only the source text. Every new
worker-side input FIFO must have a Runtime.fill(), Worker producer, or link in
the same Program variant.
```

## Disabled Debug Stream Creates Zero-Length TAP

Symptom:

```text
ValueError: All sizes must be >= 1, but got [1, 1, 1, 0]
```

Root cause:

```text
K-cache debug output size was set to zero for the O-projection checkpoint, but
the TAP was still generated from include_scores_softmax instead of the more
specific include_k_cache_debug flag.
```

Fix:

```text
When a debug stream is optional, guard object FIFOs, workers, fills, drains,
and TAPs with the same boolean.
```

