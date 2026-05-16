<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Resources Symptoms

[Back to symptom index](symptoms.md).

## Persistent QKV Cannot Place On 8 Columns

Symptom:

```text
ValueError: Failed to find a tile matching column 0: tried until column 8.
Try using a device with more columns.
```

Diagnostic:

```text
This is a placement/resource failure from resolve_program(), not a runtime
numeric issue. Reduce the persistent stage first, then scale placement
explicitly.
```

Root cause established for bring-up:

```text
The first hand-authored QKV persistent stage was too optimistic as an 8-column
layout before resource ownership was explicit.
```

Fix used for current checkpoint:

```text
Set num_aie_columns=1 for the semantic checkpoint.
```

Status:

```text
This is not a final performance placement. It is the accepted correctness
bring-up fallback.
```

## Persistent QKV Exceeds Output DMA Channels

Symptom:

```text
error: 'aie.tile' op number of output DMA channel exceeded! %tile_0_3
```

Diagnostic:

```text
Inspect the ObjectFIFO graph around the tile named in the aiecc error. Count
producer outputs from that tile before editing kernels.
```

Evidence found:

```text
weight_worker produced four separate outputs: one debug x_norm drain plus
Q, K, and V copies.
```

Root cause:

```text
The design duplicated the same logical x_norm token as multiple producer
outputs, exhausting tile DMA resources.
```

Fix:

```text
Use one ObjectFIFO with multiple consumers:

hidden_fifo -> rmsnorm_worker -> weight_worker ->
single xnorm broadcast FIFO -> Q/K/V matvec workers and optional debug drain
```

## Attention Score Worker Exceeds Input DMA Channels

Symptom:

```text
error: 'aie.tile' op number of input DMA channel exceeded! %tile_2_3
```

Diagnostic:

```bash
rg -n "tile_2_3|objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
qwen3_rc_q_rope_0  -> tile_2_3
qwen3_rc_k_rope_0  -> tile_2_3
qwen3_rc_k_cache_0 -> tile_2_3
```

Root cause:

```text
The score worker had three input ObjectFIFOs: Q RoPE, current K RoPE, and
K-cache block. The tile input DMA channels were exhausted before runtime.
```

Fix direction:

```text
Do not build score as a three-input worker. Pack Q and current K in a prior
two-input worker, then run score as qk_pair + k_cache_block.
```

## K Cache Matrix Does Not Fit In L1

Symptom:

```text
Failed to allocate buffer: "qwen3_rc_k_cache_0_cons_buff_0" with size: 65536 bytes
allocated buffers exceeded available memory
```

Diagnostic:

```text
Read the aiecc MemoryMap. It printed two 65536-byte K-cache consumer buffers on
one compute tile, before score and Q buffers were counted.
```

Root cause:

```text
The first score design tried to move a full [256, 128] bf16 K-cache head as one
ObjectFIFO object. With depth=2 that alone needs 128KB, above the tile L1 budget.
```

Fix:

```text
Stream K cache as [64, 128] blocks. This makes each K-cache object 16KB and
keeps depth=2 within L1.
```

## Debug Pass-Through FIFO Exceeds L1

Symptom:

```text
Adding an independent K-cache debug stream makes aiecc fail allocation on the
debug worker tile, even though the production K-cache stream already fits.
```

Evidence found:

```text
The debug worker carried both input and output ObjectFIFOs with [64, 128] bf16
objects. With depth=2 on both edges, the tile needs 16KB * 4 = 64KB for debug
FIFO objects before stack and other allocations.
```

Root cause:

```text
The debug copy used normal depth=2 FIFOs on both sides. For a pass-through
diagnostic worker that does no compute overlap, double-buffering both input and
output consumed the whole tile L1 budget.
```

Fix:

```text
Set both K-cache debug pass-through FIFOs to depth=1. The diagnostic still
proves the runtime TAP/stream order, while production K-cache keeps depth=2 for
the score worker.
```

## K Cache Block DMA Exhausts BD IDs

Symptom:

```text
Allocator exhausted available buffer descriptor IDs
```

Diagnostic:

```bash
rg -n "dma_configure_task_for @qwen3_rc_k_cache_0" build_qwen3_persistent/*.mlir
```

Evidence found:

```text
One Python rt.fill per q-head/cache-block generated 64 DMA tasks for the same
K-cache FIFO.
```

Root cause:

```text
The logical tiling was correct, but the runtime expression used too many
separate DMA tasks. BD exhaustion is a runtime/tap expression bug, not a math
kernel bug.
```

Fix direction:

```text
Prefer one multidimensional TensorAccessPattern for the repeated block stream.
If repeated GQA access would require illegal stride=0, reuse K blocks inside a
worker instead of asking DMA to reread the same block.
```

## Multidimensional TAP Is Legal But NPU BD Rejects It

Symptoms:

```text
Stride 2 must be a positive integer
Size 0 exceeds the [0:1023] range
```

Evidence found:

```text
[size = 2, stride = 0] was rejected.
[size = 8192, stride = 1] was rejected in a multidimensional BD.
```

Root cause:

```text
TAP can describe access maps that are semantically clear but illegal for NPU BD
lowering. Reusing a tensor region with stride=0 across a non-unit dimension is
not accepted, and individual BD dimension sizes must stay in range.
```

Fix:

```text
Express K-cache blocks as [kv_head, block, row, dim] with strides
[max_seq_len * head_dim, block_rows * head_dim, head_dim, 1].
```

## New Stage Exceeds SequentialPlacer Worker Capacity

Symptom:

```text
ValueError: Failed to find a tile matching column 3: tried until column 8.
Try using a device with more columns.
```

Root cause:

```text
The accepted context checkpoint already used 15 compute Workers on the current
NPU2 placement. Adding separate context-flatten, O-projection, and residual
Workers pushed the design past the 16 compute-tile budget used by the
SequentialPlacer in this environment.
```

Fix:

```text
Fuse context flatten into the context Worker and disable the older K-cache
debug copy Worker in the deeper O-projection checkpoint. Keep debug at the
new boundary instead: attn_context, attn_context_flat, attn_o_proj,
attn_residual.
```

Recheck:

```text
preflight: ok ... max_tile_inputs=2 max_tile_outputs=2
```

