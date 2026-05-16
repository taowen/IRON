<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Scale Metadata

Used during SageAttention bring-up.

## Goal

Move Q/K dequant scales out of a compile-time AIE `Buffer` and into runtime
metadata. Recompiling for every input scale table is not a real operator.

## Failed Attempt: `write_rtp` From Runtime Data

The first attempt used `Buffer(..., use_write_rtp=True)` and tried to copy from
the runtime sequence scale argument:

```python
scale_buffer[scale_idx] = scale_data.op[scale_idx]
```

MLIR generation failed:

```text
RuntimeError: std::bad_cast
```

`npu_write_rtp` expects an immediate attribute-like value. It cannot take a
runtime memref load as the value.

## Failed Attempt: Plain AIE Buffer Store

The second attempt removed `use_write_rtp=True` and tried a normal store into
the AIE buffer from the runtime sequence.

Verification failed:

```text
'aie.buffer' op is accessed outside of a tile
```

Plain tile-local `aie.buffer` stores are not valid from the runtime sequence.

## Fix Used

Move dequant from the QK worker to the softmax worker:

```text
runtime scale table
  -> small ObjectFIFO per query pipeline
  -> softmax worker

QK worker:
  Q_i8 x K_i8 -> int32 scores

softmax worker:
  int32 scores * runtime scale -> bf16 logits
  online softmax
```

This avoids adding a third input stream to the QK tile, which had already hit
input DMA resource limits. The softmax tile already consumes the QK score FIFO
and can also consume the small scale FIFO.

## Tradeoff

The scale metadata is now a true runtime input, but QK-to-softmax traffic is
currently int32 instead of bf16. That increased data movement and reduced the
speedup compared with the compile-time scale prototype.

The current direction is correct for operator semantics. A later optimization
should reduce the traffic, for example by packing scale with a nearby tile
descriptor or by fusing dequant without reintroducing QK tile DMA pressure.
