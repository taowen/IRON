<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ObjectFIFO Layout Mistakes

Used during SageAttention bring-up.

## Symptom

Fusing QK and softmax compiled and ran, but output verification failed badly:

```text
O: 5577 errors (68.08%) exceeds allowed rate of 0.50% (40 errors)
```

The QK external kernel had already worked in the non-fused path, so the first
suspect was not the int8 dot product itself.

## Cause

The non-fused path had this edge:

```python
outA = memA[i].cons().forward(
    name=f"sage_outA{i}",
    dims_to_stream=a_dims,
    depth=of_depth,
)
```

That ObjectFIFO forward was not just moving bytes. Its `a_dims` access pattern
converted the QK kernel's blocked accumulator layout into the row-major layout
expected by `partial_softmax`.

The fused experiment wrote QK logits into a local buffer and passed that buffer
directly to `partial_softmax`, bypassing the `a_dims` DMA layout conversion.
The data values were plausible, but they were read in the wrong order.

## Checks Used

Inspect the generated MLIR around the ObjectFIFO graph:

```bash
rg -n "sage_memA|sage_outA|sage_memP|sage_scaleOF|aie.objectfifo.link" \
  build/SageAttention_s128_d64_npu2.mlir
```

The healthy non-fused path contains:

```text
aie.objectfifo @sage_memA0(...)
aie.objectfifo @sage_outA0(... dimensionsToStream ...)
aie.objectfifo.link [@sage_memA0] -> [@sage_outA0]([] [0])
```

If a fused path removes that edge, the replacement kernel must explicitly
produce the layout expected by the next stage.

## Fix Used

Two fixes were tried:

1. Make the fused QK kernel write row-major logits before calling softmax.
2. Revert to the three-stage pipeline and let ObjectFIFO forward DMA perform
   the layout conversion.

The row-major fused version was correct, but slower. The current implementation
uses the second fix: QK, softmax, and PV remain separate Workers, and two
query-block pipelines run in parallel.
