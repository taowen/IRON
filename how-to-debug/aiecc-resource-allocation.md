<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIECC Resource Allocation Failures

Used during SageAttention bring-up.

## Case 1: Output DMA Has Too Many Dimensions

### Symptom

`aiecc` failed after MLIR generation:

```text
'aie.dma_bd' op Cannot give more than 3 dimensions for step sizes and wraps
got 4 dimensions
```

The failing operation referenced a `memref<64x64xbf16>` output DMA with four
dimension descriptors.

### Cause

The first SageAttention design simplified the output path and drained directly
from an ObjectFIFO carrying the PV worker's tiled layout. That caused the
generated DMA BD on the selected tile to receive the full four-dimensional
`dims_to_stream` layout.

### Fix Used

Mirror the existing MHA output topology:

1. PV worker produces a tiled `qk_bf16_ty` object.
2. `ObjectFifo.prod().join(...)` joins it into a memtile-facing output FIFO.
3. Runtime drains the joined FIFO back to L3 using the output tap.

This moved the layout handling back to the topology pattern that already
compiled in `iron/operators/mha/design.py`.

## Case 2: Scale Metadata as Extra FIFO Exceeds Input DMA

### Symptom

Adding Q/K quantization scales as runtime ObjectFIFOs made `aiecc` fail with an
input DMA channel allocation error on the QK tile:

```text
'aie.tile' op number of input DMA channel exceeded!
```

### Cause

The QK worker already consumed Q and K tiles. Adding Q scale and K scale as
separate ObjectFIFOs pushed the tile over the available input DMA channels.
The shapes were small, but the resource that failed was not bytes; it was the
number of independent input DMA streams.

### Interim Fix Used

For the fixed-shape bring-up, the per-block dequant scale table is embedded as
an AIE local `Buffer`:

```python
qk_dequant_scales = Buffer(
    initial_value=np.asarray(dequant_scales, dtype=np.float32),
    name=f"sage_qk_dequant_scales_{i}",
)
```

The QK kernel indexes it by `(q_block, kv_block)`. This is not the final runtime
metadata design, but it proved the data path without adding input DMA streams.

### Current Fix Used

The scale metadata now enters as a runtime input, but not through the QK tile.
QK emits int32 scores, and the softmax Worker consumes a small scale ObjectFIFO
to dequant scores before online softmax. That moves the extra metadata stream
to a tile with enough input capacity.

## Case 3: Fusion Changes Placement Pressure

### Symptom

After fusing QK and softmax into one Worker, the first placement put the fused
worker on row 2 and PV on row 4. `aiecc` failed on the PV tile:

```text
/build/SageAttention_s128_d64_npu2.mlir:4:17: error:
'aie.tile' op number of input DMA channel exceeded!
```

### Cause

The PV tile consumed three logical inputs: P, V, and softmax scale. Moving the
scale producer from row 3 to row 2 changed the route enough for allocation to
fail. The generated MLIR showed the relevant edges:

```text
sage_outP   mem_tile_0_1 -> tile_0_4
sage_memV   mem_tile_4_1 -> tile_0_4
sage_scaleOF tile_0_2   -> tile_0_4
```

### Fix Used

Moving the fused QK+softmax worker to row 3 restored the adjacent
`scaleOF` producer/consumer placement and let the design compile. The fused
version was later dropped for performance/layout reasons, but this remains a
useful resource-placement check.
