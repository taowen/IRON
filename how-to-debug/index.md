<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# How to Debug

This directory records debug methods that were actually used while bringing up
the first IRON SageAttention path.

The current SageAttention bring-up is intentionally narrow:

- NPU2 only.
- Single head.
- `d=64`.
- `64x64` tiles.
- Two query-block pipelines for `seq_len >= 128`.
- Host-side K smoothing and Q/K quantization.
- Runtime dequant scale metadata is streamed to the softmax Worker through a
  small ObjectFIFO.
- NPU-side `int8 x int8 -> int32` QK emits int32 scores; the softmax Worker
  dequants to bf16 logits before online softmax.
- ObjectFIFO forward DMA performs the QK blocked-layout to softmax-layout
  conversion before dequant.
- Existing online softmax and bf16 PV are reused as separate pipeline stages.

The performance test compares this path against the existing bf16 MHA operator
with the same `seq_len=256`, `d=64`, and one MHA pipeline.

This is closer to SAGEAttn-B than the first scalar prototype, but it is still
not a full SageAttention implementation: quantization and K smoothing are
host-side, and only one head shape is covered.

Used debug notes:

- [MLIR generation failures](mlir-generation.md)
- [AIECC resource allocation failures](aiecc-resource-allocation.md)
- [ObjectFIFO layout mistakes](objectfifo-layout.md)
- [AIE kernel compile failures](kernel-compile.md)
- [Accuracy validation](accuracy-validation.md)
- [Performance comparison failures](performance-comparison.md)
- [Runtime isolation for performance tests](runtime-isolation.md)
- [Runtime scale metadata](runtime-scale-metadata.md)
