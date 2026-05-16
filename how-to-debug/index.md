<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# How to Debug

This directory records diagnostics that were actually used while bringing up
the IRON SageAttention path and the Qwen3-0.6B decode megakernel prototype.
It is organized by observed symptom and proven diagnostic method, not by the
order in which work happened.

Start with [triage](triage.md) when a run fails. It maps symptoms to the first
check that has already found a real bug in this tree.

## Current Scope

SageAttention is a narrow SAGEAttn-B-style bring-up:

```text
NPU2, single head, d=64, 64x64 tiles
host-side K smoothing and Q/K quantization
runtime dequant scale metadata to the softmax Worker
int8 x int8 -> int32 QK, then softmax-side dequant
ObjectFIFO forward DMA for blocked-score layout conversion
online softmax and bf16 PV as separate pipeline stages
```

It has shown a small speedup over the existing bf16 MHA test at `seq_len=256`,
but it is not a full SageAttention implementation yet.

Qwen3 has two active code paths:

```text
qwen3_megakernel.py   full-ELF FusedMLIROperator correctness scaffold
qwen3_persistent.py   hand-authored IRON Program/Worker/ObjectFifo bring-up
```

The persistent path has accepted checkpoints through
`input-rmsnorm-qkv-rope-cache-scores-softmax-context`.

## SageAttention Notes

Use these when the failing symptom is in the standalone SageAttention operator:

| Symptom class | Note |
| --- | --- |
| MLIR generation fails before kernel compile | [MLIR generation failures](mlir-generation.md) |
| `aiecc` reports tile, DMA, or BD resource failure | [AIECC resource allocation failures](aiecc-resource-allocation.md) |
| C++/Peano rejects AIE API vector code | [AIE kernel compile failures](kernel-compile.md) |
| Correct-looking values are consumed in the wrong order | [ObjectFIFO layout mistakes](objectfifo-layout.md) |
| Runtime scale metadata does not fit the obvious path | [Runtime scale metadata](runtime-scale-metadata.md) |
| Quantized output must be checked against bf16 attention | [Accuracy validation](accuracy-validation.md) |
| Correct implementation is slower than baseline | [Performance comparison failures](performance-comparison.md) |
| Timing has large unexplained outliers | [Runtime isolation for performance tests](runtime-isolation.md) |

## Qwen3 Megakernel Notes

Use these when the failing symptom is in full-ELF fusion or the persistent
Qwen3 Program:

- [Qwen3 megakernel debug map](qwen3-megakernel/index.md)
- [Qwen3 symptom lookup](qwen3-megakernel/symptoms.md)
- [Qwen3 diagnostic methods](qwen3-megakernel/diagnostic-methods.md)
- [Qwen3 lessons and next preflight checks](qwen3-megakernel/lessons.md)
- [Compatibility entry](qwen3-megakernel.md)

## Rule For New Notes

Do not add a generic debugging idea here until it has diagnosed a real failure
in this repository. A new entry should name the symptom, the command or check
used, the evidence found, the root cause, and the recheck that proved the fix.
