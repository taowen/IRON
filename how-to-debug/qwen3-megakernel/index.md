<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Debug

Use these notes as a debugging map, not as a chronological log.

- [Symptom lookup](symptoms.md): known failures, evidence, root causes, fixes,
  and recheck commands.
- [Diagnostic methods](diagnostic-methods.md): reusable checks that have already
  found real bugs in this bring-up.
- [Lessons](lessons.md): design constraints and preflight checks inferred from
  the diagnosed failures.

Scope:

```text
qwen3_megakernel.py   full-ELF FusedMLIROperator correctness scaffold
qwen3_persistent.py   hand-authored IRON Program/Worker/ObjectFifo bring-up
```

The current accepted persistent checkpoints are:

```text
input-rmsnorm      hidden + norm_weight -> x_norm
input-rmsnorm-qkv  hidden + packed_weights -> packed x_norm/Q/K/V
input-rmsnorm-qkv-rope-cache
                   hidden + packed weights + rope_angles -> Q/K RoPE + KV cache write
```

The accepted score/softmax checkpoint below compiles, runs, and verifies on the
current NPU2 environment:

```text
input-rmsnorm-qkv-rope-cache-scores-softmax
input-rmsnorm-qkv-rope-cache-scores-softmax-context
input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj
post-attn-rmsnorm-mlp-gate-up
```

Score/softmax root cause that was fixed:

```text
input RMSNorm, QKV projection, Q/K RMSNorm, RoPE, and KV cache write pass.
qk_pair debug drain and independent K-cache stream debug drain pass.
Attention score failed because the score Worker acquired the same output FIFO
with acquire(1), acquire(1) before release; generated LLVM showed both logical
score outputs could point at the same FIFO object.
The fixed graph uses acquire(2) and indexed subviews, and preflight now rejects
non-advancing ObjectFIFO acquire patterns.
```

PV/context root causes that were fixed:

```text
The first V merge graph failed in aiecc resource allocation, not at runtime.
MemoryMap showed tile_3_3 held two v_cache blocks, two v_context blocks, and
one debug V block. Each block was 64x128xbf16, so the tile exceeded L1.
The accepted graph uses depth=1 for the V-cache and V-context block FIFOs.

After V stream and weights both verified, attn_context still had six large
errors. Comparing per-row bf16 accumulation and float accumulation proved the
inputs were not the failing boundary. The context kernel was fixed to
accumulate one block in local float and write bf16 once per block.
```

O-projection root causes that were fixed:

```text
The first graph had qwen3_rc_o_weight_0 consumed by a Worker but not produced
in the active Program variant. The fix was to move the Runtime.fill into the
RoPE/cache/context implementation.

The next graph exceeded the current SequentialPlacer budget: context already
used 15 Workers and the naive extension added three more. The accepted graph
fuses context flatten into the context Worker and drops the older K-cache debug
copy Worker in this deeper checkpoint.

After K-cache debug was disabled, a stale TAP was still generated with length
0. Optional debug streams now use one boolean for size, FIFO, Worker, fill,
drain, TAP, and verifier slicing.
```

MLP gate/up root causes that were diagnosed:

```text
The isolated MLP checkpoint initially showed gate/up mismatches against the
full PyTorch reference even though mlp_x_norm passed. Recomputing gate/up from
the actual NPU mlp_x_norm proved the GEMV workers and packed weights were
correct; the full-reference difference was upstream bf16 boundary drift.

The remaining SiLU mismatches were only on negative gate values. The existing
AIE SiLU kernel uses a tanh approximation, and the old standalone SiLU test
only covered positive inputs. The checkpoint now verifies SiLU with the same
local input boundary and an explicit absolute tolerance for that approximation.
```

Do not debug from final logits first. Start from the symptom, prove the failing
boundary, and only then change code.
