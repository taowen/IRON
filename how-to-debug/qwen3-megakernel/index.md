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

The in-progress checkpoint below compiles and runs, but is not accepted because
attention score/softmax numeric verification still fails:

```text
input-rmsnorm-qkv-rope-cache-scores-softmax
```

Current score/softmax failure boundary:

```text
input RMSNorm, QKV projection, Q/K RMSNorm, RoPE, and KV cache write pass.
Attention score/softmax still fails.
The next check is qk_pair drain/checksum before changing score math.
```

Do not debug from final logits first. Start from the symptom, prove the failing
boundary, and only then change code.
