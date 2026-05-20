<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Position Metadata Attempt Rejected

Goal: remove per-position AIE core immediates by carrying decode `position` and
`valid_length` through the existing attention2 metadata path.

## What Was Tried

The experiment extended QK/RoPE metadata with position fields and changed the
attention score/mask/context path to read those fields at runtime.

Three variants were tested:

1. `qk_pair -> current_v_metadata -> v_context` also carried position to V
   merge.
2. `k_rope -> current_v_metadata -> v_context` carried position to V merge
   without consuming `qk_pair`.
3. score/mask/context metadata only; V merge stayed static.

## Diagnostics

The first two variants compiled and passed preflight but timed out on the first
real NPU decode:

```text
HostRuntimeError: Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
```

The score/mask-only variant also timed out in full-layer generate. To isolate
the failure, `--attention-probe-only` was run with MLP removed. That graph
returned, proving the timeout was not a pure compile or placement failure, but
the attention output was already numerically bad:

```text
attention_probe_residual_max_abs: nan
layer0_keys_cache_current_errors: 802
```

Padding the metadata tail from 2 bf16 values to 8 bf16 values removed odd-sized
ObjectFIFO objects, but did not fix the NaN/full-layer timeout. A 16-byte
ObjectFIFO alignment preflight was kept because it catches a real blind spot,
but it was not the only root cause here.

## Decision

Reject this stream-widening runtime-position design.

The accepted graph was restored to static position arguments, and a real NPU
generate check passed again:

```text
token_step: 1
decode_position: 26
npu_next_token: 151645
ref_next_token: 151645
token_match: True
```

## Next Direction

Do not keep adding ObjectFIFO metadata branches blindly. The next per-position
reuse experiment must choose one of these with a proof plan:

1. patch AIE ELF/CDO immediate sites after locating them,
2. patch/runtime-select only bounded DMA offsets plus active-cache-block
   buckets,
3. use low-level control packets only after proving safe BD register addresses
   and DMA quiescence rules.

NpuControlPacketOp remains a low-level hardware escape hatch, not the default
IRON workflow: it can write tile registers, but IRON does not expose a safe
runtime BD rewrite abstraction.
