<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MLIR Generation Failures

Used during SageAttention bring-up.

## Symptom

`pytest iron/operators/sage_attention/test.py -s -v` failed before kernel
compilation:

```text
ValueError: Operand 0 of operation "scf.if" must be a Value (is not a Value)
```

## Cause

The Worker used `aie.helpers.dialects.scf.if_()` with a Python compile-time
boolean:

```python
with if_(num_kv_blocks > 2):
    ...
```

`scf.if_()` expects an MLIR `Value`, not a normal Python `bool`.

## Fix Used

Use normal Python `if` when the condition is known at MLIR generation time:

```python
if num_kv_blocks > 2:
    for _ in range_(num_kv_blocks - 2):
        ...
```

Keep `scf.if_()` for runtime values loaded from RTP buffers or other MLIR
values, as the existing MHA operator does.
