<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MLIR Generation Failures

Used during SageAttention and Qwen3 persistent graph bring-up.

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

## Symptom: Dataclass Fails Inside DesignGenerator

The synthetic persistent graph probe failed before MLIR was produced:

```text
AttributeError: 'NoneType' object has no attribute '__dict__'
  ...
  File ".../dataclasses.py", line 814, in _is_type
    ns = sys.modules.get(cls.__module__).__dict__
```

Cause:

```text
DesignGenerator imports a source file through importlib and executes it. In the
observed Python 3.14 path, that module was not registered in sys.modules before
execution. A top-level @dataclass in the design file asked dataclasses to look
up sys.modules[cls.__module__], which returned None.
```

Fix used:

```text
Do not put a top-level @dataclass helper in a file that DesignGenerator imports
as the MLIR design source. The graph probe now uses a normal MLIROperator class
with an explicit __init__ and name property.
```

## Symptom: ObjectFIFO Type Raises IndexError In np_ndarray_type_get_dtype

The graph probe reached `resolve_program()` and failed while resolving the
ObjectFIFO memref type:

```text
IndexError: tuple index out of range
  np_ndarray_type_get_dtype(...)
```

Cause:

```python
head_ty = np.ndarray[(head_dim,), bfloat16]
```

IRON's ndarray-to-memref helper expects the dtype wrapper form used by existing
operators:

```python
head_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
```

Fix used:

```text
Use np.dtype[dtype] for all IRON buffer/ObjectFIFO ndarray type annotations,
even when dtype is already ml_dtypes.bfloat16.
```
