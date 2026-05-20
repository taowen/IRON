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

## Symptom: TaskGroup Fails To Close After Refactor

During the `attention_design.py` single-version cleanup, the real graph probe
failed before MLIR was produced:

```text
ValueError: Failed to close task groups: TaskGroup(0)
```

The useful part was the earlier exception in the traceback:

```text
NameError: name 'layer_chunk_weight_tap' is not defined
  fill_chunk_layer_inputs(...)
```

Cause:

```text
An AST cleanup removed a nested helper that was not referenced by the parent
function body, but it was still referenced by another nested function. The
TaskGroup error was a cleanup side effect while unwinding the Runtime sequence,
not the root cause.
```

Fix used:

```text
When deleting nested helper functions from an IRON design, collect references
from nested function bodies too. Then re-run a preflight-only real graph probe
before invoking aiecc.
```

## Symptom: aiecc Cannot Copy An External Object

Generate reached `aiecc`, then failed while compiling the first n-layer generate
graph:

```text
Error: could not copy .../build_qwen3_persistent_generate_check/mul.o to ...
No such file or directory
```

The generated MLIR showed the real cause:

```text
func.func private @eltwise_mul_bf16_vector(...) attributes {link_with = "mul.o"}
```

But the active operator generated this artifact instead:

```text
build_qwen3_persistent_generate_check/qwen3_persistent_mul.o
```

Cause:

```text
After flattening the n-layer operator, one Kernel declaration in
attention_design.py still hardcoded "mul.o" instead of using the
mul_kernel_object parameter passed by Qwen3PersistentNLayerFinalOnly.
```

Fix used:

```text
Search generated MLIR/input_with_addresses.mlir for link_with and compare every
object name against files present in the build directory. Kernel declarations in
design.py/attention_design.py must use the operator's artifact-name parameter,
not a stale literal object filename.
```
