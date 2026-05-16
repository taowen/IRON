<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Numeric Diagnostic Methods

[Back to method index](diagnostic-methods.md).

## 8. Use Local Reference And Full Reference Separately

Use when a stage after an accepted stage reports numeric mismatch.

Pattern:

```text
local reference: uses actual NPU output from the previous accepted stage
full reference: uses the all-PyTorch path and measures accumulated drift
```

For persistent QKV, the decisive local check was:

```python
torch_from_npu_x = F.linear(npu_x_norm.view(1, 1, -1), W).flatten()
```

This proved Q/K/V projection was locally exact while full-reference drift came
from upstream RMSNorm precision.

## 10. Treat Layout-Only Changes As Correctness Changes

Use when a change claims to affect only layout or scratch names.

Clean A/B recheck is required because full-ELF fusion makes layout,
patch-sites, and scratch lifetime part of correctness.

Reverted examples:

```text
RoPE LUT one row per head
separate attention and MLP x_norm buffers
one RoPE(rows=1) call per head
```

## 12. Use Operator-Specific Tolerance

Use when local reference is correct but a downstream math operator reports a
small number of numeric mismatches.

Checks used:

```text
1. Compare against a host reference that follows the external kernel's operation
   order.
2. Check that operator's existing test tolerance.
3. Only then update the persistent-stage verifier.
```

This diagnosed the RoPE mismatch. Existing RoPE tests use
`rel_tol=0.05, abs_tol=0.5`; applying GEMV's `abs_tol=1e-6` to RoPE produced
false failures.

## 18. Use Error Cardinality To Find Layout Bugs

Use when a numeric mismatch has a structured count.

Example:

```text
216 attention score mismatches = 8 odd heads * 27 valid positions
```

This points at GQA pair ordering or qk-pair packing, not at all Q heads, not at
the K-cache prefix, and not at the final output drain.

## 21. Prove PV Inputs Before Changing The Context Kernel

Use when `attn_context` is wrong.

The accepted PV/context bring-up first checked every input boundary:

```text
attn_weights_errors: 0
v_context_stream_prefix_errors: 0
v_context_stream_current_errors: 0
values_cache_prefix_errors: 0
values_cache_current_errors: 0
```

Only after those passed was the context external kernel the first unproven
boundary. This avoided guessing about V-cache layout, GQA head mapping, or
softmax output order.

## 22. Match The Accumulation Boundary

Use when all input streams pass but a reduction kernel has a small number of
large, cancellation-sensitive errors.

The first context kernel updated bf16 `context[dim]` on every sequence row:

```text
context[dim] = bf16(float(context[dim]) + float(weight) * float(v))
```

The symptom was only six `attn_context` mismatches, but with large absolute
errors on elements whose expected values were near zero. A reference using the
same NPU weights and V stream showed the inputs were correct.

The accepted kernel accumulates a 64-row block in local float and writes bf16
once per block:

```text
accum[dim] = row_base == 0 ? 0 : float(context[dim])
for row in cache_block:
  accum[dim] += float(weight[row]) * float(v[row, dim])
context[dim] = bf16(accum[dim])
```

Recheck:

```text
attn_context_errors: 0
attn_context_max_abs: 0.000000
```

## 26. Rebuild Local References From The Actual FIFO Boundary

Use when the first tensor in a stage passes but every downstream tensor has
small numeric mismatches against the full model reference.

For the MLP gate/up checkpoint, `mlp_x_norm` passed, but gate/up initially
failed against the full PyTorch reference. The decisive check was:

```python
gate_local = F.linear(actual_mlp_x_norm.view(1, 1, -1), W_gate).flatten()
up_local = F.linear(actual_mlp_x_norm.view(1, 1, -1), W_up).flatten()
```

Result:

```text
ffn_gate_errors: 0
ffn_up_errors: 0
```

That proves the active boundary is `actual_mlp_x_norm`, not the higher
precision PyTorch tensor with the same semantic name.

## 27. Test Approximation Kernels On The Model's Real Input Distribution

Use when an elementwise approximation passes its standalone operator test but
fails inside a model checkpoint.

The SiLU standalone test covered positive inputs only. Qwen3 gate values include
negative inputs, and the AIE tanh-approx SiLU produced a few values with about
0.02 absolute error against PyTorch exact SiLU.

Diagnosis:

```text
1. Compare SiLU against a local reference fed by actual ffn_gate.
2. Check whether ffn_hidden matches actual_ffn_gate_silu * actual_ffn_up.
3. If the multiply passes, the boundary is the approximation kernel, not the
   downstream elementwise multiply.
```

Accepted checkpoint evidence:

```text
ffn_gate_silu_max_abs: 0.019531
ffn_gate_silu_errors: 0
ffn_hidden_errors: 0
```

