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

Clean A/B recheck is required because graph dataflow can make layout, buffer
lifetime, and patch-like runtime metadata part of correctness.

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

The A0B host-side KV writeback experiment used the same method but reached a
different conclusion. `max_summary_abs=0.062500` remained after the CPU
reference was changed to the same sequential accumulation order as the AIE
kernel, so the issue was not operator tolerance and not cache writeback. The
first bad boundary was the float-to-bfloat16 store. Adding
`aie::set_rounding(aie::rounding_mode::conv_even)` before BF16 stores made the
output exact.

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

## 28. Freeze The Full-Depth Prefill Reference For Prefix Ladders

Use when a multi-layer prefix ladder is meant to isolate the first failing
decode layer.

The wrong ladder is:

```text
--num-layers 4
--num-layers 8
--num-layers 12
```

if each run also uses `num_layers` for prefill. That changes the prefill logits,
the selected decode token, and every KV cache slice, so the runs are no longer
prefixes of one fixed full-model state.

The accepted ladder pins the prefill/token/cache reference:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage multi-layer-full-layer \
  --num-layers 16 \
  --reference-num-layers 28 \
  --verify \
  --build-dir build_qwen3_persistent_multilayer
```

Accepted evidence:

```text
num_layers=4  reference_num_layers=28 hidden_after_layers_errors: 0
num_layers=8  reference_num_layers=28 hidden_after_layers_errors: 0
num_layers=12 reference_num_layers=28 hidden_after_layers_errors: 0
num_layers=16 reference_num_layers=28 hidden_after_layers_errors: 0
num_layers=18 reference_num_layers=28 first_failure: layer_17_v_context_stream_current
```

## 29. Compare The Same Value Through Two Consumers

Use when one ObjectFIFO value feeds both a compute Worker and a Runtime drain.

For the full-layer checkpoint, `v_fifos[col]` is consumed by:

```text
v_context_merge_worker -> v_context_stream debug drain -> attention context
Runtime.drain(v_fifos[col].cons()) -> packed KV cache writeback
```

Checking only `values_cache_current` cannot tell whether the V projection
produced the wrong value, or whether the cache writeback path wrote the right
value to the wrong address. The diagnostic adds both checks:

```text
v_context_stream_current vs local V reference
values_cache_current vs local V reference
values_cache_current vs v_context_stream_current
```

Interpretation:

```text
v_context_stream_current passes, values_cache_current fails
  -> cache writeback DMA/TAP/order is the first bad boundary
both fail with the same values
  -> cache writeback is ruled out; isolate the V producer/input boundary next
values_cache_current and v_context_stream_current differ
  -> broadcast/multi-consumer FIFO or writeback path needs inspection
```

## 30. Export A Boundary Bundle And Re-run A Smaller Operator

Use when a composed full-layer checkpoint fails against PyTorch reference, but
the failing value can be recomputed by an earlier standalone operator.

Do not switch to the smaller xclbin inside the same diagnostic until tensor
lifetime is proven safe. Export a host-owned bundle, then run a fresh process:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --qkv-diagnostic-bundle build_qwen3_persistent_multilayer/diagnostics/qkv_boundary_layer_17.npz \
  --build-dir build_qwen3_persistent_multilayer
```

Evidence from layer 17:

```text
layer_17_diag_qkv_values_local_errors: 0
layer_17_diag_full_v_vs_qkv_values_errors: 0
layer_17_diag_full_v_vs_py_ref_values_errors: 14
qkv_diagnostic_result: layer_17_diag_py_ref_drift
```

Interpretation:

```text
full V == isolated QKV V
isolated QKV V == F.linear(actual NPU x_norm, W_v)
full V != F.linear(PyTorch rms_norm(hidden), W_v) within strict V tolerance
```

That rules out the full-layer V FIFO, cache writeback, and standalone V GEMV.
The active mismatch is a PyTorch-reference boundary after NPU RMSNorm/bf16
approximation, not a V dataflow bug.

## 31. Use The Debug Ladder To Validate Final-Only Chunks

Use when a final-only n-layer chunk has no intermediate drains, final hidden
passes, but a deeper layer cache/current check fails against the full reference.

The diagnostic is to run the accepted debug full-layer ladder with the same
prompt and chunk depth:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage multi-layer-full-layer \
  --num-layers 4 \
  --verify \
  --diagnose-depth \
  --build-dir build_qwen3_persistent_multilayer \
  --prompt 'Count from one to five.' \
  --raw-prompt
```

Interpretation:

```text
debug ladder local cache/current passes
  -> final-only path likely has full-reference drift; adjust the verifier
     boundary or tolerance, then prove token generation

debug ladder local cache/current fails at the same layer
  -> inspect V producer, cache writeback TAP, or hidden feedback before changing
     tolerance

final hidden fails too
  -> this is not just a cache-reference-boundary issue; start from the first
     wrong debug-drained boundary
```

Evidence from the n-layer final-only chunk=4 refactor:

```text
final-only layer3_values_cache_current_errors: 123
debug ladder layer_3_values_cache_vs_context_current_errors: 0
debug ladder layer_3_values_cache_current_errors: 0
debug ladder first_full_ref_drift: layer_0_ffn_hidden_full_ref
```

The fix was to scale final-only V-cache full-reference tolerance by layer depth
and keep the debug ladder as the dataflow proof.
