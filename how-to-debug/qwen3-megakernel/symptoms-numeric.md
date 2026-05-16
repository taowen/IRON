<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Numeric Symptoms

[Back to symptom index](symptoms.md).

## One-Step Decode Token Is Wrong

Symptom:

```text
prompt_next_token: 33067
npu_next_token: 1172 text=' only'
ref_next_token: 11853 text='imize'
logits_max_abs: 25.687500
logits_mean_abs: 4.346396
```

Diagnostic:

```text
Do not inspect final logits first. Drain stage-local debug copies and find the
first wrong semantic tensor.
```

Useful debug outputs:

```text
x_norm
queries_raw
queries_norm
queries
keys_raw
keys_norm
keys
values
attn_scores
attn_weights
attn_context
attn_out
ffn_out
logits
```

Evidence found:

```text
values had only GEMV-level error, while queries and keys were already badly
wrong after the extra Q/K path.
```

Root cause:

```text
In-place Q/K RMSNorm and RoPE were unsafe across the fused full-ELF buffer
layout:

q_norm("queries" -> "queries")
rope_q("queries" -> "queries")
k_norm("keys" -> "keys")
rope_k("keys" -> "keys")
```

Fix:

```text
Make Q/K phases explicit:

gemv_q -> queries_raw -> q_norm -> queries_norm -> rope_q -> queries
gemv_k -> keys_raw    -> k_norm -> keys_norm    -> rope_k -> keys
```

Accepted recheck:

```text
npu_next_token: 11853 text='imize'
ref_next_token: 11853 text='imize'
logits_max_abs: 0.437500
logits_mean_abs: 0.073747
```

## Layout-Only Changes Move The Token

Symptom:

```text
A graph edit that should be layout-only changes the one-layer output token.
```

Experiments that were reverted:

```text
1. Expanding RoPE LUTs from one shared angle row to one row per head.
2. Splitting the reused x_norm buffer into attn_x_norm and mlp_x_norm.
3. Replacing multi-row RoPE with one RoPE(rows=1) call per head.
```

Diagnostic:

```text
Run a clean A/B build. In full-ELF fusion, scratch layout and patch locations
are correctness-relevant.
```

Current baseline:

```text
Q/K in-place norm/RoPE removed: keep
RoPE angle_rows=1 shared LUT: keep
Multi-row RoPE run: keep
x_norm reused between attention and MLP norm: keep
```

## QKV Numeric Errors Appear After Runtime Packing

Symptom:

```text
x_norm_errors: 0
queries_raw_errors: 59
keys_raw_errors: 26
values_errors: 34
```

Diagnostic:

```text
Use the first accepted NPU tensor as the local reference input for the next
stage. Do not compare every downstream stage directly to the full PyTorch path.
```

Check used:

```python
torch_from_npu_x = F.linear(npu_x_norm.view(1, 1, -1), W).flatten()
```

Evidence found:

```text
against_torch_from_npu_x queries_raw: max=0.000000 mean=0.000000
against_torch_from_npu_x keys_raw:    max=0.000000 mean=0.000000
against_torch_from_npu_x values:      max=0.000000 mean=0.000000
```

Root cause:

```text
The verifier used the wrong boundary. Q/K/V projection was correct for the
actual NPU x_norm it consumed. The apparent Q/K/V errors were upstream RMSNorm
drift being projected by Wq/Wk/Wv.
```

Fix:

```text
Verify Q/K/V against F.linear(actual_npu_x_norm, W). Print full-reference drift
separately.
```

Accepted recheck:

```text
x_norm_max_abs: 0.007812
x_norm_errors: 0
queries_raw_max_abs: 0.000000
queries_raw_full_ref_max_abs: 0.015625
queries_raw_errors: 0
keys_raw_max_abs: 0.000000
keys_raw_full_ref_max_abs: 0.007812
keys_raw_errors: 0
values_max_abs: 0.000000
values_full_ref_max_abs: 0.005859
values_errors: 0
```

## RoPE Outputs Fail Under GEMV Tolerance

Symptom:

```text
queries_errors: 10
keys_errors: 6
queries_max_abs: 0.125000
keys_max_abs: 2.000000
```

Diagnostic:

```text
Compare NPU RoPE against both the model-level apply_rope reference and a host
reference that follows aie_kernels/generic/rope.cc two-halves ordering.
Then check the existing RoPE operator test tolerance.
```

Evidence found:

```text
q_apply max 0.125 mean 0.004813
q_cc    max 0.125 mean 0.004813
k_apply max 2.000 mean 0.011349
k_cc    max 2.000 mean 0.011349
```

The existing RoPE operator test uses:

```text
rel_tol=0.05
abs_tol=0.5
```

Root cause:

```text
The persistent stage verifier reused the strict GEMV abs_tol=1e-6 for RoPE
outputs. The RoPE layout and LUT were correct; the verifier was using the wrong
operator-specific tolerance.
```

Fix:

```text
Use rel_tol=0.05 and abs_tol=0.5 for `queries` and `keys` in
input-rmsnorm-qkv-rope-cache. Keep the stricter GEMV-style tolerance for Q/K/V
projection and cache copy checks.
```

Accepted recheck:

```text
queries_errors: 0
keys_errors: 0
keys_cache_current_errors: 0
values_cache_current_errors: 0
keys_cache_prefix_errors: 0
values_cache_prefix_errors: 0
```

## MLP Gate/Up Fails Full Reference But Passes Local Boundary

Symptom:

```text
mlp_x_norm_errors: 0
ffn_gate_errors: 108
ffn_up_errors: 145
```

Evidence after changing only the verifier to use the actual NPU
`mlp_x_norm` as the GEMV input boundary:

```text
ffn_gate_max_abs: 0.000000
ffn_gate_errors: 0
ffn_up_max_abs: 0.000000
ffn_up_errors: 0
ffn_gate_full_ref_max_abs: 0.015625
ffn_up_full_ref_max_abs: 0.007812
```

Root cause:

```text
The gate/up GEMV workers were correct. The original check compared them to a
full PyTorch reference fed by PyTorch mlp_x_norm, while the NPU workers consume
the bf16 mlp_x_norm FIFO produced by the NPU weighted RMSNorm worker.
```

Fix:

```text
For each stage, compare the first output at the full reference boundary, then
build downstream local references from the actual NPU FIFO output that the next
Worker consumes.
```

## SiLU Negative Inputs Exceed The Positive-Only Operator Tolerance

Symptom:

```text
ffn_gate_silu_errors: 3
ffn_gate_silu_max_abs: 0.019531
Mismatch in ffn_gate_silu[1331]: expected -0.090820, got -0.100586
```

Root cause:

```text
The AIE SiLU kernel uses the tanh-form approximation. The standalone SiLU test
used random positive inputs in [0, 4), but Qwen3 gate projection produces
negative values where the approximation has a larger absolute error against
PyTorch exact SiLU.
```

Fix:

```text
Verify SiLU at the actual gate FIFO boundary and use an explicit absolute
tolerance for the approximation. Do not treat the downstream ffn_hidden as a
new multiply bug if it matches actual_silu * actual_up.
```

Recheck:

```text
ffn_gate_silu_errors: 0
ffn_hidden_errors: 0
```

