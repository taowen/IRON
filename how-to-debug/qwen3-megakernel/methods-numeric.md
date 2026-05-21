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

## 30. Compare Independent Host References Before Editing Kernels

Use when NPU output matches one reference but fails another.

The production phase-owned runner keeps two reference views:

```text
phase_owned_reference:
  emulates the packed phase packet stream and external kernel math

qwen3_reference:
  fills the same output slots from real Qwen3 tensors and row-shard helpers
```

When the NPU matched `qwen3_reference` but failed `phase_owned_reference`, the
kernel was already correct. Comparing only the two host references found a stale
output offset without rerunning NPU:

```text
num_diff_gt_0.5 2668
first bad output slots: K RoPE and gate/up
```

Decode each failing flat index into:

```text
layer = index // output_values_per_layer
lane  = (index % output_values_per_layer) // output_values_per_lane
slot  = index % output_values_per_lane
```

Then map `slot` to the per-lane layout. In the diagnosed case, `gate_base` still
ignored newly inserted `q_rope` and `k_rope` slots, so the reference wrote
gate/up values into the K RoPE segment. The fix was purely host reference offset
alignment; changing the external kernel would have been a blind edit.

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

## 32. Split Output Diff By Semantic Segment

Use when a production phase-owned run passes or fails with one aggregate
`max_abs` over a joined per-lane output object.

Decode the flat output into:

```text
layer = index // output_values_per_layer
lane  = (index % output_values_per_layer) // output_values_per_lane
slot  = index % output_values_per_lane
```

Then aggregate error by semantic segment rather than by the whole tensor. The
accepted first-head chunked score/softmax/PV run used this segment map:

```text
q
k
v
q_rope
k_rope
context
attention_residual
gate_up
residual
```

Evidence:

```text
q max 0.000000
k max 0.003906
v max 0.000122
q_rope max 0.000000
k_rope max 0.000000
context max 0.312500
attention max 0.000000
gate_up max 0.000244
residual max 0.000488
```

Interpretation:

```text
The aggregate max_abs=0.312500 came from the new context segment only. That
matched the expected tolerance boundary for AIE exp2<bfloat16> online softmax;
the surrounding dataflow and row-shard kernels remained near exact BF16 error.
```

## 33. Poison Host-Fed Slices To Prove A Real Phase Handoff

Use when a downstream phase is supposed to consume an upstream NPU-produced
activation, but the packet still carries a host reference copy for the parts not
yet fully moved on-chip.

The test is:

```text
1. Keep the upstream NPU phase unchanged.
2. Overwrite the corresponding host-fed packet slice with an impossible value.
3. Recompute the reference according to the intended NPU handoff.
4. Run NPU and compare against that reference.
```

Accepted production example:

```text
phase:
  O projection should read the lane-mapped context head from the preceding
  score/softmax/PV phase. With num_lanes=8, this covers heads 0..7.

fault injection:
  for each lane, overwrite that lane's own O-packet host_attention_context
  slice with 123.0

result:
  poison_o_host_lane_heads_npu_time_us=10760.835
  poison_o_host_lane_heads_max_abs=0.017578
  poison_o_host_lane_heads_mean_abs=0.000329
  poison_o_host_lane_heads_errors_gt_0_5=0

After extending each lane to two context heads:

  overwrite both host-fed context slices claimed by the lane

  poison_o_host_two_lane_heads_npu_time_us=11171.860
  poison_o_host_two_lane_heads_max_abs=0.017578
  poison_o_host_two_lane_heads_mean_abs=0.000309
  poison_o_host_two_lane_heads_errors_gt_0_5=0

After wiring O output into gate/up for local residual rows:

  overwrite each lane's own host-fed gate_up attn_residual rows

  poison_gate_host_local_residual_npu_time_us=11293.627
  poison_gate_host_local_residual_max_abs=0.017578
  poison_gate_host_local_residual_mean_abs=0.000312
  poison_gate_host_local_residual_errors_gt_0_5=0

After wiring gate/up output into down_proj for local FFN hidden rows:

  overwrite each lane's own host-fed down_proj ffn_hidden rows

  poison_down_host_local_ffn_npu_time_us=320777.251
  poison_down_host_local_ffn_max_abs=0.437500
  poison_down_host_local_ffn_mean_abs=0.007681
  poison_down_host_local_ffn_errors_gt_0_5=0
```

Interpretation:

```text
If the O kernel still used the host-fed context slice for a lane, this poison
would produce a large O-projection error in that lane's rows. Passing the
poisoned run proves the O phase reads the lane-local context segment written by
the previous attention phase.

The same pattern applies to O->MLP handoff. If `gate_up` still reads the
host-fed local residual rows, poisoning those rows would move gate/up outputs by
a large amount. Passing the poisoned run proves those local rows are sourced
from the previous O phase.

It also applies to MLP-internal handoff. If `down_proj` still reads the
host-fed local `ffn_hidden` rows, poisoning those rows would move every dense
down row. Passing the poisoned run proves the local FFN rows are sourced from
the previous `gate_up` phase. It does not prove full FFN coverage; non-local
FFN rows remain host-fed until the design adds full-vector gather/broadcast or
partial projection reduce.
```

This catches a common false-positive in megakernel development: the aggregate
output matches because the host packet still contains the correct intermediate,
not because the on-chip phase handoff is correct.

## 34. Check Packet Reference Against Semantic Reference Before Editing Kernels

Use when the NPU matches the full model/semantic reference, but the packet-level
phase reference reports a small number of mismatches.

The production O partial-reduce bring-up produced this first accepted run:

```text
qwen3_phase_output_errors=0 at abs_tol=0.5
phase_owned_errors=1 at abs_tol=0.5
```

The bad slot decoded to:

```text
layer 25
lane 3
segment attention_residual
row 1
actual = 182.0
phase_owned_reference = 181.0
qwen3_reference = 182.0
```

The decisive diagnostic was host-only:

```text
compare phase_owned_reference against qwen3_reference without running NPU
```

It reproduced the same single difference:

```text
bad_count=1
max_abs=1.0
same layer/lane/slot
```

Interpretation:

```text
The NPU dataflow and kernel matched the semantic Qwen reference. The mismatch
was a packet-reference/reduction-boundary BF16 ULP issue. Rewriting the O
kernel or the ObjectFifo graph would have been a blind edit.
```

Fix used:

```text
Keep the semantic Qwen reference gate strict at abs_tol=0.5.
Allow the packet-level phase-owned reduce boundary one BF16 ULP:
  phase_abs_tol = max(abs_tol, 1.0)
```

Rule:

```text
When two references disagree, debug the references first. Only edit the NPU
kernel after the packet reference and semantic reference agree on which value
the kernel should produce.
```
