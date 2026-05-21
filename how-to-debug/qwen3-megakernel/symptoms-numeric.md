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
In-place Q/K RMSNorm and RoPE were unsafe across the shared scratch buffer
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
Run a clean A/B build. In this dataflow, scratch layout and buffer lifetime are
correctness-relevant.
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

## Production Phase Reference Fails But Qwen3 Reference Passes

Symptom:

```text
phase_owned_max_abs: 17.921875
phase_owned_errors: 2668
qwen3_phase_output_max_abs: 0.003906
qwen3_phase_output_errors: 0
```

Diagnostic:

```text
Do not change the NPU kernel first. Compare the two host references using the
same packed phase packets:

phase_owned_reference(packet stream) vs qwen3_reference(real model packets)
```

Evidence found:

```text
num_diff_gt_0.5 2668
first bad indices:
  layer 0 lane 0 pos 33
  layer 0 lane 0 pos 34
  layer 0 lane 0 pos 35
  layer 0 lane 0 pos 48
```

With `output_values_per_lane=64`, those positions decode as:

```text
0..7   Q shard
8..15  K shard
16..23 V shard
24..31 Q RoPE shard
32..39 K RoPE shard
40..47 attention residual shard
48..55 gate/up shard
56..63 layer residual shard
```

Root cause:

```text
The NPU output and Qwen3 reference were correct. The old
phase_owned_reference gate_base still used:

q + k + v + attention

after q_rope and k_rope output slots had been inserted. It wrote gate/up
reference values into the K RoPE slots and left the real gate/up slots as zero.
```

Fix:

```text
Update every host reference base offset when a per-lane output segment is added:

gate_base = q + k + v + q_rope + k_rope + attention
down_base = q + k + v + q_rope + k_rope + attention + gate_up
```

Accepted recheck:

```text
host-only reference compare:
  num_diff_gt_0.5 0 max 0.0 mean 0.0

NPU run:
  phase_owned_max_abs: 0.003906
  phase_owned_errors: 0
  qwen3_phase_output_max_abs: 0.003906
  qwen3_phase_output_errors: 0
```

## BF16 Output Differs By One ULP Until Rounding Mode Is Set

Symptom:

```text
present_k_max_abs: 0.003906
present_v_max_abs: 0.003906
max_summary_abs: 0.062500
```

Context:

```text
The A0B host-side KV writeback experiment had correct dataflow:
same artifact across 70 dispatches, host wrote present K/V into the packed
cache, and the next invocation read those rows back. The only failure was a
small BF16-valued numeric mismatch.
```

Diagnostics used:

```text
1. Keep the host writeback path unchanged.
2. Change the CPU reference from torch.sum to the same sequential row order as
   the AIE kernel.
3. Re-run. The error stayed at 0.062500, so reduction order was not the cause.
4. Inspect the BF16 store boundary in the C++ kernel.
```

Root cause:

```text
The C++ kernel cast float values to bfloat16 without explicitly setting the AIE
rounding mode. That did not match the CPU reference's BF16 conversion.
```

Fix:

```cpp
::aie::set_rounding(aie::rounding_mode::conv_even);
```

Add this before BF16-producing arithmetic or final BF16 stores in the external
kernel.

Accepted recheck:

```text
max_present_k_abs: 0.000000
max_present_v_abs: 0.000000
max_summary_abs: 0.000000
decision: accepted
```

## Full-Layer Attention Residual Fails Strict Full-Reference Tolerance

Symptom:

```text
attn_residual_max_abs: 0.011719
attn_residual_mean_abs: 0.002365
attn_residual_errors: 106
```

Diagnostic:

```text
Run the accepted O-projection checkpoint on the same prompt/position and compare
its full-reference drift before changing the full-layer kernels.
```

Evidence found:

```text
O-proj checkpoint:
attn_residual_full_ref_max_abs: 0.015625
attn_residual_errors: 0

Full-layer checkpoint:
attn_residual_full_ref_max_abs: 0.011719
```

Root cause:

```text
The full-layer checkpoint no longer drains attn_o_proj, so the verifier cannot
construct the local reference attn_residual = actual_hidden + actual_o_proj.
It was comparing directly against the full PyTorch reference with abs_tol=1e-6,
which is stricter than the already accepted bf16 O-projection full-reference
drift.
```

Fix:

```text
Use the measured O-projection bf16 boundary tolerance for full-layer
attn_residual: rel_tol=0.04, abs_tol=0.016.
```

Accepted recheck:

```text
attn_residual_errors: 0
ffn_hidden_errors: 0
ffn_out_errors: 0
layer_residual_errors: 0
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

## Full-Depth Multi-Layer Hidden Fails After Short Ladder Passes

Symptom:

```text
multi-layer-full-layer --num-layers 4  --reference-num-layers 28 -> hidden_after_layers_errors: 0
multi-layer-full-layer --num-layers 8  --reference-num-layers 28 -> hidden_after_layers_errors: 0
multi-layer-full-layer --num-layers 12 --reference-num-layers 28 -> hidden_after_layers_errors: 0
multi-layer-full-layer --num-layers 16 --reference-num-layers 28 -> hidden_after_layers_errors: 0
multi-layer-full-layer --num-layers 18 --reference-num-layers 28 -> layer_17 V current errors
```

Diagnostic:

```bash
source /opt/xilinx/xrt/setup.sh
PYTHONUNBUFFERED=1 .venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --stage multi-layer-full-layer \
  --num-layers 18 \
  --reference-num-layers 28 \
  --verify \
  --diagnose-depth \
  --build-dir build_qwen3_persistent_multilayer
```

Use `PYTHONUNBUFFERED=1` when a long run exits abruptly; otherwise the last
printed phase can be lost to stdout buffering.

Evidence found:

```text
preflight: ok
layer_17_residual_add_errors: 0
layer_17_ffn_out_local_errors: 0
layer_17_values_cache_vs_context_current_errors: 0
layer_17_keys_cache_current_errors: 0
layer_17_v_context_stream_current_errors: 14
layer_17_values_cache_current_errors: 14
layer_17_qkv_diagnostic_bundle: build_qwen3_persistent_multilayer/diagnostics/qkv_boundary_layer_17.npz
```

Then run the emitted QKV bundle in a separate process:

```bash
source /opt/xilinx/xrt/setup.sh
PYTHONUNBUFFERED=1 .venv/bin/python iron/applications/qwen3_0_6b/persistent/main.py \
  --model Qwen/Qwen3-0.6B \
  --qkv-diagnostic-bundle build_qwen3_persistent_multilayer/diagnostics/qkv_boundary_layer_17.npz \
  --build-dir build_qwen3_persistent_multilayer
```

Evidence found:

```text
layer_17_diag_qkv_values_local_errors: 0
layer_17_diag_full_v_vs_qkv_values_errors: 0
layer_17_diag_full_v_vs_py_ref_values_errors: 14
qkv_diagnostic_result: layer_17_diag_py_ref_drift
```

Root cause:

```text
The full-layer V stream and cache writeback are consistent with the standalone
QKV operator. The mismatch is against a PyTorch RMSNorm-derived reference at a
strict V tolerance after bf16/NPU approximation, not a V FIFO/cache dataflow
bug.
```

False leads ruled out:

```text
cache writeback DMA/TAP: values_cache_current == v_context_stream_current
full-layer V FIFO/dataflow: full V == isolated QKV V
standalone V GEMV: QKV V == F.linear(actual NPU x_norm, W_v)
```

## N-Layer Final-Only Cache Fails Full Reference But Debug Path Passes Local Boundary

Symptom:

```text
n-layer-final-only --layer-chunk-size 4 --verify
chunk_hidden_errors: 0
layer2_values_cache_current_errors: 3
layer3_values_cache_current_errors: 123
```

The same compile passed static checks:

```text
preflight: ok runtime_memrefs=5 metadata_host_bos=5
compute_cores=21 max_dma_tasks_per_fifo=4 non_advancing_acquires=0
```

Diagnostic:

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

Evidence found:

```text
layer_2_values_cache_vs_context_current_errors: 0
layer_2_values_cache_current_errors: 0
layer_3_values_cache_vs_context_current_errors: 0
layer_3_values_cache_current_errors: 0
hidden_after_layers_errors: 0
first_failure: none
first_full_ref_drift: layer_0_ffn_hidden_full_ref
```

Root cause:

```text
The final-only n-layer verifier compared deeper layer cache values directly
against the full PyTorch reference. After multiple NPU bf16/approximation
boundaries, the hidden entering layer 2/3 has already drifted enough that strict
V-cache full-reference tolerance is not the right boundary check. The debug
full-layer ladder recomputed local references from actual NPU hidden and showed
cache writeback and V stream were consistent.
```

Fix used:

```text
Keep strict cache checks for shallow layers, but scale final-only V-cache
absolute tolerance by layer depth when checking against the full reference.
Use the debug full-layer ladder for dataflow proof when the final-only path has
no intermediate drains.
```

Recheck:

```text
n-layer-final-only --layer-chunk-size 4 --verify:
layer2_values_cache_current_errors: 0
layer3_values_cache_current_errors: 0

generate --fast-generate --layer-chunk-size 4 --verify-generate:
token_match: True for positions 6 and 7
```

## Attention2 N-Layer Hidden Fails Strict Full Reference But Boundary Replay Matches

Symptom:

```text
n-layer-final-only attention_columns=2 mlp_gate_up_columns=2:

layer_iterations=18:
  chunk_hidden_errors=0
  layer17_values_cache_current_errors=77 against full reference

layer_iterations=19:
  chunk_hidden_errors=1
  first error: index=574 expected=-5.406250 got=-6.187500
```

Diagnostic:

```text
Do not rewrite TAP or placement after preflight passes. First prove whether the
failing deep-layer value is wrong relative to the actual NPU layer input.
```

Commands used:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only --verify \
  --layer-chunk-size 19 \
  --num-aie-columns 2 --attention-columns 2 --mlp-gate-up-columns 2 \
  --diagnose-nlayer-layer 17 \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers19_diag17

PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --qkv-diagnostic-bundle \
  build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers19_diag17/diagnostics/qkv_boundary_layer_17.npz \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers19_diag17_qkv

PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only --verify \
  --layer-chunk-size 19 \
  --num-aie-columns 2 --attention-columns 2 --mlp-gate-up-columns 2 \
  --diagnose-nlayer-layer 18 \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers19_diag18
```

Evidence found:

```text
layer17:
  layer_17_diag_qkv_values_local_errors: 0
  layer_17_diag_full_v_vs_qkv_values_errors: 0
  qkv_diagnostic_result: layer_17_diag_py_ref_drift

layer18:
  layer_18_diag_qkv_values_local_errors: 0
  layer_18_diag_full_v_vs_qkv_values_errors: 0
  qkv_diagnostic_result: layer_18_diag_py_ref_drift

boundary replay:
  layer18_diag_main_vs_single_layer_hidden_errors: 0
  layer18_diag_single_layer_hidden_vs_local_ref_errors: 5
  layer18_diag_single_layer_hidden_vs_local_ref_max_abs: 0.125000
```

Root cause:

```text
The deep V cache values are locally correct: the n-layer V cache matches an
isolated QKV operator run from the actual prefix NPU hidden. The layer18 output
is also locally correct: replaying layer18 as a single-layer boundary from the
prefix=18 NPU hidden matches the main n-layer19 output exactly.

The remaining one-element chunk_hidden failure is accumulated full-reference
numeric drift crossing the current strict tolerance, not an ObjectFIFO, TAP,
writeback, or persistent routing bug.
```

Fix direction:

```text
Keep the local-boundary diagnostic as the correctness gate for this branch.
Next validate generated tokens for attention2 + full MLP2 instead of blocking
performance work on a single strict full-reference element.
```

## Attention Probe Returns NaN After Metadata Stream Change

Symptom:

```text
--attention-probe-only returns, but:
  attention_probe_residual_max_abs: nan
```

Diagnostic used:

```text
Run the attention-only boundary before changing MLP or final-layer code. If
attention already returns NaN, the later full-layer timeout is a downstream
effect, not proof that MLP placement or SiLU is the root cause.
```

Evidence from runtime-position metadata attempt:

```text
attention_probe_residual_max_abs: nan
layer0_keys_cache_current_errors: 802
```

Rejected interpretation:

```text
"The dynamic metadata path compiled, so it is probably correct and the timeout
is just MLP."
```

Actual conclusion:

```text
The attention boundary was already corrupted. Restore the last token-correct
attention path before optimizing MLP or adding more metadata FIFOs.
```

## Real Phase Shards Match Local BF16 But Differ From PyTorch

Symptom:

```text
Production phase-owned real Q/gate/up shards:
  phase_owned_errors: 0
  qwen3_phase_output_max_abs: 0.250000
  qwen3_phase_output_mean_abs: 0.003669

Production phase-owned real down/residual shard, before fixing the reference:
  phase_owned_errors: 0
  qwen3_phase_output_max_abs: 1.000000
  qwen3_phase_output_errors: 2
```

Diagnostic:

```text
Compare the NPU output to two references:
  1. a local reference that mirrors the AIE kernel boundary and loop order
  2. the PyTorch Qwen3 q_proj reference
```

Evidence found:

```text
The NPU output matches the local BF16 loop reference exactly.
The PyTorch q_proj and gate/up projection references differ by up to 0.25 but
stay under the operator tolerance used for these real-shape shards.

For down/residual, the only two failing elements were:
  layer 26 lane 0 residual row 2: local BF16=243.0, PyTorch=242.0
  layer 27 lane 3 residual row 1: local BF16=-33.25, PyTorch=-34.0

Reproducing without NPU by comparing phase_owned_reference against the Qwen3
reference produced the same two mismatches. That ruled out ObjectFIFO, DMA,
placement, and external-kernel ABI.
```

Root cause:

```text
This is an accumulation-boundary difference, not a dataflow or weight-layout
bug. The AIE kernels explicitly cast BF16 operands to float in scalar loops
and store BF16 output. PyTorch's BF16 linear path can use a different
accumulation/reduction implementation; the 3072-wide down projection made this
large enough to exceed the old 0.5 threshold on two residual elements.
```

Fix:

```text
Use the local BF16 boundary reference as the hard gate for the kernel.
For independent row-shard checks, generate the Qwen3 shard reference with the
same explicit contract:

  BF16 inputs -> scalar float32 accumulation -> BF16 output

Keep PyTorch Qwen3 references as model-semantic tolerance checks, not as the
strict acceptance gate for a specific AIE row-shard kernel.
```

Accepted recheck:

```text
phase_owned_max_abs: 0.000488
phase_owned_errors: 0
qwen3_phase_output_max_abs: 0.000488
qwen3_phase_output_errors: 0
```

## Chunked O Residual Has A Few One-ULP Semantic Errors

Symptom:

```text
phase_owned_errors: 0
qwen3_phase_output_max_abs: 1.000000
qwen3_phase_output_errors: 16
```

Diagnostic:

```text
Add segment-level error counts before editing kernels. A single aggregate error
count is not enough after the output object contains Q, K, V, RoPE, context,
full attention residual, gate/up, and down residual segments.
```

Evidence from D1.5w:

```text
qwen3_segment_attention_residual_errors: 16 max_abs=1.000000

No gate_up or down_residual segment errors were reported.
The packet-level phase reference had zero errors under the existing
one-BF16-ULP phase tolerance.
```

Root cause:

```text
The full O projection now runs as 32 chunks and accumulates by lane-local
context partials plus source/target reducer sums. The PyTorch/Qwen semantic
reference uses a different reduction order for the same O projection. The
difference appears only in attention_residual and is bounded by one BF16 ULP.
```

Fix:

```text
Keep the packet-level phase reference as the strict dataflow/kernel gate.
Allow one BF16 ULP for the qwen semantic attention_residual segment:

  attention_residual tolerance = max(abs_tol, 1.0)

Keep all other qwen semantic segments at the normal abs_tol.
```

Accepted recheck:

```text
phase_owned_errors: 0
qwen3_phase_output_errors: 0
```

## Down Residual Fails After FFN Partial Handoff

Symptom:

```text
phase_owned_max_abs: 33.000000
phase_owned_errors: 96
qwen3_phase_output_max_abs: 33.000000
qwen3_phase_output_errors: 144
qwen3_segment_down_residual_errors: 144 max_abs=33.000000
```

Diagnostic:

```text
Use the segment-level qwen3 stats first. The error was only in down_residual;
Q/K/V, RoPE, context, attention_residual, and gate_up had already matched.
That narrows the bug to the new gate_up -> FFN reducer -> down path.
```

Evidence found:

```text
gate_up used packet[0] as the FFN row base when writing ffn_partial.

In the full-residual production layout, packet[0] is not the lane's FFN row
base. It is residual_row_base for replacing host attention residual with
NPU-produced rows. For D1.5w full residual visibility this field is 0 for every
lane.

Result:
  all eight lanes wrote their four FFN values into rows 0..3
  reducer summed rows 0..3 across lanes
  rows 4..31 stayed zero
  down_proj consumed corrupted ffn_reduced[0:32]
```

Root cause:

```text
Two different metadata meanings were accidentally assigned to the same packet
slot:

packet[0] = residual_row_base
packet[0] = ffn_row_base
```

Fix:

```text
Keep packet[0] as residual_row_base.
Store the FFN row base in an unused fixed metadata slot:

  gate_packet[packet_elements - 1] = lane * q_rows_per_packet

Then gate_up writes:

  ffn_partial[ffn_row_base + row] = silu(gate[row]) * up[row]
```

Accepted recheck:

```text
phase_owned_max_abs: 1.000000
phase_owned_errors: 0
qwen3_phase_output_max_abs: 1.000000
qwen3_phase_output_errors: 0
```
