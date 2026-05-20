<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Column Scaling Experiments

This is the active decision page for Qwen3 persistent megakernel performance
work. Keep historical command logs in `archive/`; keep this file as the current
baseline, current blocker, and next executable steps.

Archived evidence:

```text
archive/experiments-column-scaling-2026-05-21.md
archive/experiments-column-scaling-2026-05-21-attention2-mlp2.md
archive/experiments-column-scaling-2026-05-21-attention2-mlp2-multilayer.md
```

## Accepted Baseline

Fast-generate path that is known to produce correct tokens:

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: single-column
MLP layout: packed two-column segment-major weights
final norm / LM head: CPU
```

Correctness evidence:

```text
default prompt:          token_match=1/1, new_text='Paris'
Fibonacci prompt:        token_match=5/5, new_text=' 5, 8,'
weekdays prompt:         token_match=5/5, new_text=' Thursday, Friday, Saturday,'
numeric sequence prompt: token_match=5/5, new_text=' 10, 1'
```

Performance evidence from the same-build suite:

```text
default prompt:     138.077 ms, one NPU decode step before EOS
Fibonacci prompt:   mean 131.179 ms, min 130.028 ms, max 133.291 ms
weekdays prompt:    mean 121.899 ms, min 120.276 ms, max 123.286 ms
numeric sequence:   mean 125.676 ms, min 124.116 ms, max 127.984 ms
```

Resource evidence:

```text
compute_cores=26
total_dma_tasks=18
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
```

Banked improvements:

```text
8 + 8 + 8 + 4 host dispatch -> one layer_chunk_size=28 dispatch
single-column MLP -> packed two-column gate/up/down MLP
```

## Current Candidate

Candidate under test:

```text
stage: n-layer-final-only
layer_chunk_size: variable
num_aie_columns: 2
attention_columns: 2
mlp_gate_up_columns: 2
runtime input: per-layer hidden[1024] || QK/RoPE metadata[384]
MLP col0 weights: post_norm_i || gate0_i || up0_i per layer
MLP col1 weights: gate1_i || up1_i per layer
```

Why this candidate exists:

```text
attention-only attention2 is about 24% faster than attention1.
attention2 + MLP1 is slower than attention1 + full MLP2.
attention2 + full MLP2 with hidden+metadata fusion is slightly faster at
layer_iterations=1.
```

Measured timing ladder:

```text
attention-only, attention1:      warm mean 2163.765 us
attention-only, attention2:      warm mean 1640.395 us
full layer, attention1 + MLP1:   warm mean 6913.680 us
full layer, attention2 + MLP1:   warm mean 6522.273 us
full layer, attention1 + MLP2:   warm mean 4473.282 us
full layer, attention2 + down-only MLP2:
                                  warm mean 5684.301 us
full layer, attention2 + full MLP2, hidden+metadata fused:
                                  warm mean 4393.854 us
```

## Current Status

Resource scaling is no longer the blocker for this candidate. The full-depth
graph now preflights and compiles with 32 compute cores. The blocker moved to
full-depth numeric drift.

Preflight results:

```text
layer_iterations=2:  compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=4:  compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=8:  compute_cores=32 total_dma_tasks=29 max_dma_tasks_per_fifo=2
layer_iterations=12: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=16: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=18: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=19: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=20: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
layer_iterations=28: compute_cores=32 total_dma_tasks=21 max_dma_tasks_per_fifo=1
```

Verify results:

```text
layer_iterations=1:
  preflight ok
  chunk_hidden_errors=0
  current K/V cache errors=0
  warm mean 4393.854 us

layer_iterations=2:
  preflight ok
  iteration0 npu_time_us=11002.385
  iteration1 npu_time_us=7770.842
  chunk_hidden_errors=0
  current K/V cache errors=0

layer_iterations=4:
  preflight ok
  npu_time_us=18240.725
  chunk_hidden_errors=0
  layer0..3 current K/V cache errors=0

layer_iterations=8:
  preflight ok
  npu_time_us=33935.656
  chunk_hidden_errors=0
  current V cache errors=0
  strict current-K misses: layer2=2, layer4=2, layer5=3

layer_iterations=12:
  preflight ok
  npu_time_us=49852.199
  chunk_hidden_errors=0
  strict current-K miss: layer1=1

layer_iterations=16:
  preflight ok
  npu_time_us=60346.417
  chunk_hidden_errors=0
  strict current-K/V misses increase near layer14..15

layer_iterations=18:
  preflight ok
  npu_time_us=70722.295
  chunk_hidden_errors=0
  layer17_values_cache_current_errors=77 against full reference

layer_iterations=19:
  preflight ok
  repeat0 npu_time_us=76693.633
  repeat1 npu_time_us=72042.476
  chunk_hidden_errors=1 in both repeats
  first error is stable:
    index=574 expected=-5.406250 got=-6.187500

layer_iterations=20:
  preflight ok
  npu_time_us=79131.658
  chunk_hidden_errors=8
  layer17_values_cache_current_errors=77
  layer18_values_cache_current_errors=15
  layer19_values_cache_current_errors=91

layer_iterations=28:
  preflight ok
  npu_time_us=112198.634
  chunk_hidden_errors=65
  chunk_hidden_max_abs=12.000000
  K cache first strict miss: layer5
  V cache errors become substantial from layer19 onward
```

Current interpretation:

```text
The graph shape fits. Do not keep rewriting placement or TAPs blindly.
The first stable strict full-reference final-hidden failure is
layer_iterations=19. The last strict full-reference passing candidate is
layer_iterations=18. Local diagnostics now show the layer17/18 V path and the
layer18 n-layer routing are correct; the remaining failure is accumulated
NPU-vs-PyTorch numeric drift crossing the existing strict tolerance by one
element.
```

## Diagnosed Issues

### Hidden + Metadata Runtime Fusion

Symptom:

```text
attention2 + full MLP2, layer_iterations=1, failed placement by one more
host-to-NPU Runtime.fill output endpoint.
```

Fix:

```text
Pack hidden[1024] and QK/RoPE metadata[384] into one runtime input object,
then split it in IRON.
```

Evidence:

```text
placement_trace_counts runtime_output=16 runtime_input=6 other_input=1
chunk_hidden_errors=0
current K/V cache errors=0
```

### Attention2 Multi-Layer Weight Order

Symptom:

```text
layer_iterations=2 produced hidden and current K/V errors after the one-layer
shape was generalized.
```

Root cause:

```text
The packed QK/V artifact was layer-interleaved:

  qk_l0, v_l0, qk_l1, v_l1

but runtime TAPs expected segment-major per column:

  qk_col0_all_layers, v_col0_all_layers,
  qk_col1_all_layers, v_col1_all_layers
```

Fix:

```text
Pack attention2 weights in the same order that the Runtime.fill TAPs consume.
```

Recheck:

```text
test_qwen3_attention2_segment_major_weight_chunk_matches_layer_major_artifact
layer_iterations=2 verify: chunk_hidden_errors=0, current K/V errors=0
```

### Hidden/Metadata Split Backpressure

Symptom:

```text
layer_iterations=8 preflighted and compiled, but timed out at runtime.
```

Root cause:

```text
The hidden+metadata split emits one hidden child token and one metadata child
token per layer. The first fused initial/RMS worker consumed layer0 hidden and
then discarded all later hidden child tokens up front. For larger chunks this
could backpressure against the metadata child FIFO and stall the stream order.
```

Fix:

```text
Consume one unused hidden child token per subsequent layer immediately before
that layer consumes feedback. This keeps the split child streams advancing in
layer order.
```

Recheck:

```text
layer_iterations=8 no longer times out
chunk_hidden_errors=0
```

### Cache-Pair K/V Fill Rejected

Reason:

```text
The first cache-pair TAP generated a 6D BD and failed lowering:

  At most four data layout transformation dimensions may be provided

The corrected 4D single-active-block variant verified one active block but was
slower than hidden+metadata fusion and not general enough for later positions.
```

Decision:

```text
Do not pursue cache-pair fill until there is new evidence that runtime endpoint
pressure is again the limiting issue.
```

### Intermediate Chunk Guard Removed

Symptom:

```text
The documented layer12 diagnostic command failed before compile:

ValueError: Qwen3 n-layer final-only currently supports chunk sizes 1..8 or
the experimental full-depth chunk size 28
```

Root cause:

```text
The host-side operator guard was stale. The IRON design already uses the same
single-group cache DMA strategy for layer_iterations>8 that made chunk=28 fit.
The guard prevented diagnosis; it was not protecting a known compiler/runtime
failure.
```

Fix:

```text
Allow diagnostic chunk sizes 1..28 while keeping the hard model limit at 28.
```

Recheck:

```text
layer_iterations=12/16/18/19/20 all compile and preflight with
max_dma_tasks_per_fifo=1.
```

### Layer17/18 Local Boundary Diagnostics

Symptom:

```text
layer_iterations=19 is the first strict full-reference final-hidden failure:

chunk_hidden_errors=1
chunk_hidden_first_error: index=574 expected=-5.406250 got=-6.187500
```

Diagnostics used:

```bash
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

Evidence:

```text
layer17 isolated QKV:
  layer_17_diag_qkv_values_local_errors=0
  layer_17_diag_full_v_vs_qkv_values_errors=0
  qkv_diagnostic_result=layer_17_diag_py_ref_drift

layer18 isolated QKV:
  layer_18_diag_qkv_values_local_errors=0
  layer_18_diag_full_v_vs_qkv_values_errors=0
  qkv_diagnostic_result=layer_18_diag_py_ref_drift

layer18 boundary:
  layer18_diag_main_vs_single_layer_hidden_errors=0
  layer18_diag_single_layer_hidden_vs_local_ref_errors=5
  layer18_diag_single_layer_hidden_vs_local_ref_max_abs=0.125000
```

Diagnosis:

```text
The layer17/18 V cache full-reference errors are not V FIFO/TAP/writeback bugs:
the full n-layer V cache values match the isolated QKV operator exactly.

The layer18 final output is not an n-layer routing/state bug:
running layer18 as a single-layer boundary from the prefix=18 NPU hidden
matches the main n-layer19 output exactly.

The strict chunk_hidden failure at layer_iterations=19 is therefore a numeric
tolerance/full-reference drift issue, not a graph resource or ObjectFIFO
correctness issue.
```

## Next N Steps

Every step must end with a diagnosis: first-failing boundary, root cause, fix,
or an explicit rejected branch.

### 1. Locate The First Failing Full-Depth Boundary

Goal:

```text
Find exactly where the candidate changes from "final hidden correct" to
"final hidden wrong".
```

Commands:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only --verify \
  --layer-chunk-size 12 \
  --num-aie-columns 2 --attention-columns 2 --mlp-gate-up-columns 2 \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers12_verify

python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only --verify \
  --layer-chunk-size 16 \
  --num-aie-columns 2 --attention-columns 2 --mlp-gate-up-columns 2 \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers16_verify

python iron/applications/qwen3_0_6b/persistent/main.py \
  --stage n-layer-final-only --verify \
  --layer-chunk-size 20 \
  --num-aie-columns 2 --attention-columns 2 --mlp-gate-up-columns 2 \
  --build-dir build_qwen3_attncol_probe_attn2_mlp2_hiddenmeta_layers20_verify
```

Acceptance:

```text
Record the smallest layer_iterations with chunk_hidden_errors>0.
Record the first layer where current K/V errors become nontrivial.
Do not change graph structure during this step.
```

Status:

```text
done.
smallest stable chunk_hidden failure: layer_iterations=19
last passing chunk_hidden check: layer_iterations=18
first substantial full-reference V-cache miss: layer17 at layer_iterations=18/19
```

### 2. Add Local Boundary Diagnostics Around The First Bad Layer

Goal:

```text
Stop comparing only final hidden and cache writeback. Identify whether the
first real error is attention context, O projection, MLP gate/up, down
projection, or residual routing.
```

Method:

```text
Use local references from actual NPU inputs, not only full PyTorch references.
For layer_iterations=19, drain around layers 17 and 18:

  post-attention residual
  post-MLP RMSNorm
  gate/up shard outputs
  ffn_hidden
  down output
  layer residual
```

Status:

```text
done without adding IRON debug drains.
prefix chunk outputs were used as actual NPU layer inputs.
isolated QKV bundles proved layer17/18 V cache is locally correct.
single-layer boundary replay proved n-layer19 layer18 routing is locally correct.
```

### 3. Decide Whether Layer17 V Misses Are Local Or Full-Reference Drift

Goal:

```text
The first large cache symptom is layer17_values_cache_current_errors=77 while
layer_iterations=18 final hidden still passes. Determine whether layer17 V is
wrong relative to the actual NPU layer input, or only wrong relative to the
full PyTorch hidden after accumulated bf16 drift.
```

Method:

```text
Reuse the local-boundary diagnostic style from multi-layer-full-layer:
compare layer17 V against F.linear(actual NPU layer17 x_norm, W_v), and compare
cache current against the V stream for the same layer.
```

Status:

```text
done.
layer17 and layer18 full-layer V cache values match isolated QKV exactly.
The apparent local CPU comparison failure was the same PyTorch/NPU RMSNorm/QKV
drift pattern seen in older multi-layer diagnostics.
```

### 4. Reclassify The Strict Chunk Verifier

Goal:

```text
Stop treating the layer19 one-element full-reference miss as proof of a dataflow
bug after local boundary checks have ruled out the graph path.
```

Acceptance:

```text
n-layer-final-only verifier reports local-boundary diagnostics separately from
full-reference drift.
layer_iterations=19 no longer blocks the performance branch solely because one
element exceeds the global abs_tol by about 0.02.
```

### 5. Enable Attention2 Fast Generate Token Validation

Goal:

```text
Run the actual user-visible token check for the performance candidate:

layer_chunk_size=28
attention_columns=2
mlp_gate_up_columns=2
```

Current blocker:

```text
generate currently rejects attention_columns != 1, because fast-generate has
not yet packed the per-chunk hidden[1024] || QK/RoPE metadata[384] runtime
input.
```

Acceptance:

```text
default prompt token_match=True
Fibonacci prompt token_match=True for multiple new tokens
NPU time is compared against the accepted 120-138 ms baseline
```

### 6. Decide Whether Strict Current-K Misses Are Verifier Noise

Goal:

```text
layer_iterations=8 has correct final hidden but a few strict current-K misses.
Decide whether those are tolerance/local-boundary artifacts or real cache
writeback bugs that are masked at short depth.
```

Method:

```text
Compare K/V cache current slices against a reference recomputed from actual NPU
layer inputs. Keep the existing full-reference numbers as drift context.
```

Acceptance:

```text
If local-boundary K/V passes, relax or relabel the strict cache check.
If local-boundary K/V fails, diagnose layout/TAP/writeback for that layer.
```

### 7. Re-Verify Full-Depth Candidate Before Speed Work

Run only after steps 1-3 produce a fix.

Acceptance:

```text
layer_iterations=28:
  chunk_hidden_errors=0
  local-boundary cache checks pass
  NPU time is at or below the accepted 120-138 ms baseline
```

### 8. Recheck Tokens On Multiple Prompts

Run only after full-depth verify passes.

Acceptance:

```text
default prompt token_match=True
Fibonacci prompt token_match=True for multiple new tokens
weekdays prompt token_match=True for multiple new tokens
numeric sequence prompt token_match=True for multiple new tokens
```

### 9. Resume Performance Optimization

Only after correctness is restored:

```text
Measure repeat timing of attention2 + full MLP2 at layer_chunk_size=28.
If faster, make it the accepted fast-generate layout.
If not faster, archive it as a resource-learning branch and return to the
single-column-attention baseline.
```

Next performance branches after that:

```text
columns=4 only after columns=2 has a correct measured speedup.
runtime-position reusable artifact after one correct fast path exists.
descriptor-driven layer state machine after the current static graph is no
longer the main speed limiter.
```
