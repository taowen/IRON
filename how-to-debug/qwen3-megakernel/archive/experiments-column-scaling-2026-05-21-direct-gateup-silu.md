<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Direct Gate/Up + SiLU Fusion

This archive records the direct hidden-producing MLP gate/up+SiLU branch. The
active decision page is `../experiments-column-scaling.md`.

## Goal

Remove the separate per-layer/per-column `qwen3_silu_mul_shard_bf16` pass after
the paired gate/up matvec:

```text
old:
  qwen3_mlp_gate_up_pair_matvec4_rows_shard_bf16 -> gate shard + up shard
  qwen3_silu_mul_shard_bf16                      -> hidden shard

new:
  qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16   -> hidden shard
```

Acceptance criteria from the active plan:

```text
preflight keeps max_tile_inputs<=2 and max_tile_outputs<=2
prompt-suite token IDs match
suite mean beats 101.683 ms
work estimator shows qwen3_silu_mul_shard_bf16 calls removed
```

## Compile Failure And Fix

First full compile failed in the external kernel:

```text
error: no matching function for call to 'tanh'
candidate template ignored: could not match 'vector<float, Elems>' against 'float'
```

Root cause:

```text
The AIE API tanh used by this repo is vector-only. The first fused kernel tried
to call aie::tanh(float) after reducing one gate row to a scalar.
```

Second compile failed after the fix was placed in the wrong scope:

```text
error: use of undeclared identifier 'silu_vec_len'
error: use of undeclared identifier 'half'
error: use of undeclared identifier 'one'
```

Root cause:

```text
The vector constants were inserted into qwen3_mlp_matvec4_rows_bf16, but the
macro expansion that used them lived in
qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16.
```

Final fix:

```text
After reducing gate/up to bf16 row results, broadcast each scalar into a
16-lane bf16 vector, run the same tanh-form SiLU approximation used by
qwen3_silu_mul_bf16, then store lane 0 to the hidden shard.
```

Diagnostic rule carried forward:

```text
When an external kernel fails in clang++, read the AIE API overload boundary
literally before changing the IRON graph. Scalar code that looks valid in host
C++ may be invalid or undesirable in an AIE API kernel.
```

## Static Result

Preflight:

```text
compute_cores=30
max_fifo_buffered_bytes=32768
max_dma_tasks_per_fifo=1
max_tile_inputs=2
max_tile_outputs=2
non_advancing_acquires=0
```

Work estimator at position 26:

```text
old total external kernel calls: 76,216
new total external kernel calls: 76,160
delta:                           -56 calls/token

old MLP activation calls:
  qwen3_silu_mul_shard_bf16: 56 calls/token

new MLP gate/up calls:
  qwen3_mlp_gate_up_pair_silu4_rows_shard_bf16: 21,504 calls/token

estimated_macs_per_token:        unchanged at 443.498M
rough_bf16_element_visits/token:  538,226,304
```

`qwen3_silu_mul_shard_bf16` remains declared in MLIR because the Kernel object
exists, but there are no call sites in the direct graph.

## Prompt Suite

| prompt | positions | checked NPU steps | new text | NPU time ms |
| --- | --- | ---: | --- | --- |
| default | 26 | 1/1 | `Paris` | 105.293 |
| Fibonacci | 17-21 | 5/5 | ` 5, 8,` | min 101.794, mean 103.061, max 103.740 |
| weekdays | 6-10 | 5/5 | ` Thursday, Friday, Saturday,` | min 93.272, mean 96.995, max 98.857 |
| numeric sequence | 11-15 | 5/5 | ` 10, 1` | min 99.424, mean 101.120, max 102.777 |
| free-form opposite | 5-9 | 5/5 | ` cold, and the opposite of` | min 97.021, mean 98.336, max 99.503 |

Suite comparison:

```text
previous paired gate/up matvec average: 101.683 ms
direct gate/up+SiLU average:            100.961 ms
delta:                                  -0.722 ms
relative improvement:                   0.71%
```

Decision:

```text
Accepted, but marginal. This removes one pass and improves the prompt-suite
mean, but it does not change the main MAC count and does not materially change
the phase-level bottleneck.
```

## Phase Probe After Acceptance

Command:

```bash
python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 5 \
  --mlp-gate-up-direct-silu \
  --build-dir-prefix build_qwen3_phase_probe_direct \
  --json-output build_qwen3_phase_probe_direct_summary.json
```

Results after dropping iteration 0:

| probe | warm mean | warm median | warm min | warm max |
| --- | ---: | ---: | ---: | ---: |
| full layer, current graph, 1 layer | 3.905 ms | 3.896 ms | 3.752 ms | 4.075 ms |
| attention-only current graph, 1 layer | 1.752 ms | 1.684 ms | 1.670 ms | 1.970 ms |
| QKV standalone | 0.913 ms | 0.823 ms | 0.684 ms | 1.320 ms |
| full MLP standalone | 2.136 ms | 2.031 ms | 1.973 ms | 2.509 ms |
| MLP gate/up standalone | 0.807 ms | 0.792 ms | 0.780 ms | 0.864 ms |
| MLP down standalone | 0.796 ms | 0.800 ms | 0.744 ms | 0.842 ms |

Derived signal:

```text
current full-layer warm median:    3.896 ms
current attention-only median:     1.684 ms
current MLP-side median increment: 2.212 ms
MLP median-increment share:        56.8%

standalone QKV warm median:        0.823 ms
attention-only median minus QKV:   0.861 ms
```

Next target:

```text
Stay on MLP-side parallelism. The direct fusion removed calls but did not erase
the MLP-side phase differential. A four-way MLP gate/up branch is reasonable
only if the ffn-hidden join preserves the two-input/two-output invariant.
```
