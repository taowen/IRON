<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Current Baseline And Phase Probe Snapshot

This archive records the accepted paired gate/up baseline and the first dynamic
phase sensitivity probe. The active decision page is
`../experiments-column-scaling.md`.

## Accepted Baseline

```text
stage: generate --fast-generate
operator: n-layer-final-only
layer_chunk_size: 28
num_aie_columns: 2
attention layout: two-column attention2
attention score/softmax: fused per attention column
MLP layout: packed two-column segment-major gate/up/down weights
MLP gate/up rows: paired as [4 gate rows][4 matching up rows]
MLP gate/up compute: qwen3_mlp_gate_up_pair_matvec4_rows_shard_bf16
MLP activation: separate qwen3_silu_mul_shard_bf16
final norm / LM head: CPU F.linear path
weights: prepacked bf16 weights on disk
```

Preflight shape:

```text
compute_cores=30
ObjectFIFOs=53
total_dma_tasks=21
max_dma_tasks_per_fifo=1
max_fifo_buffered_bytes=32768
max_tile_inputs=2
max_tile_outputs=2
non_advancing_acquires=0
```

Banked improvements:

```text
8 + 8 + 8 + 4 host dispatch -> one layer_chunk_size=28 dispatch
single-column MLP -> packed two-column gate/up/down MLP
single-column attention -> two-column attention2
per-run safetensor pack -> prepacked bf16 weights on disk
unused fused-path RoPE metadata copy -> removed
separate softmax Workers -> fused score+softmax Workers, freeing 2 cores
separate gate/up matvec calls -> paired gate/up matvec calls
```

## Prompt Suite

| prompt | positions | checked NPU steps | new text | NPU time ms |
| --- | --- | ---: | --- | --- |
| default | 26 | 1/1 | `Paris` | 106.914 |
| Fibonacci | 17-21 | 5/5 | ` 5, 8,` | min 99.701, mean 102.262, max 104.871 |
| weekdays | 6-10 | 5/5 | ` Thursday, Friday, Saturday,` | min 97.140, mean 98.682, max 100.235 |
| numeric sequence | 11-15 | 5/5 | ` 10, 1` | min 99.449, mean 101.138, max 102.889 |
| free-form opposite | 5-9 | 5/5 | ` cold, and the opposite of` | min 96.750, mean 99.421, max 100.738 |

Suite comparison:

```text
score-side fused-softmax baseline average: 105.737 ms
current paired gate/up matvec average:      101.683 ms
delta:                                      -4.054 ms
relative improvement:                       3.83%
```

## Static Estimate

Static estimate at position 26:

```text
kernel_calls_per_token: 76,216
estimated_macs_per_token: 443.498M
rough_bf16_element_visits_per_token: 538,484,352
```

| category | calls/token | calls/layer | est MACs/token | rough bf16 visits/token |
| --- | ---: | ---: | ---: | ---: |
| MLP gate/up matvec | 21,504 | 768.00 | 176.161M | 198,352,896 |
| QKV projection matvec | 28,672 | 1024.00 | 117.441M | 146,915,328 |
| MLP down matvec | 7,168 | 256.00 | 88.080M | 110,129,152 |
| O-proj matvec | 14,336 | 512.00 | 58.720M | 73,457,664 |
| attention QK scores | 448 | 16.00 | 1.548M | 1,617,728 |
| attention PV context | 448 | 16.00 | 1.548M | 1,617,728 |

Change from the previous score-side fused-softmax graph:

```text
total external kernel calls: 97,720 -> 76,216
MLP gate/up matvec calls:    43,008 -> 21,504
rough bf16 element visits:   560,504,448 -> 538,484,352
estimated MACs:              unchanged at 443.498M/token
```

Interpretation:

```text
The paired matvec edit was a real speedup. The next target should not be chosen
from static estimates alone: MLP gate/up still has the largest MAC count, while
QKV projection now has the largest kernel call count.
```

## Phase Sensitivity Probe

Command:

```bash
python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 5 \
  --json-output build_qwen3_phase_probe_summary.json
```

Current-graph results after dropping iteration 0:

| probe | warm mean | warm median | warm min | warm max |
| --- | ---: | ---: | ---: | ---: |
| full layer, current graph, 1 layer | 3.927 ms | 3.801 ms | 3.749 ms | 4.357 ms |
| attention-only current graph, 1 layer | 1.697 ms | 1.643 ms | 1.635 ms | 1.867 ms |

Standalone diagnostic signals:

| probe | warm mean | warm median | warm min | warm max |
| --- | ---: | ---: | ---: | ---: |
| QKV standalone | 0.769 ms | 0.779 ms | 0.694 ms | 0.825 ms |
| full MLP standalone | 2.409 ms | 1.938 ms | 1.878 ms | 3.881 ms |
| MLP gate/up standalone | 0.847 ms | 0.834 ms | 0.816 ms | 0.902 ms |
| MLP down standalone | 0.801 ms | 0.779 ms | 0.756 ms | 0.888 ms |

Derived decision:

```text
current full-layer warm median:    3.801 ms
current attention-only median:     1.643 ms
current MLP-side median increment: 2.157 ms
MLP median-increment share:        56.8%

standalone QKV warm median:        0.779 ms
attention-only median minus QKV:   0.865 ms
```

Interpretation:

```text
The next local optimization should stay on the MLP side. QKV has the highest
static call count, but the current-graph phase differential says the MLP-side
increment is still the larger measured target.
```

## Decision Carried Forward

```text
Next active experiment:
  direct hidden-producing MLP gate/up+SiLU fusion

Why:
  qwen3_silu_mul_shard_bf16 remains a separate pass after paired matvec
  the MLP-side phase differential remains larger than the attention/QKV signal

Acceptance:
  preflight keeps max_tile_inputs<=2 and max_tile_outputs<=2
  prompt suite token IDs match
  suite mean beats 101.683 ms
  work estimator confirms qwen3_silu_mul_shard_bf16 calls are removed
```
