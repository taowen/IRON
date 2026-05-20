<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase Probe After Split Rejections

This archive records the accepted-graph phase probe after rejecting row-group 8
and three-way split gate/up as performance paths.

Command:

```bash
python iron/applications/qwen3_0_6b/persistent/phase_timing_probe.py \
  --repeat 3 \
  --mlp-gate-up-direct-silu \
  --build-dir-prefix build_qwen3_phase_probe_after_split_reject \
  --json-output build_qwen3_phase_probe_after_split_reject_summary.json
```

Results after dropping iteration 0:

| probe | warm mean | warm median | warm min | warm max |
| --- | ---: | ---: | ---: | ---: |
| full layer, current graph, 1 layer | 3.882 ms | 3.882 ms | 3.866 ms | 3.899 ms |
| attention-only current graph, 1 layer | 1.661 ms | 1.661 ms | 1.652 ms | 1.670 ms |
| QKV standalone | 1.373 ms | 1.373 ms | 0.880 ms | 1.866 ms |
| full MLP standalone | 2.179 ms | 2.179 ms | 1.936 ms | 2.421 ms |
| MLP gate/up standalone | 0.843 ms | 0.843 ms | 0.809 ms | 0.877 ms |
| MLP down standalone | 0.794 ms | 0.794 ms | 0.782 ms | 0.805 ms |

Derived signal:

```text
current full-layer warm median:    3.882 ms
current attention-only median:     1.661 ms
current MLP-side median increment: 2.221 ms
MLP median-increment share:        57.2%
```

Interpretation:

```text
The accepted graph is still MLP-side dominated, but static MLP shard-count
increases did not produce robust speedup. The next experiment should not be
another static gate/up split. It should test whether descriptor/state-machine
resource reuse can reduce static graph pressure and open room for better
placement or scheduling.
```
