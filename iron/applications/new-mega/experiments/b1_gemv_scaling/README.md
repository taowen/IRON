<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B1: Real-Shape Standalone GEMV Scaling

Status: `accepted`

Question:

```text
For real Qwen3 GEMV shapes, does increasing columns improve measured NPU time
without hitting endpoint, BD, L1, or placement limits?
```

Shapes:

```text
Q:    2048 x 1024
K:    1024 x 1024
V:    1024 x 1024
O:    1024 x 2048
gate: 3072 x 1024
up:   3072 x 1024
down: 1024 x 3072
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/b1_gemv_scaling/run.py \
  --build-dir build_new_mega_b1_gemv_scaling \
  --shapes q k v o gate up down \
  --columns 1 2 4 8 \
  --timed-iters 3
```

The script calls `aie_utils.DefaultNPURuntime.cleanup()` after each
configuration. Without this, a single process that loads many distinct xclbins
eventually failed with:

```text
DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
```

Running the failed shape in a fresh process passed, proving this was a runtime
context lifetime issue rather than a GEMV shape failure.

## Result

Accepted on NPU2. All real Qwen3 GEMV shapes compiled, ran, and matched the
bf16 reference.

```text
shape  M     K     best columns  best latency us  speedup vs 1col
q      2048  1024  4             598.101          1.40x
k      1024  1024  2             392.870          1.41x
v      1024  1024  1             972.045          1.00x
o      1024  2048  4             514.799          1.92x
gate   3072  1024  4             521.829          2.54x
up     3072  1024  8             546.144          2.41x
down   1024  3072  8             486.723          2.78x
```

Uniform-column totals for the seven standalone projections:

```text
1 column: 7347 us
2 columns: 5113 us
4 columns: 4718 us
8 columns: 5264 us
per-shape best: 4033 us
```

Conclusion:

```text
The existing standalone GEMV operator already scales for some Qwen3 shapes,
but scaling is not monotonic. A blanket 8-column GEMV policy is wrong. The next
performance experiment should investigate topology and phase-specific column
allocation, especially why small 1024x1024 K/V projections do not benefit from
more columns and why 8 columns regresses several shapes.
```
