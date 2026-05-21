<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# New Mega Production

This directory contains the production implementation path for the new Qwen3
megakernel. The sibling `experiments/` directory records mechanism proofs; code
here is the implementation surface that later stages should extend.

Current production stage:

```text
fixed-attention
```

It implements the D1.0 accepted boundary:

```text
real Qwen3 layer-0 q_norm+RoPE queries
host-owned current K/V writeback into full fixed cache
fixed max-cache K/V/mask stream
NPU chunked online softmax + PV for all 16 Q heads
attention context output [16,128]
```

Validated on NPU2:

```text
default prompt:
  decode_position=26
  npu_time_us=3732.906
  max_abs=0.015625
  errors=0

second prompt:
  decode_position=22
  npu_time_us=3551.477
  max_abs=0.019531
  errors=0
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py
```

Second prompt smoke test:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py \
  --prompt "Name the largest planet. Answer with one word."
```

Compile/preflight only:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/production/main.py --compile-only
```

## Files

```text
main.py              CLI entry point
runner.py            model/reference preparation, packing, verification
ops.py               MLIROperator definition
design.py            IRON graph
fixed_attention.cc   AIE external kernels
```

## Next Production Steps

```text
D1.1: add NPU-side QKV/RoPE and fixed present K/V outputs
D1.2: add O projection + residual
D1.3: add post-attention RMSNorm + MLP
D1.4: run full single-layer output check
```
