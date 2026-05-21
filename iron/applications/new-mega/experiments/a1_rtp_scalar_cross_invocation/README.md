<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A1: RTP Scalar Cross-Invocation Proof

## Question

Can one compiled artifact read a host-written RTP scalar and change Worker
behavior across two separate invocations without recompilation?

## Why This Is First

The dynamic-position plan depends on this. GEMM proves that a Worker can read
an RTP value after startup, but it does not prove that the same artifact can be
invoked repeatedly with different host-provided RTP values.

If this fails, do not build a Qwen3 dynamic-position graph around RTP.

## Experiment

Use the existing `Softmax` operator as the smallest available RTP probe:

```text
design.py:
  Buffer(..., use_write_rtp=True)
  WorkerRuntimeBarrier()
  Worker reads rtp[0]
  mask_bf16(..., vector_size=rtp[0], ...)
```

Compile two variants:

```text
rtp_vector_size=32
rtp_vector_size=64
```

Then:

```text
1. Compare MLIR, runtime .bin, and xclbin artifacts.
2. Run both variants on the same input.
3. Verify each output against a CPU masked-softmax reference.
4. Decide whether the RTP value is a true invocation-time value or a compiled
   runtime-sequence constant.
```

## Command

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate

python iron/applications/new-mega/experiments/a1_rtp_scalar_cross_invocation/run.py \
  --build-dir build_new_mega_a1_rtp
```

## Acceptance

```text
same xclbin and same runtime .bin for both RTP values
outputs match CPU references for both RTP values
no recompile between RTP values
no hang
```

## Result

Status: `rejected`

Run:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/a1_rtp_scalar_cross_invocation/run.py \
  --build-dir build_new_mega_a1_rtp
```

Observed:

```text
rtp32_max_abs: 0.001953
rtp32_tail_sum: 0.000000
rtp32_cpu_match: True
rtp64_max_abs: 0.002441
rtp64_tail_sum: 0.000000
rtp64_cpu_match: True
variant_output_max_abs: 0.060120
mlir_diff_bytes: 2
runtime_bin_diff_bytes: 1
xclbin_diff_bytes: 61
same_artifact: False
decision: rejected
```

Interpretation:

```text
RTP changes Worker behavior and the outputs match their CPU references.
However, changing the RTP value changes generated artifacts. In the current
high-level IRON flow, this RTP value is encoded by the generated runtime
sequence instead of being supplied as an invocation-time argument.
```

Root cause:

```text
`rt.inline_ops(set_rtps, rtps)` writes a generated runtime action with a
compiled value. It is not proof that host code can call the same loaded
artifact twice with different RTP values.
```

Impact:

```text
A2, A3, and A4 stay blocked until we find a real invocation-time RTP update
mechanism or choose a different dynamic-position mechanism.
```
