<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C2: Inactive FIFO / Phase Skip Protocol

Status: `partially-accepted`

Depends on C1. This experiment checks whether a Worker can skip a phase without
requiring dummy DMA tokens on inactive FIFOs.

If dummy tokens are required, the phase design still consumes the scarce
endpoint/BD resources it was supposed to save.

## Design

Two variants are compiled from the same source:

```text
active:
  runtime args: phase0, optional, output
  Worker acquires phase0 FIFO, then optional FIFO, then output FIFO
  output = 2 * phase0 + 1 + 3 * optional + 7

skip:
  runtime args: phase0, output
  Worker never creates or acquires optional FIFO
  output = 2 * phase0 + 1 + 7
```

This intentionally tests whether avoiding dummy tokens requires a different
static graph.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/c2_inactive_fifo_skip/run.py
```

## Result

Accepted for static graph pruning, rejected as a same-artifact dynamic skip
mechanism.

Observed:

```text
variant_result: name=active use_optional_phase=True  arg_count=3 max_abs=0.000000 errors=0
variant_result: name=skip   use_optional_phase=False arg_count=2 max_abs=0.000000 errors=0
mlir_diff_bytes: size-mismatch:3381->2602
runtime_bin_diff_bytes: size-mismatch:420->300
xclbin_diff_bytes: size-mismatch:10570->10138
same_artifact: False
skip_uses_dummy_optional_fifo: False
decision: partially-accepted
```

Conclusion:

```text
Inactive FIFO tokens can be removed only by compiling a different static
graph/ABI. This avoids dummy DMA, but it does not prove same-artifact dynamic
phase skipping.
```

Implication:

```text
Do not build a universal same-artifact Worker that expects to skip arbitrary
phase FIFOs unless there is a separate runtime-control proof. With current
IRON Runtime/ObjectFIFO usage, inactive phases should be handled by static
phase ownership or separate artifacts, not by dummy tokens.
```
