<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C1: Two-Phase Same-Worker Protocol

Status: `accepted`

Question:

```text
Can one Worker execute two host-scheduled phases in one dispatch with balanced
ObjectFifo acquire/release behavior?
```

This is required before a universal phase-based Worker design can be considered
real. `task_group` alone only proves DMA task grouping, not Worker phase
handshakes.

## Design

```text
one Worker
one dispatch
phase0 FIFO: input A
phase1 FIFO: input B
output FIFO: final output
tile-local Buffer: phase0 state
```

Worker order:

```text
acquire phase0 input
state = 2 * phase0 + 1
release phase0 input

acquire phase1 input
acquire output
output = state + 3 * phase1 + 7
release phase1 input
release output
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/c1_two_phase_worker/run.py
```

## Result

Accepted on NPU2:

```text
dispatch_count: 1
worker_phase_order: phase0_then_phase1
npu_time_us: 566.866
max_abs: 0.000000
errors: 0
decision: accepted
```

Debug note:

```text
The first run failed before MLIR compilation because the local dataclass
operator wrapper did not call MLIROperator.__init__ in __post_init__ and
therefore had no `artifacts` field. This was an experiment harness bug, not an
IRON phase-protocol failure.
```

Conclusion:

```text
One Worker can execute two FIFO-guarded phases in one dispatch when both phases
have balanced acquire/release tokens.
```

This does not prove inactive phase skipping. If a future phase is optional,
the design still needs C2 to prove that the Worker can avoid acquiring inactive
FIFOs without requiring dummy DMA tokens.
