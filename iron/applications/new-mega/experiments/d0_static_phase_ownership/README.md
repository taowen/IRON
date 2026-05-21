<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# D0: Static Phase Ownership Skeleton

Status: `accepted`

Question:

```text
Can a fixed phase-owner topology compile and run without recreating the
endpoint/BD/L1 pressure that made larger static graphs fragile?
```

This is the resource preflight that must happen before writing a new Qwen3
layer architecture. C1 proved one Worker can execute a fixed sequence of FIFO
guarded phases. C2 proved inactive phase removal requires a different static
artifact, not same-artifact dynamic skipping. D0 therefore tests the safe
direction: fixed phase ownership with a packed lane-local stream.

## Design

```text
8 lane Workers
1 input ObjectFIFO per lane
1 output ObjectFIFO per lane
1 tile-local f32 state per lane
11 fixed phase packets per lane
```

The output object is padded to eight BF16 elements even though only element 0
is semantic. AIE DMA BDs require 4-byte aligned transfer lengths; our
megakernel preflight additionally requires FIFO objects to be 16-byte aligned.
`memref<1xbf16>` fails resource allocation with `transfer length must be
multiple of 4`, and `memref<2xbf16>` fails preflight as under-aligned.

The packet stream is intentionally homogeneous:

```text
lane 0: q, k, v, o, gate, up, down, attn0, attn1, attn2, attn3
lane 1: q, k, v, o, gate, up, down, attn0, attn1, attn2, attn3
...
```

Each Worker consumes its lane-local packet stream in order, accumulates a small
state, and emits one output. This does not implement Qwen3 math. It isolates
the graph/resource question:

```text
Can we keep each compute tile to one input FIFO and one output FIFO while
representing a fixed multi-phase decode layer?
```

The default packet size is `16896` BF16 elements (`33792` bytes), chosen to be
near a real max per-lane phase payload while staying under a 64 KB L1 object:

```text
attention chunk payload:
  Q pair     = 2 * 128
  K chunk    = 64 * 128
  V chunk    = 64 * 128
  mask       = 64
  total      = 16704 BF16 elements

D0 default:
  packet     = 16896 BF16 elements
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/d0_static_phase_ownership/run.py
```

Compile/preflight only:

```bash
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/d0_static_phase_ownership/run.py \
  --skip-run
```

## Acceptance

```text
preflight passes
full aiecc passes
NPU run completes
max compute tile inputs <= 2
max compute tile outputs <= 2
max FIFO buffered bytes <= 64 KB
max DMA tasks per FIFO <= 8
```

## Result

Accepted on NPU2:

```text
num_lanes: 8
num_phase_packets: 11
packet_elements: 16896
packet_bytes: 33792
lane_packet_stream_bytes: 371712
total_input_bytes: 2973696
packet_fits_l1_64k: True

preflight_runtime_memrefs: 2
preflight_arg_specs: 2
preflight_metadata_host_bos: 5
preflight_compute_cores: 8
preflight_max_fifo_buffered_bytes: 33792
preflight_total_dma_tasks: 16
preflight_max_dma_tasks_per_fifo: 1
preflight_max_compute_tile_inputs: 1
preflight_max_compute_tile_outputs: 1
preflight_non_advancing_acquires: 0

npu_time_us: 5538.108
max_abs: 0.000000
errors: 0
decision: accepted
```

Debug note:

```text
The first compile failed with memref<1xbf16> output because DMA BD transfer
lengths must be 4-byte aligned. Padding to memref<2xbf16> reached preflight,
which then rejected the object as under the stricter 16-byte FIFO alignment
rule. The accepted output object is memref<8xbf16>.
```

## Interpretation

D0 means the next architecture can use a fixed phase-owner graph without
immediately exhausting per-tile FIFO endpoint resources, if each lane receives
a packed homogeneous stream instead of many phase-specific FIFOs. It does not
prove high utilization. Utilization still depends on replacing this accumulate
skeleton with real GEMV/attention/MLP kernels and assigning enough useful work
to each lane.
