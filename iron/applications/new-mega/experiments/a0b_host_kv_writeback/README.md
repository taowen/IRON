<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A0B Host-Side KV Writeback

Status: accepted.

## Question

Can the NPU decode blob avoid dynamic KV-cache write offsets entirely by
returning fixed-shape `present_k/present_v` tensors and letting the host copy
them into the persistent cache between dispatches?

## Hypothesis

Yes. The NPU should be a pure function over fixed-shape inputs:

```text
(current_token_state, full_past_kv_cache) -> (present_k, present_v, logits_or_debug)
```

The NPU never writes the cache. It always drains `present_k/present_v` at fixed
offset zero. The host owns the dynamic writeback:

```text
kv_cache[position] = present_kv
position += 1
```

## Minimal Design

The experiment uses a small fixed blob:

```text
input 0: current[head_dim]
input 1: packed cache chunks
         K[chunk,head_dim] || V[chunk,head_dim] || mask[chunk]
output : present_k[head_dim] || present_v[head_dim] || cache_summary[head_dim]
```

`cache_summary` is a deterministic masked sum of past K/V rows. It proves that
the next NPU invocation reads the row that the host wrote after the previous
invocation.

The host loop is:

```text
for position in decode steps:
  run same xclbin with full packed cache input
  verify present_k/present_v and cache_summary
  host writes present_k/present_v into packed cache at dynamic row offset
  host sets mask[position] = 1
```

## Acceptance

```text
same xclbin and runtime .bin are reused for all steps
NPU does not receive position as an argument
NPU does not drain full KV cache or write dynamic offsets
host writeback makes the next-step cache_summary match CPU reference
crosses a chunk boundary
```

## Run

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
python iron/applications/new-mega/experiments/a0b_host_kv_writeback/run.py
```

## Result

Accepted on NPU2.

Command:

```bash
source /opt/xilinx/xrt/setup.sh
. .venv/bin/activate
PYTHONUNBUFFERED=1 python -X faulthandler \
  iron/applications/new-mega/experiments/a0b_host_kv_writeback/run.py
```

Observed:

```text
artifact_reused_for_all_steps: True
npu_writes_kv_cache: False
host_writeback_bytes_per_step: 512
position=0  present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
position=1  present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
position=2  present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
position=63 present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
position=64 present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
position=69 present_k_max_abs=0.000000 present_v_max_abs=0.000000 summary_max_abs=0.000000
decision: accepted
```

Debug note:

```text
The first run failed with max_summary_abs=0.062500 after 70 steps. That was not
a cache-offset failure. It was a BF16 rounding-boundary mismatch. Adding
aie::set_rounding(aie::rounding_mode::conv_even) before BF16 stores made the
NPU and CPU reference exact.
```

Conclusion:

```text
Current-token KV cache writeback does not need an NPU dynamic offset. The blob
can output fixed present K/V tensors, and the host can memcpy them into the
persistent cache between dispatches.
```

For one KV head with `head_dim=128`, host writeback is 512 bytes per step
(`K+V`). For full Qwen3-0.6B, the same pattern is roughly 224 KiB per token
for 28 layers and 16 KV heads, which is small compared with decode dispatch
latency and model weight traffic.
