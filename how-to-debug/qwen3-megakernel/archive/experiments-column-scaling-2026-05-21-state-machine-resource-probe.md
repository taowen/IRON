<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# State-Machine Resource Probe

This archive records the first check before starting a descriptor/state-machine
rewrite.

## Question

The active plan said the current graph was statically expanded over 28 layers.
Before rewriting it, we checked whether compute cores, endpoints, and FIFO
limits actually grow with `layer_iterations`.

Command shape:

```bash
python iron/applications/qwen3_0_6b/persistent/real_graph_probe.py \
  --stages n-layer-final-only \
  --columns 2 \
  --attention-columns 2 \
  --mlp-gate-up-columns 2 \
  --mlp-gate-up-pair-rows \
  --mlp-gate-up-direct-silu \
  --layer-iterations <1|8|28> \
  --preflight-only \
  --trace-placement \
  --build-dir build_qwen3_state_machine_resource_probe_l<layers> \
  --clean-build
```

## Results

| layers | compute cores | total DMA tasks | max DMA tasks/FIFO | runtime outputs | runtime inputs | max tile inputs | max tile outputs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 22 | 1 | 16 | 6 | 2 | 2 |
| 8 | 30 | 29 | 2 | 16 | 5 | 2 | 2 |
| 28 | 30 | 21 | 1 | 16 | 5 | 2 | 2 |

## Interpretation

The accepted direct-SiLU graph already reuses the same Worker/ObjectFIFO graph
across layers. The main resource counts do not grow linearly with
`layer_iterations`:

```text
compute cores: 29 -> 30 -> 30
runtime outputs: 16 -> 16 -> 16
max tile ports: 2/2 across all probes
```

So a descriptor/state-machine rewrite should not be framed as "reduce one
Worker per layer"; that is already not the current failure mode.

The remaining state-machine target is narrower:

```text
reduce position-specific compilation and static runtime constants
reduce or patch position/cache-dependent TAP metadata
make weight/cache streams reusable without changing host dispatch count
preserve the accepted token-correct two-way direct-SiLU graph body
```

This changes the next experiment from broad graph replacement to a runtime
metadata/patching experiment.
