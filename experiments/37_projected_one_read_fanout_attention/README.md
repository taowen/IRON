# Experiment 37: Projected One-Read Fanout Attention

Exp36 still supplied query and current K/V from the host. Exp37 closes the
attention subgraph on NPU:

```text
hidden + Q4NX Q/K/V weights
  -> four worker columns compute Q heads
  -> column 0 also computes shared current K/V
  -> current K/V are written into the cache BO
  -> one central row1 memtile scans updated K/V history once
  -> row1 static token-major to dim-group-major reshape
  -> four fanout streams feed the attention workers
  -> four-head GQA output
```

## What It Proves

- Q/K/V projection can feed current-write plus one-read fanout attention in one
  runtime sequence.
- Current K/V is produced on NPU, written into the cache BO, and then observed
  by the central one-read K/V scan.
- The four Q heads share one K/V history scan: runtime history patches remain
  `2 * num_tiles`, not `4 * 2 * num_tiles`.
- The viable phase-handoff shape is MyLM-like: hidden and K/V history must be
  sent from the same row1 fabric into each worker input channel.

## Negative Evidence

The first implementation tried to route local hidden delivery and central K/V
fanout into the same worker DMA port as two separate static sources. MLIR-AIE
rejected that at routing time:

```text
targets same dst as another connect op
```

That is the important resource lesson from this experiment. Time-separated
producer phases are not enough if the static route graph has multiple sources
for one worker DMA destination. The fix is to move the phase switch to a single
row1 source: central row1 sends hidden first, then K/V history.

## Scope

This is still a single 4Q:1KV GQA group, not a full transformer layer:

- no RMSNorm, Q/K norm, or RoPE,
- no O projection or FFN,
- no MyLM edge/aux attention split,
- projection kernels are still scalar/vector-dot reference kernels rather than
  the optimized 16-tile Q4 online MVM fabric.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/37_projected_one_read_fanout_attention/run_npu.py
```

Observed real-NPU result:

```text
L=17 PASS time=8467.8us  max_abs=0.000005
L=31 PASS time=8556.1us  max_abs=0.000003
L=32 PASS time=8521.8us  max_abs=0.000003
L=79 PASS time=11590.3us max_abs=0.000001
```
