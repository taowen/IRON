# Experiment 31: Full GQA Attention Group

This experiment challenges the whole attention core for one MyLM-style GQA
group:

```text
hidden bf16[4096]
  -> Q projection: 4 query heads x 128 dims = 512 rows
  -> K projection: 1 KV head x 128 dims
  -> V projection: 1 KV head x 128 dims
  -> current K/V cache write
  -> rounded KV history scan
  -> online GQA softmax attention
  -> attention output bf16/f32-equivalent[512]
```

## What It Proves

- Q/K/V use different projection widths while sharing one physical worker and
  one static row1 weight chunking ring.
- Q consumes a full `512 x 4096` logical phase, while K and V each consume a
  full `128 x 4096` logical phase.
- Runtime still uses one phase-sized BD for Q, one for K, and one for V; the
  static ring re-chunks the full-K streams into Q4NX `32 x 256` chunks.
- GQA semantics are verified: four Q heads attend to one shared K/V head.
- Current K/V are written to the cache BO, then the same cache is scanned for
  online softmax attention with tail masking.

## Scope

This is the complete attention core for one GQA group, not the whole transformer
attention layer:

- one GQA group, not all 8 KV groups,
- one worker tile, so it proves the dataflow contract rather than throughput,
- no O projection yet,
- pair-tile KV cache layout `[K tile | V tile]`, not the final four-plane MyLM
  cache layout.

The next scaling step is to replicate this contract across the 8 KV groups and
replace the single-worker serial execution with the MyLM-style 16-main-tile
projection fabric plus edge attention workers.

## Result

Verified on real NPU for `L=17`, `31`, `32`, and `79`.

The run covers a full `512 x 4096` Q phase, two full `128 x 4096` K/V phases,
current K/V cache writeback, non-16-aligned tail masking, and a 5-tile KV
history scan. All 512 attention output values matched the CPU reference with
max absolute error below `5e-6`.

Measured latency was about `14.6ms` to `25.1ms` for this single-worker serial
contract. That is expected: this experiment proves the full GQA dataflow
contract, not the final throughput shape. The performance path is to spread the
same contract across MyLM's projection fabric and attention edge tiles.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/31_gqa_full_attention/run_npu.py
```
