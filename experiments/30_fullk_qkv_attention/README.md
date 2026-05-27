# Experiment 30: Full-K Q/K/V Projection Into Attention

This experiment connects the contracts from exp29 and exp23:

```text
hidden bf16[4096]
  -> Q/K/V Q4NX projection phases
  -> current K/V cache write
  -> read rounded KV cache tiles
  -> online softmax attention
```

## What It Proves

- The projection reduction dimension is full-size `K=4096`.
- Runtime uses one phase-sized weight BD for each of `Q`, `K`, and `V`.
  Each phase BD is `0x14000` bytes for one 32-row output shard, equivalent to
  16 Q4NX chunks.
- The static row1 weight ring re-chunks each long phase stream into 256-column
  Q4NX chunks.
- Q, K, and V reuse the same physical worker and weight ring over time.
- Current K/V are written to a cache BO by the NPU, then the same BO is read
  back for attention after a token sync.
- Attention output is verified against a CPU reference on real NPU.

## Scope

This is still a reduced attention block:

- one worker tile,
- one 32-dim head,
- pair-tile KV cache layout `[K tile | V tile]`,
- no O projection,
- no FFN,
- no text-generation runner.

The point is to prove the full-K projection-to-attention handoff before scaling
the dataflow back out to the larger MyLM-style fabric.

## Result

Verified on real NPU for `L=17`, `31`, `32`, and `79`.

The run covers non-16-aligned tails, exact 16-token tile boundary, and a 5-tile
history scan. The measured latencies were about `2.16ms` to `2.80ms` for this
single-worker reduced contract, and every output element matched the CPU
reference with max absolute error below `2e-5`.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/30_fullk_qkv_attention/run_npu.py
```
