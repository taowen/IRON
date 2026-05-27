# Experiment 66: MyLM Real Attention Producer Into O Phase

This experiment keeps exp65's full fused-layer schedule and replaces the O
phase deterministic attention replay with a real data-dependent attention
producer.

The MyLM evidence in `~/projects/MyLM` shows that the production layer does not
round-trip Q/K/V intermediates through host-visible buffers. Exp66 therefore
keeps the exp65 static graph shape and uses the existing main-to-edge sideband
as the diagnostic handoff: each sideband payload now carries the producing
main tile's `32` bf16 projection outputs.

## Contract

- Main projection fabric remains `c2..c5/r2..r5`.
- Edge/aux fabric remains `c0/c1/c6/c7`, rows `2..5`.
- The phase order remains `Q, K, V, O, UP, GATE, DOWN`.
- The global weight BO remains the exact `608` MyLM/Qwen patch manifest.
- Q/K/V phase records carry real per-tile projection outputs in the `16 i32`
  payload, interpreted as `32` bf16 values.
- Each edge tile caches its local Q/K/V slices, scans an `L=31` rounded
  two-tile KV history, applies online softmax, computes weighted V, and streams
  the result directly into O.
- There is no debug drain between attention and O.

## Schedule

| phase | input dim | output dim | 512-row blocks | K chunks | patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 4096 | 4096 | 8 | 16 | 64 |
| K | 4096 | 1024 | 2 | 16 | 16 |
| V | 4096 | 1024 | 2 | 16 | 16 |
| O | 4096 | 4096 | 8 | 16 | 64 |
| UP | 4096 | 12288 | 24 | 16 | 192 |
| GATE | 4096 | 12288 | 24 | 16 | 192 |
| DOWN | 12288 | 4096 | 8 | 48 | 64 |

Total: `608` exact MyLM patches.

## What This Proves

- The O phase no longer depends on deterministic attention replay.
- Q/K/V projection outputs can be carried through the existing compact
  main-to-edge handoff without adding a host-visible scratch BO.
- The exp65 full seven-phase patch schedule can continue after an online
  softmax/weighted-V attention producer feeds O directly.

## What This Does Not Prove

- Full MyLM global Q/K/V fanout. This diagnostic keeps Q/K/V state local to each
  edge tile because exp65 has no global all-to-all Q/K/V collector yet.
- Production attention microkernel performance.
- Direct CDO/transaction ownership.

## Run

```bash
.venv/bin/python experiments/66_mylm_real_attention_o_phase/run_npu.py
```
