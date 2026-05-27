# Experiment 55: MyLM-Style Linked BD Chain

Experiment 54 proved that the real Qwen3 patch schedule fits the fused-layer
fabric when weights are provided as one contiguous stream. This experiment
isolates the missing MyLM-style runtime mechanism: a coarse host transaction
that patches several shim DMA descriptors, links them with `next_bd`, and pushes
only the head descriptor.

## Contract

- One shim column and one compute tile.
- The compute tile owns a static ping-pong S2MM ring.
- The host input is split into 8 chunks of `64 i32`.
- Runtime patches 8 shim MM2S BDs:
  - `BD0 -> BD1 -> ... -> BD7`
  - `use_next_bd = 1` for `BD0..BD6`
  - `use_next_bd = 0` for `BD7`
- Runtime pushes only `BD0` on shim MM2S channel 0, then waits once.
- The core records `(header, first, last, checksum)` for each chunk and drains
  one compact output buffer.

## What This Proves

- Raw AIEX can express the MyLM-style coarse descriptor submission pattern:
  patch a linked descriptor chain, push the head, and let the hardware walk the
  chain while the compute tile consumes a continuous stream.
- The compute-side static ping-pong ring does not need host involvement at
  every chunk boundary.

## What This Does Not Prove

- Full MyLM layer scheduling.
- Qwen3 math or phase handoff ABI.
- Multi-column phase traffic.

## Run

```bash
.venv/bin/python experiments/55_mylm_linked_bd_chain/run_npu.py
```
