# Experiment 18: Integrated Fused FFN Contract

This experiment consolidates the FFN lessons from experiments 16 and 17 into a
single clean contract test.

## What This Proves

- A single physical worker group can run multiple projection phases over time:
  gate, up, SwiGLU, gathered broadcast, and down.
- Gate and up projection results stay in core-local buffers until SwiGLU. They
  do not round-trip through DDR.
- The memtile can gather four core-local SwiGLU slices, then broadcast the
  gathered vector back to all four cores for the down projection.
- Ping and pong weight slots need separate empty/full lock pairs on both the
  memtile and the compute tile.
- Runtime shim weight pushes need distinct BD descriptors while they are queued.
  Rewriting a queued descriptor before synchronization can corrupt earlier
  transfers.

## What This Does Not Prove

- Attention/KV scan, GQA fanout, or online softmax.
- 16-tile or 32-tile scale-out.
- A complete Qwen layer or text generation.
- Production numerical accuracy under large activation ranges. `run_npu.py`
  uses bounded inputs so the smoke test checks dataflow ownership rather than
  SwiGLU saturation.

## Topology

- 1 shim tile
- 1 memtile
- 4 compute tiles in one column
- `K_HIDDEN = 512`
- gate/up use two 256-wide Q4NX chunks
- down consumes the gathered 128-wide intermediate

## Verification

Run from this directory with the IRON environment:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python phase_probe.py gate-buffer-extra
/var/home/taowen/projects/IRON/.venv/bin/python phase_probe.py gate-after-up
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

The phase probes are the important regression checks for descriptor lifetime and
slot ownership. The full run is a bounded complete-FFN smoke.
