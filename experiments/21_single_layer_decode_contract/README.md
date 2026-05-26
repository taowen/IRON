# Experiment 21: Single-Layer Decode Contract

This experiment integrates the stable pieces from experiments 19 and 20 into a
small single-layer decode contract.

The important correction versus the abandoned first attempt is that Q is not a
new host/runtime input.  In a FastFlowLM/MyLM-style layer engine, Q is produced
by an earlier projection phase and remains local for attention.  This experiment
models that boundary by deriving a current-token query inside the worker from
the current K/V inputs, then keeping it in a core-local buffer.

## What This Proves

- Current K/V still write into the same KV cache BO before history scan.
- Rounded history scan and final-tile masking still work for arbitrary `L`.
- A local Q phase can run before attention without adding another runtime DMA
  stream.
- `running_max`, `running_sum`, and the attention output accumulator stay
  core-local across KV history tiles.
- A post-attention layer epilogue consumes the attention output and local query
  without a DDR round-trip.
- The program still uses raw MLIR-AIE `writebd`, `address_patch`, `push_queue`,
  static tile/memtile BD chains, and explicit locks.

## What This Does Not Prove

- Real Qwen3 RMSNorm, Q/K/V projection weights, QK norm, RoPE, O projection, or
  FFN weights.
- FastFlowLM's exact 16-main-tile placement or edge/auxiliary attention mapping.
- A vectorized high-performance softmax or Q4NX projection kernel.
- Full text generation.

The local Q phase and epilogue are deterministic reduced-model stand-ins.  They
exist to prove phase residency and dataflow ownership, not model quality.

## Shape

```text
current layout = k03 | v03 | k47 | v47
cache layout   = k03 | v03 | k47 | v47
one plane tile = 16 tokens x 4 heads x 32 dim x f32
worker output  = 4 heads x 32 dim x f32
```

## Verification

Run from this directory:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

The script checks both the current-token cache write and the final
post-attention layer output against a CPU reference.
