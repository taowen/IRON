# Experiment 23: Q/K/V Current Generation + Cache-Backed Attention

This experiment joins the projection and attention contracts into the next
MyLM/FastFlowLM-style decode boundary:

1. `hidden` enters the worker once.
2. Q, K, and V are generated on the worker from packed Q4NX weights.
3. generated K/V are written into the KV cache at token `L - 1`.
4. the same cache is then scanned in rounded 16-token tiles.
5. generated Q stays local and drives online softmax attention.

There is no host-side current K/V or query input.

## Scope

This is still a reduced contract, not the full `layer.xclbin`:

- one worker and one KV head
- `HEAD_DIM = 32`
- `HIDDEN_DIM = 512`
- Q/K/V each produce `1 head x 32 dim = 32` f32 values
- each 32-row Q4NX rowblock is one runtime weight patch
- each rowblock contains all K chunks; the AIE kernel performs K-chunk
  accumulation internally

The size reduction keeps the single-worker L1 footprint realistic while
preserving the hard contracts:

- phase reuse over Q, K, and V
- one hidden replay across all projection phases
- generated current K/V cache writeback before history scan
- phase transition on one input channel: Q/K/V weight chunks first, then V history
- online attention with a 16-token rounded cache tile

## Run

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```
