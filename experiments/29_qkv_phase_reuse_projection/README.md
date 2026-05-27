# Experiment 29: Q/K/V Phase-Reuse Projection

This experiment validates the QKV projection shape observed in MyLM:

```text
same physical projection fabric
  phase 0: Q weights
  phase 1: K weights
  phase 2: V weights
```

It intentionally does not map Q, K, and V onto three parallel compute-tile
groups.  The four compute tiles in this reduced experiment hold one hidden
activation in local memory and consume a phase-ordered Q4NX weight stream:
`Q chunks -> K chunks -> V chunks`.

## What It Proves

- One activation DMA per column feeds all Q/K/V phases.
- The same compute tile program runs all three projection phases.
- The same memtile weight ping-pong ring carries all Q/K/V chunks in order.
- Runtime uses one long weight BD per projection phase per column.  The row1
  static memtile ring re-chunks each phase stream into fat chunks.  Descriptor
  count is therefore independent of the number of K chunks.
- Output BDs are armed before the weight phases.  This matters for long
  streams: the cores must be able to drain Q output before K/V weight streams
  are allowed to make forward progress.
- `K=4096`, matching the full Qwen3-8B hidden dimension.
- Outputs are laid out as `[Q, K, V]` per tile and verified against a CPU
  reference.

## What It Does Not Prove

- The full MyLM 16-main-tile fabric.
- MyLM's exact 16-main-tile patch order.  This experiment solves the same K
  chunk descriptor pressure on the reduced projection fabric.
- Q/K norm, RoPE, current packetization, KV cache writeback, or attention.
- Exact Qwen3-8B dimensions.

This is the narrow contract needed before wiring Q/K/V current outputs into the
shape-A/shape-B attention path from experiments 27 and 28.

## Result

Run:

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/29_qkv_phase_reuse_projection/run_npu.py
```

Observed result:

```text
SUCCESS: Q/K/V phase-reuse projection produces CORRECT output.
4 tiles (2 cols x 2 rows) x Q/K/V = 384 outputs verified.
Latency: 2260.2 us
```

The first implementation tried `K=1024` with two reused shim weight BD ids.  It
compiled, but the kernel timed out at runtime because the descriptor was
rewritten while queued DMA work could still reference it.  A second attempt used
one all-QKV BD per column; that also timed out because the stream boundary was
too coarse across projection phases.

The MyLM trace showed the right boundary: each weight patch is one full-K
64-row shard (`0x28000` bytes for `K=4096`), and the row1 static memtile ring
replays that long stream as 16 Q4NX fat chunks.  The working solution sends one
phase-sized stream per column (`Q`, then `K`, then `V`) and lets the static
memtile ring handle chunking and ping-pong backpressure inside each phase.
Trying to solve this with intermediate `MM2S` syncs was the wrong boundary and
still timed out.
