# Experiment 48: Main16 Full-K Q4NX Phase Replay

This experiment replaces exp47's toy `i32` phase weights with real full-K Q4NX
weight streams on the same main16 fabric.

```text
Q4NX weight BO
  -> shim phase-sized MM2S descriptors
  -> row1 memtile ping-pong fat chunks
  -> 4 row streams per column
  -> main16 compute tiles
  -> O, gate, up, down full-K Q4NX phases
  -> tile-local phase combine
  -> row1 packet gather and final drain
```

The experiment is still a phase/dataflow contract. It does not try to implement
real Llama/Qwen FFN math: every phase uses the same deterministic local
activation vector so the test isolates full-K Q4NX streaming and phase reuse.

## Contract

- Runtime-visible input:
  - `weights`: 1,310,720 `i32` words, covering 4 main columns.
  - Per column: 4 full-K phase streams.
  - Per phase stream: 16 row1 fat chunks.
  - Per fat chunk: 4 row shards, each a 32x256 Q4NX chunk.
- Runtime-visible output:
  - `output`: 256 `i32` words, interpreted as 512 `bf16` values.
- Internal-only data:
  - Per main tile local activation: 4096 `bf16`.
  - Per main tile ping-pong Q4NX chunk buffers.
  - Per main tile O/gate/up/down `bf16[32]` phase outputs.
  - Final `bf16[32]` output packet-gathered into row1.

## What This Proves

- The main16 physical projection fabric can consume realistic full-K Q4NX
  streams for four consecutive phases.
- A phase-sized runtime descriptor can feed a static row1 chunking ring; the
  runtime does not issue one DMA task per 256-wide K chunk.
- Row1 can split each 4-row fat chunk into four compute-row streams while
  keeping ping-pong DMA/compute overlap structure.
- The final output remains the only host-visible intermediate.

## What This Does Not Prove

- Real post-attention activation as the O input.
- Real gate/up input from FFN norm or down input from SwiGLU.
- Full FFN hidden dimension expansion (`12288`) or cross-row reduction.
- Real online attention and packet14/15 edge state.

Those are separate contracts. This experiment closes the gap between exp47's
phase handoff and the Q4NX full-K weight streams required by the final fused
layer engine.

## Result

Real NPU verification passes:

```text
PASS: main16 full-K Q4NX O/gate/up/down replay verified.
```

Measured runtime for this contract was about `5123 us` on NPU2.

One routing detail is intentionally different from exp47: this experiment has
no edge-return route, so packet gather arrives in normal `row0,row1,row2,row3`
order for every main column. Exp47's middle-column `row0,row1,row3,row2` order
is specific to the edge-return topology.
