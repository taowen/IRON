# Experiment 47: Main16 Multi-Phase FFN Replay

This experiment extends exp46 by splitting the main16 tail weights into four
separate phases on the same physical input stream:

```text
main16 projection fabric
  -> 17-dword sideband records
  -> edge/aux attention path
  -> returned attention shards
  -> main16 O phase
  -> main16 gate phase
  -> main16 up phase
  -> tile-local SwiGLU
  -> main16 down phase
  -> row1 gather and final drain
```

The experiment is deterministic `i32` arithmetic. It is a phase-handoff and
resource-reuse contract, not a Q4NX, full softmax, or full-rank FFN performance
test.

## Contract

- Runtime-visible input:
  - `weights`: 2048 `i32`, one 128-dword shard per main tile.
  - Each shard is four 32-dword phase slices: O, gate, up, down.
- Runtime-visible output:
  - `output`: 512 `i32`, gathered from the same 16 main tiles.
- Internal-only data:
  - 16 records of 17 dwords (`1 header + 16 payload`), emitted by main tiles.
  - 16 edge/aux 32-dword attention-like shards.
  - 16 returned 32-dword attention shards, routed back to the corresponding main tile.
  - 16 O outputs, gate buffers, up buffers, and SwiGLU buffers, all tile-local.

There is no host-visible sideband, attention, O-output, gate, up, or SwiGLU
buffer.

## What This Proves

- The same `c2..c5, r2..r5` main16 fabric can run O, gate, up, SwiGLU, and down
  in sequence after edge attention returns.
- A single main tile S2MM weight channel can be phase-reused by a BD chain:
  O -> gate -> up -> down.
- Phase ordering is enforced with locks, not by host scheduling. Gate/up/down
  empty locks start at zero and are released only after the previous phase has
  consumed its local state.
- Row1 memtiles can stream one 128-dword per-row shard and let each main tile
  slice it into phase-local buffers without making those phase buffers
  runtime-visible.

## What This Does Not Prove

- Real Q4NX projection math.
- Real online softmax attention.
- Cross-row edge aggregation or exact MyLM packet 14/15 state routing.
- Full all-to-all O/gate/up/down projection over the complete hidden vector.

Those are still separate contracts. This experiment specifically resolves the
operator-level question after exp46: O/gate/up/down can be represented as phase
replay over the same main16 physical resources rather than separate compute
fabrics or host-visible intermediate buffers.

## Result

Real NPU verification passes:

```text
PASS: main16 multi-phase O/gate/up/down replay verified.
```

Measured runtime for this scalar `i32` contract was about `3972 us` on NPU2.
