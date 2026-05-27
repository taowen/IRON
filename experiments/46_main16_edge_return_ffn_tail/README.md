# Experiment 46: Main16 Edge Return FFN Tail

This experiment extends exp45 by keeping the post-attention tail on the same
main16 physical projection fabric:

```text
main16 projection fabric
  -> 17-dword sideband records
  -> edge/aux attention path
  -> returned attention shards
  -> the same main16 fabric runs O phase
  -> the same main16 fabric runs a local FFN tail
  -> row1 gather and final drain
```

The experiment is deterministic `i32` arithmetic. It is a dataflow contract,
not a Q4NX, full softmax, or full-rank FFN performance test.

## Contract

- Runtime-visible input:
  - `weights`: 2560 `i32`, one 160-dword shard per main tile.
  - Each shard is `32` dwords of O-phase weights plus `128` dwords of FFN-tail weights.
- Runtime-visible output:
  - `output`: 512 `i32`, gathered from the same 16 main tiles.
- Internal-only data:
  - 16 records of 17 dwords (`1 header + 16 payload`), emitted by main tiles.
  - 16 edge/aux 32-dword attention-like shards.
  - 16 returned 32-dword attention shards, routed back to the corresponding main tile.
  - 16 O-phase outputs, 16 gate buffers, 16 up buffers, and 16 SwiGLU buffers, all tile-local.
  - Final 32-dword main outputs, packet-gathered into row1 for host drain.

There is no host-visible sideband buffer, attention buffer, O-output buffer,
gate buffer, up buffer, or SwiGLU buffer.

## What This Proves

- The full `c2..c5, r2..r5` main16 fabric can act as projection producer,
  O-phase consumer, and FFN-tail consumer in sequence.
- The 16 non-main compute tiles can serve as edge/aux workers and return
  per-row attention shards to the main16 projection fabric.
- O and FFN tail do not require extra compute tiles; they time-reuse the same
  main16 physical tiles.
- Row1 memtiles can gather the returned main16 final outputs without exposing
  the intermediate attention/O/FFN data.
- Packet gather writes in S2MM arrival order, not packet-id order. The two
  middle columns use the observed `row0,row1,row3,row2` route order in their BD
  offsets; outer columns use `row0,row1,row2,row3`.

## What This Does Not Prove

- Real Q4NX projection math.
- Real online softmax attention.
- Cross-row edge aggregation or exact MyLM packet 14/15 state routing.
- Full all-to-all O projection over 4096 attention dimensions.
- Full-rank gate/up/down FFN across the complete hidden vector.

Those are still separate contracts. This experiment specifically resolves the
resource-placement question after exp45: after attention returns to main16, the
same main16 tiles can continue into an FFN-tail phase without allocating a new
compute fabric.

## Result

Real NPU verification passes:

```text
PASS: main16 -> edge -> main16 O+FFN tail verified.
```

Measured runtime for this scalar `i32` contract was about `3701 us` on NPU2.
