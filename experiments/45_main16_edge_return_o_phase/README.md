# Experiment 45: Main16 Edge Return O Phase

This experiment replaces the earlier "add more tail workers" idea with the
resource split that matches the MyLM reverse-engineering result:

```text
main16 projection fabric
  -> 17-dword sideband records
  -> edge/aux attention path
  -> returned attention shards
  -> the same main16 fabric runs O phase
  -> row1 gather and final drain
```

The experiment is deterministic `i32` arithmetic. It is a dataflow contract,
not a Q4NX or softmax-performance test.

## Contract

- Runtime-visible input:
  - `weights`: 512 `i32`, one 32-dword O-weight shard per main tile.
- Runtime-visible output:
  - `output`: 512 `i32`, gathered from the same 16 main tiles.
- Internal-only data:
  - 16 records of 17 dwords (`1 header + 16 payload`), emitted by main tiles.
  - 16 edge/aux 32-dword attention-like shards.
  - 16 returned 32-dword attention shards, routed back to the corresponding
    main tile by direct tile-to-tile flows.
  - Final 32-dword main outputs, packet-gathered into row1 for host drain.

There is no host-visible sideband buffer and no host-visible attention buffer.
The same physical main tiles that emit projection records later consume the
returned attention shard plus their O-weight shard.

## What This Proves

- The full `c2..c5, r2..r5` main16 fabric can act as both projection producer
  and O-phase consumer.
- The 16 non-main compute tiles can serve as edge/aux workers and return
  per-row attention shards to the main16 projection fabric.
- O phase does not require extra compute tiles; it time-reuses the main16
  physical tiles.
- Row1 memtiles can gather the returned main16 outputs into the final host
  output without exposing the intermediate attention data.
- Packet gather writes in S2MM arrival order, not packet-id order. The two
  middle columns use the observed `row0,row1,row3,row2` route order in their BD
  offsets; outer columns use `row0,row1,row2,row3`.

## What This Does Not Prove

- Real Q4NX projection math.
- Real online softmax attention.
- Cross-row edge aggregation or exact MyLM packet 14/15 state routing.
- Full all-to-all O projection over 4096 attention dimensions.
- FFN tail reuse after O phase.

Those are still separate contracts. This experiment specifically resolves the
resource-placement question: after attention, the layer must return to the
main16 projection fabric instead of allocating a new O/FFN fabric.

## Result

Real NPU verification passes:

```text
PASS: main16 -> edge -> main16 O-phase handoff verified.
```

Measured runtime for this scalar `i32` contract was about `3734 us` on NPU2.
