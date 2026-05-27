# Experiment 41: Edge Shape A -> Shape B Handoff Probe

This experiment isolates the MyLM/FastFlowLM edge-path question that exp40
did not answer: can one edge/aux compute tile produce a compact per-token
state and hand it directly to another edge tile without making that state a
host-visible DDR buffer?

It is not a full attention implementation.  The math is intentionally
deterministic integer arithmetic so the NPU output can be checked exactly.

## Contract

- `shape_a` receives:
  - `current`: 512 `i32` dwords from DDR.
  - `history_a`: 2048 `i32` dwords from DDR.
- `shape_a` computes a 17-dword compact state:
  - one header dword.
  - sixteen payload dwords.
- The compact state flows directly from `shape_a` to `shape_b`.
- `shape_b` receives:
  - the 17-dword compact state from `shape_a`.
  - `history_b`: 2048 `i32` dwords from DDR.
- `shape_b` computes 512 `i32` output dwords and drains them to DDR.

The runtime sequence has only four host buffers:

```text
current, history_a, history_b, output
```

There is deliberately no runtime buffer for the 17-dword compact state.  That
state must stay inside the AIE dataflow graph.

## Why This Matters

MyLM's edge attention path has shape-A-like workers, shape-B-like workers, and
small `17`-dword sideband/state streams.  Previous experiments proved pieces of
attention and projection, but the remaining ABI question is the cross-tile
handoff itself: whether a compact state can be produced by one edge phase and
consumed by the next phase without a DDR round trip.

Passing this experiment gives a concrete template for the missing edge
phase-handoff boundary before attaching real softmax state, value accumulation,
and O projection.
