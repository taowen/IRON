# Experiment 53: Full Layer Phase Chain Contract

This experiment connects the previously isolated contracts into a single
MyLM-style fused-layer dataflow skeleton.

The goal is not exact Qwen3 math yet. The goal is to prove that the physical
resource shape can run the whole layer as one phase chain:

```text
seed hidden record
  -> edge replays Q activation slices
  -> Q full-K Q4NX phase on main16
  -> compact sideband
  -> edge replays K activation slices
  -> K full-K Q4NX phase
  -> compact sideband
  -> edge replays V/O/gate/up/down slices through the same ABI
  -> O/gate/up/down full-K Q4NX phases on the same main16 tiles
  -> core-local SwiGLU after up
  -> final debug record drain
```

## Contract

- Main projection fabric remains `c2..c5/r2..r5`.
- Edge/aux fabric remains `c0/c1/c6/c7` rows `2..5`.
- Main tile input shape remains:
  - `S2MM ch0`: one `256 bf16` activation slice (`128 dwords`).
  - `S2MM ch1`: one Q4NX `32 x 256` chunk (`1280 dwords`).
- Every projection phase consumes `16` activation slices and `16` Q4NX chunks.
- The same main tile executes all `Q,K,V,O,gate,up,down` phases in sequence.
- A single compact `17 dword` sideband record gates each edge replay phase.
- The main core program uses one phase loop instead of seven flat copies of the
  Q4NX inner loop, matching the resource lesson from MyLM's reusable projection
  microprogram.
- The final host output is only a debug drain with `(group,row)` headers. It is
  not treated as MyLM's internal layer handoff ABI.

## What This Proves

- A full layer-shaped phase chain can fit in 32 compute tiles by temporal
  reuse, rather than allocating separate tiles per operator.
- The exp52 edge-slice replay contract can be repeated across multiple phase
  boundaries.
- The row1 weight fat-chunk ring can stream seven full-K phase descriptors while
  edge tiles supply activation slices on `ch0`.
- Gate/up intermediates can stay core-local through the simplified SwiGLU
  boundary.
- The runtime should not be modeled as seven independent phase submissions for
  this experiment. A single continuous per-column weight descriptor feeds the
  static ring, and phase boundaries are enforced by the worker loop.

## What This Does Not Prove

- Exact Qwen3 attention math.
- Exact MyLM edge shape-A/shape-B arithmetic.
- Exact MyLM sideband semantics. The `17 dword` record is deterministic by
  `(group,row,phase,lane)` so the test validates the handoff lifetime and
  routing contract without amplifying Q4/bf16 numerical noise into a different
  next-phase activation stream.
- Runtime-level phase descriptor submission. The failed intermediate version
  that queued one weight BD per phase timed out on real NPU; the passing version
  uses one continuous layer-shaped weight stream per column.
- Real Qwen3 down-projection expanded dimension (`12288`). This experiment
  keeps every projection at `K=4096` to isolate phase handoff/resource shape.

## Result

Real NPU run:

```text
NPU time: 130601.8 us
max_abs_err=0.062500
PASS
```

## Run

```bash
.venv/bin/python experiments/53_full_layer_phase_chain_contract/run_npu.py
```
