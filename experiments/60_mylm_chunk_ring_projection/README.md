# Experiment 60: MyLM Chunk-Ring Projection

Experiment 59 validated MyLM-sized `64-row x K=4096` patch descriptors, but it
kept a full `0x28000` patch resident in row1.  That was useful for ABI bring-up,
but it is not the MyLM resource shape: MyLM's main tile consumes one `5120B`
Q4NX chunk at a time.

This experiment keeps the exact patch descriptor size while shrinking row1
residency to a small streaming ring.  It is intentionally narrowed to one
`64-row` patch so the unresolved two-patch/multi-channel scheduling issue does
not hide the row1 residency question.

## Contract

- One main column subset: `c2/r2..r3`.
- Two simplified edge/source tiles: `c0/r2..r3`.
- Runtime submits one full patch descriptor:
  - patch0: `64 rows x 4096 K`, `0x28000` bytes, rows `0..63`
- Row1 does not allocate the full patch.  It owns one small ping-pong ring:
  - patch0 ring: `2 x (2 Q4NX chunks)` for rows `0,1`
- Main tile channel shape follows the MyLM evidence:
  - `ch0`: activation slice, `128 dwords = 256 bf16`
  - `ch1`: Q4NX weight chunk, `1280 dwords = 5120B`
  - debug output uses a proven packet drain path rather than MyLM's exact
    `ch2` sideband route, because the high-level pathfinder does not currently
    accept that route shape.

## What This Proves

- The generated design compiles with a full `0x28000` patch descriptor and row1
  chunk-sized buffers instead of full-patch row1 buffers.
- Row1 expresses the intended `2 x Q4NX chunk` ping-pong split into two compute
  rows.
- The main compute tile can use the MyLM input ABI: activation on `ch0`, weight
  on `ch1`.

## What This Does Not Prove

- MyLM's exact route8/route15 sideband semantics.
- Exact MyLM same-channel patch0/patch1 descriptor scheduling.
- Runtime completion.  Current status: compilation succeeds, but real NPU
  execution times out, which points to a remaining row1/core lock or DMA
  phase-order mismatch in the small-ring handoff.
- Four-column full N-block scaling; exp59 already proved that at the descriptor
  level.

## Run

```bash
.venv/bin/python experiments/60_mylm_chunk_ring_projection/run_npu.py
```

Last checked status:

```text
Compilation completed successfully.
Runtime timed out with ERT_CMD_STATE_TIMEOUT.
```
