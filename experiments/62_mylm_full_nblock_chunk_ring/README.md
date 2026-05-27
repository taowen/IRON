# Experiment 62: MyLM Full N-Block Chunk Ring

This experiment folds the planned exp62/exp63/exp64 steps into one hardware
probe. Exp61 proved one `0x28000` patch can feed two rows through a small row1
chunk ring. This experiment scales that directly to the full main16 N-block:
four main columns, four rows per column, two exact MyLM patches per column, and
no full-patch row1 buffers.

## Contract

- Main projection fabric: `c2..c5/r2..r5`.
- Edge/source fabric: `c0/c1/c6/c7/r2..r5`.
- Each main column receives two linked shim `MM2S ch0` descriptors:
  - patch0: output rows `0..63`, `0x28000` bytes
  - patch1: output rows `64..127`, `0x28000` bytes
- Row1 does not allocate a full patch. Each memtile owns two slot-locked rings:
  - patch0 ping/pong: `2 x (2 Q4NX chunks)` for rows `0,1`
  - patch1 ping/pong: `2 x (2 Q4NX chunks)` for rows `2,3`
- The memtile input DMA uses a finite static BD phase:
  - 16 chunk-pair BDs fill patch0 ping/pong
  - 16 chunk-pair BDs fill patch1 ping/pong
  - the last BD loops back to patch0 for future reuse
- Row stream BDs preserve the proven row-channel bank mapping:
  - row0: `2/3`
  - row1: `24/25`
  - row2: `4/5`
  - row3: `26/27`
- Output packets route to memtile `DMA : 2`, and the collector starts on
  `S2MM ch2`.

The default `full` and `onecol` variants intentionally keep both patches on the
same shim/memtile input channel to reproduce the exact MyLM-style question. The
`split_onecol` and `split_full` diagnostics put patch0 on channel 0 and patch1
on channel 1 while preserving the same row1 chunk-ring and compute contract.

## What This Proves

- Exp61's small ring schedule scales to the full `512-row x 4096 K` main16
  N-block when each patch phase uses a legal memtile DMA channel/BD bank.
- The full N-block can avoid exp59's full-patch row1 residency.
- The exp60/exp62 same-channel timeout has a concrete root cause: on memtiles,
  even DMA channels use BD slots `0..23`, and odd DMA channels use `24..47`.
  The same-channel finite phase needs 32 input BDs on channel 0, so patch1
  crosses into the odd-channel BD bank. High-level MLIR-AIE does not reject that
  next-BD chain, but the hardware stalls when it reaches those descriptors.
- The diagnosis is not a four-column scaling issue: `onecol` times out, while
  `split_onecol` passes with the same rows and kernels after moving patch1 to
  channel 1 / BD `28,29`.

## What This Does Not Prove

- Production attention ABI.
- Q/K/V to edge attention and O-phase handoff.
- Multi-phase Q/K/V/O/up/gate/down sequencing over the exact chunk-ring ABI.
- Direct same-channel patch0/patch1 execution through high-level MLIR-AIE. That
  likely needs direct CDO/transaction ownership or another legal finite-phase
  mechanism.

## Run

```bash
# Reproduce the same-channel failure mode.
EXP62_VARIANT=onecol .venv/bin/python experiments/62_mylm_full_nblock_chunk_ring/run_npu.py

# Prove the full main16 chunk-ring schedule with legal channel/BD ownership.
EXP62_VARIANT=split_full .venv/bin/python experiments/62_mylm_full_nblock_chunk_ring/run_npu.py
```

Status:

```text
onecol:      compiles, then NPU timeout; same-channel ch0 crosses into BD 28+.
full:        compiles, then NPU timeout for the same reason at main16 scale.
split_onecol PASS, NPU time 2330.2 us.
split_full   PASS, NPU time 7132.1 us.
```
