# Experiment 58: MyLM Patch-Pair Row1 Split

Experiment 57 fixed the exact MyLM weight patch manifest: each main column
receives two 64-row patches per 512-row N-block. Experiment 56 still coalesced
those two patches into one 128-row descriptor before row1/memtile split.

This experiment validates the missing hardware ABI directly.

## Contract

- One main column: `c2`.
- Four compute rows: `r2..r5`.
- Runtime submits two linked shim MM2S descriptors:
  - patch pair 0: rows 0 and 1
  - patch pair 1: rows 2 and 3
- Row1/memtile receives two separate `64-row` patch-pair buffers and splits:
  - patch0 offset 0 -> row0
  - patch0 offset 1 chunk -> row1
  - patch1 offset 0 -> row2
  - patch1 offset 1 chunk -> row3
- Each compute tile checks the raw bf16 words it received and emits a compact
  `(row, first, last, checksum)` record.

## What This Proves

- MyLM's 64-row patch pair can feed the four-row main fabric without first
  coalescing into a 128-row fat chunk.
- The row1/memtile split can map two independent patch buffers to four compute
  rows.
- A linked BD chain can deliver the two patch pairs as a coarse host
  transaction.
- The memtile BD bank mapping matters: row `MM2S` channel 1/3 must use the high
  BD bank (`24`, `26` here), matching the pattern seen in the larger phase-chain
  experiments.

## What This Does Not Prove

- Full-K projection math.
- The full 608-patch layer transaction.
- Exact MyLM core program.

## Run

```bash
.venv/bin/python experiments/58_mylm_patch_pair_row1_split/run_npu.py
```

Last verified on real NPU:

```text
NPU time: 841.0 us
PASS: two 64-row MyLM patch pairs were split by row1 into four compute rows.
```
