# Experiment 59: MyLM Exact N-Block Projection

Experiment 58 proved only the first K-chunk of MyLM's two `64-row` patch
descriptors. This experiment combines the next two planned steps: it uses the
real `64 rows x K=4096` patch size and scales directly to the full four-column
main projection fabric.

## Contract

- Main projection fabric: `c2..c5/r2..r5`.
- Edge/source fabric: `c0/c1/c6/c7/r2..r5` generates deterministic activation
  slices and streams them to the matching main tile.
- Each main column receives two linked shim MM2S descriptors:
  - patch 0: output rows `0..63` for that column
  - patch 1: output rows `64..127` for that column
- Each patch is a real full-K MyLM-sized patch:
  - `64 output rows x 4096 input columns`
  - `0x28000` bytes
  - internally stored as 16 K-chunks, each chunk containing two `32-row` Q4NX
    shards.
- Row1/memtile uses strided static BDs to split each full patch into row-local
  `32 x 256` Q4NX chunks.
- Each compute tile consumes 16 activation slices and 16 weight chunks, runs
  Q4NX online dequant+MAC, and emits one debug output record.

## What This Proves

- The exact MyLM 64-row patch descriptor can be consumed without coalescing into
  a 128-row fat chunk.
- A full `512-row x 4096` N-block maps to `4 columns x 4 compute rows`.
- Row1 can split full patches into per-row K-chunk streams with static BD
  strides while the main tile consumes activation/weight phase streams.
- This is the first hardware experiment where the exact patch ABI and the full
  main16 projection fabric are validated together.

## What This Does Not Prove

- Q/K/V/O/up/gate/down phase sequencing over the exact patch ABI.
- Real attention output as the activation source.
- MyLM's exact core program.
- MyLM's row1 streaming storage efficiency. This experiment buffers one full
  `0x28000` patch in row1; the compiler falls back from bank-aware allocation to
  sequential memtile allocation. The next MyLM-oriented step should turn this
  into smaller static rings instead of full-patch residency.

## Run

```bash
.venv/bin/python experiments/59_mylm_exact_nblock_projection/run_npu.py
```

Last verified on real NPU:

```text
NPU time: 4721.8 us
PASS: exact MyLM 64-row patches fed a full 512-row main16 projection N-block.
```
