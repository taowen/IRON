# Experiment 57: MyLM Exact Patch Manifest

Experiment 56 proved that linked BD batches can feed the full Qwen3 layer
schedule, but it still coalesces two MyLM weight patches into one descriptor per
main column and 512-row output block.

This experiment fixes the contract at the transaction-manifest level. It
generates the exact MyLM-style weight patch list:

- phase order: `Q, K, V, O, UP, GATE, DOWN`
- one N-block = 512 output rows
- one N-block = 8 DDR patches:
  - `c2`: patch pair `0, 1`
  - `c3`: patch pair `0, 1`
  - `c4`: patch pair `0, 1`
  - `c5`: patch pair `0, 1`
- one patch = 64 output rows x full input dimension
- BD bank alternates every N-block
- total weight patches = 608

## What This Proves

- The exact MyLM patch unit is 64 output rows, not the 128-row per-column
  coalesced descriptor used by experiment 56.
- The full Qwen3 layer has the expected patch counts and byte sizes:
  - `4096` input dim patch: `0x28000` bytes
  - `12288` input dim patch: `0x78000` bytes
- The generated patch order matches the documented MyLM transaction shape:
  projection-major, then N-block, then column, then the two row-pair patches.

## What This Does Not Prove

- NPU execution.
- The row1 memtile ABI for splitting a 64-row patch into compute-tile streams.
- Exact CDO binary encoding.

The next hardware experiment should consume this manifest shape instead of the
coalesced exp56 descriptor shape.

## Run

```bash
.venv/bin/python experiments/57_mylm_exact_patch_manifest/run.py
```
