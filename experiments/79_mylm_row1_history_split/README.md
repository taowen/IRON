# Experiment 79: MyLM Row1 History Split

This experiment resolves the static rounded-history carrier from row0 KV-cache
scan into row1 memtile and then into shape-A/shape-B edge consumers.

## Why This Exists

Exp78 showed packet14/15 are current K/V writeback, and that the later history
scan is grouped differently:

```text
c0r0 scans k03/v03 history windows
c7r0 scans k47/v47 history windows
```

The remaining question was whether row1 broadcasts a generic history sideband or
performs a precise K/V split for the edge attention fabric.

## Run

```bash
.venv/bin/python experiments/79_mylm_row1_history_split/run.py --reuse
```

The artifact is written under `/tmp/iron_exp79_mylm_row1_history_split`:

- `row1_history_split_summary.txt`

## Result

For Qwen3, one rounded `16-token x 4-KV-head x 128-dim bf16` plane is:

```text
16 * 4 * 128 bf16 = 4096 dwords
```

The MyLM row1 memtiles use two independent ping-pong rings per side:

```text
c0r1/c7r1 K ring: 0x20000/0x24000
c0r1/c7r1 V ring: 0x28000/0x2c000
```

Each 4096-dword ring is split into two 2048-dword streams:

```text
K -> shape-A heads0/1 and heads2/3
V -> shape-B heads0/1 and heads2/3
```

The physical paths are:

```text
c0r0 k03/v03 -> c0r1 -> c0r2/c0r4 shape-A K, c0r3/c0r5 shape-B V
c7r0 k47/v47 -> c7r1 -> c7r2/c7r4 shape-A K, c7r3/c7r5 shape-B V
```

So the rounded-history ABI is no longer a global Q/K/V sideband collector. It is
grouped by KV group at row0, then split by K versus V and head-pair in row1. The
remaining unresolved attention boundary is the hidden shape-A to shape-B state
payload and the exact phase order that fills and publishes the four shape-B
return windows.
