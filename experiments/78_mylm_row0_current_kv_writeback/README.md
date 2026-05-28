# Experiment 78: MyLM Row0 Current K/V Writeback

This experiment resolves the row0 side of packet14/15: they are not generic
row0/FIFO traffic. They line up with the arg4 current K/V cache-writeback
descriptors.

## Why This Exists

Exp77 fixed row0 shim decoding and showed packet14/15 terminate at row0 shim
`SOUTH_2` packet masters:

```text
packet14 -> c0r0 SOUTH_2
packet15 -> c7r0 SOUTH_2
```

The missing question was semantic: do those packet streams actually feed the
current-token K/V cache writeback, or are they another row0-side path?

## Run

```bash
.venv/bin/python experiments/78_mylm_row0_current_kv_writeback/run.py --reuse
```

The artifact is written under `/tmp/iron_exp78_mylm_row0_current_kv_writeback`:

- `row0_current_kv_writeback_summary.txt`

## Result

The static evidence closes by size, endpoint, and transaction descriptor:

```text
packet14 sources:
  c1r3.bd2 len=256
  c1r3.bd4 len=256
  total=512 dwords

packet15 sources:
  c1r3.bd3 len=256
  c1r3.bd5 len=256
  total=512 dwords
```

Qwen3 current K is `1024 bf16 = 512 dwords`; current V is also `512` dwords.

The row0 current-write descriptors are:

```text
c0r0.bd0 len=0x100 arg4 offsets:
  k03@0x7800
  k47@0x7800

c7r0.bd0 len=0x100 arg4 offsets:
  v03@0x7800
  v47@0x7800
```

Those match packet endpoints:

```text
packet14 -> c0r0 NORTH_1.slot0 -> SOUTH_2 drop_header=1
packet15 -> c7r0 NORTH_3.slot0 -> SOUTH_2 drop_header=1
```

Therefore the strongest static ABI is:

```text
packet14 = current K writeback, two 256-dword KV-group chunks
packet15 = current V writeback, two 256-dword KV-group chunks
```

The later history scan is organized differently:

```text
c0r0 scans k03/v03 history windows
c7r0 scans k47/v47 history windows
```

So the fused engine should emit packet14/15 as current K and current V into row0
shim S2MM/cache-writeback, then run the rounded history scan grouped by KV group.
The remaining unsolved part is no longer packet14/15 semantics; it is exact phase
ordering and the row1 history split into shape-A/shape-B consumers.
