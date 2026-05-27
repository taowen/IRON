# Experiment 36: Current-Write One-Read KV Fanout Attention

Exp35 proved that one central row1 memtile can read K/V history once, reshape it,
and fan it out to four attention workers. Exp36 adds the missing decode
ordering boundary: current K/V is written into the cache BO first, then the same
one-read KV fanout path scans the updated cache.

```text
current K/V input -> current writer tile -> S2MM into kv_cache current slot
  -> sync current writer
  -> central KV reader scans updated K/V cache once
  -> row1 static reshape + four-channel fanout
  -> four one-head attention workers
```

## What It Proves

- NPU-side current K/V writeback can be sequenced before one-read KV scan in a
  single runtime sequence.
- The central KV reader sees the just-written current token.
- The four fanout attention workers still match the same CPU reference for
  `L=17/31/32/79`.
- Runtime K/V history scan remains one-read: `2 * num_tiles` patches, plus two
  current-write S2MM patches.

## Scope

This is still not the full MyLM layer engine:

- current K/V and query are host-provided rather than produced by Q/K/V
  projection,
- no Q/K norm or RoPE,
- no edge/aux attention split.

The experiment isolates the decode cache boundary that must exist before
projection can be reattached to the one-read fanout path.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/36_current_write_one_read_fanout_attention/run_npu.py
```

Observed real-NPU result:

```text
L=17 PASS max_abs=0.0000242 cache_max_abs=0.0000000
L=31 PASS max_abs=0.0000197 cache_max_abs=0.0000000
L=32 PASS max_abs=0.0000182 cache_max_abs=0.0000000
L=79 PASS max_abs=0.0000144 cache_max_abs=0.0000000
```
