# Experiment 35: One-Read KV Fanout Attention

Exp34 proved four-column GQA-head attention, but each column still read the
same logical K/V history from DDR. Exp35 removes that duplicate runtime KV scan:
one central row1 memtile reads each K and V tile once, reshapes the token-major
plane with static BD dimensions, then fans the reshaped K/V streams out to four
attention workers.

```text
query head 0..3 -> local row1 query path -> worker 0..3

K/V history DDR -> column 4 row1 ping-pong buffers
  -> static BD token-major to dim-group-major reshape
  -> four MM2S fanout channels
  -> worker 0..3 one-head online attention
  -> output head slices
```

## What It Proves

- The exp33 row1 reshape can be driven from one shared K/V reader instead of
  one reader per query head.
- The shared row1 ping-pong lock protocol can safely handle four consumers:
  input DMAs acquire four empty credits, and each fanout channel releases one
  credit after consuming a buffer.
- Worker-side attention is unchanged from exp34, so output still matches the
  same CPU reference for `L=17/31/32/79`.
- Runtime K/V descriptor count drops from `4 * 2 * num_tiles` to
  `2 * num_tiles`.

## Scope

This is still not the full MyLM layer engine:

- no Q/K/V projection,
- no current-token cache writeback,
- no edge/aux attention split,
- query and output still use per-column local paths.

The experiment isolates the key post-exp34 gap: removing repeated K/V DDR reads
while preserving four parallel attention consumers.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/35_one_read_kv_fanout_attention/run_npu.py
```

Observed real-NPU result:

```text
L=17 PASS max_abs=0.0000285
L=31 PASS max_abs=0.0000171
L=32 PASS max_abs=0.0000157
L=79 PASS max_abs=0.0000111
```
