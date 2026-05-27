# Experiment 38: Full Attention Fabric

Exp37 closed one 4Q:1KV GQA group after NPU-side Q/K/V projection. Exp38
scales the attention fabric itself to the full Qwen-style attention shape:
`32Q:8KV`.

```text
8 columns = 8 KV groups
4 compute rows per column = 4 Q heads sharing that KV group

per column:
  query[4 heads] + K/V history for one KV group
    -> row1 reads K/V once
    -> row1 reshapes token-major K/V to dim-group-major
    -> row1 fans query slice + K/V stream to 4 workers
    -> 4 workers produce one head each
    -> packet gather back to row1
    -> row1 drains a 512-dword output block
```

## What It Proves

- A full `32Q/8KV` attention fabric maps naturally to `8 columns x 4 rows`.
- Every KV group reads K/V history once per tile; fanout to four Q heads happens
  inside row1.
- Worker input is a single phase-ordered stream from row1: query first, then
  K/V history tiles. This preserves the static-route lesson from exp37.
- Four worker outputs per column can be gathered through packet switching into
  row1 and drained as a single per-group output block.

## Scope

This experiment intentionally keeps query and cache host-provided. Exp37 already
proved that NPU-side Q/K/V projection and current-write can feed the one-read
fanout path. Exp38 focuses on the full attention resource map:

- no Q/K/V projection,
- no current-token writeback,
- no Q/K norm or RoPE,
- no O projection or FFN.

The contract is still important because it answers whether the full attention
shape fits the physical `8 x 4` compute tile fabric before reattaching
projection and the rest of the layer.

## Result

Real NPU run:

```text
L=17  PASS  time=6117.1us  max_abs=0.0007520
L=31  PASS  time=6858.3us  max_abs=0.0003763
L=32  PASS  time=6888.0us  max_abs=0.0003101
L=79  PASS  time=9697.8us  max_abs=0.0002061
```

The important conclusion is resource-level, not latency-level: all 32 compute
tiles, 8 row1 fanout/gather blocks, and 32 packet-gather outputs fit and route.
This answers the outstanding attention-fabric question left by exp37.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/38_full_attention_fabric/run_npu.py
```
