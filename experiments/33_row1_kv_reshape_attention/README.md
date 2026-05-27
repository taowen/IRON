# Experiment 33: Row1 KV Reshape Attention

This experiment isolates the MyLM-style row1 KV-cache boundary before adding
projection phases, fanout across many columns, or edge-tile aggregation.

It proves that a token-major KV history tile can be loaded into row1 memtile
storage and reshaped by static BD dimensions before the compute tile consumes
it for online attention.

## Contract

- Input query is four Q heads, `4 x 128` `f32`.
- KV cache is two planes, K then V, each rounded to 16-token tiles.
- Each raw plane tile is token-major: `[16 tokens][128 dim]`.
- Row1 memtile sends each tile as dim-group-major:
  `[32 dim-groups][16 tokens][4 dims]`.
- The worker runs online softmax attention for a `4Q:1KV` GQA group.
- Non-16-aligned context lengths are tail-masked in the worker, so padding
  sentinels in the rounded tile must not affect the output.

The critical static BD reshape is:

```mlir
[<size = 32, stride = 4>, <size = 16, stride = 128>, <size = 4, stride = 1>]
```

which emits:

```text
for dim_group in 0..31:
  for token in 0..15:
    for lane in 0..3:
      raw[token * 128 + dim_group * 4 + lane]
```

## Scope

This is not a complete layer. It intentionally excludes Q/K/V projection,
current-token cache writeback, four-column fanout, and edge-worker packet
aggregation. Those pieces were covered by earlier experiments. This experiment
focuses on the missing row1 reshape contract that explains why MyLM does not
feed raw token-major KV directly into attention.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/33_row1_kv_reshape_attention/run_npu.py
```
