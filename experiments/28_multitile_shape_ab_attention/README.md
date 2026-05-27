# Experiment 28: Multi-Tile Shape-A/Shape-B Online Attention

Exp27 proved a single 16-token shape-A/shape-B attention boundary.  Exp28
extends the same physical contract across multiple KV history tiles.

## Goal

Verify that a compact 17-word sideband can drive tiled online attention:

```text
current query-side vector
  -> c1r3 packet14
  -> c0r2 shape-A
  current is sent once and reused for every history tile

K history tiles
  -> shim0
  -> row1 memtile static ring
  -> c0r2 shape-A

c0r2 shape-A, per tile:
  scores[16] = Q dot K_tile
  sideband[0:16] = exp(scores - local_max)
  sideband[16] = local_max

sideband
  -> c6r2 relay/debug
  -> c6r3 shape-B

V history tiles
  -> shim7
  -> row1 memtile static ring
  -> c6r3 shape-B

c6r3 shape-B:
  maintains running_max, running_sum, and running_output
  emits final normalized attention vector
```

## What It Proves

- current packet is not repeated per KV tile;
- K/V history scale with runtime descriptor length, not with new runtime tasks;
- row1 static rings can feed multiple history tiles into shape-A and shape-B;
- 17-word compact state is sufficient for online softmax merging across tiles;
- tail masking works for non-16-aligned context lengths.

## Run

```bash
cd /var/home/taowen/projects/IRON/experiments/28_multitile_shape_ab_attention
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

Expected result:

```text
L=17   PASS
L=31   PASS
L=32   PASS
L=64   PASS
L=128  PASS
L=129  PASS
SUCCESS: exp28 multi-tile shape-A/B online attention verified on NPU.
```

## Non-Goals

- no Q4 projection;
- no packet15/right-side mirror;
- no 4-Q-head GQA group;
- no full layer engine.

The next useful step is to expand this contract from one logical head to a
4-Q-head/1-KV-head GQA group while preserving the same K/V ring and compact
sideband structure.
