# Experiment 27: Shape-A/Shape-B Single-Tile Attention Contract

Exp27 is the first runnable attention experiment after the exp25/exp26 edge
calibration work.  It deliberately keeps the scope to one 16-token tile so the
shape-A/shape-B split can be verified before scaling to all KV groups.

## Goal

Prove this minimal MyLM-style attention boundary on real NPU:

```text
current query-side vector
  -> c1r3 packet14
  -> c0r2 shape-A current input

K history tile
  -> shim0
  -> row1 memtile ring
  -> c0r2 shape-A history input

c0r2 shape-A
  -> 17-float compact softmax sideband
  -> c6r2 sideband relay/debug
  -> c6r3 shape-B sideband input

V history tile
  -> shim7
  -> row1 memtile ring
  -> c6r3 shape-B history input

c6r3 shape-B
  -> 512-float attention output
```

## What It Proves

- packet14 current routing can feed a shape-A-like two-input worker;
- shape-A can consume current + K history and emit only a compact 17-word state;
- the compact state can cross to a shape-B-like worker without a host-visible
  intermediate tensor;
- shape-B can consume compact state + V history and produce a 512-word output;
- tail masking for L=17/31/32/128 works through the compact state.

The 17-word sideband is modeled as 16 unnormalized softmax weights plus one
denominator.  That is not claimed to be MyLM's exact representation; it is the
smallest useful contract that exercises the same physical route shape.

## Run

```bash
cd /var/home/taowen/projects/IRON/experiments/27_shape_ab_attention_contract
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

Expected result:

```text
L=17   PASS
L=31   PASS
L=32   PASS
L=128  PASS
SUCCESS: exp27 shape-A/shape-B attention contract verified on NPU.
```

## Non-Goals

- no Q4 projection;
- no multi-tile online accumulation;
- no GQA fanout;
- no full layer engine.

The next experiment should extend this from one 16-token tile to repeated
history tiles while keeping the same packet current, row1 history ring, and
shape-A/shape-B compact-state handoff.
