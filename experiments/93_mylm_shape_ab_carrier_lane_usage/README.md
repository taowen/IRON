# 93_mylm_shape_ab_carrier_lane_usage

This experiment tightens the remaining Shape-A/B carrier ambiguity after exp87.

Exp87 proved the hidden handoff protocol and split:

```text
base[0x100] + scalar[0x40]
```

Exp93 asks how Shape-B actually consumes those bytes.

## Run

```bash
.venv/bin/python experiments/93_mylm_shape_ab_carrier_lane_usage/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp93_mylm_shape_ab_carrier_lane_usage/shape_ab_carrier_lane_usage.txt
```

## Result

Shape-B has two relevant carrier consumers:

```text
0x410: reads scalar through p1 and base through p0
0x600: reads base through p0 and scalar dwords through p3
```

Both consumers read `base[0x100]` as the same four 0x40 blocks:

```text
0x00, 0x40, 0xc0, 0x80
```

So the Shape-B side should model `base[0x100]` as four vector-load blocks, not
as one opaque 0x100-byte blob. A 0x40 block exactly fits:

```text
2 query heads x 16 tokens x bf16
```

The likely structure is therefore four head-pair blocks. The exact physical
head-pair order, head order inside a pair, and token lane order still need a
value-calibration experiment.

`scalar[0x40]` also needs more careful wording than "max/sum":

- target `0x410` vector-loads the full 0x40 scalar block;
- target `0x600` reads the first eight dwords from offsets `0x00..0x1c`;
- first-principles online softmax says the block must carry per-head
  rescale/normalization state, but the exact lane labels are not statically
  proven.

The next value experiment should keep `scalar[0x40]` as 16 opaque dword lanes
until it maps them to online-softmax variables.

Exp94 refines the producer side: Shape-A writes `base[0x100]` as eight
0x20-byte head-sized records, which Shape-B consumes as four 0x40 head-pair
blocks.
