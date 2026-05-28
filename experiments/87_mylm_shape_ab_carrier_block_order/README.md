# 87_mylm_shape_ab_carrier_block_order

This experiment narrows the unresolved Shape-A/B hidden-carrier ABI.

Exp81 proved the carrier capacity:

```text
0x140 bytes = 0x100 bytes score/weight state + 0x40 bytes online-softmax state
```

Exp84 proved the aliases and locks:

```text
Shape-A aliases 0x62400/0x66000
Shape-B locals  0x72400/0x76000
Shape-A local ready L7/L5, empty L4
Shape-B north-neighbor acq 0x7/0x5, rel 0x4
```

Exp87 adds the block-phase order:

```text
Shape-A:
  rel L7 first
  acq L4
  helper 0x650 stores base[0x100]
  post-call store through dj0=0x100
  rel L5 second

Shape-B:
  acq north L7
  acq north L5
  hidden consume window splits p0=base, p1/p3=base+0x100
  rel north L4
```

## Run

```bash
.venv/bin/python experiments/87_mylm_shape_ab_carrier_block_order/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp87_mylm_shape_ab_carrier_block_order/shape_ab_carrier_block_order.txt
```

## Result

The engine should implement Shape-A/B carrier handoff as a two-ready protocol
with an explicit `base[0x100] + scalar[0x40]` split, not as one monolithic
buffer and not as a full attention vector.

What is now proven:

- Shape-A publishes two ready phases in order: L7 then L5.
- Shape-B waits on both ready phases before using the carrier and releases L4
  after consumption.
- Shape-B's call-slot setup splits the carrier into `p0=base` and
  `p1/p3=base+0x100`.
- The same-bundle pointer rule is required: `mov p0,p1` reads the
  pre-increment base while `paddb [p1], #0x100` advances p1 for the scalar
  block.

Implementation model:

- `base[0x100]`: 8 heads x 16 token bf16 softmax weights.
- `scalar[0x40]`: per-head fp32 online-softmax scale/normalization state.

What remains unknown:

- head/token order inside `base[0x100]`;
- exact scalar lane semantics inside `scalar[0x40]`.

Exp93 later narrows `base[0x100]` to four Shape-B 0x40 consumer blocks and
keeps `scalar[0x40]` opaque until value calibration names the lanes.
