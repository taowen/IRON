# 94_mylm_shape_a_carrier_producer_layout

This experiment refines the producer side of the Shape-A/B carrier ABI.

Exp93 proved that Shape-B consumes `base[0x100]` as four 0x40 vector blocks.
Exp94 asks how Shape-A writes that base window.

## Run

```bash
.venv/bin/python experiments/94_mylm_shape_a_carrier_producer_layout/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp94_mylm_shape_a_carrier_producer_layout/shape_a_carrier_producer_layout.txt
```

## Result

Shape-A helper `0x650` contains eight visible carrier-base stores:

```text
8 x vst wl0 = 8 x 0x20 bytes = 0x100 bytes
```

These are separate from the temporary stack stores of `wl6/wl10`. Combined with
exp93, the useful carrier model is:

```text
base[0x100]:
  8 records x 0x20 bytes
  each record fits one query head x 16 tokens x bf16

Shape-B consumer:
  4 vector blocks x 0x40 bytes
  each block consumes two adjacent 0x20 records
```

The Shape-B pipeline reads those pair blocks in this order:

```text
0x00, 0x40, 0xc0, 0x80
```

This read order should not be mistaken for a different physical allocation.
The physical base window is still the eight-record `0x100` byte region.

First-principles Qwen3 mapping says those eight records should correspond to
the eight local query heads in the current Q window. With the known Q-window
route, the likely global head groups are:

```text
c0r2: heads 0..7
c0r4: heads 8..15
c7r2: heads 16..23
c7r4: heads 24..31
```

This global-head labeling is the right implementation assumption for the fused
engine skeleton, but it is still a value-calibration target rather than a
disassembly-proven lane label.

Remaining unknowns:

- local head order of the eight 0x20 records;
- token lane order inside one 0x20 record;
- scalar[0x40] lane semantics.
