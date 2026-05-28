# 84_mylm_shape_ab_carrier_lock

Tightens the remaining Shape-A/B handoff question after exp81.

The key correction is that `0x76000..0x76140` should not be described as the
only hidden carrier. Shape-A uses two hidden aliases, `0x62400` and `0x66000`;
Shape-B sees the matching local bases as `0x72400` and `0x76000`, with a
consistent `+0x10000` vertical-neighbor delta. These are the hidden carrier
ping-pong aliases for the Shape-A/B pair.

The lock ownership is now clear enough to reproduce:

```text
Shape-A owns local hidden locks:
  ready L7, ready L5, empty L4

Shape-B accesses the same north-neighbor locks:
  acq 0x7, acq 0x5, rel 0x4
```

The exact byte order inside the compact carrier is still unresolved. The old
capacity model, `0x100` bytes of 16-token weights plus `0x40` bytes of fp32
running max/sum, remains the right size, but it must be treated as a payload
hypothesis rather than a decoded ABI field order.
