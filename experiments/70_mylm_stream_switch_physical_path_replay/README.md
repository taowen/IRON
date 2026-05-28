# Experiment 70: MyLM Stream-Switch Physical Path Replay

This experiment turns the MyLM stream-switch decode into explicit physical path
evidence. It is not a new fused-engine variant. Its purpose is to answer which
paths are packet paths, which paths are circuit-switched paths, and which parts
of the attention ingress remain unresolved.

## Why This Exists

The reverse tools corrected a bad assumption: the old "route selector" field is
really an AIE2P packet slave-slot `arbitor + msel` field. Destination ports come
from local packet-master rows, not from that low byte.

That changes the MyLM attention model:

- packet14/15 from `c1r3` do not directly feed `c0r2/c7r2` shape-A current DMA;
  those rows are packet transit hops toward row0 shim paths.
- `c1r3`'s 2048-dword `ch2` output is a circuit path, not a packet route.
- shape-A current `DMA_0` input is circuit-switched locally, but its upstream
  source crosses packet/circuit distributor logic in the row1/aux area.

The next engine cannot be a global Q/K/V sideband collector. It also cannot
connect packet14/15 directly into shape-A just because `c0r2/c7r2` mention those
packet ids.

## Evidence Generated

The script dumps or reuses the MyLM layer contract, then runs
`~/projects/MyLM/tools/re/aie_physical_path_trace.py`:

```bash
.venv/bin/python experiments/70_mylm_stream_switch_physical_path_replay/run.py
```

Artifacts are written under `/tmp/iron_exp70_mylm_path_replay`:

- `packet_14_15.txt`: hard packet-source paths from `c1r3`.
- `packet_0_2.txt`: packet sources from `c6r1`, useful for row1/aux distributor
  context.
- `circuit_c1r3.txt`: forward circuit path for the `c1r3` 2048-dword output.
- `reverse_shape_a_current.txt`: reverse circuit/packet trace from shape-A
  current inputs `c0r2 DMA_0` and `c7r2 DMA_0`.

## Current Result

Confirmed packet14:

```text
c1r3 DMA_1.slot1 pkt14 -> SOUTH_1
  c1r2 NORTH_1.slot0 pkt14 -> WEST_0
    c0r2 EAST_0.slot0 pkt14 -> SOUTH_1
      c0r1 NORTH_1.slot0 pkt14 -> SOUTH_1
        c0r0 NORTH_1.slot0 pkt14 -> SOUTH_2
```

Confirmed packet15:

```text
c1r3 DMA_1.slot0 pkt15 -> EAST_0
  c2r3 WEST_0.slot0 pkt15 -> EAST_2
    c3r3 WEST_2.slot0 pkt15 -> EAST_3
      c4r3 WEST_3.slot0 pkt15 -> EAST_3
        c5r3 WEST_3.slot0 pkt15 -> EAST_1
          c6r3 WEST_1.slot0 pkt15 -> SOUTH_2
            c6r2 NORTH_2.slot0 pkt15 -> EAST_2
              c7r2 WEST_2.slot0 pkt15 -> SOUTH_3
                c7r1 NORTH_3.slot0 pkt15 -> SOUTH_3
                  c7r0 NORTH_3.slot0 pkt15 -> SOUTH_2
```

Confirmed `c1r3` 2048-dword circuit path:

```text
c1r3 DMA_0 -> EAST_2
  c2r3 WEST_2 -> EAST_0
    c3r3 WEST_0 -> EAST_2
      c4r3 WEST_2 -> EAST_2
        c5r3 WEST_2 -> EAST_2
          c6r3 WEST_2 -> SOUTH_1
            c6r2 NORTH_1 -> SOUTH_3
              c6r1 NORTH_3 -> DMA_1
```

This is a stronger correction than exp69: the `2048`-dword `c1r3` sibling output
is not the shape-A current input. It lands in the right row1/aux distributor
area, where later local circuit paths can send data toward row0 or edge tiles.

Shape-A current reverse trace:

```text
c0r2 DMA_0 <= circuit EAST_1
c7r2 DMA_0 <= circuit WEST_3
```

Tracing those inputs backward reaches row1/aux distributor logic: left side goes
through `c1r2/c1r1`, right side goes through `c6r2/c6r1`. Exp72 later fixes an
over-broad inference in this reverse trace: a directional port used as a circuit
source must first cross to the neighbor output port, rather than being treated
as a same-tile packet-master destination. With that correction, this remains a
distributor clue, not a payload ABI: the exact current-window source, head
grouping, and packet/circuit arbitration order remain open.

## Value

This experiment removes a wrong implementation direction:

- Do not feed shape-A current from packet14/15 directly.
- Do not model the `c1r3` 2048-dword output as a packet route.
- Do not build a global Q/K/V sideband collector.

The deleted exp71-77 probes are folded into the current model: the distributor
is a Q-window path, packet14/15 are K/V row0 writeback, `c6r1` owns the four-way
Q split, and `c1r1/c1r2` are part of the packet2 return bridge rather than a
missing left-side Q-window mirror.
