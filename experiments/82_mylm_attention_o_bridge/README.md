# Experiment 82: MyLM Attention-O Bridge Quantum

This experiment closes a subtle return-side layout question from exp80: how the
2048-dword packet2 attention vector becomes the 16 O-phase activation chunks
that the main16 projection workers consume.

## Run

```bash
.venv/bin/python experiments/82_mylm_attention_o_bridge/run.py --reuse
```

The artifact is written under `/tmp/iron_exp82_mylm_attention_o_bridge`:

- `attention_o_bridge_summary.txt`

## Result

`c6r1.bd29` publishes one packet2 block:

```text
c6r1.bd29 base=0x28000 len=2048 packet2
```

`c1r1` receives and republishes this through a 256-dword ping-pong bridge:

```text
input ring:
  bd6  base=0x28000 len=256 next=bd7  L69:-1 -> L70:+1
  bd7  base=0x2c000 len=256 next=bd6  L69:-1 -> L70:+1

output ring:
  bd28 base=0x28000 len=256 next=bd29 L70:-1 -> L69:+1
  bd29 base=0x2c000 len=256 next=bd28 L70:-1 -> L69:+1
```

Every main16 tile has the same activation-input ring:

```text
bd0 base=0x78000 len=128 next=bd1
bd1 base=0x7c000 len=128 next=bd0
```

Therefore the return layout is:

```text
packet2 2048 dwords / c1r1 256-dword bridge quantum = 8 bridge iterations
each c1r1 quantum / main16 128-dword activation BD = 2 O chunks
total O chunks = 16
```

So the exact order is linear packet2 order, two O chunks per bridge iteration:

```text
bridge_iter0 -> O chunks 0,1
bridge_iter1 -> O chunks 2,3
...
bridge_iter7 -> O chunks 14,15
```

`c1r1 DMA_1` multicast reaches all 16 main16 DMA0 activation inputs, so each O
chunk is broadcast to every main worker, matching the projection K-chunk model.

## Implication

The fused engine should not create 16 separate return descriptors for attention
output. It should publish a linear 2048-dword packet2 block, use the c1r1
256-dword ping-pong bridge, and let each main tile's `bd0->bd1` ring split the
bridge quantum into the two consecutive 128-dword O input chunks.
