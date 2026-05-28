# 90_mylm_main16_activation_bridge

This experiment closes the main16 activation-return bridge for both O and down
projection inputs.

Before this experiment, exp82 had already closed packet2 attention output into
O, and exp89 had shown that c6r1 publishes the 12288-bf16 SwiGLU output as
packet0 for down. The missing bridge question was whether packet0/down used a
different row1 path or the same activation ingress that O uses.

## Run

```bash
.venv/bin/python experiments/90_mylm_main16_activation_bridge/run.py --reuse
```

The artifact is written under `/tmp/iron_exp90_mylm_main16_activation_bridge`:

- `main16_activation_bridge.txt`

## Result

Both sources land in `c1r1 DMA_4`:

```text
packet2/O:
  c6r1.bd29 len=2048 packet2
  ... -> c1r1 NORTH_0 slot packet2 -> DMA_4

packet0/down:
  c6r1.bd1 len=6144 packet0
  ... -> c1r1 NORTH_1 slot packet0 -> DMA_4
```

`c1r1` then uses the same 256-dword ping-pong bridge for both:

```text
input ring on DMA4:
  bd6  base=0x28000 len=256 next=bd7  L69:-1 -> L70:+1
  bd7  base=0x2c000 len=256 next=bd6  L69:-1 -> L70:+1

output ring on DMA1:
  bd28 base=0x28000 len=256 next=bd29 L70:-1 -> L69:+1
  bd29 base=0x2c000 len=256 next=bd28 L70:-1 -> L69:+1
```

Every main16 tile consumes the same 128-dword activation ring:

```text
bd0 base=0x78000 len=128 next=bd1
bd1 base=0x7c000 len=128 next=bd0
```

Therefore one `c1r1` bridge quantum becomes two consecutive main16 activation
chunks:

```text
O:    2048 dwords / 256 =  8 bridge iterations -> 16 chunks
down: 6144 dwords / 256 = 24 bridge iterations -> 48 chunks
```

This exactly matches Qwen3:

```text
O input:    4096 bf16 = 2048 dwords = 16 x 256-bf16 chunks
down input: 12288 bf16 = 6144 dwords = 48 x 256-bf16 chunks
```

## Implication

The fused engine should model the main16 activation ingress as one reused
physical bridge:

```text
packet source -> c1r1 DMA4 256-dword input ring
              -> c1r1 DMA1 multicast
              -> main16 DMA0 128-dword bd0/bd1 activation ring
```

O and down are not separate transport mechanisms. They differ in phase and
source length only: O publishes eight bridge quanta; down publishes twenty-four.

Exp91 later closes the normal `c1r2` producer phase order around this bridge:
post-O residual/RMSNorm feeds the 48 up/gate replays, and final down residual
feeds the single hidden-output boundary transfer. The bridge itself is closed
well enough to implement the transport side of a fused layer engine.
