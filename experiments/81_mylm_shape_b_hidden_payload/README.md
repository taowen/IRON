# Experiment 81: MyLM Shape-B Hidden Payload Inference

This experiment tightens the unresolved exp80 question: what Shape-A hands to
Shape-B through the hidden lock protocol.

It is still a static reverse-engineering experiment. It does not prove exact
field order inside the hidden buffer, but it constrains the payload size and
lifecycle enough to guide the next calibration kernel.

## Run

```bash
.venv/bin/python experiments/81_mylm_shape_b_hidden_payload/run.py --reuse
```

The artifact is written under `/tmp/iron_exp81_mylm_shape_b_hidden_payload`:

- `shape_b_hidden_payload_inference.txt`

## Result

The Shape-A/B pair naturally covers:

```text
8 query heads
2 KV heads
16 tokens per rounded history tile
GQA = 4 query heads per KV head
```

The static evidence now lines up exactly:

```text
Shape-A current input: 512 dwords = 8 query heads x 128 dim x bf16
Shape-A K history:    2048 dwords = 16 tokens x 2 KV heads x 128 dim x bf16
Shape-B V history:    2048 dwords = same V layout
Shape-B output:        512 dwords = 8 query heads x 128 dim x bf16
```

The strongest hidden-payload clue is one Shape-B local carrier/accumulator
window:

```text
0x76000..0x76140 = 0x140 bytes carrier-capacity window
0x76140..0x77140 = 0x1000 bytes
```

First-principles attention math gives:

```text
16-token softmax weights for 8 query heads:
  8 * 16 * bf16 = 0x100 bytes

online-softmax scalar state for 8 query heads:
  8 * 2 * fp32 = 0x40 bytes

total compact carrier:
  0x100 + 0x40 = 0x140 bytes

output accumulator:
  8 * 128 * fp32 = 0x1000 bytes
```

The disassembly confirms the output half. Every Shape-B tile performs a final
conversion window:

```text
0x1750..0x191a:
  64 vlda bmll0, [p2], #0x40      -> 0x1000 bytes read
  64 vst.conv.bf16.fp32, #0x20    -> 0x800 bytes written
```

`0x800` bytes is exactly the 512-dword Shape-B DMA output.

The alias evidence from exp80 also matches, and exp84 later refines these as
the two hidden ping-pong carrier aliases:

```text
Shape-A 0x62400 -> Shape-B 0x72400, delta 0x10000
Shape-A 0x66000 -> Shape-B 0x76000, delta 0x10000
```

## Implication

The next fused-engine calibration should not try to stream a full attention
vector from Shape-A to Shape-B. It should model Shape-A as the score/online
softmax carrier producer:

```text
Q window + K history -> 0x140-byte hidden carrier
hidden carrier + V history -> 0x1000-byte fp32 accumulator -> 512-dword output
```

The exact byte order inside the compact carrier is still not fully decoded.
Exp84 resolves the carrier aliases and local/north-neighbor lock ownership.
Exp93 later refines the 0x40-byte state as online-softmax scale/normalization
lanes; static code does not prove the lanes are literal raw max/sum values.
