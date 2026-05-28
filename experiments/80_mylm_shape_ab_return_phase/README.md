# Experiment 80: MyLM Shape-A/B Handoff and Return Phase

This experiment closes the next static attention boundary after exp79. It
summarizes the core lock protocol between paired Shape-A and Shape-B tiles, then
checks the exact `c6r1` return-window phase that publishes the four Shape-B
outputs to O phase.

## Why This Exists

Exp79 resolved the rounded-history carrier:

```text
row0 history scan -> row1 K/V split -> shape-A K history and shape-B V history
```

The remaining static questions were:

- how the non-DMA-visible Shape-A to Shape-B state handoff is synchronized;
- whether `c6r1` streams each Shape-B output onward immediately or gathers four
  windows before publishing packet2.

## Run

```bash
.venv/bin/python experiments/80_mylm_shape_ab_return_phase/run.py --reuse
```

The artifact is written under `/tmp/iron_exp80_mylm_shape_ab_return_phase`:

- `shape_ab_return_phase_summary.txt`

## Result

Shape-A and Shape-B use a paired hidden-lock protocol:

```text
Shape-A:
  current input 512 dwords at 0x78000, locks L0 -> L1
  K-history input 2048 dwords at 0x70400, locks L2 -> L3
  core releases hidden-ready L7 and L5
  core waits on hidden-empty L4 before the main body
  no DMA-visible output

Shape-B:
  waits on hidden-ready imm 0x7 and 0x5
  waits on V-history L3
  releases hidden-empty imm 0x4
  publishes 512-dword DMA-visible output through locks L0/L1
```

The immediate values show that the low lock IDs pair as `L7/L5` ready and `L4`
empty, while the high bits differ across the vertical neighbor boundary. The
program constants also line up:

```text
Shape-A uses 0x62400 / 0x66000
Shape-B local state uses 0x72400 / 0x76000
```

The strongest current model is therefore a lock-synchronized hidden handoff
through a neighbor-local alias or related local/switch path. The exact hidden
payload layout is still not decoded, so this must not be modeled as a generic
packet stream or host-visible buffer.

The return phase order is now explicit:

```text
c6r1 ch2 bd4   base=0x28000 len=512  L71:-8 -> L72:+1
c6r1 ch3 bd27  base=0x28200 len=512  L72:-1 -> L73:+1
c6r1 ch4 bd5   base=0x28400 len=512  L73:-1 -> L74:+1
c6r1 ch5 bd28  base=0x28600 len=512  L74:-1 -> L75:+8
c6r1 ch11 bd29 base=0x28000 len=2048 packet2 L75:-1 -> L71:+1
```

So packet2 is not a per-Shape-B streaming pass-through. Four 512-dword Shape-B
outputs fill `c6r1:0x28000..0x28800`, and only then `c6r1.bd29` publishes one
2048-dword packet2 block. `c1r1` fans that block out as 16 O-input slices of
`256 bf16 = 128 dwords`.

## Implication

The fused-layer engine should implement attention return as a four-window gather
with an explicit publish barrier. The next implementation risk is the hidden
payload layout and local/remote lock encoding for the Shape-A to Shape-B
handoff, not the packet2/O return bridge.
