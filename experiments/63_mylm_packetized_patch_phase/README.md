# Experiment 63: MyLM Packetized Patch Phase

Exp62 proved that the full main16 small chunk-ring schedule is correct when
patch0 and patch1 use separate physical input channels. It also proved why the
same-channel variant times out: memtile `S2MM ch0` cannot legally execute BD
slots `24..47`.

Exp63 tests the next physical question: can one host-side logical patch chain
still feed legal row1 BD banks if each linked shim descriptor carries a packet
ID?

## Contract

- Host/runtime pushes one linked `MM2S ch0` queue per main column.
- Shim BD0 carries patch0 and packet ID `16 + group * 2`.
- Shim BD1 carries patch1 and packet ID `17 + group * 2`.
- Stream switch packet routes send:
  - patch0 packet -> memtile `S2MM ch0`, BD `0/1`
  - patch1 packet -> memtile `S2MM ch1`, BD `28/29`
- Row1 still uses the exp62 small slot-locked chunk rings:
  - patch0 ping/pong feeds rows `0,1`
  - patch1 ping/pong feeds rows `2,3`
- Output packets still return to memtile `DMA : 2`; output drain uses shim
  `S2MM ch0`.

This keeps the host-visible ABI close to a single MyLM-style descriptor chain
while avoiding the hardware-illegal `S2MM ch0 -> BD 24..47` path.

## What This Proves

- High-level MLIR-AIE can express packetized shim descriptors for
  host-to-memtile routing.
- One runtime input queue can drive two row1 physical DMA channels without host
  chunk pushes.
- Packetized descriptors are enough for this same-logical-channel patch phase.
  Direct CDO/transaction generation is not required for this specific handoff.
- The remaining direct-CDO question should move to later cases where packetized
  descriptors cannot express the needed phase ownership.

## Run

```bash
# Small diagnostic first.
EXP63_VARIANT=packet_onecol .venv/bin/python experiments/63_mylm_packetized_patch_phase/run_npu.py

# Full main16 version.
EXP63_VARIANT=packet_full .venv/bin/python experiments/63_mylm_packetized_patch_phase/run_npu.py
```

Status:

```text
packet_onecol PASS, NPU time 2841.5 us.
packet_full   PASS, NPU time 8955.4 us.
```
