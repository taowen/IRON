# Experiment 65: MyLM Fused Layer Engine V0

This experiment starts the fused layer engine instead of another isolated
single-boundary probe.

It combines the retained results that now matter:

- exp54's real Qwen3 phase/block/chunk schedule,
- exp57's exact 608-patch MyLM manifest shape,
- exp63's packetized logical patch queue,
- exp64's direct attention-result-to-O handoff.

## Contract

- Main projection fabric is fixed to `c2..c5/r2..r5`.
- Edge/aux fabric is fixed to `c0/c1/c6/c7`, rows `2..5`.
- Phase order is `Q, K, V, O, UP, GATE, DOWN`.
- The global weight BO is ordered as exact MyLM patches:
  phase -> 512-row block -> main column -> 64-row patch pair.
- Each main column receives `152` packetized patch descriptors:
  `76` logical output blocks x `2` patch pairs per column.
- Runtime patches linked shim BD batches of up to `14` descriptors, pushes only
  the batch head, and waits once per batch.
- Shim packet IDs route each patch descriptor to legal row1 channels:
  patch pair 0 -> memtile `S2MM ch0`, patch pair 1 -> memtile `S2MM ch1`.
- Row1 uses small Q4NX chunk rings, not fat chunks or full-patch residency.
- O phase activation slices come from a deterministic attention-result replay
  producer and stream directly into O. There is no host-visible attention drain.
- Final output is one compact record per main tile for validation.

## Schedule

| phase | input dim | output dim | 512-row blocks | K chunks | patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 4096 | 4096 | 8 | 16 | 64 |
| K | 4096 | 1024 | 2 | 16 | 16 |
| V | 4096 | 1024 | 2 | 16 | 16 |
| O | 4096 | 4096 | 8 | 16 | 64 |
| UP | 4096 | 12288 | 24 | 16 | 192 |
| GATE | 4096 | 12288 | 24 | 16 | 192 |
| DOWN | 12288 | 4096 | 8 | 48 | 64 |

Total: `608` exact MyLM patches.

Patch sizes:

```text
4096 input-dim patch:  0x28000 bytes
12288 input-dim patch: 0x78000 bytes
```

## What This Proves

- The real Qwen3 seven-phase patch schedule can run through packetized exact
  MyLM patch descriptors, not only exp54's contiguous fat stream.
- The exp63 packetized patch ABI scales from one O block to the full 608-patch
  layer-shaped schedule.
- The exp64 attention-result-to-O stream composes with the full phase schedule.
- High-level MLIR-AIE can still generate the required descriptor/RTP program for
  this v0 engine; direct CDO generation is not required for this contract yet.

## What This Does Not Prove

- Production attention math. The O activation producer is deterministic replay,
  not rounded KV scan, online softmax, or weighted V.
- Real data dependencies between Q/K/V, attention, O, UP/GATE/DOWN.
- High-throughput Q4NX performance. Kernels remain contract kernels.
- Final transaction-builder quality. The descriptor program is still generated
  from Python strings rather than a reusable CDO builder.

## Run

```bash
.venv/bin/python experiments/65_mylm_fused_layer_engine_v0/run_npu.py
```

Status:

```text
PASS, NPU time 709976.3 us.
```
