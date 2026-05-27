# Experiment 50: Edge Slice Replay To Q4NX O Phase

This experiment targets the ABI risk found after exp49: the edge/attention path
must not be modeled as returning a full activation tensor to the main projection
tile. MyLM evidence points to a compact edge shard that is replayed in chunks.

The contract tested here is:

```text
main16 tile
  -> 17-dword compact sideband
  -> paired edge tile
  -> 4 x 128-dword activation slices
  -> same main16 tile pairs each slice with one Q4NX 32x256 chunk
  -> accumulate O output locally
  -> row1 packet gather and final drain
```

The total edge-returned data is still a 512-dword shard, but it is never stored
as a `1024xbf16` activation buffer. It is produced and consumed as four
`256xbf16` slices, matching the Q4NX chunk width.

## Contract

- Runtime-visible input:
  - `weights`: packed Q4NX O weights.
  - Per main column: one phase-sized stream.
  - Per stream: four row1 fat chunks.
  - Per fat chunk: four row shards, each a `32 x 256` Q4NX chunk.
- Runtime-visible output:
  - `output`: final gathered `bf16` O-phase rows.
- Internal-only data:
  - 16 sideband records of 17 `i32` dwords.
  - 16 edge producers replaying `4 x 256 bf16` slices.
  - Per-main-tile ping-pong activation-slice buffers.
  - Per-main-tile ping-pong Q4NX chunk buffers.

## What This Proves

- The main16 fabric can emit MyLM-sized 17-dword sideband records, then resume
  Q4NX projection work on the same physical tile.
- The edge path can replay a 512-dword shard as four 128-dword slices.
- Each edge slice can be paired with the matching `32 x 256` Q4NX weight chunk
  and accumulated as the O projection input.
- The runtime sequence exposes only weights and final output; sideband, edge
  slices, and projection activations stay internal to the AIE graph.

## What This Does Not Prove

- Real Q/K/V projection values.
- Real softmax attention.
- The exact MyLM shape-A/shape-B attention arithmetic.
- Full cross-layer execution. This is the O-phase handoff ABI after edge
  output, not the whole transformer layer.

## Run

```bash
.venv/bin/python experiments/50_edge_slice_replay_q4nx_o_phase/run_npu.py
```
