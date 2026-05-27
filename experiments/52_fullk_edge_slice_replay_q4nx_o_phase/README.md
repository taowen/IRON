# Experiment 52: Full-K Edge Slice Replay To Q4NX O Phase

This experiment keeps the corrected exp50 ABI but scales it to the full O
projection input width.

The important distinction from older experiments:

- exp42 proved a toy edge-output-to-O route, but not Q4NX full-K projection.
- exp45 proved main16/edge resource reuse, but not full `K=4096` O weights.
- exp49 was discarded because it made the edge return look like a full shard
  buffer.
- exp50 fixed the ABI, but only for one `K=1024` shard.

Exp52 proves the next contract:

```text
main16 tile
  -> 17-dword compact sideband
  -> paired edge tile
  -> 4 x 512-dword edge shards
  -> each shard replays as 4 x 128-dword activation slices
  -> same main16 tile pairs each slice with one Q4NX 32x256 O-weight chunk
  -> accumulate full K=4096 O output locally
  -> debug packet drain with per-row headers
```

There is no host-visible attention BO, no `4096xbf16` activation buffer, and no
`1024xbf16` shard buffer. The edge path produces only `256xbf16` slices.

## Contract

- Runtime-visible input:
  - `weights`: packed Q4NX O weights.
  - Per main column: one full-K O stream.
  - Per stream: sixteen row1 fat chunks.
  - Per fat chunk: four row shards, each a `32 x 256` Q4NX chunk.
- Runtime-visible output:
  - `output`: debug records. Each record is `bf16(group), bf16(row), 32
    bf16 O values`.
- Internal-only data:
  - 16 sideband records of 17 `i32` dwords.
  - 16 edge producers replaying `16 x 256 bf16` slices.
  - Per-main-tile ping-pong activation-slice buffers.
  - Per-main-tile ping-pong Q4NX chunk buffers.

## What This Proves

- The corrected edge-slice ABI scales from one shard to the full `K=4096`
  O-projection input.
- The main16 fabric can run full-K Q4NX O projection from edge-replayed slices
  without materializing the full attention vector.
- The row1 full-K fat-chunk ring and edge slice stream can be consumed together
  by the same physical main tile.
- The final debug drain may receive packets in hardware arrival order. The
  `(group,row)` header is the ABI for verification; row1 packet order is not
  treated as a MyLM layer handoff.

## What This Does Not Prove

- Real softmax attention math.
- The exact MyLM shape-A/shape-B arithmetic.
- O output handoff into FFN. That is the next boundary after full-K O is stable.
- A row1 packet gather ordering contract. MyLM does not appear to use that as
  the O-projection ABI.

## Run

```bash
.venv/bin/python experiments/52_fullk_edge_slice_replay_q4nx_o_phase/run_npu.py
```
