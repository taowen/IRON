# Experiment 22: Q4NX Query Projection + KV Attention

This experiment joins the two pieces that matter for a MyLM/FastFlowLM-style
decode layer:

1. A packed Q4NX projection consumes hidden state and produces the query on the
   compute tile.
2. The same worker then writes current K/V into the cache and scans the rounded
   history cache with online softmax attention.

The query is not a host input. It is produced from `hidden + q_weight` inside the
worker and stays core-local until the attention phase consumes it.

## Scope

This is a reduced one-head, one-worker contract. It intentionally proves the
phase boundary before scaling:

- `hidden` is bf16 and loaded once for the Q projection phase.
- `q_weight` is Q4NX: 32 output rows x 256 K columns per chunk, 4 chunks for
  `K=1024`.
- `current K/V` and `KV cache` are f32, matching exp21's stable attention path.
- The same two worker input DMA channels are reused across phases:
  - channel 0: hidden -> current K -> history K
  - channel 1: Q4NX weight chunks -> current V -> history V
- current K/V are written to the cache before the history scan is queued.

This does not yet prove the full MyLM 4-head plane, 16-tile projection fabric,
or all Q/K/V/O/up/gate/down phases. It proves the critical bridge: online Q4NX
projection output becomes a resident attention query without a host round trip.

## Run

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```
