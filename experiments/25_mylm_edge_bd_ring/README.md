# Experiment 25: MyLM Edge BD Ring Contract

This experiment starts from the new MyLM reverse-engineering conclusion:
exp24 failed because the edge/KV path was simplified into one raw-history
worker. MyLM uses static BD rings on the edge and auxiliary tiles before the
attention workers see the stream.

The goal is to make that boundary explicit, generated, and runnable before
writing another attention kernel.

## Final Goal

Exp25 is done when one checked table of MyLM edge BD rows can generate and run a
minimal NPU skeleton that proves the edge/KV boundary.

The final runnable program should be:

```text
runtime descriptors for L
  -> row0 shim current/history queues
  -> c0/c7 row1 static ping-pong rings
  -> c1/c6 auxiliary packet/control path
  -> deterministic checksum workers
  -> host-visible checksum/output
```

The pass criteria are:

- the BD table compares cleanly against MyLM's decoded BD CSV for the covered
  rows;
- the emitted MLIR is generated from that table, not hand-written again;
- L=17, L=31, L=32, and L=128 run on NPU without deadlock;
- the checksum proves every rounded 16-token history tile arrives in the same
  phase/order as the table describes;
- no raw K/V history tile is fed directly into a single edge worker;
- no attention math is required yet.

Exp25 explicitly does not implement full attention, GQA fanout, Q/K/V
projection, or the whole fused layer. Its output is the reliable edge-ring
skeleton that the next experiment can attach attention math to.

## Current Status

Achieved runnable checkpoint:

```text
row0 shim history descriptors
  -> c0/c7 row1 static ping-pong rings
  -> 4096-dword plane tile split into two 2048-dword stream phases
  -> c1/c6 checksum workers
  -> host-visible checksum output
```

The real NPU run passes L=17, L=31, L=32, and L=128. L=128 verifies eight
2048-dword half-tile arrivals per plane.

This runnable skeleton intentionally keeps attention math out. Current-token
writeback remains covered by earlier KV experiments; in exp25 the current
write offsets remain part of the audited contract table, while the runnable
path focuses on the row1 history ring that broke exp24.

## Core Question

Can we reproduce the MyLM edge/KV boundary as a static BD-ring contract?

The contract must separate:

```text
dynamic runtime descriptors:
  current K/V write offsets
  rounded history read lengths

static fabric descriptors:
  c0/c7 row1 ping-pong rings
  c1/c6 auxiliary rings
  packet ids and lock handoff
```

If this cannot be represented cleanly, a full fused layer will keep failing
from wrong ping-pong phase, packet id, channel parity, or lock balance.

## What This Experiment Proves

- `L` only affects arg4-style runtime offsets and lengths.
- The row1 edge ring stays static across context lengths.
- A 4096-dword raw 16-token/4-head plane tile is not the worker-visible unit;
  row1 emits 2048-dword reshaped streams first.
- `c1/c6` auxiliary rings are part of the contract, not optional debug noise.
- The next generated MLIR should come from a table, not hand-written repeated
  BD fragments.

## First Checkpoint

Run the contract/audit tool:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python bd_ring_spec.py --l 31
```

Expected output includes:

```text
current_inside=0x7800
history_len=0x2000 dwords
c0r1/c7r1 bd0/1 4096-dword rings
c1r3 packet ids 14/15
c6r1 6144-dword aggregate path
```

This is not yet a runnable NPU test. It is the development tool needed before
the runnable static-ring skeleton.

If a MyLM BD CSV has been produced with `tools/re/aie2ps_bd_decode.py`, compare
the table against it:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python bd_ring_spec.py \
  --l 31 \
  --compare-bd-csv /tmp/mylm_re_exp24_refresh/qwen3_8b_layer_L31.bd.csv
```

Expected result:

```text
validation: PASS
compare: PASS
```

## Next Checkpoint

The same table can emit a low-level MLIR-AIE fragment:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python bd_ring_spec.py \
  --emit-mlir-fragment > /tmp/mylm_edge_bd_ring.mlir.inc
```

This emitted text is a checked fragment, not a complete `aie.device` module.
The first runnable skeleton should wrap this contract with explicit buffers,
flows, and deterministic checksum workers. Only after the ring/packet boundary
is correct should attention math be reintroduced.

The runnable skeleton is now implemented:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

Expected result:

```text
L=17   PASS
L=31   PASS
L=32   PASS
L=128  PASS
SUCCESS: edge/KV static-ring checksum skeleton verified on NPU.
```
