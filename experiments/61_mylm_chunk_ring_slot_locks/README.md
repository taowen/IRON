# Experiment 61: MyLM Chunk-Ring Slot Locks

Experiment 60 changed row1 from full-patch residency to a small Q4NX chunk
ring, but it timed out on real hardware. This experiment keeps exp60's
physical shape and diagnoses the timeout with two narrow changes:

- the row1 ping and pong buffers get independent slot locks,
- the memtile output collector is started on the same DMA channel used by the
  packet route.

## Contract

- One main column subset: `c2/r2..r3`.
- Two simplified edge/source tiles: `c0/r2..r3`.
- Runtime submits one full patch descriptor:
  - patch0: `64 rows x 4096 K`, `0x28000` bytes, rows `0..63`
- Row1 owns a small ping-pong ring:
  - patch0 ping: `2 Q4NX chunks` for rows `0,1`
  - patch0 pong: `2 Q4NX chunks` for rows `0,1`
- Ping and pong have independent slot locks:
  - `patch0_ping_empty/full`
  - `patch0_pong_empty/full`
- Each slot full lock is released by the producer with count `2`, then row0
  and row1 each acquire one token before the producer can reuse that slot.
- Main tile channel shape remains:
  - `ch0`: activation slice, `128 dwords = 256 bf16`
  - `ch1`: Q4NX weight chunk, `1280 dwords = 5120B`

## What This Proves

- Exp60's timeout had a concrete route/DMA mismatch: the main-tile packet route
  targets `packet_dest<%mt0, DMA : 2>`, but exp60 started the memtile output
  collector on `S2MM` channel `4`. Channel 2 therefore never had a listening
  DMA BD, and channel 4 had no producer.
- This experiment fixes that mismatch by starting output collection on memtile
  `S2MM` channel `2`.
- With the route fixed, the full `0x28000` host patch descriptor feeds a small
  row1 Q4NX chunk ring without full-patch row1 residency.
- The MyLM-shaped main tile ABI works for this narrowed case: activation on
  `ch0`, weight on `ch1`.

## What This Does Not Prove

- Exact MyLM same-channel patch0/patch1 descriptor scheduling.
- Four-column full N-block scaling; exp59 already proved the descriptor-level
  scaling with full-patch residency.
- Production attention ABI or attention-to-O handoff.

## Run

```bash
.venv/bin/python experiments/61_mylm_chunk_ring_slot_locks/run_npu.py
```

Last verified on real NPU:

```text
NPU time: 2148.7 us
PASS: exact MyLM patch descriptors feed row1 slot-locked chunk rings and MyLM-shaped main inputs.
```
