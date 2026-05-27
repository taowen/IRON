# Experiment 54: Real Qwen3 Contiguous Patch Stream Contract

Experiment 53 proved that a full layer-shaped phase chain can run on the
`main16 + edge16` fabric with simplified `K=4096` phases. This experiment keeps
the same physical dataflow but replaces the simplified phase schedule with the
real Qwen3 projection patch schedule.

This is deliberately not a runtime-descriptor stress test. Earlier diagnostic
variants showed that splitting one shim MM2S channel into many independent raw
`aiex.npu.push_queue` tasks can deadlock even for tiny schedules. That runtime
queueing question should be isolated in a smaller probe. This experiment proves
the fused-layer dataflow contract by feeding each main column with one
contiguous, already ordered Q4NX stream.

## Contract

- Main projection fabric remains `c2..c5/r2..r5`.
- Edge/aux fabric remains `c0/c1/c6/c7`, rows `2..5`.
- Runtime submits one contiguous weight MM2S descriptor per main column. The
  core and memtile fabric consume that stream as the real logical phase/block
  schedule.
- Phase sequencing is inside the worker/BD dataflow, not host-driven.
- Every main tile still consumes:
  - `S2MM ch0`: one `256 bf16` activation slice.
  - `S2MM ch1`: one Q4NX `32 x 256` weight chunk.
- The schedule uses real Qwen3 block/chunk counts:

| phase | input dim | output dim | 512-row blocks | K chunks | patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 4096 | 4096 | 8 | 16 | 64 |
| K | 4096 | 1024 | 2 | 16 | 16 |
| V | 4096 | 1024 | 2 | 16 | 16 |
| O | 4096 | 4096 | 8 | 16 | 64 |
| UP | 4096 | 12288 | 24 | 16 | 192 |
| GATE | 4096 | 12288 | 24 | 16 | 192 |
| DOWN | 12288 | 4096 | 8 | 48 | 64 |

Total: `608` patches.

## What This Proves

- The real Qwen3 projection/FFN patch schedule can be expressed as one
  layer-level stream over the same physical fabric.
- The long row1 fat-chunk ring can consume the full `608`-patch logical schedule
  without exposing phase boundaries to the core worker program.
- `DOWN`'s `K=12288` case is represented by `48` chunks per output block rather
  than the simplified `16` chunks used by earlier phase-chain experiments.
- The fused-layer fabric is not blocked by physical tile count, lock count, or
  memtile ring capacity for this real phase schedule.

## What This Does Not Prove

- Exact Qwen3 attention math.
- Real up/gate/down numerical dependency. The output is a deterministic summary
  record proving schedule consumption and phase/block ordering.
- Final high-throughput microkernel performance. The Q4NX kernel is still the
  contract kernel used by prior experiments.
- Raw AIEX multi-descriptor queue semantics for splitting the weight stream at
  logical block boundaries.

## Run

```bash
.venv/bin/python experiments/54_real_qwen_patch_schedule/run_npu.py
```
