# Experiment 56: MyLM-Style Linked Qwen3 Patch Schedule

Experiment 53 proved that a full layer-shaped phase chain can run on the
`main16 + edge16` fabric with simplified `K=4096` phases. Experiment 54 proved
the real Qwen3 patch schedule with one contiguous weight descriptor per main
column. Experiment 55 proved a MyLM-style linked shim BD chain on a small stream.

This experiment combines those two results: it feeds the real Qwen3 schedule
through linked BD batches instead of one giant contiguous descriptor or one
push per logical block.

## Contract

- Main projection fabric remains `c2..c5/r2..r5`.
- Edge/aux fabric remains `c0/c1/c6/c7`, rows `2..5`.
- Runtime submits one linked weight BD chain per batch and per main column.
  Each batch patches up to 14 shim BDs, links them with `next_bd`, pushes only
  the head BD, and then waits once on the MM2S token.
- The core and memtile fabric consume those batches as the real logical
  phase/block schedule.
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
- MyLM-style linked BD batches can feed the real schedule without host
  involvement at every logical block boundary.

## What This Does Not Prove

- Exact Qwen3 attention math.
- Real up/gate/down numerical dependency. The output is a deterministic summary
  record proving schedule consumption and phase/block ordering.
- Final high-throughput microkernel performance. The Q4NX kernel is still the
  contract kernel used by prior experiments.
- The final FastFlowLM/MyLM transaction generator. This still uses Python string
  generation rather than a reusable CDO/transaction builder.

## Run

```bash
.venv/bin/python experiments/56_mylm_linked_qwen_schedule/run_npu.py
```
