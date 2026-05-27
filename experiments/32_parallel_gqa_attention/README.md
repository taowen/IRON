# Experiment 32: Parallel GQA Attention Heads

Exp31 proved one full GQA attention group on a single worker. Exp32 keeps the
same math, but changes the physical schedule:

```text
column 0: Q head 0 + shared K/V projection + current K/V cache write
column 1: Q head 1
column 2: Q head 2
column 3: Q head 3

all columns:
  scan the same rounded KV cache tiles
  run one-head online softmax attention
  drain one 128-dim output slice
```

## What It Proves

- The four Q heads of one GQA group can run on four physical workers in
  parallel.
- One worker can produce the shared current K/V, write it into the cache BO, and
  the other workers can safely read the updated cache after the runtime sync.
- The same KV history BO can be scanned by multiple columns in one dispatch.
- The output BO can be assembled from four independent S2MM drains into the
  final 512-dim GQA attention output.

## Scope

This is still one GQA group, not the full Qwen layer:

- 4 Q heads and 1 shared KV head,
- one worker per query head,
- one column per worker to avoid row-level hidden fanout while proving the
  cross-column schedule,
- no O projection,
- pair-tile KV cache layout `[K tile | V tile]`.

The next step is to replace the per-column duplicate hidden/KV DMA with the
MyLM-style row1 memtile fanout/edge path, then replicate the group across all
8 KV groups.

## Result

Verified on real NPU for `L=17`, `31`, `32`, and `79`.

All 512 output values matched the CPU reference with max absolute error below
`5e-6`. Measured latency was:

| Context | Tiles | Time |
|---------|-------|------|
| 17 | 2 | `7.36ms` |
| 31 | 2 | `7.86ms` |
| 32 | 2 | `7.88ms` |
| 79 | 5 | `9.98ms` |

Compared with exp31's single-worker full GQA group (`14.6ms` to `25.1ms`), this
shows that query-head parallelism is already useful even before the final MyLM
row1 fanout/edge path is reproduced.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/32_parallel_gqa_attention/run_npu.py
```
