# Experiment 34: Parallel Row1 KV Reshape Attention

Exp33 proved the row1 memtile KV reshape contract on one worker. Exp34 keeps
that row1 static-BD reshape, but runs the four query heads of one GQA group on
four physical columns.

```text
column 0: Q head 0
column 1: Q head 1
column 2: Q head 2
column 3: Q head 3

each column:
  query head slice -> row1 -> worker
  token-major K/V history -> row1 static BD reshape -> worker
  one-head online attention -> output slice
```

## What It Proves

- The row1 dim-group-major KV reshape from exp33 composes with multi-column
  query-head parallelism.
- Four workers can consume the same logical token-major K/V history while each
  row1 memtile reshapes it independently for its local attention worker.
- Tail masking still works for non-16-aligned lengths.
- The final `512`-element GQA output can be assembled from four independent
  worker drains.
- The real-NPU output matches the CPU reference for `L=17/31/32/79` with
  `4e-5` absolute tolerance.

## Scope

This is still not the full MyLM layer engine:

- no Q/K/V projection,
- no current-token cache writeback,
- no one-read shared row1 fanout across columns,
- no edge/aux aggregation path.

The experiment deliberately checks the next physical boundary after exp33:
parallel consumers of the row1 reshape contract. The runtime still duplicates
the K/V DDR reads per column; removing that duplication needs the later
MyLM-style edge/fanout path.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/34_parallel_row1_reshape_attention/run_npu.py
```
