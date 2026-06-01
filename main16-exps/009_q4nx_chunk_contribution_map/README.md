# 009 Q4NX Chunk Contribution Map

Experiment 008 falsified the simplest uniform formula for one active chunk:

```text
expected first_chunk_only = 128 * 1 * (1/64) * 1 = 2
observed first_chunk_only = 4
```

The full-record and first-record-only cases still matched:

```text
16 active chunks -> 32
```

So the issue is not direct-entry numeric observability or record-major ordering.
The next question is whether the 16 chunks inside a record contribute uniformly.
This experiment activates exactly one record0 stream chunk at a time and maps
the first-record payload contribution.

