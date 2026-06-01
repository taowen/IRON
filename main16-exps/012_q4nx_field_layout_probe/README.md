# 012 Q4NX Field Layout Probe

This experiment starts the production Q4NX layout work.

Experiments 008..011 showed the scalar synthetic contract and the paired
even-chunk schedule. This experiment fixes one active pair:

```text
activation chunk 0 active
weight chunk 0 active
activation = bf16 1.0
scale      = bf16 1/64
nibble     = 1
zero       = 0
```

Then it varies one field inside the active Q4NX chunk:

- baseline: all scale slots and all q4 data words active;
- one scale dword active at a time;
- one q4 data dword active at a time for the first 32 q4 dwords and a sparse
  set across the rest of the 1024-dword q4 data region.

The output records all 16 payload dwords of record0. This is the first isolated
probe for mapping host Q4NX fields to compact payload lanes.
