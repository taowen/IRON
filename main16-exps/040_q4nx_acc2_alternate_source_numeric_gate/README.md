# Q4NX Acc2 Alternate Source Numeric Gate

Experiment 039 showed that the representative zero/offset site does not require
the exact original source quadrant:

```text
0x700: vmov bmhh1, bmhh2
```

For `zero0_allq4`, `bmll2`, `bmlh2`, `bmhl2`, and `bmhh2` all work as sources,
while `nopm` and `bmhh1` self-copy fail.

This experiment runs the alternate `acc2` source mutations against every
direct-QKV synthetic case. A passing alternate source is the first non-byte-copy
source-bundle mutation that survives the current NPU numeric gate.

Run:

```bash
python main16-exps/040_q4nx_acc2_alternate_source_numeric_gate/run.py
```
