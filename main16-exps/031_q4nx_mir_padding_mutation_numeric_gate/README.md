# Q4NX MIR Padding Mutation Numeric Gate

This experiment is the first intentional hot-loop byte-diff after the
byte-exact no-op gate in experiment 030.

It patches MyLM raw program address `0x2ba`, replacing four zero bytes in the
first explicit hot-loop padding gap with `mov r31, r31`:

```text
00 00 00 00 -> f8 a0 df 1f
```

The direct-QKV NPU harness then checks whether the compact record payload still
matches the MyLM baseline.

Expected interpretation:

- pass: this specific padding gap has enough slack for a one-cycle shorter
  no-op mutation;
- fail: padding bytes are part of the timing contract, so future mutations must
  preserve cycle count exactly.

Run:

```bash
python3 main16-exps/031_q4nx_mir_padding_mutation_numeric_gate/run.py --max-cases 0
```
