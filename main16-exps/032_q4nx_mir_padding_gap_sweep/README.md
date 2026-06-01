# Q4NX MIR Padding Gap Sweep

This experiment applies the same byte-level no-op mutation to each explicit
padding gap discovered by experiment 029:

```text
00 00 00 00 -> f8 a0 df 1f   ; mov r31, r31
```

Each site is patched independently into the MyLM raw program and run through the
direct-QKV NPU numeric gate.

Run:

```bash
python3 main16-exps/032_q4nx_mir_padding_gap_sweep/run.py
```

Expected interpretation:

- mismatch: the padding gap is part of the cycle-level timing contract;
- match: the site can serve as a byte-diff packaging canary, but not as proof
  that real Q4NX bundles can be reordered.
