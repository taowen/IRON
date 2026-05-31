# Experiment 103: MyLM Q4NX Register Flow

This experiment is the next step after exp101/102:

- exp101 proved that a MyLM-looking opcode skeleton can be numerically wrong.
- exp102 proved the immediate cause for one class of wrong results:
  `vextbcst.16` has visible use latency, and the following `vmac.f` must be
  scheduled after independent work.

The useful artifact here is a register-flow table for the first software
pipeline window of MyLM's Q4NX hot loop, `0x260..0x52a`.

Run:

```bash
python3 experiments/103_mylm_q4nx_register_flow/run.py
```

It writes:

```text
experiments/103_mylm_q4nx_register_flow/mylm_q4nx_0x260_0x52a_register_flow.md
```

The table is a working reverse-engineering artifact, not a proof of a complete
decompiler. The point is to make defs/uses and alias families explicit before
writing any more production-style Q4NX assembly.

The first generated window already explains why the simple exp101 body is the
wrong abstraction:

```text
range 0x260..0x52a
instruction slots = 192
vmac.f = 28
vextbcst.16 = 32
vconv.bf16.fp32 = 16
vups.4x = 8
vunpack = 12
lda.s16 = 2
```

So group0 is not a closed 33-MAC unit. MyLM starts with a pipeline fill window;
middle groups have 33 MACs, and the final group drains with 38 MACs. The next
generator has to model that cross-group register/dataflow schedule instead of
repeating one isolated group body eight times.
