# Experiment 114: MyLM Q4NX Half-Register Alias

Exp113 traces `vmac.f` operands at register-family granularity. That is useful
for seeing cross-group carry, but it is still too coarse for writing assembly:
`x4`, `wl4`, and `wh4` are not the same cell, and `dm3`, `cml3`, `cmh3`,
`bmll3`, `bmlh3`, `bmhl3`, and `bmhh3` do not overwrite the same slice.

This experiment keeps a cell-level producer map:

- `xN` is split into `vecN.lo` and `vecN.hi`;
- `wlN` and `whN` update one vector half;
- `dmN` is split into four accumulator quadrants;
- `cmlN` and `cmhN` cover lower/upper accumulator halves;
- `bmllN`, `bmlhN`, `bmhlN`, and `bmhhN` update one quadrant.

The generated report asks a narrower question than a full decompiler: when a
steady-state `vmac.f` consumes a vector register, did both vector halves come
from the same producer, or is the operand composed from separately scheduled
half-register producers?

Run:

```bash
python3 experiments/114_mylm_q4nx_half_register_alias/run.py
```

It writes:

```text
experiments/114_mylm_q4nx_half_register_alias/mylm_q4nx_half_register_alias.md
```

This is the next piece needed before generating a MyLM-shaped body. A generator
must preserve half-register composition and boundary carry state; a whole-vector
`xN` table is not enough.

Current result:

```text
whole hot loop: mixed vector operands = 202/528
steady group1:  mixed vector operands = 25/66
steady group1:  vector operands with cross-group cells = 9/66
steady group1:  accumulator operands with cross-group cells = 2/33
```

This report is deliberately conservative. It shows which half-register cells
are live, but it does not yet decode the `vmac.f` configuration word to say
which lanes from each `xN` operand are actually selected.

Exp102 already proved the primitive `vmac.f #0x33c` operand model on real NPU:
single `vextbcst.16` lanes produce the expected first-lane value, the
group-sum correction produces `64`, and summing 32 activation lanes produces
`528`. The remaining gap is the scheduled mixed-half state in the MyLM loop,
not whether `#0x33c` can multiply broadcasted BF16 operands.
