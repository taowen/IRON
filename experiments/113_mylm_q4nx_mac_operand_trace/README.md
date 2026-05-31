# Experiment 113: MyLM Q4NX MAC Operand Trace

Exp103/104/107 proved the MyLM Q4NX hot loop is a software-pipelined
template, not eight independent groups. This experiment looks at the same
loop from the point of view of the `vmac.f` instructions.

For every `vmac.f` in the hot loop, the script records the latest visible
producer for both vector operands and the accumulator source operand. The
report focuses on group1, the canonical steady-state group, because that is
the template we would need to reproduce before writing production assembly.

Run:

```bash
python3 experiments/113_mylm_q4nx_mac_operand_trace/run.py
```

It writes:

```text
experiments/113_mylm_q4nx_mac_operand_trace/mylm_q4nx_mac_operand_trace.md
```

The analysis intentionally stays conservative. It tracks register families
such as `x4/wl4/wh4` as one vector family and `dm3/bmll3/cml3` as one
accumulator family. That is not a full alias decompiler, but it is enough to
answer the first production question: which operands are local to the current
steady group, and which ones must be carried across the software-pipeline
boundary.

Current result:

```text
whole hot loop:   vmac.f=264, crossing vector operands=42/528
steady group1:    vmac.f=33,  crossing vector operands=6/66
steady group1:    accumulator source carries=2/33
```

This is the concrete reason a self-contained 33-MAC group macro is the wrong
implementation unit. The generator needs to emit a scheduled operand graph with
fill, steady, pre-drain, and drain boundary state.
