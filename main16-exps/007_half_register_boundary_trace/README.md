# 007 Half-Register Boundary Trace

This experiment refines `006_q4nx_alias_lifetime_graph`.

Experiment 006 proves the steady-state cross-group live-through families. This
experiment splits those families into vector halves and accumulator quadrants:

- `xN` -> `vecN.lo`, `vecN.hi`
- `wlN` -> `vecN.lo`
- `whN` -> `vecN.hi`
- `dmN` -> four accumulator quadrants
- `cmlN/cmhN` -> low/high accumulator halves
- `bmll/bmlh/bmhl/bmhh` -> one accumulator quadrant

The output focuses on group1 because it is the first steady-state group after
the group0 fill. It reports:

- cells produced in group0 and first consumed in group1;
- every group1 `vmac.f` operand with cell-level producers;
- mixed-half vector operands, which are the part a MyLM-style code generator
  must preserve instead of treating `xN` as an indivisible value.

