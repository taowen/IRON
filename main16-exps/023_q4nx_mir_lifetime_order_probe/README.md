# 023 Q4NX MIR Lifetime Order Probe

Experiment 022 proved direct AIE2P MIR is viable, but the naive one-group
dependency graph did not get a postpipeliner schedule.

This experiment compares that naive order with a MyLM-derived group1 projection
from experiment 006:

- keep only `vups.4x`, `vextbcst.16`, and `vmac.f`;
- preserve the actual MyLM group1 instruction order and registers;
- compile the projected loop with Peano `llc`;
- compare schedule remarks and objdump counts.

Run:

```bash
python3 main16-exps/023_q4nx_mir_lifetime_order_probe/run.py
```

Outputs:

- `q4nx_mir_lifetime_order_probe.json`
- `q4nx_mir_lifetime_order_probe.md`
- `build/*.mir`
- `build/*.s`
- `build/*.o`
- `build/*.objdump`

