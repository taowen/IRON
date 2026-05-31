# Exp121: Q4NX Modified Body Numeric Gate

Goal: turn the exp120 generator contract into a concrete numerical branch
decision before changing production assembly.

The MyLM section contract preserves the fast `32 main MAC + 1 zero/group-sum
correction MAC` group shape. Current IRON exact parity instead requires zero to
be recomposed into each coefficient before the MAC. This experiment compares the
two candidate bodies on deterministic synthetic chunks and records which one can
be migrated under which correctness contract.

Run:

```bash
python3 experiments/121_q4nx_modified_body_numeric_gate/run.py
```

Inputs:

- `experiments/120_mylm_q4nx_generator_contract/mylm_q4nx_generator_contract.json`

Outputs:

- `q4nx_modified_body_numeric_gate.md`
- `q4nx_modified_body_numeric_gate.json`
