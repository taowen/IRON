# Exp122: Q4NX Exact Section Rewrite Contract

Goal: convert exp121's numerical decision into a section-level rewrite contract
for the next source-assembly body generator.

Exp120 preserves MyLM's section boundaries and `264` static group MACs. Exp121
shows the production-safe branch is exact coefficient recomposition, not the
MyLM-like group correction. This experiment derives the exact target section
shape by removing one zero-correction MAC from each logical group while keeping
the exp120 live-state boundaries.

Run:

```bash
python3 experiments/122_q4nx_exact_section_rewrite_contract/run.py
```

Inputs:

- `experiments/120_mylm_q4nx_generator_contract/mylm_q4nx_generator_contract.json`
- `experiments/121_q4nx_modified_body_numeric_gate/q4nx_modified_body_numeric_gate.json`

Outputs:

- `q4nx_exact_section_rewrite_contract.md`
- `q4nx_exact_section_rewrite_contract.json`
- `generated_q4nx_exact_section_rewrite.s.inc`
