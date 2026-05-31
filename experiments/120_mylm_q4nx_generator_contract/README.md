# Exp120: MyLM Q4NX Generator Contract

Goal: merge the full MAC operand graph and boundary liveness into the contract
the next Q4NX assembly generator should consume.

Exp117 can replay the MyLM hot loop from templates. Exp118 describes live state
at each group boundary. Exp119 describes every `vmac.f` operand. This
experiment combines those artifacts and emits a sectionized assembly include
whose comments name the live-in/live-out contract for each macro.

Run:

```bash
python3 experiments/120_mylm_q4nx_generator_contract/run.py
```

Inputs:

- `experiments/118_mylm_q4nx_cell_liveness/mylm_q4nx_cell_liveness.json`
- `experiments/119_mylm_q4nx_full_operand_graph/mylm_q4nx_full_operand_graph.json`
- `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

Outputs:

- `mylm_q4nx_generator_contract.md`
- `mylm_q4nx_generator_contract.json`
- `generated_mylm_q4nx_contract_sections.s.inc`
