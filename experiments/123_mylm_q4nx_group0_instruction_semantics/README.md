# Exp123: MyLM Q4NX Group0 Instruction Semantics

Goal: learn the MyLM Q4NX assembly before changing the production hot body.

This experiment annotates the first full Q4NX section, `0x260..0x52a`, at
instruction-slot granularity. It records each slot's opcode, semantic role,
half-register/accumulator-cell defs, uses, and latest visible producer.

Run:

```bash
python3 experiments/123_mylm_q4nx_group0_instruction_semantics/run.py
```

Inputs:

- `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`
- `experiments/115_mylm_q4nx_operand_graph/run.py`
- `experiments/118_mylm_q4nx_cell_liveness/mylm_q4nx_cell_liveness.json`

Outputs:

- `mylm_q4nx_group0_instruction_semantics.md`
- `mylm_q4nx_group0_instruction_semantics.json`
- `mylm_q4nx_group0_instruction_semantics.tsv`
