# Exp118: MyLM Q4NX Cell Liveness

Goal: learn the MyLM Q4NX hot-loop assembly before changing production code.

This experiment turns the half-register operand graph from exp115 into a full
hot-loop liveness table. It records which vector halves, accumulator quadrants,
pointers, and scalar cells stay live across group boundaries. The output is a
Markdown report for reading and a JSON artifact for a later assembly generator.

Run:

```bash
python3 experiments/118_mylm_q4nx_cell_liveness/run.py
```

Inputs:

- `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`
- `experiments/115_mylm_q4nx_operand_graph/run.py`

Outputs:

- `mylm_q4nx_cell_liveness.md`
- `mylm_q4nx_cell_liveness.json`
