# Exp119: MyLM Q4NX Full Operand Graph

Goal: extend the exp115 operand graph from one steady group to the full MyLM
Q4NX hot loop.

Exp115 proves the canonical steady group has a useful half-register operand
graph, but production assembly work needs all `vmac.f` slots, including fill,
pre-drain, and drain. This experiment emits a machine-readable graph for every
MyLM `vmac.f` in `0x260..0x1850` and checks the steady group signatures.

Run:

```bash
python3 experiments/119_mylm_q4nx_full_operand_graph/run.py
```

Inputs:

- `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`
- `experiments/115_mylm_q4nx_operand_graph/run.py`

Outputs:

- `mylm_q4nx_full_operand_graph.md`
- `mylm_q4nx_full_operand_graph.json`
