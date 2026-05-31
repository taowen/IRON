# Exp126: MyLM Main16 Whole-Core Contract

Goal: turn "move toward MyLM" into an executable migration contract for the
main16 projection role.

This experiment does not change the active `qwen3-layer` runtime path. It
audits the current IRON main16 role object against the MyLM-style target:

- one shared Q4NX body;
- phase bodies for Q/K/V, O, up/gate, and down;
- a dispatcher that owns phase order and record emission;
- no partial QKV-only nocall scheduler in the active path.

Run:

```bash
python3 experiments/126_mylm_main16_whole_core_contract/run.py
```

Inputs:

- `qwen3-layer/main_projection_q4nx_fast.o`
- `experiments/119_mylm_q4nx_full_operand_graph/mylm_q4nx_full_operand_graph.json`
- `experiments/120_mylm_q4nx_generator_contract/mylm_q4nx_generator_contract.json`
- `qwen3-layer/main16_q4nx_mylm_compare.md`
- `MyLM/tools/re/fused-layer-engine/current-understanding.md`

Outputs:

- `mylm_main16_whole_core_contract.md`
- `mylm_main16_whole_core_contract.json`
