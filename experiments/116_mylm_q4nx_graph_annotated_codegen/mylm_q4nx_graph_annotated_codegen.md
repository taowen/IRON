# MyLM Q4NX Graph-Annotated Codegen

This experiment consumes the exp115 JSON operand graph and emits a
steady-state assembly include with generated half-register comments.

## Checks

- Group1 slots: `189`
- Group1 `vmac.f`: `33`
- Graph `vmac.f` records: `33`
- Instruction text match after stripping comments: `True`
- Stable steady boundary: `True`
- Steady boundary hash: `5121c8604a8461c4`
- Missing graph records: `0`

## Generated Include

- `/var/home/taowen/projects/IRON/experiments/116_mylm_q4nx_graph_annotated_codegen/generated_mylm_q4nx_steady_graph.s.inc`

## Why This Matters

- The generated include proves exp115 is a usable generator input, not only a report.
- Each `vmac.f` now carries machine-generated half-register state next to the instruction that consumes it.
- The next step is to generate a modified body from graph records and run a numeric gate, rather than manually editing raw assembly.
