# MyLM Q4NX Full Graph Codegen

This experiment expands the full MyLM Q4NX hot loop from templates and
annotates the five steady sections using the exp115 operand graph.

## Checks

- Original slots: `1532`
- Generated slots: `1532`
- Original hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Generated hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Instruction text match after stripping comments: `True`
- Graph `vmac.f` records per steady group: `33`
- Annotated steady `vmac.f`: `165`
- Missing graph records: `0`
- Steady boundary hash: `5121c8604a8461c4`

## Opcode Counts

| Op | Count |
| --- | ---: |
| `vmac.f` | 264 |
| `vextbcst.16` | 256 |
| `vunpack` | 64 |
| `vups.4x` | 64 |
| `vconv.bf16.fp32` | 136 |
| `vst` | 0 |
| `vlda` | 11 |
| `vldb` | 46 |
| `lda.s16` | 8 |
| `vbcst.16` | 8 |

## Generated Include

- `/var/home/taowen/projects/IRON/experiments/117_mylm_q4nx_full_graph_codegen/generated_mylm_q4nx_hot_loop_graph.s.inc`

## Production Boundary

- The full hot loop now has a graph-annotated codegen artifact.
- This is still MyLM's numerical contract; it is not the active IRON exact-Q4NX body.
- The next step is to emit a modified graph-derived body and run a synthetic numeric gate before touching production.
