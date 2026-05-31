# Experiment 115: MyLM Q4NX Operand Graph

Exp114 proved that MyLM's Q4NX `vmac.f` operands cannot be modeled as whole
`xN` registers. This experiment turns that observation into the first
generator-shaped operand graph:

- group boundary state is represented as vector-half and accumulator-quadrant
  cells;
- `fill -> steady` and `steady -> steady` boundaries are listed separately;
- steady group `vmac.f` operands are summarized by whether they consume local
  cells, carried cells, or mixed-half values;
- a JSON artifact is emitted so the next assembly generator can consume the
  same state instead of scraping Markdown.

Run:

```bash
python3 experiments/115_mylm_q4nx_operand_graph/run.py
```

It writes:

```text
experiments/115_mylm_q4nx_operand_graph/mylm_q4nx_operand_graph.md
experiments/115_mylm_q4nx_operand_graph/mylm_q4nx_operand_graph.json
```

The graph is still pre-lane-select: exp102 has already validated the primitive
`vmac.f #0x33c` lane model on real NPU, while this experiment focuses on which
half-register cells are live at each scheduled MAC site.

Current result:

```text
group1 vmac.f = 33
group1 mixed vector operands = 25/66
boundary data cells into group1 = 23
steady boundary hashes for groups 2..6 = 5121c8604a8461c4
```

So `fill -> steady` and `steady -> steady` now have explicit state records.
The steady-to-steady data-cell signature is stable, which is the first concrete
shape a generator can consume.
