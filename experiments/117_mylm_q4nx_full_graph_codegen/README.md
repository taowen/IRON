# Experiment 117: MyLM Q4NX Full Graph Codegen

Exp116 proves the exp115 operand graph can annotate one steady group. This
experiment expands that into the full MyLM Q4NX hot-loop shape:

```text
fill(group0)
steady_template(group1) * 5
pre_drain(group6)
drain(group7)
```

The generated include contains one expanded macro with all hot-loop
instructions. Steady sections are annotated from the exp115 JSON graph. Fill,
pre-drain, and drain sections keep the raw template instructions without graph
comments because their boundary state differs from the steady transition.

Run:

```bash
python3 experiments/117_mylm_q4nx_full_graph_codegen/run.py
```

It writes:

```text
experiments/117_mylm_q4nx_full_graph_codegen/mylm_q4nx_full_graph_codegen.md
experiments/117_mylm_q4nx_full_graph_codegen/generated_mylm_q4nx_hot_loop_graph.s.inc
```

The verifier strips comments and macro wrappers and requires the remaining
instruction stream to exact-match MyLM `0x260..0x1850`.
