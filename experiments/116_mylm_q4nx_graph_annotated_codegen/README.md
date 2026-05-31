# Experiment 116: MyLM Q4NX Graph-Annotated Codegen

Exp115 emits a machine-readable operand graph. This experiment consumes that
JSON and generates a steady-state assembly include:

```text
generated_mylm_q4nx_steady_graph.s.inc
```

The include keeps MyLM group1 instruction text byte-for-byte at the instruction
level, but inserts generated comments before every `vmac.f` describing the
half-register cells that feed the accumulator, left vector, and right vector.

This is not a production kernel yet. The value is the boundary: the next
generator can consume the same JSON state instead of scraping Markdown or
manually copying reverse-engineering notes.

Run:

```bash
python3 experiments/116_mylm_q4nx_graph_annotated_codegen/run.py
```

It writes:

```text
experiments/116_mylm_q4nx_graph_annotated_codegen/mylm_q4nx_graph_annotated_codegen.md
experiments/116_mylm_q4nx_graph_annotated_codegen/generated_mylm_q4nx_steady_graph.s.inc
```

The verifier removes comments and macro wrappers from the generated include and
requires the remaining instruction stream to exactly match MyLM group1.
