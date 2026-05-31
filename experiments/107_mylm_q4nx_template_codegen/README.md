# Experiment 107: MyLM Q4NX Template Codegen

Exp106 identified the repeatable structure of the MyLM Q4NX hot loop:

```text
fill(group0)
steady_template(group1) * 5
pre_drain(group6)
drain(group7)
```

This experiment turns that observation into a generator check. It emits the
full hot-loop instruction stream from those four templates and compares it
against the original MyLM disassembly text for `0x260..0x1850`.

Passing means we have a generator-shaped representation of the MyLM schedule.
It does not mean the generated body is production-ready for IRON: the body
still uses MyLM's group-sum numerical contract, register plan, and phase-body
scratch setup.

Run:

```bash
python3 experiments/107_mylm_q4nx_template_codegen/run.py
```

It writes:

```text
experiments/107_mylm_q4nx_template_codegen/mylm_q4nx_template_codegen.md
experiments/107_mylm_q4nx_template_codegen/generated_mylm_q4nx_hot_loop.s.inc
```
