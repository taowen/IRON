# Experiment 106: MyLM Q4NX Steady-State Template

Exp104 proves the full MyLM Q4NX hot loop is a cross-group software pipeline.
This experiment answers the next generator question: which part of that loop is
a repeatable template?

Current result:

- group1, group2, group3, group4, and group5 have identical instruction text;
- group6 is a pre-drain variant with the same prefix and a scalar `p3` save /
  rewind sequence near the tail;
- group0 is the fill window;
- group7 is the drain window.

This means the production generator should not repeat a 33-MAC group eight
times. The correct shape is:

```text
fill(group0)
steady_template(group1) * 5
pre_drain(group6)
drain(group7)
```

Run:

```bash
python3 experiments/106_mylm_q4nx_steady_state_template/run.py
```

It writes:

```text
experiments/106_mylm_q4nx_steady_state_template/mylm_q4nx_steady_state_template.md
```
