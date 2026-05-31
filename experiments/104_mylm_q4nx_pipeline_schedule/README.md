# Experiment 104: MyLM Q4NX Pipeline Schedule

Exp103 annotates the first MyLM Q4NX software-pipeline window. This experiment
turns the full hot loop, `0x260..0x1850`, into the smaller schedule facts needed
before writing more assembly:

- per-group opcode counts for all eight activation groups;
- incoming register families used before local definition;
- cross-group carry at each group boundary;
- `vextbcst.16` to first `vmac.f` consumer slot distance.

The goal is to learn the schedule, not to change production code. The important
check is that the full loop is not eight independent copies of one middle group:
group0 is a fill window, group1..6 are steady-state windows, and group7 is a
drain window.

Current summary:

```text
group0: vmac.f=28, vconv.bf16.fp32=16, vunpack=12, lda.s16=2
group1..6: vmac.f=33, vconv.bf16.fp32=17, vunpack=8, lda.s16=1
group7: vmac.f=38, vconv.bf16.fp32=18, vunpack=4, lda.s16=0
```

The generated report also records families such as `acc1`, `acc4`, `vec0`,
`vec8`, `vec9`, and `vec10` crossing group boundaries. That is the concrete
reason an isolated 33-MAC group skeleton is the wrong implementation unit.

Run:

```bash
python3 experiments/104_mylm_q4nx_pipeline_schedule/run.py
```

It writes:

```text
experiments/104_mylm_q4nx_pipeline_schedule/mylm_q4nx_pipeline_schedule.md
```
