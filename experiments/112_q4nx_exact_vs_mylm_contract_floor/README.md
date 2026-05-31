# Experiment 112: Q4NX Exact vs MyLM Contract Floor

Exp110 proved exact parity requires zero to enter the dequant coefficient before
the MAC. Exp111 proved the active body already has the right MAC count and is
slow because coefficient construction is heavy.

This experiment computes a lower-bound cost model for two different numerical
contracts:

- current exact:
  `bf16(bf16(q * scale) + zero) * activation`
- MyLM-like group correction:
  `bf16(q * scale) * activation + zero * sum(activation_group)`

The key question is whether exact parity can ever reach MyLM-like instruction
shape. If the exact lower bound is still far from MyLM, the performance path
must include a deliberate numerical contract decision plus multi-layer token
validation.

Run:

```bash
python3 experiments/112_q4nx_exact_vs_mylm_contract_floor/run.py
```

It writes:

```text
experiments/112_q4nx_exact_vs_mylm_contract_floor/q4nx_exact_vs_mylm_contract_floor.md
```

Current result:

```text
active_exact:       vmac=512 vmul=512 conversions=2048
exact_lower_bound:  vmac=512 vmul=512 conversions=1536
mylm_group_contract: vmac=528 vmul=16 conversions=272
```

So exact-parity assembly can still remove redundant unpack/control/conversion
traffic, but it cannot reach the MyLM hot-loop instruction shape unless there
is an unknown fused path for the two BF16 roundings. The MyLM-like route is a
different numerical contract and must be gated at multi-layer token level.
