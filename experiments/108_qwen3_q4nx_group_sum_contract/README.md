# Experiment 108: Qwen3 Q4NX Group-Sum Contract

Exp107 proves the MyLM Q4NX hot loop is generator-shaped. The next production
decision is numerical: should the generated body preserve IRON's current exact
Q4NX contract, or switch to MyLM's group-sum contract?

This experiment answers that on real Qwen3-8B-NPU2 weights and real layer
activations. It compares:

```text
current exact:
  sum(bf16(q * scale + offset) * activation)

MyLM-style group sum:
  sum(bf16(q * scale) * activation) + offset * bf16(sum(activation_group))
```

The report includes isolated projection deltas for Q/K/V/O/UP/GATE/DOWN and a
full single-layer forward where every Q4NX projection uses group-sum math.

Run:

```bash
python3 experiments/108_qwen3_q4nx_group_sum_contract/run.py
```

It writes:

```text
experiments/108_qwen3_q4nx_group_sum_contract/qwen3_q4nx_group_sum_contract.md
```

To test a real hidden activation dumped from the full CPU reference:

```bash
python3 experiments/108_qwen3_q4nx_group_sum_contract/run.py \
  --layer 35 \
  --token 0 \
  --hidden-bf16 qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16 \
  --output experiments/108_qwen3_q4nx_group_sum_contract/qwen3_q4nx_group_sum_contract_layer35.md
```

Current results:

- layer0/token31 synthetic hidden stays within the current hidden_out `1e-2`
  absolute gate: max_abs `0.0078125`.
- layer35/token0 real dumped hidden is not a silent replacement:
  `full_hidden_out max_abs=44.0` and `>1e-2=4008`.

So group-sum may be MyLM's intended production contract, but it cannot be
introduced as a drop-in change under the current exact reference. Either the
new assembly body must preserve exact per-dim rounding, or the full reference
must intentionally switch to group-sum and revalidate multi-layer token quality.
