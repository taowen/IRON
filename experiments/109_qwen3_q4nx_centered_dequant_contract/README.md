# Experiment 109: Qwen3 Q4NX Centered-Dequant Contract

Exp108 showed that this group-sum form is not a drop-in replacement:

```text
sum(bf16(q * scale) * activation) + zero * sum(activation_group)
```

This experiment checks the stronger MyLM-style algebraic shape that may explain
the `vadd/vsub/vconv/vmac` chain seen in the raw Q4NX body:

```text
centered = bf16(bf16(q * scale) + zero) - zero
exact    = sum(centered * activation) + zero * sum(activation_group)
```

The point is to learn whether the fast schedule can keep the current Q4NX
numerical contract while still using a group-sum correction path. The formula is
algebraically exact over real numbers, but it can differ from the current direct
per-dim reference on fp32 hardware because `zero * sum(activation_group)`
contracts 32 separate `zero * activation_dim` products.

Run the default synthetic layer0 case:

```bash
python3 experiments/109_qwen3_q4nx_centered_dequant_contract/run.py
```

Run the real layer35 hidden dump:

```bash
python3 experiments/109_qwen3_q4nx_centered_dequant_contract/run.py \
  --layer 35 \
  --token 0 \
  --hidden-bf16 qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16 \
  --output experiments/109_qwen3_q4nx_centered_dequant_contract/qwen3_q4nx_centered_dequant_contract_layer35.md
```

Current results:

- layer0/token31 synthetic hidden stays inside the `1e-2` full-layer gate:
  `centered_fp32_sum hidden_out max_abs=0.00390625`.
- layer35/token0 real hidden removes the exp108 systematic drift
  (`44.0 -> 0.0625` max_abs), but still leaves `2` hidden values over `1e-2`.

So the fast body should not use the weak scaled-only group-sum form. The useful
target is centered-dequant. If the final gate requires stricter parity than the
remaining residual, the zero correction cannot be a single contracted group-sum
MAC; it must preserve more of the direct per-dim accumulation order.
