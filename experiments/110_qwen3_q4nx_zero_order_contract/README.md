# Experiment 110: Qwen3 Q4NX Zero-Order Contract

Exp109 showed that centered-dequant plus one `zero * group_sum` correction is
close to the current exact Q4NX reference, but not identical on real layer35
activations. This experiment isolates the remaining residual by keeping the
same centered coefficient and changing only the zero-correction accumulation
order.

Variants:

- `centered_group_sum_after_group`: 32 centered MACs, then one contracted
  `zero * sum(activation_group)` MAC.
- `centered_zero_after_group_dims`: 32 centered MACs, then 32 explicit
  `zero * activation_dim` MACs.
- `centered_zero_interleaved`: for every dim, do centered MAC then zero MAC.
- `centered_recompose_coeff`: recombine `centered + zero` before one MAC,
  checking whether centered coefficient construction itself loses precision.

Run:

```bash
python3 experiments/110_qwen3_q4nx_zero_order_contract/run.py
```

Run the real layer35 hidden dump:

```bash
python3 experiments/110_qwen3_q4nx_zero_order_contract/run.py \
  --layer 35 \
  --token 0 \
  --hidden-bf16 qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16 \
  --output experiments/110_qwen3_q4nx_zero_order_contract/qwen3_q4nx_zero_order_contract_layer35.md
```

Current results on the layer35 dump:

- `centered_group_sum_after_group`: `hidden_out max_abs=0.125`, `>1e-2=10`.
- `centered_zero_after_group_dims`: `hidden_out max_abs=0.125`, `>1e-2=10`.
- `centered_recompose_coeff`: exact against the current reference,
  `hidden_out max_abs=0`.

So the centered coefficient itself is not lossy. The remaining mismatch appears
when zero is added as a separate accumulator contribution. Exact parity requires
zero to be recomposed into the dequant coefficient before the MAC, equivalent to
the current direct `bf16(bf16(q*scale)+zero) * activation` order. The
MyLM-like `32 main MAC + 1 zero MAC` group shape is still a plausible speed
tradeoff, but it is a deliberate numerical contract change that needs a wider
multi-layer token gate.
