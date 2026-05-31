# Qwen3 Q4NX Zero-Order Contract

- Layer: `35`
- Current token: `0`
- Hidden source: `qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16`
- Elapsed seconds: `15.455`

## Formula

```text
centered = bf16(bf16(q * scale) + zero) - zero

contracted:
  for group:
    acc += sum(centered_dim * activation_dim)
    acc += zero * sum(activation_dim)

split per-dim zero:
  for group, dim:
    acc += centered_dim * activation_dim
    acc += zero * activation_dim

recomposed coefficient:
  for group, dim:
    acc += (centered_dim + zero) * activation_dim
```

## Cost Shape

| Variant | Main MACs/group | Zero MACs/group | Total MACs/group | MyLM 33-MAC shape |
| --- | ---: | ---: | ---: | --- |
| `centered_group_sum_after_group` | 32 | 1 | 33 | yes |
| `centered_zero_after_group_dims` | 32 | 32 | 64 | no |
| `centered_zero_interleaved` | 32 | 32 | 64 | no |
| `centered_recompose_coeff` | 32 | 0 | 32 | no |

## Isolated Projection Delta

| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated_Q` | `centered_group_sum_after_group` | `(4096,)` | 0.000976562 | 0.000000000 | 0.000000479 | 0 | 0 | 0 |
| `isolated_Q` | `centered_zero_after_group_dims` | `(4096,)` | 0.000976562 | 0.000000000 | 0.000000240 | 0 | 0 | 0 |
| `isolated_Q` | `centered_zero_interleaved` | `(4096,)` | 0.000976562 | 0.000000000 | 0.000000238 | 0 | 0 | 0 |
| `isolated_Q` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_K` | `centered_group_sum_after_group` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_K` | `centered_zero_after_group_dims` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_K` | `centered_zero_interleaved` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_K` | `centered_recompose_coeff` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_V` | `centered_group_sum_after_group` | `(1024,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `isolated_V` | `centered_zero_after_group_dims` | `(1024,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `isolated_V` | `centered_zero_interleaved` | `(1024,)` | 0.000488281 | 0.000000000 | 0.000000477 | 0 | 0 | 0 |
| `isolated_V` | `centered_recompose_coeff` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_O` | `centered_group_sum_after_group` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_O` | `centered_zero_after_group_dims` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_O` | `centered_zero_interleaved` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_O` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_UP` | `centered_group_sum_after_group` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000000701 | 1 | 0 | 0 |
| `isolated_UP` | `centered_zero_after_group_dims` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000000765 | 1 | 0 | 0 |
| `isolated_UP` | `centered_zero_interleaved` | `(12288,)` | 0.015625000 | 0.000000000 | 0.000001272 | 1 | 1 | 0 |
| `isolated_UP` | `centered_recompose_coeff` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_GATE` | `centered_group_sum_after_group` | `(12288,)` | 0.000244141 | 0.000000000 | 0.000000022 | 0 | 0 | 0 |
| `isolated_GATE` | `centered_zero_after_group_dims` | `(12288,)` | 0.000488281 | 0.000000000 | 0.000000062 | 0 | 0 | 0 |
| `isolated_GATE` | `centered_zero_interleaved` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_GATE` | `centered_recompose_coeff` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_DOWN` | `centered_group_sum_after_group` | `(4096,)` | 0.001953125 | 0.000000000 | 0.000000477 | 1 | 0 | 0 |
| `isolated_DOWN` | `centered_zero_after_group_dims` | `(4096,)` | 0.001953125 | 0.000000000 | 0.000000715 | 1 | 0 | 0 |
| `isolated_DOWN` | `centered_zero_interleaved` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_DOWN` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |

## Full Layer Delta

| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_q_raw` | `centered_group_sum_after_group` | `(4096,)` | 0.000976562 | 0.000000000 | 0.000000479 | 0 | 0 | 0 |
| `full_k_raw` | `centered_group_sum_after_group` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_v_raw` | `centered_group_sum_after_group` | `(1024,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `full_attention` | `centered_group_sum_after_group` | `(4096,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `full_o` | `centered_group_sum_after_group` | `(4096,)` | 0.003906250 | 0.000012040 | 0.000007567 | 7 | 0 | 0 |
| `full_up` | `centered_group_sum_after_group` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000002536 | 5 | 0 | 0 |
| `full_gate` | `centered_group_sum_after_group` | `(12288,)` | 0.062500000 | 0.000000000 | 0.000005107 | 1 | 1 | 1 |
| `full_swiglu` | `centered_group_sum_after_group` | `(12288,)` | 0.015625000 | 0.000000000 | 0.000001400 | 1 | 1 | 0 |
| `full_down` | `centered_group_sum_after_group` | `(4096,)` | 0.125000000 | 0.000244141 | 0.000291198 | 30 | 16 | 10 |
| `full_hidden_out` | `centered_group_sum_after_group` | `(4096,)` | 0.125000000 | 0.000000000 | 0.000229836 | 16 | 10 | 8 |
| `full_q_raw` | `centered_zero_after_group_dims` | `(4096,)` | 0.000976562 | 0.000000000 | 0.000000240 | 0 | 0 | 0 |
| `full_k_raw` | `centered_zero_after_group_dims` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_v_raw` | `centered_zero_after_group_dims` | `(1024,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `full_attention` | `centered_zero_after_group_dims` | `(4096,)` | 0.000488281 | 0.000000000 | 0.000000536 | 0 | 0 | 0 |
| `full_o` | `centered_zero_after_group_dims` | `(4096,)` | 0.003906250 | 0.000012040 | 0.000007567 | 7 | 0 | 0 |
| `full_up` | `centered_zero_after_group_dims` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000002546 | 5 | 0 | 0 |
| `full_gate` | `centered_zero_after_group_dims` | `(12288,)` | 0.062500000 | 0.000000000 | 0.000005148 | 1 | 1 | 1 |
| `full_swiglu` | `centered_zero_after_group_dims` | `(12288,)` | 0.015625000 | 0.000000000 | 0.000001405 | 1 | 1 | 0 |
| `full_down` | `centered_zero_after_group_dims` | `(4096,)` | 0.125000000 | 0.000244141 | 0.000291198 | 30 | 16 | 10 |
| `full_hidden_out` | `centered_zero_after_group_dims` | `(4096,)` | 0.125000000 | 0.000000000 | 0.000229836 | 16 | 10 | 8 |
| `full_q_raw` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_k_raw` | `centered_recompose_coeff` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_v_raw` | `centered_recompose_coeff` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_attention` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_o` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_up` | `centered_recompose_coeff` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_gate` | `centered_recompose_coeff` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_swiglu` | `centered_recompose_coeff` | `(12288,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_down` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_hidden_out` | `centered_recompose_coeff` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |

## Decision

`centered coefficient construction is sound; exact parity requires zero to be recomposed into the coefficient before the MAC`.

Contracted full hidden: max_abs `0.125000000`, >1e-2 `10`.
Split per-dim zero full hidden: max_abs `0.125000000`, >1e-2 `10`.
Recomposed full hidden: max_abs `0.000000000`, >1e-2 `0`.

If contracted zero is accepted, the assembly target can keep the MyLM-like
`32 main MAC + 1 zero MAC` group shape. If exact direct-order parity is
required, zero must enter the dequant coefficient before the MAC; adding
zero as a separate accumulator contribution, even per dimension, is not
the same fp32 order as the current reference.
