# Qwen3 Q4NX Centered-Dequant Contract

- Layer: `0`
- Current token: `31`
- Hidden source: `synthetic make_reference_inputs`
- Elapsed seconds: `14.763`

## Formula

```text
exact current reference:
  sum(bf16(bf16(q * scale) + zero) * activation)

weak group sum from exp108:
  sum(bf16(q * scale) * activation) + zero * sum(activation_group)

centered-dequant group sum:
  centered = bf16(bf16(q * scale) + zero) - zero
  exact    = sum(centered * activation) + zero * sum(activation_group)
```

The last line is algebraically exact over real numbers. On NPU fp32 it is
not necessarily bit-identical to the current direct order, because one
`zero * sum(activation_group)` correction contracts 32 separate
`zero * activation_dim` products into a different floating-point order.

## Isolated Projection Delta

| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated_Q` | `scaled_bf16_sum` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000063986 | 0 | 0 | 0 |
| `isolated_Q` | `scaled_fp32_sum` | `(4096,)` | 0.000244141 | 0.000122070 | 0.000013066 | 0 | 0 | 0 |
| `isolated_Q` | `centered_bf16_sum` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000063947 | 0 | 0 | 0 |
| `isolated_Q` | `centered_fp32_sum` | `(4096,)` | 0.000061035 | 0.000000000 | 0.000000015 | 0 | 0 | 0 |
| `isolated_K` | `scaled_bf16_sum` | `(1024,)` | 0.000488281 | 0.000244141 | 0.000073189 | 0 | 0 | 0 |
| `isolated_K` | `scaled_fp32_sum` | `(1024,)` | 0.000244141 | 0.000122070 | 0.000014263 | 0 | 0 | 0 |
| `isolated_K` | `centered_bf16_sum` | `(1024,)` | 0.000488281 | 0.000244141 | 0.000075233 | 0 | 0 | 0 |
| `isolated_K` | `centered_fp32_sum` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_V` | `scaled_bf16_sum` | `(1024,)` | 0.000244141 | 0.000244141 | 0.000066034 | 0 | 0 | 0 |
| `isolated_V` | `scaled_fp32_sum` | `(1024,)` | 0.000244141 | 0.000122070 | 0.000014756 | 0 | 0 | 0 |
| `isolated_V` | `centered_bf16_sum` | `(1024,)` | 0.000244141 | 0.000244141 | 0.000065976 | 0 | 0 | 0 |
| `isolated_V` | `centered_fp32_sum` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_O` | `scaled_bf16_sum` | `(4096,)` | 0.000762939 | 0.000260163 | 0.000118498 | 0 | 0 | 0 |
| `isolated_O` | `scaled_fp32_sum` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000016548 | 0 | 0 | 0 |
| `isolated_O` | `centered_bf16_sum` | `(4096,)` | 0.000732422 | 0.000244141 | 0.000117884 | 0 | 0 | 0 |
| `isolated_O` | `centered_fp32_sum` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_UP` | `scaled_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000572819 | 803 | 0 | 0 |
| `isolated_UP` | `scaled_fp32_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000140597 | 166 | 0 | 0 |
| `isolated_UP` | `centered_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000572504 | 733 | 0 | 0 |
| `isolated_UP` | `centered_fp32_sum` | `(12288,)` | 0.000244141 | 0.000000000 | 0.000000020 | 0 | 0 | 0 |
| `isolated_GATE` | `scaled_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000609098 | 1029 | 0 | 0 |
| `isolated_GATE` | `scaled_fp32_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000146172 | 198 | 0 | 0 |
| `isolated_GATE` | `centered_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000608770 | 890 | 0 | 0 |
| `isolated_GATE` | `centered_fp32_sum` | `(12288,)` | 0.000122070 | 0.000000000 | 0.000000015 | 0 | 0 | 0 |
| `isolated_DOWN` | `scaled_bf16_sum` | `(4096,)` | 0.007812500 | 0.000976562 | 0.000343619 | 12 | 0 | 0 |
| `isolated_DOWN` | `scaled_fp32_sum` | `(4096,)` | 0.000976562 | 0.000488281 | 0.000046908 | 0 | 0 | 0 |
| `isolated_DOWN` | `centered_bf16_sum` | `(4096,)` | 0.007812500 | 0.000976562 | 0.000341818 | 12 | 0 | 0 |
| `isolated_DOWN` | `centered_fp32_sum` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |

## Full Layer Delta

| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_q_raw` | `centered_bf16_sum` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000063947 | 0 | 0 | 0 |
| `full_k_raw` | `centered_bf16_sum` | `(1024,)` | 0.000488281 | 0.000244141 | 0.000075233 | 0 | 0 | 0 |
| `full_v_raw` | `centered_bf16_sum` | `(1024,)` | 0.000244141 | 0.000244141 | 0.000065976 | 0 | 0 | 0 |
| `full_attention` | `centered_bf16_sum` | `(4096,)` | 0.000244141 | 0.000122070 | 0.000043718 | 0 | 0 | 0 |
| `full_o` | `centered_bf16_sum` | `(4096,)` | 0.001068115 | 0.000367738 | 0.000100206 | 1 | 0 | 0 |
| `full_up` | `centered_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000327329 | 428 | 0 | 0 |
| `full_gate` | `centered_bf16_sum` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000339539 | 502 | 0 | 0 |
| `full_swiglu` | `centered_bf16_sum` | `(12288,)` | 0.000976562 | 0.000488281 | 0.000047470 | 0 | 0 | 0 |
| `full_down` | `centered_bf16_sum` | `(4096,)` | 0.001953125 | 0.000976562 | 0.000221442 | 5 | 0 | 0 |
| `full_hidden_out` | `centered_bf16_sum` | `(4096,)` | 0.007812500 | 0.003906250 | 0.000302136 | 318 | 0 | 0 |
| `full_q_raw` | `centered_fp32_sum` | `(4096,)` | 0.000061035 | 0.000000000 | 0.000000015 | 0 | 0 | 0 |
| `full_k_raw` | `centered_fp32_sum` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_v_raw` | `centered_fp32_sum` | `(1024,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_attention` | `centered_fp32_sum` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_o` | `centered_fp32_sum` | `(4096,)` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `full_up` | `centered_fp32_sum` | `(12288,)` | 0.000244141 | 0.000000000 | 0.000000020 | 0 | 0 | 0 |
| `full_gate` | `centered_fp32_sum` | `(12288,)` | 0.000122070 | 0.000000000 | 0.000000015 | 0 | 0 | 0 |
| `full_swiglu` | `centered_fp32_sum` | `(12288,)` | 0.000061035 | 0.000000000 | 0.000000005 | 0 | 0 | 0 |
| `full_down` | `centered_fp32_sum` | `(4096,)` | 0.000976562 | 0.000007629 | 0.000001271 | 0 | 0 | 0 |
| `full_hidden_out` | `centered_fp32_sum` | `(4096,)` | 0.003906250 | 0.000000000 | 0.000001140 | 1 | 0 | 0 |

## Coefficient Shape

| Phase | Center Delta Values | Center Delta Nonzero | Center Delta Max | Center Delta Mean | Zero-Sum-Round Values | Zero-Sum-Round Nonzero | Zero-Sum-Round Max | Zero-Sum-Round Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q` | 16777216 | 877511 | 0.001953125 | 0.000004767 | 524288 | 495616 | 0.000050552 | 0.000002590 |
| `K` | 4194304 | 221843 | 0.001464844 | 0.000005432 | 131072 | 123904 | 0.000042729 | 0.000002907 |
| `V` | 4194304 | 215756 | 0.000488281 | 0.000004696 | 131072 | 123904 | 0.000029892 | 0.000002617 |
| `O` | 16777216 | 862195 | 0.001953125 | 0.000004437 | 524288 | 524288 | 0.000418089 | 0.000009793 |
| `UP` | 50331648 | 2596904 | 0.000976562 | 0.000004387 | 1572864 | 1536000 | 0.000719726 | 0.000029407 |
| `GATE` | 50331648 | 2611911 | 0.000976562 | 0.000004697 | 1572864 | 1536000 | 0.001349017 | 0.000031206 |
| `DOWN` | 50331648 | 2602804 | 0.001953125 | 0.000004882 | 1572864 | 1564672 | 0.000293642 | 0.000007393 |

## Decision

`centered_fp32_sum stays within the current full-layer 1e-2 gate`.

This means the next assembly target should not be the scaled-only
group-sum body. It should use a centered-dequant coefficient path. If the
production gate requires tighter than this residual, the zero correction
must also preserve the direct per-dim accumulation order instead of a
single contracted group-sum MAC.
