# Qwen3 Q4NX Centered-Dequant Contract

- Layer: `35`
- Current token: `0`
- Hidden source: `qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16`
- Elapsed seconds: `14.931`

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
| `isolated_Q` | `scaled_bf16_sum` | `(4096,)` | 0.250000000 | 0.062500000 | 0.004059391 | 950 | 361 | 97 |
| `isolated_Q` | `scaled_fp32_sum` | `(4096,)` | 0.250000000 | 0.062500000 | 0.002519577 | 297 | 226 | 74 |
| `isolated_Q` | `centered_bf16_sum` | `(4096,)` | 0.250000000 | 0.031250000 | 0.002120472 | 808 | 200 | 35 |
| `isolated_Q` | `centered_fp32_sum` | `(4096,)` | 0.000007629 | 0.000000000 | 0.000000002 | 0 | 0 | 0 |
| `isolated_K` | `scaled_bf16_sum` | `(1024,)` | 0.500000000 | 0.015625000 | 0.002683013 | 639 | 13 | 4 |
| `isolated_K` | `scaled_fp32_sum` | `(1024,)` | 0.500000000 | 0.004611224 | 0.001025668 | 90 | 7 | 3 |
| `isolated_K` | `centered_bf16_sum` | `(1024,)` | 0.250000000 | 0.013828278 | 0.002088390 | 665 | 11 | 1 |
| `isolated_K` | `centered_fp32_sum` | `(1024,)` | 0.000030518 | 0.000000000 | 0.000000032 | 0 | 0 | 0 |
| `isolated_V` | `scaled_bf16_sum` | `(1024,)` | 0.015625000 | 0.007812500 | 0.002114028 | 760 | 2 | 0 |
| `isolated_V` | `scaled_fp32_sum` | `(1024,)` | 0.015625000 | 0.003906250 | 0.000236854 | 49 | 1 | 0 |
| `isolated_V` | `centered_bf16_sum` | `(1024,)` | 0.015625000 | 0.007812500 | 0.002130143 | 779 | 3 | 0 |
| `isolated_V` | `centered_fp32_sum` | `(1024,)` | 0.000122070 | 0.000000000 | 0.000000194 | 0 | 0 | 0 |
| `isolated_O` | `scaled_bf16_sum` | `(4096,)` | 0.031250000 | 0.007812500 | 0.001390401 | 1404 | 30 | 0 |
| `isolated_O` | `scaled_fp32_sum` | `(4096,)` | 0.031250000 | 0.007812500 | 0.000730504 | 687 | 13 | 0 |
| `isolated_O` | `centered_bf16_sum` | `(4096,)` | 0.031250000 | 0.007812500 | 0.001255928 | 1311 | 30 | 0 |
| `isolated_O` | `centered_fp32_sum` | `(4096,)` | 0.000000238 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `isolated_UP` | `scaled_bf16_sum` | `(12288,)` | 1.000000000 | 0.125000000 | 0.035669681 | 11292 | 11279 | 1334 |
| `isolated_UP` | `scaled_fp32_sum` | `(12288,)` | 0.500000000 | 0.031250000 | 0.001820939 | 296 | 163 | 92 |
| `isolated_UP` | `centered_bf16_sum` | `(12288,)` | 1.000000000 | 0.125000000 | 0.034583542 | 11255 | 11240 | 1298 |
| `isolated_UP` | `centered_fp32_sum` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000000704 | 1 | 0 | 0 |
| `isolated_GATE` | `scaled_bf16_sum` | `(12288,)` | 0.500000000 | 0.066406250 | 0.031423990 | 10625 | 10610 | 1721 |
| `isolated_GATE` | `scaled_fp32_sum` | `(12288,)` | 0.062500000 | 0.003906250 | 0.000233151 | 156 | 82 | 14 |
| `isolated_GATE` | `centered_bf16_sum` | `(12288,)` | 0.500000000 | 0.066406250 | 0.031423669 | 10619 | 10604 | 1729 |
| `isolated_GATE` | `centered_fp32_sum` | `(12288,)` | 0.015625000 | 0.000000000 | 0.000001590 | 2 | 1 | 0 |
| `isolated_DOWN` | `scaled_bf16_sum` | `(4096,)` | 2.000000000 | 0.312500000 | 0.086788878 | 2750 | 2738 | 2505 |
| `isolated_DOWN` | `scaled_fp32_sum` | `(4096,)` | 2.000000000 | 0.250000000 | 0.020519972 | 1178 | 1104 | 597 |
| `isolated_DOWN` | `centered_bf16_sum` | `(4096,)` | 2.000000000 | 0.250000000 | 0.084520072 | 2734 | 2719 | 2499 |
| `isolated_DOWN` | `centered_fp32_sum` | `(4096,)` | 0.015625000 | 0.000000000 | 0.000004351 | 2 | 1 | 0 |

## Full Layer Delta

| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_q_raw` | `centered_bf16_sum` | `(4096,)` | 0.250000000 | 0.031250000 | 0.002120472 | 808 | 200 | 35 |
| `full_k_raw` | `centered_bf16_sum` | `(1024,)` | 0.250000000 | 0.013828278 | 0.002088390 | 665 | 11 | 1 |
| `full_v_raw` | `centered_bf16_sum` | `(1024,)` | 0.015625000 | 0.007812500 | 0.002130143 | 779 | 3 | 0 |
| `full_attention` | `centered_bf16_sum` | `(4096,)` | 0.015625000 | 0.007812500 | 0.002130143 | 3116 | 12 | 0 |
| `full_o` | `centered_bf16_sum` | `(4096,)` | 0.062500000 | 0.015625000 | 0.004746794 | 2965 | 252 | 3 |
| `full_up` | `centered_bf16_sum` | `(12288,)` | 1.000000000 | 0.125000000 | 0.036286458 | 11364 | 11358 | 1493 |
| `full_gate` | `centered_bf16_sum` | `(12288,)` | 0.500000000 | 0.069462776 | 0.032849219 | 10815 | 10809 | 1929 |
| `full_swiglu` | `centered_bf16_sum` | `(12288,)` | 12.000000000 | 2.000000000 | 0.069763057 | 7043 | 2451 | 656 |
| `full_down` | `centered_bf16_sum` | `(4096,)` | 44.000000000 | 6.080471039 | 1.591383815 | 4018 | 4016 | 3998 |
| `full_hidden_out` | `centered_bf16_sum` | `(4096,)` | 44.000000000 | 6.002346039 | 1.591674328 | 4005 | 4004 | 3988 |
| `full_q_raw` | `centered_fp32_sum` | `(4096,)` | 0.000007629 | 0.000000000 | 0.000000002 | 0 | 0 | 0 |
| `full_k_raw` | `centered_fp32_sum` | `(1024,)` | 0.000030518 | 0.000000000 | 0.000000032 | 0 | 0 | 0 |
| `full_v_raw` | `centered_fp32_sum` | `(1024,)` | 0.000122070 | 0.000000000 | 0.000000194 | 0 | 0 | 0 |
| `full_attention` | `centered_fp32_sum` | `(4096,)` | 0.000122070 | 0.000000000 | 0.000000194 | 0 | 0 | 0 |
| `full_o` | `centered_fp32_sum` | `(4096,)` | 0.001953125 | 0.000000000 | 0.000001128 | 1 | 0 | 0 |
| `full_up` | `centered_fp32_sum` | `(12288,)` | 0.007812500 | 0.000000000 | 0.000000704 | 1 | 0 | 0 |
| `full_gate` | `centered_fp32_sum` | `(12288,)` | 0.015625000 | 0.000000000 | 0.000001590 | 2 | 1 | 0 |
| `full_swiglu` | `centered_fp32_sum` | `(12288,)` | 0.000976562 | 0.000000000 | 0.000000108 | 0 | 0 | 0 |
| `full_down` | `centered_fp32_sum` | `(4096,)` | 0.062500000 | 0.000000000 | 0.000030398 | 6 | 3 | 1 |
| `full_hidden_out` | `centered_fp32_sum` | `(4096,)` | 0.062500000 | 0.000000000 | 0.000031471 | 3 | 2 | 2 |

## Coefficient Shape

| Phase | Center Delta Values | Center Delta Nonzero | Center Delta Max | Center Delta Mean | Zero-Sum-Round Values | Zero-Sum-Round Nonzero | Zero-Sum-Round Max | Zero-Sum-Round Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q` | 16777216 | 872876 | 0.000976562 | 0.000004485 | 524288 | 516096 | 0.025100946 | 0.000096047 |
| `K` | 4194304 | 216688 | 0.001464844 | 0.000004453 | 131072 | 129024 | 0.017315626 | 0.000095295 |
| `V` | 4194304 | 208909 | 0.000976562 | 0.000005335 | 131072 | 129024 | 0.005906105 | 0.000116028 |
| `O` | 16777216 | 874999 | 0.001953125 | 0.000004659 | 524288 | 507904 | 0.009338379 | 0.000182266 |
| `UP` | 50331648 | 2641563 | 0.000976562 | 0.000005389 | 1572864 | 1523712 | 0.332426548 | 0.000333888 |
| `GATE` | 50331648 | 2648775 | 0.000976562 | 0.000005090 | 1572864 | 1523712 | 0.256343007 | 0.000308810 |
| `DOWN` | 50331648 | 2429722 | 0.002929688 | 0.000004430 | 1572864 | 1572864 | 0.892879725 | 0.003268891 |

## Decision

`centered_fp32_sum removes the systematic scaled-only drift but still has a small contracted-sum residual against the current exact order`.

This means the next assembly target should not be the scaled-only
group-sum body. It should use a centered-dequant coefficient path. If the
production gate requires tighter than this residual, the zero correction
must also preserve the direct per-dim accumulation order instead of a
single contracted group-sum MAC.
