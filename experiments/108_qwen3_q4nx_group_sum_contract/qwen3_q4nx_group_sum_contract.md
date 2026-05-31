# Qwen3 Q4NX Group-Sum Contract

- Layer: `0`
- Current token: `31`
- Hidden source: `synthetic make_reference_inputs`
- Elapsed seconds: `5.339`

## Formula

```text
current exact = sum(bf16(q * scale + offset) * activation)
group sum    = sum(bf16(q * scale) * activation) + offset * bf16(sum(activation_group))
```

## Isolated Projection Delta

| Stage | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated_Q` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000063986 | 0 | 0 | 0 |
| `isolated_K` | `(1024,)` | 0.000488281 | 0.000244141 | 0.000073189 | 0 | 0 | 0 |
| `isolated_V` | `(1024,)` | 0.000244141 | 0.000244141 | 0.000066034 | 0 | 0 | 0 |
| `isolated_O` | `(4096,)` | 0.000762939 | 0.000260163 | 0.000118498 | 0 | 0 | 0 |
| `isolated_UP` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000572819 | 803 | 0 | 0 |
| `isolated_GATE` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000609098 | 1029 | 0 | 0 |
| `isolated_DOWN` | `(4096,)` | 0.007812500 | 0.000976562 | 0.000343619 | 12 | 0 | 0 |

## Full Layer Delta

| Stage | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_q_raw` | `(4096,)` | 0.000488281 | 0.000244141 | 0.000063986 | 0 | 0 | 0 |
| `full_k_raw` | `(1024,)` | 0.000488281 | 0.000244141 | 0.000073189 | 0 | 0 | 0 |
| `full_v_raw` | `(1024,)` | 0.000244141 | 0.000244141 | 0.000066034 | 0 | 0 | 0 |
| `full_attention` | `(4096,)` | 0.000244141 | 0.000183105 | 0.000041655 | 0 | 0 | 0 |
| `full_o` | `(4096,)` | 0.001617432 | 0.000428773 | 0.000115102 | 1 | 0 | 0 |
| `full_up` | `(12288,)` | 0.003906250 | 0.001953125 | 0.000691634 | 1451 | 0 | 0 |
| `full_gate` | `(12288,)` | 0.007812500 | 0.001953125 | 0.000740844 | 1853 | 0 | 0 |
| `full_swiglu` | `(12288,)` | 0.002929688 | 0.000488281 | 0.000091562 | 2 | 0 | 0 |
| `full_down` | `(4096,)` | 0.001953125 | 0.001464844 | 0.000380665 | 124 | 0 | 0 |
| `full_hidden_out` | `(4096,)` | 0.007812500 | 0.003906250 | 0.000425354 | 469 | 0 | 0 |

## Decision

`group-sum stays within the current hidden_out 1e-2 absolute gate for this case`.

For production, this means the MyLM four-template schedule cannot simply
replace the active exact Q4NX body unless we either preserve the exact
rounding contract inside that schedule or intentionally move the full
reference to group-sum math and validate multi-layer token quality.
