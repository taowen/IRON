# Qwen3 Q4NX Group-Sum Contract

- Layer: `35`
- Current token: `0`
- Hidden source: `qwen3-layer/build/reference-dump-layer35/pos0000.layer35.hidden_in.bf16`
- Elapsed seconds: `5.363`

## Formula

```text
current exact = sum(bf16(q * scale + offset) * activation)
group sum    = sum(bf16(q * scale) * activation) + offset * bf16(sum(activation_group))
```

## Isolated Projection Delta

| Stage | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated_Q` | `(4096,)` | 0.250000000 | 0.062500000 | 0.004059391 | 950 | 361 | 97 |
| `isolated_K` | `(1024,)` | 0.500000000 | 0.015625000 | 0.002683013 | 639 | 13 | 4 |
| `isolated_V` | `(1024,)` | 0.015625000 | 0.007812500 | 0.002114028 | 760 | 2 | 0 |
| `isolated_O` | `(4096,)` | 0.031250000 | 0.007812500 | 0.001390401 | 1404 | 30 | 0 |
| `isolated_UP` | `(12288,)` | 1.000000000 | 0.125000000 | 0.035669681 | 11292 | 11279 | 1334 |
| `isolated_GATE` | `(12288,)` | 0.500000000 | 0.066406250 | 0.031423990 | 10625 | 10610 | 1721 |
| `isolated_DOWN` | `(4096,)` | 2.000000000 | 0.312500000 | 0.086788878 | 2750 | 2738 | 2505 |

## Full Layer Delta

| Stage | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_q_raw` | `(4096,)` | 0.250000000 | 0.062500000 | 0.004059391 | 950 | 361 | 97 |
| `full_k_raw` | `(1024,)` | 0.500000000 | 0.015625000 | 0.002683013 | 639 | 13 | 4 |
| `full_v_raw` | `(1024,)` | 0.015625000 | 0.007812500 | 0.002114028 | 760 | 2 | 0 |
| `full_attention` | `(4096,)` | 0.015625000 | 0.007812500 | 0.002114028 | 3040 | 8 | 0 |
| `full_o` | `(4096,)` | 0.031250000 | 0.015625000 | 0.003928017 | 2690 | 160 | 0 |
| `full_up` | `(12288,)` | 1.000000000 | 0.125000000 | 0.036834564 | 11362 | 11352 | 1459 |
| `full_gate` | `(12288,)` | 0.500000000 | 0.070312500 | 0.032558296 | 10775 | 10767 | 1888 |
| `full_swiglu` | `(12288,)` | 12.000000000 | 2.000000000 | 0.069212310 | 7018 | 2429 | 657 |
| `full_down` | `(4096,)` | 44.000000000 | 6.000000000 | 1.598496199 | 4026 | 4026 | 4005 |
| `full_hidden_out` | `(4096,)` | 44.000000000 | 6.000000000 | 1.597509384 | 4008 | 4008 | 3990 |

## Decision

`group-sum is not a silent drop-in for the current exact reference`.

For production, this means the MyLM four-template schedule cannot simply
replace the active exact Q4NX body unless we either preserve the exact
rounding contract inside that schedule or intentionally move the full
reference to group-sum math and validate multi-layer token quality.
