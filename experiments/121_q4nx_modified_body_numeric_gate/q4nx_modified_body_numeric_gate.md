# Q4NX Modified Body Numeric Gate

This experiment uses the exp120 section contract as the body boundary and
tests the two realistic numerical branches before changing production asm.

## Exp120 Contract

- Sections: `5`
- Original slots: `1532`
- Total static group MACs: `264`
- Exact instruction replay: `True`
- Stable boundary pressure: `True`

## Candidate Body Shapes

| Candidate | Static MACs | Dynamic MACs | Keeps exp120 Shape | Exact Contract | Note |
| --- | ---: | ---: | --- | --- | --- |
| `mylm_group_correction` | 264 | 528 | yes | no | Preserves the MyLM 32+1 MAC/group shape; requires token-quality acceptance. |
| `exact_recompose_coeff` | 256 | 512 | no | yes | Exact parity route; zero must be recomposed into the coefficient before MAC. |
| `split_zero_per_dim` | 512 | 1024 | no | no | Diagnostic only: per-dim split zero changes fp32 accumulation order. |

## Synthetic Numeric Gate

- Samples: `256`
- Rows/sample: `32`
- Groups: `8`
- Group size: `32`
- Scenarios: `nominal, stress`

| Scenario | Candidate | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nominal` | `mylm_group_correction` | 0.000005722 | 0.000001907 | 0.000000345 | 0 | 0 | 0 |
| `nominal` | `exact_recompose_coeff` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `nominal` | `split_zero_per_dim` | 0.000005722 | 0.000001907 | 0.000000342 | 0 | 0 | 0 |
| `stress` | `mylm_group_correction` | 0.015625000 | 0.007812500 | 0.001518205 | 3431 | 16 | 0 |
| `stress` | `exact_recompose_coeff` | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 | 0 |
| `stress` | `split_zero_per_dim` | 0.015625000 | 0.007812500 | 0.001630444 | 3611 | 26 | 0 |

## Decision

- `exact route is numerically valid for this gate`.
- `MyLM-like route needs multi-layer/token acceptance, not exact migration`.

The first production-safe modified body should therefore target
`exact_recompose_coeff`: preserve exp120 section boundaries and live state,
but accept that the MyLM 33-MAC/group count changes to an exact 32-MAC/group
body unless a later real layer/token gate explicitly accepts the MyLM-like numerical contract.
