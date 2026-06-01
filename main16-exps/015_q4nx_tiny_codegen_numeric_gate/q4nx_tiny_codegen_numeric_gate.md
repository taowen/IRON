# Q4NX Tiny Codegen Numeric Gate

- Status: `passed`

## Results

| case | formula == MyLM | tiny == expected | tiny == MyLM | tiny status |
| --- | --- | --- | --- | --- |
| `q4word0_allnibbles` | `True` | `True` | `True` | `record_observed` |
| `q4word512_allnibbles` | `True` | `True` | `True` | `record_observed` |
| `q4word0_nibble3` | `True` | `True` | `True` | `record_observed` |
| `zero0_allq4` | `True` | `True` | `True` | `record_observed` |

## Interpretation

This is an emit-only generated whole-core source-assembly gate. It proves that the observed Q4NX layout/parity formula can be turned into a standalone raw AIE2P program that consumes the same stream count and emits a compact record exactly matching the MyLM raw body. The next step is to replace the generated constants with dynamic loads and arithmetic for the same tiny cases.
