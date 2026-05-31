# MyLM Q4NX Generator Contract

This experiment combines exp118 boundary liveness with exp119 full MAC
operand graph records. The generated assembly include is still MyLM's
instruction text, but it is now split by the generator sections that a
modified numerical body must preserve.

## Checks

- Original slots: `1532`
- Generated slots: `1532`
- Original hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Generated hash: `9093b2372c1324ace60247f52e7edbec7ed77f880bc75c86a705f1f4c515e66b`
- Instruction text match: `True`
- Sections: `5`
- Total group MACs: `264`
- Stable steady-to-steady signature: `True`
- Stable boundary pressure: `True`

## Sections

| Section | Macro | Groups | Template | MACs | Live In | Live Out | Signature |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `fill` | `MYLM_Q4NX_FILL` | `0` | 0 | 28 | `entry` | `boundary1` | `` |
| `fill_to_steady` | `MYLM_Q4NX_FILL_TO_STEADY` | `1` | 1 | 33 | `boundary1` | `boundary2` | `070ae3b9658fd953` |
| `steady_to_steady` | `MYLM_Q4NX_STEADY_TO_STEADY` | `2,3,4,5` | 2 | 33 | `boundary2` | `boundary6` | `5466ff6129c65f6b` |
| `pre_drain` | `MYLM_Q4NX_PRE_DRAIN` | `6` | 6 | 33 | `boundary6` | `boundary7` | `` |
| `drain` | `MYLM_Q4NX_DRAIN` | `7` | 7 | 38 | `boundary7` | `exit` | `` |

## Boundary Pressure

| Boundary | Data Cells | Control Cells |
| ---: | ---: | ---: |
| 1 | 27 | 12 |
| 2 | 27 | 12 |
| 3 | 27 | 12 |
| 4 | 27 | 12 |
| 5 | 27 | 12 |
| 6 | 27 | 12 |
| 7 | 27 | 12 |

## Generated Include

- `/var/home/taowen/projects/IRON/experiments/120_mylm_q4nx_generator_contract/generated_mylm_q4nx_contract_sections.s.inc`

## Next Step

Use this contract as the single input to a modified Q4NX body generator.
The first candidate should preserve the section boundaries and live-state
contract, then replace only the coefficient construction path and run a
synthetic numerical gate before production `qwen3-layer` changes.
