# Q4NX Generated MIR Contiguous Window Gate

Status: `passed`

- case set: `asymmetric-zero`

## Generated Variant

- patched raw: `main16-exps/044_q4nx_generated_mir_contiguous_window_gate/build/generated_window_variant/mylm_c2r2_program.generated_mir_window_variant.bin`
- object: `main16-exps/044_q4nx_generated_mir_contiguous_window_gate/build/generated_window_variant/q4nx_hotloop_window_variant_padded.o`
- window: `0x6ee..0x704`
- replacements: `3`
- text bytes: `5616`
- byte diffs vs MyLM hot loop: `3`

| address | original MIR | variant MIR |
| --- | --- | --- |
| `0x6ee` | `$bmlh1 = VMOV_alu_mv_mv_x $bmlh2` | `$bmlh1 = VMOV_alu_mv_mv_x $bmll2` |
| `0x6f6` | `$bmhl1 = VMOV_alu_mv_mv_x $bmhl2` | `$bmhl1 = VMOV_alu_mv_mv_x $bmll2` |
| `0x700` | `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` | `$bmhh1 = VMOV_alu_mv_mv_x $bmll2` |

## Byte Diffs

| address | original | generated |
| --- | --- | --- |
| `0x6f0` | `49` | `48` |
| `0x6fc` | `8a` | `88` |
| `0x702` | `cb` | `c8` |

## Numeric Gate

| case | status | matches reference | observed first words |
| --- | --- | --- | --- |
| `zero00_low_half` | `record_observed` | `True` | `0x1, 0x40804090, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero00_high_half` | `record_observed` | `True` | `0x1, 0x40904080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero01_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804090, 0x40804080, 0x40804080, 0x40804080` |
| `zero01_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40904080, 0x40804080, 0x40804080, 0x40804080` |
| `zero02_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804090, 0x40804080, 0x40804080` |
| `zero02_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40904080, 0x40804080, 0x40804080` |
| `zero03_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804090, 0x40804080` |
| `zero03_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40904080, 0x40804080` |
| `zero04_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804090` |
| `zero04_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40904080` |
| `zero05_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero05_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero06_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero06_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero07_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero07_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero08_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero08_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero09_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero09_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero10_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero10_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero11_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero11_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero12_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero12_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero13_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero13_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero14_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero14_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero15_low_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |
| `zero15_high_half` | `record_observed` | `True` | `0x1, 0x40804080, 0x40804080, 0x40804080, 0x40804080, 0x40804080` |

## Interpretation

- This is a contiguous producer/consumer window mutation, while preserving the original bundle schedule shape.
- Passing the gate would show this local window can be generated and changed without breaking current MyLM-reference coverage.
- Failing the gate identifies the first selected case that distinguishes the window's accumulator quadrant values.
