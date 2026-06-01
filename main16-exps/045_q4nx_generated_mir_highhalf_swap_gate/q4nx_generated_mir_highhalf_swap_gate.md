# Q4NX Generated MIR High-Half Swap Gate

Status: `passed`

- case set: `asymmetric-zero`

## Generated Variant

- patched raw: `main16-exps/045_q4nx_generated_mir_highhalf_swap_gate/build/generated_highhalf_swap/mylm_c2r2_program.generated_mir_highhalf_swap.bin`
- object: `main16-exps/045_q4nx_generated_mir_highhalf_swap_gate/build/generated_highhalf_swap/q4nx_hotloop_highhalf_swap_padded.o`
- window: `0x6f6..0x700`
- replacements: `2`
- text bytes: `5616`
- byte diffs vs MyLM hot loop: `2`

| address | original MIR | variant MIR |
| --- | --- | --- |
| `0x6f6` | `$bmhl1 = VMOV_alu_mv_mv_x $bmhl2` | `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` |
| `0x700` | `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` | `$bmhl1 = VMOV_alu_mv_mv_x $bmhl2` |

## Byte Diffs

| address | original | generated |
| --- | --- | --- |
| `0x6fc` | `8a` | `cb` |
| `0x702` | `cb` | `8a` |

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

- This is a schedule-changing mutation: the `bmhl1` and `bmhh1` move timing is swapped while bundle sizes are preserved.
- A passed gate means the MIR route tolerates this local high-half move reorder under the selected numeric coverage.
- A failed gate identifies the first selected case that depends on the original high-half move order.
