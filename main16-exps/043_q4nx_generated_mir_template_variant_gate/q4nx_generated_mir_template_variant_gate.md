# Q4NX Generated MIR Template Variant Gate

Status: `passed`

- case set: `asymmetric-zero`
- original MIR: `      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2`
- variant MIR: `      $bmhh1 = VMOV_alu_mv_mv_x $bmll2`

## Generated Variant

- patched raw: `main16-exps/043_q4nx_generated_mir_template_variant_gate/build/generated_template_variant/mylm_c2r2_program.generated_mir_template_variant.bin`
- object: `main16-exps/043_q4nx_generated_mir_template_variant_gate/build/generated_template_variant/q4nx_hotloop_template_variant_padded.o`
- text bytes: `5616`
- mutation sites: `15`
- mutation addresses: `0x44e, 0x4ae, 0x700, 0x760, 0x9b4, 0xa14, 0xc68, 0xcc8, 0xf1c, 0xf7c, 0x11d0, 0x1230, 0x1484, 0x14e4, 0x173e`
- byte diffs vs MyLM hot loop: `15`

| address | original | generated |
| --- | --- | --- |
| `0x450` | `cb` | `c8` |
| `0x4b0` | `cb` | `c8` |
| `0x702` | `cb` | `c8` |
| `0x762` | `cb` | `c8` |
| `0x9b6` | `cb` | `c8` |
| `0xa16` | `cb` | `c8` |
| `0xc6a` | `cb` | `c8` |
| `0xcca` | `cb` | `c8` |
| `0xf1e` | `cb` | `c8` |
| `0xf7e` | `cb` | `c8` |
| `0x11d2` | `cb` | `c8` |
| `0x1232` | `cb` | `c8` |
| `0x1486` | `cb` | `c8` |
| `0x14e6` | `cb` | `c8` |
| `0x1740` | `cb` | `c8` |

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

- This regenerates the full hot loop and changes every matching source template bundle.
- Passing the gate shows the MIR route can carry a template-level generated variant.
- The generated variant is still functionally equivalent under the selected case set; it is not a speedup yet.
