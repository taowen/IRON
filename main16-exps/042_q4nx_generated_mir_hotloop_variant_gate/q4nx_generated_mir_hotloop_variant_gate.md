# Q4NX Generated MIR Hotloop Variant Gate

Status: `passed`

- case set: `asymmetric-zero`
- mutation address: `0x700`
- original MIR: `      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2`
- variant MIR: `      $bmhh1 = VMOV_alu_mv_mv_x $bmll2`

## Generated Variant

- patched raw: `main16-exps/042_q4nx_generated_mir_hotloop_variant_gate/build/generated_variant/mylm_c2r2_program.generated_mir_variant.bin`
- object: `main16-exps/042_q4nx_generated_mir_hotloop_variant_gate/build/generated_variant/q4nx_hotloop_variant_padded.o`
- text bytes: `5616`
- byte diffs vs MyLM hot loop: `1`

| address | original | generated |
| --- | --- | --- |
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

- This replaces the whole hot-loop range with generated pre-bundled MIR bytes, not a direct 4-byte patch.
- Passing the gate means the MIR generator route can produce a controlled non-byte-copy hot-loop variant.
- The generated variant is still functionally equivalent under the current direct-QKV cases; it is not a speedup yet.
