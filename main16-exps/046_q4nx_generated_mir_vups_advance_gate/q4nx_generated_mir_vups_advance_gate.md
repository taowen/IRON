# Q4NX Generated MIR VUPS Advance Gate

Status: `failed`

- case set: `direct`

## Generated Variant

- patched raw: `main16-exps/046_q4nx_generated_mir_vups_advance_gate/build/generated_vups_advance/mylm_c2r2_program.generated_mir_vups_advance.bin`
- object: `main16-exps/046_q4nx_generated_mir_vups_advance_gate/build/generated_vups_advance/q4nx_hotloop_vups_advance_padded.o`
- window: `0x700..0x704`
- replacements: `2`
- text bytes: `5616`
- byte diffs vs MyLM hot loop: `4`

| address | original MIR | variant MIR |
| --- | --- | --- |
| `0x700` | `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2` | `$dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0` |
| `0x704` | `$dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0<br>$dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0` | `$bmhh1 = VMOV_alu_mv_mv_x $bmhh2<br>$dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0` |

## Byte Diffs

| address | original | generated |
| --- | --- | --- |
| `0x701` | `12` | `04` |
| `0x702` | `cb` | `0c` |
| `0x709` | `04` | `12` |
| `0x70a` | `0c` | `cb` |

## Numeric Gate

| case | status | matches reference | observed first words |
| --- | --- | --- | --- |
| `q4word0_allnibbles` | `record_observed` | `True` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80, 0x3c803c80, 0x0` |
| `q4word512_allnibbles` | `record_observed` | `True` | `0x1, 0x0, 0x0, 0x0, 0x0, 0x0` |
| `q4word0_nibble3` | `record_observed` | `True` | `0x1, 0x0, 0x3c800000, 0x0, 0x0, 0x0` |
| `zero0_allq4` | `record_observed` | `False` | `0x1, 0xc8c2c8c2, 0xc8c2c8c2, 0xc8c2c8c2, 0xc8c2c8c2, 0xc8c2c8c2` |

## Interpretation

- This is a schedule-changing mutation: `vups.4x` is issued one bundle earlier and the high-half move is delayed.
- A passed gate means this local `vups -> vadd` spacing change preserves selected MyLM-reference numeric coverage.
- A failed gate means the original immediate `vups.4x` placement is part of the local numeric contract.
