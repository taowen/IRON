# Q4NX MIR Dead Vmov NOPM Numeric Gate

Status: `failed`

## Patch

- patched raw: `main16-exps/034_q4nx_mir_dead_vmov_nopm_numeric_gate/build/patched_raw/mylm_c2r2_program.dead_vmov_to_nopm.bin`
- address: `0x44e`
- original: `f812cb19` / `vmov bmhh1, bmhh2`
- replacement: `f84a0318` / `nopm`
- static proof: acc1.bmhh is overwritten by the next event at 0x452 before use

## Cases

| case | status | matches expected | observed first words |
| --- | --- | --- | --- |
| `q4word0_allnibbles` | `record_observed` | `True` | `0x1, 0x3c803c80, 0x3c803c80, 0x3c803c80` |
| `q4word512_allnibbles` | `record_observed` | `True` | `0x1, 0x0, 0x0, 0x0` |
| `q4word0_nibble3` | `record_observed` | `True` | `0x1, 0x0, 0x3c800000, 0x0` |
| `zero0_allq4` | `record_observed` | `False` | `0x1, 0x408f408f, 0x407f407f, 0x407f407f` |

## Interpretation

- This is the first real source-bundle mutation selected by the bundle manifest.
- If all cases pass, the weak cell-level dead-destination proof is good enough for this site.
- If any case fails, the manifest is still missing data-dependent alias, rounding, or offset-path semantics.
- Current failed cases: `zero0_allq4`.
