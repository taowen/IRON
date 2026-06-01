# Q4NX MIR Lifetime Order Probe

Status: `passed`

This experiment compares naive one-group MIR ordering with a projection of MyLM group1's actual
`vups.4x`, `vextbcst.16`, and `vmac.f` order/registers.

## Results

### `naive_x2d_order`

Status: `pass`

Naive x2d one-group order with the same macro counts as MyLM group1.

- `llc_returncode`: `0`
- `objdump_returncode`: `0`
- `vups_4x`: `8`
- `vups_2x`: `0`
- `vextbcst_16`: `32`
- `vextbcst_32`: `0`
- `vmac_f`: `33`
- `vst`: `0`
- `vlda`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `4`
- `bundle_count_remarks`: `6`
- `loop_bundle_count`: `70`

### `mylm_group1_projected_order`

Status: `pass`

MyLM group1 projection preserving actual order and registers for vups/vext/vmac.

- `llc_returncode`: `0`
- `objdump_returncode`: `0`
- `vups_4x`: `8`
- `vups_2x`: `0`
- `vextbcst_16`: `32`
- `vextbcst_32`: `0`
- `vmac_f`: `33`
- `vst`: `0`
- `vlda`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `4`
- `bundle_count_remarks`: `6`
- `loop_bundle_count`: `94`

## Interpretation

- Direct MIR can preserve the MyLM instruction vocabulary and register names.
- The key comparison is loop bundle count and postpipeliner status, not only opcode counts.
- If the projected order still does not schedule, the missing part is the full interleaved dependency graph including `vlda/vldb/vunpack/vadd/vsub/vconv/vmov`, not only the three hot op classes.

## Next Step

Generate a full group1 MIR projection with the extra producer instructions that feed the mixed-half operands in experiment 007, then rerun the same schedule gate.
