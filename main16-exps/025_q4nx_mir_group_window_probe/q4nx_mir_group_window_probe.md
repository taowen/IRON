# Q4NX MIR Group Window Probe

Status: `passed`

This experiment checks direct MIR generation for steady-state MyLM group windows.

## Results

### `mylm_groups_1_2_window`

Status: `pass`

Compile MyLM steady-state groups (1, 2) as one direct MIR loop body.

- `groups`: `(1, 2)`
- `source_event_count`: `378`
- `translated_event_count`: `352`
- `source_span_bytes`: `1382`
- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `0`
- `add_nc`: `1`
- `movx`: `0`
- `mov_scl`: `0`
- `paddb`: `0`
- `lda_s16`: `2`
- `vlda`: `2`
- `vldb`: `12`
- `vunpack`: `16`
- `vups_4x`: `16`
- `vadd`: `16`
- `vsub_f`: `16`
- `vconv_bf16_fp32`: `34`
- `vextbcst_16`: `64`
- `vextbcst_32`: `0`
- `vbcst_16`: `2`
- `vmul_f`: `2`
- `vmac_f`: `66`
- `vmov`: `100`
- `vmov_d`: `4`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `2`
- `loop_bundle_count`: `240`
- `loop_byte_count`: `1392`

### `mylm_groups_1_2_3_window`

Status: `pass`

Compile MyLM steady-state groups (1, 2, 3) as one direct MIR loop body.

- `groups`: `(1, 2, 3)`
- `source_event_count`: `567`
- `translated_event_count`: `528`
- `source_span_bytes`: `2074`
- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `0`
- `add_nc`: `1`
- `movx`: `0`
- `mov_scl`: `0`
- `paddb`: `0`
- `lda_s16`: `3`
- `vlda`: `3`
- `vldb`: `18`
- `vunpack`: `24`
- `vups_4x`: `24`
- `vadd`: `24`
- `vsub_f`: `24`
- `vconv_bf16_fp32`: `51`
- `vextbcst_16`: `96`
- `vextbcst_32`: `0`
- `vbcst_16`: `3`
- `vmul_f`: `3`
- `vmac_f`: `99`
- `vmov`: `150`
- `vmov_d`: `6`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `2`
- `loop_bundle_count`: `360`
- `loop_byte_count`: `2080`

## Interpretation

- Direct MIR still preserves the MyLM steady-state opcode vocabulary across group boundaries.
- There is still no `vextbcst.32` fallback and no vector store spill.
- Peano's postpipeliner still reports no schedule for these large already-ordered windows; the useful output here is the object/assembly shape, not an automatic new schedule.
- The next useful step is to compare this generated object against the MyLM hot range at the bundle/byte level, then package a tiny generated MIR object in the same direct-QKV harness for numeric readback.
