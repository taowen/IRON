# Q4NX MIR Full Hotloop Probe

Status: `passed`

This experiment compiles the complete MyLM Q4NX hot-loop event stream as direct AIE2P MIR.
It checks whether the llvm-aie route can preserve the full object-level instruction shape before any NPU numeric packaging.

## Source Op Counts

- `add.nc`: `4`
- `lda.s16`: `8`
- `lshl`: `2`
- `mov`: `4`
- `movx`: `1`
- `nop`: `108`
- `nopa`: `1`
- `nopb`: `1`
- `paddb`: `2`
- `vadd`: `64`
- `vbcst.16`: `8`
- `vconv.bf16.fp32`: `136`
- `vextbcst.16`: `256`
- `vlda`: `11`
- `vldb`: `46`
- `vmac.f`: `264`
- `vmov`: `400`
- `vmov.d`: `16`
- `vmul.f`: `8`
- `vsub.f`: `64`
- `vunpack`: `64`
- `vups.4x`: `64`

## Results

### `mylm_full_hotloop_straightline`

Status: `pass`

Compile the complete MyLM Q4NX hot-loop event order as one straight-line MIR block.

- `source_event_count`: `1532`
- `translated_event_count`: `1422`
- `expected_translated_event_count`: `1422`
- `source_span_bytes`: `5610`
- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `2`
- `add_nc`: `4`
- `movx`: `1`
- `mov_scl`: `4`
- `paddb`: `2`
- `lda_s16`: `8`
- `vlda`: `11`
- `vldb`: `46`
- `vunpack`: `64`
- `vups_4x`: `64`
- `vadd`: `64`
- `vsub_f`: `64`
- `vconv_bf16_fp32`: `136`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vbcst_16`: `8`
- `vmul_f`: `8`
- `vmac_f`: `264`
- `vmov`: `400`
- `vmov_d`: `16`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `0`
- `bundle_count_entries`: `1`
- `loop_bundle_count`: `-1`
- `loop_byte_count`: `-1`

### `mylm_full_hotloop_zol`

Status: `pass`

Compile the complete MyLM Q4NX hot-loop event order inside a hardware-loop MIR block.

- `source_event_count`: `1532`
- `translated_event_count`: `1422`
- `expected_translated_event_count`: `1422`
- `source_span_bytes`: `5610`
- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `2`
- `add_nc`: `5`
- `movx`: `1`
- `mov_scl`: `4`
- `paddb`: `2`
- `lda_s16`: `8`
- `vlda`: `11`
- `vldb`: `46`
- `vunpack`: `64`
- `vups_4x`: `64`
- `vadd`: `64`
- `vsub_f`: `64`
- `vconv_bf16_fp32`: `136`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vbcst_16`: `8`
- `vmul_f`: `8`
- `vmac_f`: `264`
- `vmov`: `400`
- `vmov_d`: `16`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `2`
- `bundle_count_entries`: `3`
- `loop_bundle_count`: `988`
- `loop_byte_count`: `5600`

## Interpretation

- The direct MIR route now covers the complete hot-loop instruction vocabulary, including scalar pointer setup and drain instructions.
- Passing this gate means the compiler can assemble the MyLM-shaped full hot body without falling back to `vextbcst.32` or spilling with `vst`.
- This still is not a replacement kernel: the next gate is packaging the generated object into a direct-QKV numeric harness and comparing payloads against MyLM raw `0x1870`.
