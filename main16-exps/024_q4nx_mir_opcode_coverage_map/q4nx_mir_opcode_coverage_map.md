# Q4NX MIR Opcode Coverage Map

Status: `passed`

This experiment maps MyLM group1 Q4NX events to explicit AIE2P machine MIR.
It checks backend opcode coverage before attempting a full numeric replacement kernel.

## MyLM Group1 Op Counts

- `lda.s16`: `1`
- `nop`: `13`
- `vadd`: `8`
- `vbcst.16`: `1`
- `vconv.bf16.fp32`: `17`
- `vextbcst.16`: `32`
- `vlda`: `1`
- `vldb`: `6`
- `vmac.f`: `33`
- `vmov`: `50`
- `vmov.d`: `2`
- `vmul.f`: `1`
- `vsub.f`: `8`
- `vunpack`: `8`
- `vups.4x`: `8`

## Results

### `opcode_coverage_block`

Status: `pass`

Compile one representative MIR instruction for every MyLM group1 opcode/register-class form.

- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `0`
- `add_nc`: `0`
- `movx`: `0`
- `mov_scl`: `0`
- `paddb`: `0`
- `lda_s16`: `1`
- `vlda`: `1`
- `vldb`: `1`
- `vunpack`: `1`
- `vups_4x`: `1`
- `vadd`: `1`
- `vsub_f`: `1`
- `vconv_bf16_fp32`: `1`
- `vextbcst_16`: `1`
- `vextbcst_32`: `0`
- `vbcst_16`: `1`
- `vmul_f`: `1`
- `vmac_f`: `1`
- `vmov`: `2`
- `vmov_d`: `1`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `0`
- `bundle_count_entries`: `1`
- `loop_bundle_count`: `-1`

### `mylm_group1_full_projection`

Status: `pass`

Compile the full MyLM group1 event order, excluding only explicit nops.

- `llc_returncode`: `0`
- `asm_returncode`: `0`
- `objdump_returncode`: `0`
- `lshl`: `0`
- `add_nc`: `1`
- `movx`: `0`
- `mov_scl`: `0`
- `paddb`: `0`
- `lda_s16`: `1`
- `vlda`: `1`
- `vldb`: `6`
- `vunpack`: `8`
- `vups_4x`: `8`
- `vadd`: `8`
- `vsub_f`: `8`
- `vconv_bf16_fp32`: `17`
- `vextbcst_16`: `32`
- `vextbcst_32`: `0`
- `vbcst_16`: `1`
- `vmul_f`: `1`
- `vmac_f`: `33`
- `vmov`: `50`
- `vmov_d`: `2`
- `vst`: `0`
- `schedule_found`: `0`
- `schedule_missed`: `2`
- `bundle_count_entries`: `3`
- `loop_bundle_count`: `120`

## Interpretation

- The useful part of `llvm-aie` is now concrete: it can compile a direct MIR form for the full MyLM group1 opcode vocabulary.
- This avoids the C++ instruction-selection drift that produced `vups.2x`/extra `vmul.f` in experiment 020.
- Passing this gate does not mean the replacement kernel is ready. The remaining work is a schedulable fill/steady/drain MIR graph with the same data dependencies and a numeric NPU comparison against MyLM direct `0x1870`.

## Next Step

Generate a full steady-state group pair, not one isolated group, so the postpipeliner sees the same cross-group producers and consumers that MyLM uses to hide latency.
