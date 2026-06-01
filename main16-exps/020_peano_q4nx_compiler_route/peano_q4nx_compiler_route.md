# Peano Q4NX Compiler Route

Status: `partial`

This experiment tries the compiler route for the main16 Q4NX hot body.
It uses Peano/llvm-aie to compile narrow C++ intrinsic candidates and scores the resulting AIE2P assembly.

## MyLM Target

- `vmac_f`: `264`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vups_4x`: `64`
- `vunpack`: `64`
- `vst_hot_loop`: `0`
- `vconv_bf16_fp32`: `136`
- `vmul_f`: `8`

## Candidate Results

### `peano_mac_signed_probe`

Status: `pass`

Key counts:
- `instruction_lines`: `11`
- `vmac_f`: `1`
- `vextbcst_16`: `1`
- `vextbcst_32`: `0`
- `vunpack`: `0`
- `vups`: `0`
- `vups_2x`: `0`
- `vups_4x`: `0`
- `vmul_f`: `0`
- `vconv_bf16_fp32`: `0`
- `vconv_fp32_bf16`: `0`
- `vldb`: `0`
- `vst`: `1`
- `control_33c`: `1`
- `hardware_loop_regs`: `0`

### `peano_q4_group_sum_chunk_lane_unrolled`

Status: `fail`

Reasons:
- still scale-multiplies per dim: vmul.f=256 > 32
- conversion pressure too high: vconv.bf16.fp32=384 > 160

Key counts:
- `instruction_lines`: `1155`
- `vmac_f`: `264`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vunpack`: `256`
- `vups`: `128`
- `vups_2x`: `128`
- `vups_4x`: `0`
- `vmul_f`: `256`
- `vconv_bf16_fp32`: `384`
- `vconv_fp32_bf16`: `1`
- `vldb`: `145`
- `vst`: `5`
- `control_33c`: `1`
- `hardware_loop_regs`: `0`

### `peano_q4_group_sum_chunk_lane_loop`

Status: `fail`

Reasons:
- still scale-multiplies per dim: vmul.f=256 > 32
- conversion pressure too high: vconv.bf16.fp32=384 > 160
- loop schedule uses too many vector local loads: vlda=153 > 64

Key counts:
- `instruction_lines`: `1095`
- `vmac_f`: `264`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vunpack`: `256`
- `vups`: `128`
- `vups_2x`: `128`
- `vups_4x`: `0`
- `vmul_f`: `256`
- `vconv_bf16_fp32`: `384`
- `vconv_fp32_bf16`: `1`
- `vldb`: `0`
- `vst`: `2`
- `control_33c`: `1`
- `hardware_loop_regs`: `3`

### `peano_q4_exact_chunk_lane_unrolled`

Status: `fail`

Reasons:
- exact rounding form is not viable as compiler route: vst=1043 > 128

Key counts:
- `instruction_lines`: `2698`
- `vmac_f`: `256`
- `vextbcst_16`: `256`
- `vextbcst_32`: `0`
- `vunpack`: `256`
- `vups`: `128`
- `vups_2x`: `128`
- `vups_4x`: `0`
- `vmul_f`: `256`
- `vconv_bf16_fp32`: `640`
- `vconv_fp32_bf16`: `257`
- `vldb`: `144`
- `vst`: `1043`
- `control_33c`: `1`
- `hardware_loop_regs`: `0`

## Conclusion

Peano can generate the signed vextbcst.16/vmac.f primitive and the 264/256 group-sum macro shape, but the current C++ intrinsic form still has too much scale multiplication/conversion and does not match MyLM's register-resident vups.4x schedule.

## Next Step

Keep Peano as backend, but generate a narrower DSL/MIR-level Q4NX body that exposes MyLM's vups.4x dequant pipeline instead of emitting per-dim C++ scale/dequant operations.
