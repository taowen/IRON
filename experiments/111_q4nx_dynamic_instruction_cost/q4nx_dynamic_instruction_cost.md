# Q4NX Dynamic Instruction Cost

- IRON asm: `/var/home/taowen/projects/IRON/qwen3-layer/main_projection_q4nx_asm.s`
- MyLM disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

## Model Notes

### IRON active exact asm

- Loop shape: `2 lane passes * 8 groups * 8 exact4 blocks`
- Preserves current exact bf16(q*scale+zero) coefficient before MAC.
- Counts are source-expanded because the active body is generated from macros.

### MyLM raw Q4NX hot loop

- Loop shape: `static 0x260..0x1850 * lc=2`
- Uses 512 main MACs plus 16 offset/group-sum correction MACs per chunk.
- This is the fast numerical contract candidate, not current exact parity.

### Exact recomposed-coeff target floor

- Loop shape: `2 lane passes * 8 groups * 32 direct coefficient MACs`
- From exp110: exact parity requires centered+zero recomposed before MAC.
- This is a semantic floor, not an assembled schedule; dequant/rounding ops are still unknown.

## Dynamic Op Counts Per Q4NX Chunk

| Op | IRON Active | MyLM Raw | Target Floor | IRON/MyLM |
| --- | ---: | ---: | ---: | ---: |
| `vmac.f` | 512 | 528 | 512 | 0.97x |
| `vextbcst.16` | 512 | 512 | 512 | 1.00x |
| `vunpack` | 512 | 128 | 0 | 4.00x |
| `vups.4x` | 0 | 128 | 0 | 0.00x |
| `vups.2x` | 256 | 0 | 0 | n/a |
| `vmul.f` | 512 | 16 | 0 | 32.00x |
| `vadd.f` | 512 | 0 | 0 | n/a |
| `vsub.f` | 256 | 128 | 0 | 2.00x |
| `vconv.bf16.fp32` | 1280 | 272 | 0 | 4.71x |
| `vconv.fp32.bf16` | 768 | 0 | 0 | n/a |
| `vlda` | 2 | 22 | 0 | 0.09x |
| `vldb` | 256 | 92 | 0 | 2.78x |
| `vldb.128` | 256 | 0 | 0 | n/a |
| `vst` | 2 | 0 | 0 | n/a |
| `crunpacksize` | 512 | 0 | 0 | n/a |
| `crupsmode` | 128 | 0 | 0 | n/a |
| `vbcst.16` | 128 | 16 | 0 | 8.00x |
| `vmov` | 384 | 800 | 0 | 0.48x |
| `nop` | 3088 | 216 | 0 | 14.30x |

## Interpretation

- Active exact has `512` dynamic `vmac.f`, which is the expected 512 direct coefficient MACs.
- MyLM has `528` dynamic `vmac.f`: 512 main MACs plus 16 group correction MACs.
- Active exact pays `1280` dynamic `vconv.bf16.fp32` and `768` dynamic `vconv.fp32.bf16`; MyLM pays `272` and `0` in the hot loop.
- Therefore the next exact-parity assembly work should not chase MAC count; it should reduce coefficient construction and conversion traffic while keeping zero inside the coefficient before MAC.
- The MyLM `32+1 MAC/group` path remains a separate numerical contract because exp110 showed split zero accumulation is not exact.
