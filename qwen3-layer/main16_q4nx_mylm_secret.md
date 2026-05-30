# Main16 Q4NX MyLM Secret Report

This report is generated from the local MyLM disassembly and the selected IRON role objects.

## Inputs

- MyLM disassembly: `/tmp/mylm_solidify_L31/disasm/c2r2.s`
- MyLM BD CSV: `/tmp/mylm_solidify_L31/layer_bd.csv`
- MyLM program segments: `/tmp/mylm_solidify_L31/programs/program_segments.tsv`
- MyLM program images: `/tmp/mylm_solidify_L31/programs/program_images.tsv`

## MyLM Raw Program Layout

- 16 main16 images, bytes=14868

| offset | bytes | source |
| --- | --- | --- |
| 0x0 | 492 | dma_0029_addr_4220000.bin |
| 0x1f0 | 5760 | dma_0030_addr_42201f0.bin |
| 0x1870 | 1544 | dma_0031_addr_4221870.bin |
| 0x1e80 | 1544 | dma_0032_addr_4221e80.bin |
| 0x2490 | 1544 | dma_0033_addr_4222490.bin |
| 0x2aa0 | 1560 | dma_0034_addr_4222aa0.bin |
| 0x30c0 | 1544 | dma_0035_addr_42230c0.bin |
| 0x36d0 | 504 | dma_0036_addr_42236d0.bin |
| 0x38d0 | 324 | dma_0037_addr_42238d0.bin |

The c2r2 program is a raw segmented core program. The Q4NX microkernel is loaded once at `0x1f0`; the visible fused phase bodies call into it instead of embedding separate C++-style hot loops per phase.

## MyLM Phase Body Shape

| phase | range | bytes | records | q4 calls | jl | acq | rel | jnz | lc/ls/le lines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q/K/V | 0x1870-0x1e80 | 1552 | 12 | 1 | 1 | 3 | 3 | 2 | 3 |
| O | 0x1e80-0x2490 | 1552 | 8 | 1 | 1 | 3 | 3 | 2 | 3 |
| up/gate | 0x2490-0x2aa0 | 1552 | 48 | 1 | 1 | 3 | 3 | 2 | 3 |
| down | 0x2aa0-0x30c0 | 1568 | 8 | 1 | 1 | 3 | 3 | 2 | 3 |
| alternate | 0x30c0-0x36d0 | 1552 | 304 | 1 | 1 | 3 | 3 | 2 | 3 |

Each normal phase body has one scheduled `jl #0x1f0` into the shared Q4NX microkernel. The compact-record replay count is encoded by the body entry setup, not by cloning the per-chunk lock choreography.

## MyLM Main16 BD Contract

| role | bd | len | base | next | acquire | release |
| --- | --- | --- | --- | --- | --- | --- |
| activation_ping | bd0 | 128 | 0x78000 | bd1 | L0:-1 | L1:1 |
| activation_pong | bd1 | 128 | 0x7c000 | bd0 | L0:-1 | L1:1 |
| weight_ping | bd2 | 1280 | 0x72800 | bd3 | L2:-1 | L3:1 |
| weight_pong | bd3 | 1280 | 0x74000 | bd2 | L2:-1 | L3:1 |
| record_ping | bd4 | 17 | 0x73c1c | bd5 | L5:-1 | L4:1 |
| record_pong | bd5 | 17 | 0x7541c | bd4 | L5:-1 | L4:1 |

The outer ABI matches the active IRON design: DMA0 activation, DMA1 Q4NX weight, and a 17-dword compact record output. The performance gap is inside the core program consuming that ABI.

## Q4NX Call-Site Evidence

| call | branch-slot setup |
| --- | --- |
| 0x1d70 | 1d76: 02 70 90 34 00 60 e1 71      	movs	p3, r15;		mov	p0, r18<br>1d7e: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80 |
| 0x2380 | 2386: 02 70 90 34 00 60 e1 71      	movs	p3, r15;		mov	p0, r18<br>238e: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80 |
| 0x2990 | 2996: 02 70 90 34 00 60 e1 71      	movs	p3, r15;		mov	p0, r18<br>299e: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80 |
| 0x2fb0 | 2fb6: 02 70 90 34 00 60 61 71      	movs	p3, r11;		mov	p0, r18<br>2fbe: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80 |
| 0x35c0 | 35c6: 02 70 90 34 00 60 e1 71      	movs	p3, r15;		mov	p0, r18<br>35ce: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80 |

The arguments are prepared in the branch-slot window after `jl #0x1f0`; MyLM is using a raw scheduled core body, not a normal C++ call boundary.

## IRON Full Main Core Shape

| core | q4 calls | jl | acq | rel | jnz | lc/ls/le lines | instruction lines | op slots | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iron_full_main_core | 129 | 157 | 232 | 232 | 11 | 9 | 2796 | 4191 | qwen3-layer/build/main_core_2_2.after-direct-emit.s |

This is the active full-layer main core shape produced by MLIR-AIE. It is the gate that must collapse toward the MyLM phase-body shape before a 5x main16 improvement is credible.

## Fixed Address Evidence

```text
1f0: ba 10 32 36 ce 01 00 40 45 00	mova	lc, #0x2;		movxm	p4, #0x73c64
238: 44 c0 e4 01 00 00    	movxm	ls, #0x260
23e: 44 a0 f0 16 00 00    	movxm	le, #0x1850
260: b6 a8 42 0c 6e 21 27 e8 3d 72 c3 03  	vlda	 x8, [p0], #0x40;		vldb	 x11, [p1], #0x40;		lshl	 r18, r19, r2;		add.nc	r16, r17, r1
1d70: 04 01 00 f8 00 00    	jl	#0x1f0
1d76: 02 70 90 34 00 60 e1 71      	movs	p3, r15;		mov	p0, r18
1d7e: 3a 11 40 36 cd 01 00 60 01 32	movs	p1, r16;		movxm	p2, #0x73c80
1d90: 64 fd bf 08 19 c8    	rel	#0x32, r12;		mov	r17, #-0x1
1d9c: 18 c8 00 16  	rel	#0x30, r12
1db0: 44 c0 d8 34 07 00    	movxm	p2, #0x73c60
1db6: 18 18 83 16  	acq	#0x34, r17
1dfa: 00 00        	nop
1dfe: 98 f2 04 0a  	vst.conv.bf16.fp32	 bmhh1, [p2, #0x0]
1e02: 98 d2 14 0a  	vst.conv.bf16.fp32	 bmhl1, [p2, #0x20]
```

## Opcode Shape

| range | instruction lines | op slots | top ops |
| --- | --- | --- | --- |
| mylm_q4_microkernel | 976 | 1560 | vmov=400, vmac.f=264, vextbcst.16=256, vconv.bf16.fp32=136, nop=108, vunpack=64, vups.4x=64, vadd=64, vsub.f=64, vldb=46, vmov.d=16, vlda=11, vbcst.16=9, mov=8, lda.s16=8, vmul.f=8 |
| mylm_q4_hot_loop | 963 | 1532 | vmov=400, vmac.f=264, vextbcst.16=256, vconv.bf16.fp32=136, nop=108, vunpack=64, vups.4x=64, vadd=64, vsub.f=64, vldb=46, vmov.d=16, vlda=11, lda.s16=8, vmul.f=8, vbcst.16=8, add.nc=4 |
| mylm_qkv_body | 264 | 393 | vmov=72, nop=44, vadd.f=40, vshift=32, st=20, lda=19, movxm=12, mov=11, movx=9, nopa=9, nopb=9, nopv=8, vlda.conv.fp32.bf16=8, st.s16=8, vconv.bf16.fp32=8, vextract.16=8 |
| iron_baseline_iron_fast_q4_hot_body | 575 | 1179 | vst=154, mov=110, vconv.bf16.fp32=110, vlda=75, nop=71, vadd.f=46, vbcst.16=45, vconv.fp32.bf16=45, vunpack=44, vmul.f=44, lda.s16=44, vmac.f=44, lshl=42, mova=36, or=34, add=26 |
| iron_baseline_iron_fast_q4_function | 6 | 8 | nop=2, movs=2, mov=2, j=1, movxm=1 |
| iron_baseline_iron_fast_q4_block_function | 23 | 34 | nop=8, mov=3, nopb=2, nops=2, nopxm=2, nopv=2, paddxm=2, movs=2, mova=1, ltu=1, jnz=1, st=1, jl=1, lshl=1, movxm=1, padda=1 |
| iron_baseline_iron_fast_perf_fill | 30 | 104 | nopb=16, nopv=16, nops=15, nopa=15, nopxm=8, movxm=6, nopm=4, mova=3, nop=3, ge=2, jnz=2, add.nc=2, lshl=2, nopx=2, mov=2, add=2 |

## IRON Object Size

| object | file bytes | text bytes | path |
| --- | --- | --- | --- |
| baseline | 9576 | 5568 | qwen3-layer/main_projection_q4nx_fast.o |

## Conclusion

MyLM's main16 advantage is a raw zero-overhead Q4NX loop with scheduled vector dequant/MAC/output packing and caller-side branch-slot setup. The active IRON kernel is numerically correct, but the full main core still exposes per-chunk lock/control structure that the compiler does not collapse into the same phase-body schedule.

The next performance step should generate a fixed-schedule main16 core body for the existing DMA0/DMA1/record ABI while preserving the verified Q4NX numerical contract `int4 * scale + offset`. Small Python generator cleanup cannot close this gap by itself.
