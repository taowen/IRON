# Q4NX Hot Body Schedule

- Status: `passed`
- Source ELF: `experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf`
- Hot loop: `0x260..0x1850`
- Loop count: `2`
- Hot instruction lines: `963`
- Hot op slots: `1532`

## Key Counts

| op | static | dynamic |
| --- | ---: | ---: |
| `vldb` | `46` | `92` |
| `lda.s16` | `8` | `16` |
| `vextbcst.16` | `256` | `512` |
| `vbcst.16` | `8` | `16` |
| `vmac.f` | `264` | `528` |
| `vunpack` | `64` | `128` |
| `vups.4x` | `64` | `128` |
| `vconv.bf16.fp32` | `136` | `272` |
| `vst` | `0` | `0` |
| `vst.conv.bf16.fp32` | `0` | `0` |

## Activation Lane Groups

| group | range | lanes | complete | vmac.f | vextbcst.16 | vups.4x | vconv.bf16.fp32 | vst |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `0x260..0x52a` | `32` | `True` | `28` | `32` | `8` | `16` | `0` |
| `1` | `0x52a..0x7de` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `2` | `0x7de..0xa92` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `3` | `0xa92..0xd46` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `4` | `0xd46..0xffa` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `5` | `0xffa..0x12ae` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `6` | `0x12ae..0x1566` | `32` | `True` | `33` | `32` | `8` | `17` | `0` |
| `7` | `0x1566..0x1850` | `32` | `True` | `38` | `32` | `8` | `18` | `0` |

## Phase Group-Sum Producer

- Activation vector loads: `8`
- Scalar extracts: `8`
- Scratch stores: `8`

## Register Roles

- `p0`: selected Q4NX weight buffer at microkernel call
- `p1`: selected activation chunk buffer at microkernel call
- `p2`: fixed local scratch/accumulator base 0x73c80
- `p3`: phase-body group-sum scratch stream
- `x11`: 32-lane activation vector whose lanes feed vextbcst.16
- `r7`: current 16-bit activation group sum loaded from p3

## First Group Flow

This is a best-effort def/use table from the first activation-lane group. Memory side effects and packed vector aliases still need manual review.

| address | op | defs | uses | fragment |
| --- | --- | --- | --- | --- |
| `0x260` | `vlda` | `x8` | `p0` | `vlda	 x8, [p0], #0x40` |
| `0x260` | `vldb` | `x11` | `p1` | `vldb	 x11, [p1], #0x40` |
| `0x260` | `lshl` | `r18` | `r19, r2` | `lshl	 r18, r19, r2` |
| `0x260` | `add.nc` | `r16` | `r17, r1` | `add.nc	r16, r17, r1` |
| `0x26c` | `vlda` | `x7` | `p0` | `vlda	 x7, [p0], #0x40` |
| `0x26c` | `vunpack` | `x9` | `wl8` | `vunpack	x9, wl8, unpacksign0` |
| `0x26c` | `lshl` | `r20` | `r19, r3` | `lshl	 r20, r19, r3` |
| `0x26c` | `add.nc` | `p5` | `r17, r18` | `add.nc	p5, r17, r18` |
| `0x278` | `vlda` | `x0` | `p0` | `vlda	 x0, [p0], #0x40` |
| `0x278` | `vunpack` | `x8` | `wh8` | `vunpack	x8, wh8, unpacksign0` |
| `0x278` | `movx` | `r19` | `` | `movx	r19, #0x10` |
| `0x278` | `mov` | `r20` | `` | `mov	dj0, r20` |
| `0x284` | `vlda` | `p2` | `` | `vlda	 lfh0, [p2, dj0]` |
| `0x284` | `vunpack` | `x5` | `wl7` | `vunpack	x5, wl7, unpacksign0` |
| `0x284` | `add.nc` | `p4` | `r18, r16` | `add.nc	p4, r18, r16` |
| `0x28e` | `lda.s16` | `r7` | `p3` | `lda.s16	 r7, [p3], #0x2` |
| `0x28e` | `vunpack` | `x10` | `wh7` | `vunpack	x10, wh7, unpacksign0` |
| `0x294` | `vunpack` | `x1` | `wl0` | `vunpack	x1, wl0, unpacksign0` |
| `0x298` | `nop` | `` | `` | `nop` |
| `0x29a` | `vextbcst.16` | `x9` | `x11` | `vextbcst.16	 x9, x11, #0x1` |
| `0x29e` | `vups.4x` | `dm2` | `x9, s0` | `vups.4x	dm2, x9, s0, upssign0` |
| `0x29e` | `vadd` | `dm1` | `dm2, dm0, r0` | `vadd	dm1, dm2, dm0, r0` |
| `0x2a6` | `vups.4x` | `dm2` | `x8, s0` | `vups.4x	dm2, x8, s0, upssign0` |
| `0x2a6` | `vadd` | `dm2` | `dm2, dm0, r0` | `vadd	dm2, dm2, dm0, r0` |
| `0x2ae` | `vextbcst.16` | `x7` | `x11` | `vextbcst.16	 x7, x11, #0x0` |
| `0x2ae` | `vsub.f` | `dm1` | `dm1, dm0, r5` | `vsub.f	dm1, dm1, dm0, r5` |
| `0x2b6` | `vsub.f` | `dm3` | `dm2, dm0, r5` | `vsub.f	dm3, dm2, dm0, r5` |
| `0x2c2` | `vmov` | `bmll2` | `bmll1` | `vmov	bmll2, bmll1` |
| `0x2c6` | `vmov` | `bmlh2` | `bmlh1` | `vmov	bmlh2, bmlh1` |
| `0x2ca` | `vmov` | `bmhl2` | `bmhl1` | `vmov	bmhl2, bmhl1` |
| `0x2ce` | `vconv.bf16.fp32` | `x5` | `cml2` | `vconv.bf16.fp32	 x5, cml2` |
| `0x2ce` | `vmov` | `bmhh2` | `bmhh1` | `vmov	bmhh2, bmhh1` |
| `0x2d6` | `vups.4x` | `dm1` | `x5, s0` | `vups.4x	dm1, x5, s0, upssign0` |
| `0x2d6` | `vadd` | `dm4` | `dm1, dm0, r0` | `vadd	dm4, dm1, dm0, r0` |
| `0x2de` | `vconv.bf16.fp32` | `x3` | `` | `vconv.bf16.fp32	 x3, cmh2` |
| `0x2de` | `vmov` | `wl6` | `wh5` | `vmov	wl6, wh5` |
| `0x2e6` | `vmov` | `bmhl1` | `bmhl3` | `vmov	bmhl1, bmhl3` |
| `0x2e6` | `vsub.f` | `dm2` | `dm4, dm0, r5` | `vsub.f	dm2, dm4, dm0, r5` |
| `0x2ee` | `vmov` | `wl2` | `wh3` | `vmov	wl2, wh3` |
| `0x2f2` | `vmov` | `bmhh1` | `bmhh3` | `vmov	bmhh1, bmhh3` |
| `0x2f6` | `vldb` | `x6` | `p0` | `vldb	 x6, [p0], #0x40` |
| `0x2fa` | `vconv.bf16.fp32` | `x8` | `` | `vconv.bf16.fp32	 x8, cmh1` |
| `0x2fa` | `vmov` | `bmll1` | `bmll3` | `vmov	bmll1, bmll3` |
| `0x302` | `vups.4x` | `dm3` | `x10, s0` | `vups.4x	dm3, x10, s0, upssign0` |
| `0x302` | `vadd` | `dm3` | `dm3, dm0, r0` | `vadd	dm3, dm3, dm0, r0` |
| `0x30a` | `vextbcst.16` | `x10` | `x11` | `vextbcst.16	 x10, x11, #0x2` |
| `0x30a` | `vmul.f` | `dm3` | `x5, x7, r4` | `vmul.f	dm3, x5, x7, r4` |
| `0x312` | `vmov` | `bmlh1` | `bmlh3` | `vmov	bmlh1, bmlh3` |
| `0x312` | `vsub.f` | `dm4` | `dm3, dm0, r5` | `vsub.f	dm4, dm3, dm0, r5` |
| `0x31a` | `vextbcst.16` | `x7` | `x11` | `vextbcst.16	 x7, x11, #0x3` |
| `0x31e` | `vconv.bf16.fp32` | `x4` | `cml1` | `vconv.bf16.fp32	 x4, cml1` |
| `0x31e` | `vmov` | `bmhh1` | `bmhh2` | `vmov	bmhh1, bmhh2` |
| `0x31e` | `vmac.f` | `dm3` | `dm3, x6, x9, r4` | `vmac.f	dm3, dm3, x6, x9, r4` |
| `0x328` | `vextbcst.16` | `x9` | `x11` | `vextbcst.16	 x9, x11, #0x8` |
| `0x32c` | `vmov` | `wl10` | `wh4` | `vmov	wl10, wh4` |
| `0x330` | `vmov` | `bmll1` | `bmll2` | `vmov	bmll1, bmll2` |
| `0x330` | `vmac.f` | `dm3` | `dm3, x3, x10, r4` | `vmac.f	dm3, dm3, x3, x10, r4` |
| `0x338` | `vmov` | `bmll2` | `bmll4` | `vmov	bmll2, bmll4` |
| `0x33c` | `vunpack` | `x1` | `wh0` | `vunpack	x1, wh0, unpacksign0` |
| `0x33c` | `vextbcst.16` | `x7` | `x11` | `vextbcst.16	 x7, x11, #0x4` |
| `0x342` | `vmov` | `bmlh1` | `bmlh2` | `vmov	bmlh1, bmlh2` |
| `0x342` | `vmac.f` | `dm3` | `dm3, x2, x7, r4` | `vmac.f	dm3, dm3, x2, x7, r4` |
| `0x34a` | `vmov` | `bmlh2` | `bmlh4` | `vmov	bmlh2, bmlh4` |
| `0x34e` | `vconv.bf16.fp32` | `x5` | `cml1` | `vconv.bf16.fp32	 x5, cml1` |
| `0x34e` | `vextbcst.16` | `x4` | `x11` | `vextbcst.16	 x4, x11, #0x5` |
| `0x356` | `vmov` | `bmhl1` | `bmhl2` | `vmov	bmhl1, bmhl2` |
| `0x356` | `vmac.f` | `dm3` | `dm3, x4, x7, r4` | `vmac.f	dm3, dm3, x4, x7, r4` |
| `0x35e` | `vmov` | `bmhl2` | `bmhl4` | `vmov	bmhl2, bmhl4` |
| `0x362` | `vconv.bf16.fp32` | `x2` | `` | `vconv.bf16.fp32	 x2, cmh1` |
| `0x362` | `vups.4x` | `dm4` | `x1, s0` | `vups.4x	dm4, x1, s0, upssign0` |
| `0x362` | `vadd` | `dm1` | `dm4, dm0, r0` | `vadd	dm1, dm4, dm0, r0` |
| `0x36c` | `vextbcst.16` | `x3` | `x11` | `vextbcst.16	 x3, x11, #0x6` |
| `0x370` | `vmov` | `bmhh2` | `bmhh4` | `vmov	bmhh2, bmhh4` |
| `0x370` | `vsub.f` | `dm4` | `dm1, dm0, r5` | `vsub.f	dm4, dm1, dm0, r5` |
| `0x378` | `vextbcst.16` | `x0` | `x11` | `vextbcst.16	 x0, x11, #0x7` |
| `0x378` | `vmac.f` | `dm4` | `dm3, x10, x4, r4` | `vmac.f	dm4, dm3, x10, x4, r4` |
| `0x380` | `vextbcst.16` | `x4` | `x11` | `vextbcst.16	 x4, x11, #0x9` |
| `0x384` | `vconv.bf16.fp32` | `x8` | `cml2` | `vconv.bf16.fp32	 x8, cml2` |
| `0x384` | `vmov` | `wl3` | `wh8` | `vmov	wl3, wh8` |
| `0x384` | `vmov.d` | `dm1` | `dm4` | `vmov.d	dm1, dm4` |

## Interpretation

The MyLM hot body is not just a `vextbcst.16` probe. It has a fixed eight-group activation-lane schedule, no hot-loop `vst` spill, and a separate phase-body group-sum producer feeding `p3`. Matching this shape requires preserving the register lifetime plan across group boundaries, not only replacing individual broadcasts in the current IRON body.
