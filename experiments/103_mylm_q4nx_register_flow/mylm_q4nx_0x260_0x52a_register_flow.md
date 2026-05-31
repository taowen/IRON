# MyLM Q4NX Register Flow 0x260..0x52a

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This table is intentionally register-level. It is the input for a real
MyLM-style generator: every production instruction must be scheduled
with its defs, uses, alias family, and latency in mind.

## Summary

- Range: `0x260..0x52a`
- Instruction slots: `192`
- `vmac.f`: `28`
- `vextbcst.16`: `32`
- `vconv.bf16.fp32`: `16`
- `vups.4x`: `8`
- `vunpack`: `12`
- `lda.s16`: `2`

## Key Rules Learned

- `vextbcst.16` is not consume-immediate; exp102 proves a following
  `vmac.f` reads the previous broadcast value unless latency is filled.
- `dmN`, `bmllN/bmlhN/bmhlN/bmhhN`, and `cmlN/cmhN` are one accumulator
  alias family. A `vups.4x` or `vconv` into that family can overwrite
  values later consumed by `vmac.f`.
- MyLM uses the software pipeline to fill those latency gaps with real
  `vldb/vunpack/vups/vconv/vmov/vmac` work across group boundaries.

## Table

| Addr | Slot | Instruction | Defs | Uses | Alias Families | Semantics |
| --- | ---: | --- | --- | --- | --- | --- |
| `0x260` | 0 | `vlda	 x8, [p0], #0x40` | `x8` | `p0` | `x8->vec8` | load vector/accumulator from local memory |
| `0x260` | 1 | `vldb	 x11, [p1], #0x40` | `x11` | `p1` | `x11->vec11` | load 32-lane activation vector into x11 |
| `0x260` | 2 | `lshl	 r18, r19, r2` | `` | `r18, r19, r2` | `` | unclassified |
| `0x260` | 3 | `add.nc	r16, r17, r1` | `r16` | `r16, r17, r1` | `` | pointer/scalar update |
| `0x26c` | 0 | `vlda	 x7, [p0], #0x40` | `x7` | `p0` | `x7->vec7` | load vector/accumulator from local memory |
| `0x26c` | 1 | `vunpack	x9, wl8, unpacksign0` | `x9` | `wl8, unpacksign0` | `x9->vec9, wl8->vec8` | unpack packed integer lanes into vector register |
| `0x26c` | 2 | `lshl	 r20, r19, r3` | `` | `r20, r19, r3` | `` | unclassified |
| `0x26c` | 3 | `add.nc	p5, r17, r18` | `p5` | `p5, r17, r18` | `` | pointer/scalar update |
| `0x278` | 0 | `vlda	 x0, [p0], #0x40` | `x0` | `p0` | `x0->vec0` | load vector/accumulator from local memory |
| `0x278` | 1 | `vunpack	x8, wh8, unpacksign0` | `x8` | `wh8, unpacksign0` | `x8->vec8, wh8->vec8` | unpack packed integer lanes into vector register |
| `0x278` | 2 | `movx	r19, #0x10` | `r19` | `` | `` | scalar/control register setup |
| `0x278` | 3 | `mov	dj0, r20` | `` | `r20` | `` | scalar/control register setup |
| `0x284` | 0 | `vlda	 lfh0, [p2, dj0]` | `p2` | `p2` | `` | load vector/accumulator from local memory |
| `0x284` | 1 | `vunpack	x5, wl7, unpacksign0` | `x5` | `wl7, unpacksign0` | `x5->vec5, wl7->vec7` | unpack packed integer lanes into vector register |
| `0x284` | 2 | `add.nc	p4, r18, r16` | `p4` | `p4, r18, r16` | `` | pointer/scalar update |
| `0x28e` | 0 | `lda.s16	 r7, [p3], #0x2` | `r7, p3` | `p3` | `` | load one activation group-sum scratch scalar |
| `0x28e` | 1 | `vunpack	x10, wh7, unpacksign0` | `x10` | `wh7, unpacksign0` | `x10->vec10, wh7->vec7` | unpack packed integer lanes into vector register |
| `0x294` | 0 | `vunpack	x1, wl0, unpacksign0` | `x1` | `wl0, unpacksign0` | `x1->vec1, wl0->vec0` | unpack packed integer lanes into vector register |
| `0x298` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x29a` | 0 | `vextbcst.16	 x9, x11, #0x1` | `x9` | `x11` | `x9->vec9, x11->vec11` | extract+broadcast 16-bit lane #0x1; consumer must respect latency |
| `0x29e` | 0 | `vups.4x	dm2, x9, s0, upssign0` | `dm2` | `x9, s0, upssign0` | `dm2->acc2, x9->vec9` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x29e` | 1 | `vadd	dm1, dm2, dm0, r0` | `dm1` | `dm2, dm0, r0` | `dm1->acc1, dm2->acc2, dm0->acc0` | vector arithmetic |
| `0x2a6` | 0 | `vups.4x	dm2, x8, s0, upssign0` | `dm2` | `x8, s0, upssign0` | `dm2->acc2, x8->vec8` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x2a6` | 1 | `vadd	dm2, dm2, dm0, r0` | `dm2` | `dm2, dm0, r0` | `dm2->acc2, dm0->acc0` | vector arithmetic |
| `0x2ae` | 0 | `vextbcst.16	 x7, x11, #0x0` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x0; consumer must respect latency |
| `0x2ae` | 1 | `vsub.f	dm1, dm1, dm0, r5` | `dm1` | `dm1, dm0, r5` | `dm1->acc1, dm0->acc0` | vector arithmetic |
| `0x2b6` | 0 | `vsub.f	dm3, dm2, dm0, r5` | `dm3` | `dm2, dm0, r5` | `dm3->acc3, dm2->acc2, dm0->acc0` | vector arithmetic |
| `0x2c2` | 0 | `vmov	bmll2, bmll1` | `bmll2` | `bmll1` | `bmll2->acc2, bmll1->acc1` | move between aliasing vector/accumulator segments |
| `0x2c6` | 0 | `vmov	bmlh2, bmlh1` | `bmlh2` | `bmlh1` | `bmlh2->acc2, bmlh1->acc1` | move between aliasing vector/accumulator segments |
| `0x2ca` | 0 | `vmov	bmhl2, bmhl1` | `bmhl2` | `bmhl1` | `bmhl2->acc2, bmhl1->acc1` | move between aliasing vector/accumulator segments |
| `0x2ce` | 0 | `vconv.bf16.fp32	 x5, cml2` | `x5` | `cml2` | `x5->vec5, cml2->acc2` | convert between accumulator segment and BF16 vector segment |
| `0x2ce` | 1 | `vmov	bmhh2, bmhh1` | `bmhh2` | `bmhh1` | `bmhh2->acc2, bmhh1->acc1` | move between aliasing vector/accumulator segments |
| `0x2d6` | 0 | `vups.4x	dm1, x5, s0, upssign0` | `dm1` | `x5, s0, upssign0` | `dm1->acc1, x5->vec5` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x2d6` | 1 | `vadd	dm4, dm1, dm0, r0` | `dm4` | `dm1, dm0, r0` | `dm4->acc4, dm1->acc1, dm0->acc0` | vector arithmetic |
| `0x2de` | 0 | `vconv.bf16.fp32	 x3, cmh2` | `x3` | `cmh2` | `x3->vec3, cmh2->acc2` | convert between accumulator segment and BF16 vector segment |
| `0x2de` | 1 | `vmov	wl6, wh5` | `wl6` | `wh5` | `wl6->vec6, wh5->vec5` | move between aliasing vector/accumulator segments |
| `0x2e6` | 0 | `vmov	bmhl1, bmhl3` | `bmhl1` | `bmhl3` | `bmhl1->acc1, bmhl3->acc3` | move between aliasing vector/accumulator segments |
| `0x2e6` | 1 | `vsub.f	dm2, dm4, dm0, r5` | `dm2` | `dm4, dm0, r5` | `dm2->acc2, dm4->acc4, dm0->acc0` | vector arithmetic |
| `0x2ee` | 0 | `vmov	wl2, wh3` | `wl2` | `wh3` | `wl2->vec2, wh3->vec3` | move between aliasing vector/accumulator segments |
| `0x2f2` | 0 | `vmov	bmhh1, bmhh3` | `bmhh1` | `bmhh3` | `bmhh1->acc1, bmhh3->acc3` | move between aliasing vector/accumulator segments |
| `0x2f6` | 0 | `vldb	 x6, [p0], #0x40` | `x6` | `p0` | `x6->vec6` | load packed/Q4 or scale vector |
| `0x2fa` | 0 | `vconv.bf16.fp32	 x8, cmh1` | `x8` | `cmh1` | `x8->vec8, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x2fa` | 1 | `vmov	bmll1, bmll3` | `bmll1` | `bmll3` | `bmll1->acc1, bmll3->acc3` | move between aliasing vector/accumulator segments |
| `0x302` | 0 | `vups.4x	dm3, x10, s0, upssign0` | `dm3` | `x10, s0, upssign0` | `dm3->acc3, x10->vec10` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x302` | 1 | `vadd	dm3, dm3, dm0, r0` | `dm3` | `dm3, dm0, r0` | `dm3->acc3, dm0->acc0` | vector arithmetic |
| `0x30a` | 0 | `vextbcst.16	 x10, x11, #0x2` | `x10` | `x11` | `x10->vec10, x11->vec11` | extract+broadcast 16-bit lane #0x2; consumer must respect latency |
| `0x30a` | 1 | `vmul.f	dm3, x5, x7, r4` | `dm3` | `x5, x7, r4` | `dm3->acc3, x5->vec5, x7->vec7` | vector arithmetic |
| `0x312` | 0 | `vmov	bmlh1, bmlh3` | `bmlh1` | `bmlh3` | `bmlh1->acc1, bmlh3->acc3` | move between aliasing vector/accumulator segments |
| `0x312` | 1 | `vsub.f	dm4, dm3, dm0, r5` | `dm4` | `dm3, dm0, r5` | `dm4->acc4, dm3->acc3, dm0->acc0` | vector arithmetic |
| `0x31a` | 0 | `vextbcst.16	 x7, x11, #0x3` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x3; consumer must respect latency |
| `0x31e` | 0 | `vconv.bf16.fp32	 x4, cml1` | `x4` | `cml1` | `x4->vec4, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x31e` | 1 | `vmov	bmhh1, bmhh2` | `bmhh1` | `bmhh2` | `bmhh1->acc1, bmhh2->acc2` | move between aliasing vector/accumulator segments |
| `0x31e` | 2 | `vmac.f	dm3, dm3, x6, x9, r4` | `dm3` | `dm3, x6, x9, r4` | `dm3->acc3, x6->vec6, x9->vec9` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x328` | 0 | `vextbcst.16	 x9, x11, #0x8` | `x9` | `x11` | `x9->vec9, x11->vec11` | extract+broadcast 16-bit lane #0x8; consumer must respect latency |
| `0x32c` | 0 | `vmov	wl10, wh4` | `wl10` | `wh4` | `wl10->vec10, wh4->vec4` | move between aliasing vector/accumulator segments |
| `0x330` | 0 | `vmov	bmll1, bmll2` | `bmll1` | `bmll2` | `bmll1->acc1, bmll2->acc2` | move between aliasing vector/accumulator segments |
| `0x330` | 1 | `vmac.f	dm3, dm3, x3, x10, r4` | `dm3` | `dm3, x3, x10, r4` | `dm3->acc3, x3->vec3, x10->vec10` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x338` | 0 | `vmov	bmll2, bmll4` | `bmll2` | `bmll4` | `bmll2->acc2, bmll4->acc4` | move between aliasing vector/accumulator segments |
| `0x33c` | 0 | `vunpack	x1, wh0, unpacksign0` | `x1` | `wh0, unpacksign0` | `x1->vec1, wh0->vec0` | unpack packed integer lanes into vector register |
| `0x33c` | 1 | `vextbcst.16	 x7, x11, #0x4` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x4; consumer must respect latency |
| `0x342` | 0 | `vmov	bmlh1, bmlh2` | `bmlh1` | `bmlh2` | `bmlh1->acc1, bmlh2->acc2` | move between aliasing vector/accumulator segments |
| `0x342` | 1 | `vmac.f	dm3, dm3, x2, x7, r4` | `dm3` | `dm3, x2, x7, r4` | `dm3->acc3, x2->vec2, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x34a` | 0 | `vmov	bmlh2, bmlh4` | `bmlh2` | `bmlh4` | `bmlh2->acc2, bmlh4->acc4` | move between aliasing vector/accumulator segments |
| `0x34e` | 0 | `vconv.bf16.fp32	 x5, cml1` | `x5` | `cml1` | `x5->vec5, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x34e` | 1 | `vextbcst.16	 x4, x11, #0x5` | `x4` | `x11` | `x4->vec4, x11->vec11` | extract+broadcast 16-bit lane #0x5; consumer must respect latency |
| `0x356` | 0 | `vmov	bmhl1, bmhl2` | `bmhl1` | `bmhl2` | `bmhl1->acc1, bmhl2->acc2` | move between aliasing vector/accumulator segments |
| `0x356` | 1 | `vmac.f	dm3, dm3, x4, x7, r4` | `dm3` | `dm3, x4, x7, r4` | `dm3->acc3, x4->vec4, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x35e` | 0 | `vmov	bmhl2, bmhl4` | `bmhl2` | `bmhl4` | `bmhl2->acc2, bmhl4->acc4` | move between aliasing vector/accumulator segments |
| `0x362` | 0 | `vconv.bf16.fp32	 x2, cmh1` | `x2` | `cmh1` | `x2->vec2, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x362` | 1 | `vups.4x	dm4, x1, s0, upssign0` | `dm4` | `x1, s0, upssign0` | `dm4->acc4, x1->vec1` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x362` | 2 | `vadd	dm1, dm4, dm0, r0` | `dm1` | `dm4, dm0, r0` | `dm1->acc1, dm4->acc4, dm0->acc0` | vector arithmetic |
| `0x36c` | 0 | `vextbcst.16	 x3, x11, #0x6` | `x3` | `x11` | `x3->vec3, x11->vec11` | extract+broadcast 16-bit lane #0x6; consumer must respect latency |
| `0x370` | 0 | `vmov	bmhh2, bmhh4` | `bmhh2` | `bmhh4` | `bmhh2->acc2, bmhh4->acc4` | move between aliasing vector/accumulator segments |
| `0x370` | 1 | `vsub.f	dm4, dm1, dm0, r5` | `dm4` | `dm1, dm0, r5` | `dm4->acc4, dm1->acc1, dm0->acc0` | vector arithmetic |
| `0x378` | 0 | `vextbcst.16	 x0, x11, #0x7` | `x0` | `x11` | `x0->vec0, x11->vec11` | extract+broadcast 16-bit lane #0x7; consumer must respect latency |
| `0x378` | 1 | `vmac.f	dm4, dm3, x10, x4, r4` | `dm4` | `dm3, x10, x4, r4` | `dm4->acc4, dm3->acc3, x10->vec10, x4->vec4` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x380` | 0 | `vextbcst.16	 x4, x11, #0x9` | `x4` | `x11` | `x4->vec4, x11->vec11` | extract+broadcast 16-bit lane #0x9; consumer must respect latency |
| `0x384` | 0 | `vconv.bf16.fp32	 x8, cml2` | `x8` | `cml2` | `x8->vec8, cml2->acc2` | convert between accumulator segment and BF16 vector segment |
| `0x384` | 1 | `vmov	wl3, wh8` | `wl3` | `wh8` | `wl3->vec3, wh8->vec8` | move between aliasing vector/accumulator segments |
| `0x384` | 2 | `vmov.d	dm1, dm4` | `dm1` | `dm4` | `dm1->acc1, dm4->acc4` | move between aliasing vector/accumulator segments |
| `0x38e` | 0 | `vextbcst.16	 x7, x11, #0xa` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0xa; consumer must respect latency |
| `0x38e` | 1 | `vmac.f	dm4, dm4, x8, x3, r4` | `dm4` | `dm4, x8, x3, r4` | `dm4->acc4, x8->vec8, x3->vec3` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x396` | 0 | `vextbcst.16	 x10, x11, #0xb` | `x10` | `x11` | `x10->vec10, x11->vec11` | extract+broadcast 16-bit lane #0xb; consumer must respect latency |
| `0x396` | 1 | `vmov.d	dm3, dm1` | `dm3` | `dm1` | `dm3->acc3, dm1->acc1` | move between aliasing vector/accumulator segments |
| `0x39e` | 0 | `vmov	wl3, wh2` | `wl3` | `wh2` | `wl3->vec3, wh2->vec2` | move between aliasing vector/accumulator segments |
| `0x3a2` | 0 | `vmac.f	dm4, dm4, x3, x0, r4` | `dm4` | `dm4, x3, x0, r4` | `dm4->acc4, x3->vec3, x0->vec0` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x3a6` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x3a8` | 0 | `vmov	wl5, wh5` | `wl5` | `wh5` | `wl5->vec5, wh5->vec5` | move between aliasing vector/accumulator segments |
| `0x3ac` | 0 | `vmov	wl9, wh8` | `wl9` | `wh8` | `wl9->vec9, wh8->vec8` | move between aliasing vector/accumulator segments |
| `0x3ac` | 1 | `vmac.f	dm4, dm4, x5, x9, r4` | `dm4` | `dm4, x5, x9, r4` | `dm4->acc4, x5->vec5, x9->vec9` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x3b4` | 0 | `vmov	bmhl1, bmhl3` | `bmhl1` | `bmhl3` | `bmhl1->acc1, bmhl3->acc3` | move between aliasing vector/accumulator segments |
| `0x3b8` | 0 | `vmov	bmhh1, bmhh3` | `bmhh1` | `bmhh3` | `bmhh1->acc1, bmhh3->acc3` | move between aliasing vector/accumulator segments |
| `0x3bc` | 0 | `vconv.bf16.fp32	 x1, cmh2` | `x1` | `cmh2` | `x1->vec1, cmh2->acc2` | convert between accumulator segment and BF16 vector segment |
| `0x3bc` | 1 | `vmov	bmll1, bmll3` | `bmll1` | `bmll3` | `bmll1->acc1, bmll3->acc3` | move between aliasing vector/accumulator segments |
| `0x3bc` | 2 | `vmac.f	dm4, dm4, x5, x4, r4` | `dm4` | `dm4, x5, x4, r4` | `dm4->acc4, x5->vec5, x4->vec4` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x3c6` | 0 | `vconv.bf16.fp32	 x0, cmh1` | `x0` | `cmh1` | `x0->vec0, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x3c6` | 1 | `vups.4x	dm3, x1, s0, upssign0` | `dm3` | `x1, s0, upssign0` | `dm3->acc3, x1->vec1` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x3c6` | 2 | `vadd	dm2, dm3, dm0, r0` | `dm2` | `dm3, dm0, r0` | `dm2->acc2, dm3->acc3, dm0->acc0` | vector arithmetic |
| `0x3d0` | 0 | `vextbcst.16	 x4, x11, #0xc` | `x4` | `x11` | `x4->vec4, x11->vec11` | extract+broadcast 16-bit lane #0xc; consumer must respect latency |
| `0x3d4` | 0 | `vunpack	x4, wl6, unpacksign0` | `x4` | `wl6, unpacksign0` | `x4->vec4, wl6->vec6` | unpack packed integer lanes into vector register |
| `0x3d4` | 1 | `vmov	bmlh1, bmlh3` | `bmlh1` | `bmlh3` | `bmlh1->acc1, bmlh3->acc3` | move between aliasing vector/accumulator segments |
| `0x3d4` | 2 | `vmac.f	dm3, dm4, x2, x7, r4` | `dm3` | `dm4, x2, x7, r4` | `dm3->acc3, dm4->acc4, x2->vec2, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x3de` | 0 | `vunpack	x6, wh6, unpacksign0` | `x6` | `wh6, unpacksign0` | `x6->vec6, wh6->vec6` | unpack packed integer lanes into vector register |
| `0x3de` | 1 | `vextbcst.16	 x5, x11, #0xd` | `x5` | `x11` | `x5->vec5, x11->vec11` | extract+broadcast 16-bit lane #0xd; consumer must respect latency |
| `0x3de` | 2 | `vsub.f	dm2, dm2, dm0, r5` | `dm2` | `dm2, dm0, r5` | `dm2->acc2, dm0->acc0` | vector arithmetic |
| `0x3e8` | 0 | `vconv.bf16.fp32	 x2, cml1` | `x2` | `cml1` | `x2->vec2, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x3e8` | 1 | `vextbcst.16	 x10, x11, #0xe` | `x10` | `x11` | `x10->vec10, x11->vec11` | extract+broadcast 16-bit lane #0xe; consumer must respect latency |
| `0x3f0` | 0 | `vextbcst.16	 x3, x11, #0xf` | `x3` | `x11` | `x3->vec3, x11->vec11` | extract+broadcast 16-bit lane #0xf; consumer must respect latency |
| `0x3f0` | 1 | `vmac.f	dm1, dm3, x3, x10, r4` | `dm1` | `dm3, x3, x10, r4` | `dm1->acc1, dm3->acc3, x3->vec3, x10->vec10` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x3f8` | 0 | `vextbcst.16	 x7, x11, #0x10` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x10; consumer must respect latency |
| `0x3fc` | 0 | `vmov	wl8, wh1` | `wl8` | `wh1` | `wl8->vec8, wh1->vec1` | move between aliasing vector/accumulator segments |
| `0x400` | 0 | `vextbcst.16	 x4, x11, #0x15` | `x4` | `x11` | `x4->vec4, x11->vec11` | extract+broadcast 16-bit lane #0x15; consumer must respect latency |
| `0x400` | 1 | `vmac.f	dm3, dm1, x8, x4, r4` | `dm3` | `dm1, x8, x4, r4` | `dm3->acc3, dm1->acc1, x8->vec8, x4->vec4` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x408` | 0 | `vups.4x	dm4, x4, s0, upssign0` | `dm4` | `x4, s0, upssign0` | `dm4->acc4, x4->vec4` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x408` | 1 | `vadd	dm4, dm4, dm0, r0` | `dm4` | `dm4, dm0, r0` | `dm4->acc4, dm0->acc0` | vector arithmetic |
| `0x410` | 0 | `vmov	wl5, wh2` | `wl5` | `wh2` | `wl5->vec5, wh2->vec2` | move between aliasing vector/accumulator segments |
| `0x414` | 0 | `vextbcst.16	 x9, x11, #0x11` | `x9` | `x11` | `x9->vec9, x11->vec11` | extract+broadcast 16-bit lane #0x11; consumer must respect latency |
| `0x414` | 1 | `vmac.f	dm3, dm3, x9, x5, r4` | `dm3` | `dm3, x9, x5, r4` | `dm3->acc3, x9->vec9, x5->vec5` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x41c` | 0 | `vsub.f	dm4, dm4, dm0, r5` | `dm4` | `dm4, dm0, r5` | `dm4->acc4, dm0->acc0` | vector arithmetic |
| `0x420` | 0 | `vextbcst.16	 x1, x11, #0x12` | `x1` | `x11` | `x1->vec1, x11->vec11` | extract+broadcast 16-bit lane #0x12; consumer must respect latency |
| `0x424` | 0 | `vmov	bmll1, bmll2` | `bmll1` | `bmll2` | `bmll1->acc1, bmll2->acc2` | move between aliasing vector/accumulator segments |
| `0x424` | 1 | `vmac.f	dm3, dm3, x1, x10, r4` | `dm3` | `dm3, x1, x10, r4` | `dm3->acc3, x1->vec1, x10->vec10` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x42c` | 0 | `vextbcst.16	 x10, x11, #0x18` | `x10` | `x11` | `x10->vec10, x11->vec11` | extract+broadcast 16-bit lane #0x18; consumer must respect latency |
| `0x430` | 0 | `vmov	wl3, wh0` | `wl3` | `wh0` | `wl3->vec3, wh0->vec0` | move between aliasing vector/accumulator segments |
| `0x434` | 0 | `vextbcst.16	 x8, x11, #0x13` | `x8` | `x11` | `x8->vec8, x11->vec11` | extract+broadcast 16-bit lane #0x13; consumer must respect latency |
| `0x434` | 1 | `vmac.f	dm3, dm3, x8, x3, r4` | `dm3` | `dm3, x8, x3, r4` | `dm3->acc3, x8->vec8, x3->vec3` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x43c` | 0 | `vmov	bmlh1, bmlh2` | `bmlh1` | `bmlh2` | `bmlh1->acc1, bmlh2->acc2` | move between aliasing vector/accumulator segments |
| `0x440` | 0 | `vextbcst.16	 x7, x11, #0x14` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x14; consumer must respect latency |
| `0x444` | 0 | `vconv.bf16.fp32	 x2, cml1` | `x2` | `cml1` | `x2->vec2, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x444` | 1 | `vmov	bmhl1, bmhl2` | `bmhl1` | `bmhl2` | `bmhl1->acc1, bmhl2->acc2` | move between aliasing vector/accumulator segments |
| `0x444` | 2 | `vmac.f	dm3, dm3, x2, x7, r4` | `dm3` | `dm3, x2, x7, r4` | `dm3->acc3, x2->vec2, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x44e` | 0 | `vmov	bmhh1, bmhh2` | `bmhh1` | `bmhh2` | `bmhh1->acc1, bmhh2->acc2` | move between aliasing vector/accumulator segments |
| `0x452` | 0 | `vups.4x	dm1, x6, s0, upssign0` | `dm1` | `x6, s0, upssign0` | `dm1->acc1, x6->vec6` | upshift/expand unpacked Q4 lanes into FP32 accumulator family |
| `0x452` | 1 | `vadd	dm2, dm1, dm0, r0` | `dm2` | `dm1, dm0, r0` | `dm2->acc2, dm1->acc1, dm0->acc0` | vector arithmetic |
| `0x45a` | 0 | `vconv.bf16.fp32	 x5, cmh1` | `x5` | `cmh1` | `x5->vec5, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x45a` | 1 | `vextbcst.16	 x6, x11, #0x16` | `x6` | `x11` | `x6->vec6, x11->vec11` | extract+broadcast 16-bit lane #0x16; consumer must respect latency |
| `0x45a` | 2 | `vmac.f	dm3, dm3, x5, x9, r4` | `dm3` | `dm3, x5, x9, r4` | `dm3->acc3, x5->vec5, x9->vec9` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x464` | 0 | `vextbcst.16	 x9, x11, #0x17` | `x9` | `x11` | `x9->vec9, x11->vec11` | extract+broadcast 16-bit lane #0x17; consumer must respect latency |
| `0x464` | 1 | `vsub.f	dm2, dm2, dm0, r5` | `dm2` | `dm2, dm0, r5` | `dm2->acc2, dm0->acc0` | vector arithmetic |
| `0x46c` | 0 | `vextbcst.16	 x0, x11, #0x19` | `x0` | `x11` | `x0->vec0, x11->vec11` | extract+broadcast 16-bit lane #0x19; consumer must respect latency |
| `0x470` | 0 | `vmov	bmll1, bmll4` | `bmll1` | `bmll4` | `bmll1->acc1, bmll4->acc4` | move between aliasing vector/accumulator segments |
| `0x470` | 1 | `vmac.f	dm3, dm3, x0, x1, r4` | `dm3` | `dm3, x0, x1, r4` | `dm3->acc3, x0->vec0, x1->vec1` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x478` | 0 | `vmov	bmll4, lfh0` | `bmll4` | `` | `bmll4->acc4` | move between aliasing vector/accumulator segments |
| `0x47c` | 0 | `vmov	bmhh1, bmhh4` | `bmhh1` | `bmhh4` | `bmhh1->acc1, bmhh4->acc4` | move between aliasing vector/accumulator segments |
| `0x480` | 0 | `vmov	wl8, wh5` | `wl8` | `wh5` | `wl8->vec8, wh5->vec5` | move between aliasing vector/accumulator segments |
| `0x480` | 1 | `vmac.f	dm3, dm3, x3, x8, r4` | `dm3` | `dm3, x3, x8, r4` | `dm3->acc3, x3->vec3, x8->vec8` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x488` | 0 | `vmov	bmlh1, bmlh4` | `bmlh1` | `bmlh4` | `bmlh1->acc1, bmlh4->acc4` | move between aliasing vector/accumulator segments |
| `0x48c` | 0 | `vmov	wl2, wh2` | `wl2` | `wh2` | `wl2->vec2, wh2->vec2` | move between aliasing vector/accumulator segments |
| `0x490` | 0 | `vconv.bf16.fp32	 x3, cml1` | `x3` | `cml1` | `x3->vec3, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x490` | 1 | `vmov	bmhl1, bmhl4` | `bmhl1` | `bmhl4` | `bmhl1->acc1, bmhl4->acc4` | move between aliasing vector/accumulator segments |
| `0x490` | 2 | `vmac.f	dm3, dm3, x2, x7, r4` | `dm3` | `dm3, x2, x7, r4` | `dm3->acc3, x2->vec2, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x49a` | 0 | `vmov	bmll1, bmll2` | `bmll1` | `bmll2` | `bmll1->acc1, bmll2->acc2` | move between aliasing vector/accumulator segments |
| `0x49e` | 0 | `vconv.bf16.fp32	 x1, cmh1` | `x1` | `cmh1` | `x1->vec1, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x49e` | 1 | `vmov	bmlh1, bmlh2` | `bmlh1` | `bmlh2` | `bmlh1->acc1, bmlh2->acc2` | move between aliasing vector/accumulator segments |
| `0x4a6` | 0 | `vmov	bmhl1, bmhl2` | `bmhl1` | `bmhl2` | `bmhl1->acc1, bmhl2->acc2` | move between aliasing vector/accumulator segments |
| `0x4a6` | 1 | `vmac.f	dm3, dm3, x2, x4, r4` | `dm3` | `dm3, x2, x4, r4` | `dm3->acc3, x2->vec2, x4->vec4` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x4ae` | 0 | `vmov	bmhh1, bmhh2` | `bmhh1` | `bmhh2` | `bmhh1->acc1, bmhh2->acc2` | move between aliasing vector/accumulator segments |
| `0x4b2` | 0 | `vconv.bf16.fp32	 x5, cml1` | `x5` | `cml1` | `x5->vec5, cml1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x4b2` | 1 | `vextbcst.16	 x2, x11, #0x1a` | `x2` | `x11` | `x2->vec2, x11->vec11` | extract+broadcast 16-bit lane #0x1a; consumer must respect latency |
| `0x4ba` | 0 | `vconv.bf16.fp32	 x4, cmh1` | `x4` | `cmh1` | `x4->vec4, cmh1->acc1` | convert between accumulator segment and BF16 vector segment |
| `0x4ba` | 1 | `vextbcst.16	 x6, x11, #0x1b` | `x6` | `x11` | `x6->vec6, x11->vec11` | extract+broadcast 16-bit lane #0x1b; consumer must respect latency |
| `0x4ba` | 2 | `vmac.f	dm2, dm3, x5, x6, r4` | `dm2` | `dm3, x5, x6, r4` | `dm2->acc2, dm3->acc3, x5->vec5, x6->vec6` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x4c4` | 0 | `vextbcst.16	 x7, x11, #0x1c` | `x7` | `x11` | `x7->vec7, x11->vec11` | extract+broadcast 16-bit lane #0x1c; consumer must respect latency |
| `0x4c8` | 0 | `vmov	wl8, wh1` | `wl8` | `wh1` | `wl8->vec8, wh1->vec1` | move between aliasing vector/accumulator segments |
| `0x4cc` | 0 | `vmov	wl9, wh5` | `wl9` | `wh5` | `wl9->vec9, wh5->vec5` | move between aliasing vector/accumulator segments |
| `0x4cc` | 1 | `vmac.f	dm1, dm2, x8, x9, r4` | `dm1` | `dm2, x8, x9, r4` | `dm1->acc1, dm2->acc2, x8->vec8, x9->vec9` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x4d4` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x4d6` | 0 | `lda.s16	 r7, [p3], #0x2` | `r7, p3` | `p3` | `` | load one activation group-sum scratch scalar |
| `0x4d6` | 1 | `vmov	wl3, wh3` | `wl3` | `wh3` | `wl3->vec3, wh3->vec3` | move between aliasing vector/accumulator segments |
| `0x4dc` | 0 | `vldb	 wl2, [p5], #0x40` | `wl2` | `p5` | `wl2->vec2` | load packed/Q4 or scale vector |
| `0x4dc` | 1 | `vextbcst.16	 x10, x11, #0x1d` | `x10` | `x11` | `x10->vec10, x11->vec11` | extract+broadcast 16-bit lane #0x1d; consumer must respect latency |
| `0x4dc` | 2 | `vmac.f	dm1, dm1, x3, x10, r4` | `dm1` | `dm1, x3, x10, r4` | `dm1->acc1, x3->vec3, x10->vec10` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x4e6` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x4e8` | 0 | `vextbcst.16	 x0, x11, #0x1e` | `x0` | `x11` | `x0->vec0, x11->vec11` | extract+broadcast 16-bit lane #0x1e; consumer must respect latency |
| `0x4ec` | 0 | `vlda	 x8, [p0], #0x40` | `x8` | `p0` | `x8->vec8` | load vector/accumulator from local memory |
| `0x4ec` | 1 | `vldb	 wl6, [p4], #0x40` | `wl6` | `p4` | `wl6->vec6` | load packed/Q4 or scale vector |
| `0x4ec` | 2 | `vextbcst.16	 x11, x11, #0x1f` | `x11` | `x11` | `x11->vec11` | extract+broadcast 16-bit lane #0x1f; consumer must respect latency |
| `0x4ec` | 3 | `vmac.f	dm1, dm1, x3, x0, r4` | `dm1` | `dm1, x3, x0, r4` | `dm1->acc1, x3->vec3, x0->vec0` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x4f8` | 0 | `vmov	wl3, wh4` | `wl3` | `wh4` | `wl3->vec3, wh4->vec4` | move between aliasing vector/accumulator segments |
| `0x4fc` | 0 | `vbcst.16	 x1, r7` | `x1` | `r7` | `x1->vec1` | scalar broadcast to BF16 vector |
| `0x500` | 0 | `vldb	 x7, [p0], #0x40` | `x7` | `p0` | `x7->vec7` | load packed/Q4 or scale vector |
| `0x500` | 1 | `vmac.f	dm1, dm1, x1, x2, r4` | `dm1` | `dm1, x1, x2, r4` | `dm1->acc1, x1->vec1, x2->vec2` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x508` | 0 | `vunpack	x5, wl7, unpacksign0` | `x5` | `wl7, unpacksign0` | `x5->vec5, wl7->vec7` | unpack packed integer lanes into vector register |
| `0x50c` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x50e` | 0 | `vunpack	x9, wl8, unpacksign0` | `x9` | `wl8, unpacksign0` | `x9->vec9, wl8->vec8` | unpack packed integer lanes into vector register |
| `0x50e` | 1 | `vmac.f	dm1, dm1, x8, x6, r4` | `dm1` | `dm1, x8, x6, r4` | `dm1->acc1, x8->vec8, x6->vec6` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x516` | 0 | `vunpack	x8, wh8, unpacksign0` | `x8` | `wh8, unpacksign0` | `x8->vec8, wh8->vec8` | unpack packed integer lanes into vector register |
| `0x51a` | 0 | `vunpack	x10, wh7, unpacksign0` | `x10` | `wh7, unpacksign0` | `x10->vec10, wh7->vec7` | unpack packed integer lanes into vector register |
| `0x51e` | 0 | `vldb	 x0, [p0], #0x40` | `x0` | `p0` | `x0->vec0` | load packed/Q4 or scale vector |
| `0x51e` | 1 | `vmac.f	dm1, dm1, x5, x7, r4` | `dm1` | `dm1, x5, x7, r4` | `dm1->acc1, x5->vec5, x7->vec7` | FP MAC; first operand is written accumulator, second is source accumulator |
| `0x526` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
| `0x528` | 0 | `nop` | `` | `` | `` | bundle padding / hazard spacing |
