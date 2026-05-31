// Generated from MyLM disasm templates by exp107.
// This is a schedule artifact, not a standalone function.

.macro MYLM_Q4NX_FILL
	vlda	 x8, [p0], #0x40
	vldb	 x11, [p1], #0x40
	lshl	 r18, r19, r2
	add.nc	r16, r17, r1
	vlda	 x7, [p0], #0x40
	vunpack	x9, wl8, unpacksign0
	lshl	 r20, r19, r3
	add.nc	p5, r17, r18
	vlda	 x0, [p0], #0x40
	vunpack	x8, wh8, unpacksign0
	movx	r19, #0x10
	mov	dj0, r20
	vlda	 lfh0, [p2, dj0]
	vunpack	x5, wl7, unpacksign0
	add.nc	p4, r18, r16
	lda.s16	 r7, [p3], #0x2
	vunpack	x10, wh7, unpacksign0
	vunpack	x1, wl0, unpacksign0
	nop
	vextbcst.16	 x9, x11, #0x1
	vups.4x	dm2, x9, s0, upssign0
	vadd	dm1, dm2, dm0, r0
	vups.4x	dm2, x8, s0, upssign0
	vadd	dm2, dm2, dm0, r0
	vextbcst.16	 x7, x11, #0x0
	vsub.f	dm1, dm1, dm0, r5
	vsub.f	dm3, dm2, dm0, r5
	vmov	bmll2, bmll1
	vmov	bmlh2, bmlh1
	vmov	bmhl2, bmhl1
	vconv.bf16.fp32	 x5, cml2
	vmov	bmhh2, bmhh1
	vups.4x	dm1, x5, s0, upssign0
	vadd	dm4, dm1, dm0, r0
	vconv.bf16.fp32	 x3, cmh2
	vmov	wl6, wh5
	vmov	bmhl1, bmhl3
	vsub.f	dm2, dm4, dm0, r5
	vmov	wl2, wh3
	vmov	bmhh1, bmhh3
	vldb	 x6, [p0], #0x40
	vconv.bf16.fp32	 x8, cmh1
	vmov	bmll1, bmll3
	vups.4x	dm3, x10, s0, upssign0
	vadd	dm3, dm3, dm0, r0
	vextbcst.16	 x10, x11, #0x2
	vmul.f	dm3, x5, x7, r4
	vmov	bmlh1, bmlh3
	vsub.f	dm4, dm3, dm0, r5
	vextbcst.16	 x7, x11, #0x3
	vconv.bf16.fp32	 x4, cml1
	vmov	bmhh1, bmhh2
	vmac.f	dm3, dm3, x6, x9, r4
	vextbcst.16	 x9, x11, #0x8
	vmov	wl10, wh4
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x3, x10, r4
	vmov	bmll2, bmll4
	vunpack	x1, wh0, unpacksign0
	vextbcst.16	 x7, x11, #0x4
	vmov	bmlh1, bmlh2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmlh2, bmlh4
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x4, x11, #0x5
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x4, x7, r4
	vmov	bmhl2, bmhl4
	vconv.bf16.fp32	 x2, cmh1
	vups.4x	dm4, x1, s0, upssign0
	vadd	dm1, dm4, dm0, r0
	vextbcst.16	 x3, x11, #0x6
	vmov	bmhh2, bmhh4
	vsub.f	dm4, dm1, dm0, r5
	vextbcst.16	 x0, x11, #0x7
	vmac.f	dm4, dm3, x10, x4, r4
	vextbcst.16	 x4, x11, #0x9
	vconv.bf16.fp32	 x8, cml2
	vmov	wl3, wh8
	vmov.d	dm1, dm4
	vextbcst.16	 x7, x11, #0xa
	vmac.f	dm4, dm4, x8, x3, r4
	vextbcst.16	 x10, x11, #0xb
	vmov.d	dm3, dm1
	vmov	wl3, wh2
	vmac.f	dm4, dm4, x3, x0, r4
	nop
	vmov	wl5, wh5
	vmov	wl9, wh8
	vmac.f	dm4, dm4, x5, x9, r4
	vmov	bmhl1, bmhl3
	vmov	bmhh1, bmhh3
	vconv.bf16.fp32	 x1, cmh2
	vmov	bmll1, bmll3
	vmac.f	dm4, dm4, x5, x4, r4
	vconv.bf16.fp32	 x0, cmh1
	vups.4x	dm3, x1, s0, upssign0
	vadd	dm2, dm3, dm0, r0
	vextbcst.16	 x4, x11, #0xc
	vunpack	x4, wl6, unpacksign0
	vmov	bmlh1, bmlh3
	vmac.f	dm3, dm4, x2, x7, r4
	vunpack	x6, wh6, unpacksign0
	vextbcst.16	 x5, x11, #0xd
	vsub.f	dm2, dm2, dm0, r5
	vconv.bf16.fp32	 x2, cml1
	vextbcst.16	 x10, x11, #0xe
	vextbcst.16	 x3, x11, #0xf
	vmac.f	dm1, dm3, x3, x10, r4
	vextbcst.16	 x7, x11, #0x10
	vmov	wl8, wh1
	vextbcst.16	 x4, x11, #0x15
	vmac.f	dm3, dm1, x8, x4, r4
	vups.4x	dm4, x4, s0, upssign0
	vadd	dm4, dm4, dm0, r0
	vmov	wl5, wh2
	vextbcst.16	 x9, x11, #0x11
	vmac.f	dm3, dm3, x9, x5, r4
	vsub.f	dm4, dm4, dm0, r5
	vextbcst.16	 x1, x11, #0x12
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x1, x10, r4
	vextbcst.16	 x10, x11, #0x18
	vmov	wl3, wh0
	vextbcst.16	 x8, x11, #0x13
	vmac.f	dm3, dm3, x8, x3, r4
	vmov	bmlh1, bmlh2
	vextbcst.16	 x7, x11, #0x14
	vconv.bf16.fp32	 x2, cml1
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmhh1, bmhh2
	vups.4x	dm1, x6, s0, upssign0
	vadd	dm2, dm1, dm0, r0
	vconv.bf16.fp32	 x5, cmh1
	vextbcst.16	 x6, x11, #0x16
	vmac.f	dm3, dm3, x5, x9, r4
	vextbcst.16	 x9, x11, #0x17
	vsub.f	dm2, dm2, dm0, r5
	vextbcst.16	 x0, x11, #0x19
	vmov	bmll1, bmll4
	vmac.f	dm3, dm3, x0, x1, r4
	vmov	bmll4, lfh0
	vmov	bmhh1, bmhh4
	vmov	wl8, wh5
	vmac.f	dm3, dm3, x3, x8, r4
	vmov	bmlh1, bmlh4
	vmov	wl2, wh2
	vconv.bf16.fp32	 x3, cml1
	vmov	bmhl1, bmhl4
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmll1, bmll2
	vconv.bf16.fp32	 x1, cmh1
	vmov	bmlh1, bmlh2
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x4, r4
	vmov	bmhh1, bmhh2
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x2, x11, #0x1a
	vconv.bf16.fp32	 x4, cmh1
	vextbcst.16	 x6, x11, #0x1b
	vmac.f	dm2, dm3, x5, x6, r4
	vextbcst.16	 x7, x11, #0x1c
	vmov	wl8, wh1
	vmov	wl9, wh5
	vmac.f	dm1, dm2, x8, x9, r4
	nop
	lda.s16	 r7, [p3], #0x2
	vmov	wl3, wh3
	vldb	 wl2, [p5], #0x40
	vextbcst.16	 x10, x11, #0x1d
	vmac.f	dm1, dm1, x3, x10, r4
	nop
	vextbcst.16	 x0, x11, #0x1e
	vlda	 x8, [p0], #0x40
	vldb	 wl6, [p4], #0x40
	vextbcst.16	 x11, x11, #0x1f
	vmac.f	dm1, dm1, x3, x0, r4
	vmov	wl3, wh4
	vbcst.16	 x1, r7
	vldb	 x7, [p0], #0x40
	vmac.f	dm1, dm1, x1, x2, r4
	vunpack	x5, wl7, unpacksign0
	nop
	vunpack	x9, wl8, unpacksign0
	vmac.f	dm1, dm1, x8, x6, r4
	vunpack	x8, wh8, unpacksign0
	vunpack	x10, wh7, unpacksign0
	vldb	 x0, [p0], #0x40
	vmac.f	dm1, dm1, x5, x7, r4
	nop
	nop
.endm

.macro MYLM_Q4NX_STEADY
	vldb	 x11, [p1], #0x40
	vmac.f	dm1, dm1, x9, x10, r4
	vups.4x	dm2, x9, s0, upssign0
	vadd	dm1, dm2, dm0, r0
	vups.4x	dm2, x8, s0, upssign0
	vmac.f	dm3, dm1, x4, x0, r4
	vsub.f	dm1, dm1, dm0, r5
	vadd	dm2, dm2, dm0, r0
	vmac.f	dm3, dm3, x3, x11, r4
	vextbcst.16	 x7, x11, #0x0
	vsub.f	dm3, dm2, dm0, r5
	vextbcst.16	 x9, x11, #0x1
	nop
	nop
	vunpack	x1, wl0, unpacksign0
	vmov	bmll2, bmll1
	vconv.bf16.fp32	 wl4, bmll3
	vmov	bmlh2, bmlh1
	vmov	bmhl2, bmhl1
	vconv.bf16.fp32	 x5, cml2
	vups.4x	dm1, x5, s0, upssign0
	vmac.f	dm4, dm4, x4, x2, r4
	nop
	vmov	bmhh2, bmhh1
	vmov	wl6, wh5
	vmac.f	dm4, dm4, x6, x1, r4
	vconv.bf16.fp32	 x3, cmh2
	nop
	vmov	wl2, wh3
	vadd	dm4, dm1, dm0, r0
	nop
	vmov	lfh0, bmll4
	vsub.f	dm2, dm4, dm0, r5
	nop
	vmov	bmhl1, bmhl3
	nop
	vldb	 x6, [p0], #0x40
	vmov	bmhh1, bmhh3
	vmov	bmll1, bmll3
	vconv.bf16.fp32	 x8, cmh1
	vups.4x	dm3, x10, s0, upssign0
	vadd	dm3, dm3, dm0, r0
	vextbcst.16	 x10, x11, #0x2
	vmul.f	dm3, x5, x7, r4
	vmov	bmlh1, bmlh3
	vsub.f	dm4, dm3, dm0, r5
	vextbcst.16	 x7, x11, #0x3
	vconv.bf16.fp32	 x4, cml1
	vmov	bmhh1, bmhh2
	vmac.f	dm3, dm3, x6, x9, r4
	vextbcst.16	 x9, x11, #0x8
	vmov	wl10, wh4
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x3, x10, r4
	vmov	bmll2, bmll4
	vunpack	x1, wh0, unpacksign0
	vextbcst.16	 x7, x11, #0x4
	vmov	bmlh1, bmlh2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmlh2, bmlh4
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x4, x11, #0x5
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x4, x7, r4
	vmov	bmhl2, bmhl4
	vconv.bf16.fp32	 x2, cmh1
	vups.4x	dm4, x1, s0, upssign0
	vadd	dm1, dm4, dm0, r0
	vextbcst.16	 x3, x11, #0x6
	vmov	bmhh2, bmhh4
	vsub.f	dm4, dm1, dm0, r5
	vextbcst.16	 x0, x11, #0x7
	vmac.f	dm4, dm3, x10, x4, r4
	vextbcst.16	 x4, x11, #0x9
	vconv.bf16.fp32	 x8, cml2
	vmov	wl3, wh8
	vmov.d	dm1, dm4
	vextbcst.16	 x7, x11, #0xa
	vmac.f	dm4, dm4, x8, x3, r4
	vextbcst.16	 x10, x11, #0xb
	vmov.d	dm3, dm1
	vmov	wl3, wh2
	vmac.f	dm4, dm4, x3, x0, r4
	nop
	vmov	wl5, wh5
	vmov	wl9, wh8
	vmac.f	dm4, dm4, x5, x9, r4
	vmov	bmhl1, bmhl3
	vmov	bmhh1, bmhh3
	vconv.bf16.fp32	 x1, cmh2
	vmov	bmll1, bmll3
	vmac.f	dm4, dm4, x5, x4, r4
	vconv.bf16.fp32	 x0, cmh1
	vups.4x	dm3, x1, s0, upssign0
	vadd	dm2, dm3, dm0, r0
	vextbcst.16	 x4, x11, #0xc
	vunpack	x4, wl6, unpacksign0
	vmov	bmlh1, bmlh3
	vmac.f	dm3, dm4, x2, x7, r4
	vunpack	x6, wh6, unpacksign0
	vextbcst.16	 x5, x11, #0xd
	vsub.f	dm2, dm2, dm0, r5
	vconv.bf16.fp32	 x2, cml1
	vextbcst.16	 x10, x11, #0xe
	vextbcst.16	 x3, x11, #0xf
	vmac.f	dm1, dm3, x3, x10, r4
	vextbcst.16	 x7, x11, #0x10
	vmov	wl8, wh1
	vextbcst.16	 x4, x11, #0x15
	vmac.f	dm3, dm1, x8, x4, r4
	vups.4x	dm4, x4, s0, upssign0
	vadd	dm4, dm4, dm0, r0
	vmov	wl5, wh2
	vextbcst.16	 x9, x11, #0x11
	vmac.f	dm3, dm3, x9, x5, r4
	vsub.f	dm4, dm4, dm0, r5
	vextbcst.16	 x1, x11, #0x12
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x1, x10, r4
	vextbcst.16	 x10, x11, #0x18
	vmov	wl3, wh0
	vextbcst.16	 x8, x11, #0x13
	vmac.f	dm3, dm3, x8, x3, r4
	vmov	bmlh1, bmlh2
	vextbcst.16	 x7, x11, #0x14
	vconv.bf16.fp32	 x2, cml1
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmhh1, bmhh2
	vups.4x	dm1, x6, s0, upssign0
	vadd	dm2, dm1, dm0, r0
	vconv.bf16.fp32	 x5, cmh1
	vextbcst.16	 x6, x11, #0x16
	vmac.f	dm3, dm3, x5, x9, r4
	vextbcst.16	 x9, x11, #0x17
	vsub.f	dm2, dm2, dm0, r5
	vextbcst.16	 x0, x11, #0x19
	vmov	bmll1, bmll4
	vmac.f	dm3, dm3, x0, x1, r4
	nop
	vmov	wl8, wh5
	vmov	bmlh1, bmlh4
	vmac.f	dm3, dm3, x3, x8, r4
	vmov	bmhh1, bmhh4
	vconv.bf16.fp32	 x3, cml1
	vmov	wl2, wh2
	vmov	bmhl1, bmhl4
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmll1, bmll2
	vconv.bf16.fp32	 x1, cmh1
	vmov	bmlh1, bmlh2
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x4, r4
	vmov	bmhh1, bmhh2
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x2, x11, #0x1a
	vconv.bf16.fp32	 x4, cmh1
	vextbcst.16	 x6, x11, #0x1b
	vmac.f	dm2, dm3, x5, x6, r4
	vextbcst.16	 x7, x11, #0x1c
	vmov	wl8, wh1
	vmov	wl9, wh5
	vmac.f	dm1, dm2, x8, x9, r4
	vmov	bmll4, lfh0
	lda.s16	 r7, [p3], #0x2
	vmov	wl3, wh3
	vldb	 wl2, [p5], #0x40
	vextbcst.16	 x10, x11, #0x1d
	vmac.f	dm1, dm1, x3, x10, r4
	nop
	vextbcst.16	 x0, x11, #0x1e
	vlda	 x8, [p0], #0x40
	vldb	 wl6, [p4], #0x40
	vextbcst.16	 x11, x11, #0x1f
	vmac.f	dm1, dm1, x3, x0, r4
	vmov	wl3, wh4
	vbcst.16	 x1, r7
	vldb	 x7, [p0], #0x40
	vmac.f	dm1, dm1, x1, x2, r4
	vunpack	x5, wl7, unpacksign0
	nop
	vunpack	x9, wl8, unpacksign0
	vmac.f	dm1, dm1, x8, x6, r4
	vunpack	x8, wh8, unpacksign0
	vunpack	x10, wh7, unpacksign0
	vldb	 x0, [p0], #0x40
	vmac.f	dm1, dm1, x5, x7, r4
	nop
	nop
.endm

.macro MYLM_Q4NX_PRE_DRAIN
	vldb	 x11, [p1], #0x40
	vmac.f	dm1, dm1, x9, x10, r4
	vups.4x	dm2, x9, s0, upssign0
	vadd	dm1, dm2, dm0, r0
	vups.4x	dm2, x8, s0, upssign0
	vmac.f	dm3, dm1, x4, x0, r4
	vsub.f	dm1, dm1, dm0, r5
	vadd	dm2, dm2, dm0, r0
	vmac.f	dm3, dm3, x3, x11, r4
	vextbcst.16	 x7, x11, #0x0
	vsub.f	dm3, dm2, dm0, r5
	vextbcst.16	 x9, x11, #0x1
	nop
	nop
	vunpack	x1, wl0, unpacksign0
	vmov	bmll2, bmll1
	vconv.bf16.fp32	 wl4, bmll3
	vmov	bmlh2, bmlh1
	vmov	bmhl2, bmhl1
	vconv.bf16.fp32	 x5, cml2
	vups.4x	dm1, x5, s0, upssign0
	vmac.f	dm4, dm4, x4, x2, r4
	nop
	vmov	bmhh2, bmhh1
	vmov	wl6, wh5
	vmac.f	dm4, dm4, x6, x1, r4
	vconv.bf16.fp32	 x3, cmh2
	nop
	vmov	wl2, wh3
	vadd	dm4, dm1, dm0, r0
	nop
	vmov	lfh0, bmll4
	vsub.f	dm2, dm4, dm0, r5
	nop
	vmov	bmhl1, bmhl3
	nop
	vldb	 x6, [p0], #0x40
	vmov	bmhh1, bmhh3
	vmov	bmll1, bmll3
	vconv.bf16.fp32	 x8, cmh1
	vups.4x	dm3, x10, s0, upssign0
	vadd	dm3, dm3, dm0, r0
	vextbcst.16	 x10, x11, #0x2
	vmul.f	dm3, x5, x7, r4
	vmov	bmlh1, bmlh3
	vsub.f	dm4, dm3, dm0, r5
	vextbcst.16	 x7, x11, #0x3
	vconv.bf16.fp32	 x4, cml1
	vmov	bmhh1, bmhh2
	vmac.f	dm3, dm3, x6, x9, r4
	vextbcst.16	 x9, x11, #0x8
	vmov	wl10, wh4
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x3, x10, r4
	vmov	bmll2, bmll4
	vunpack	x1, wh0, unpacksign0
	vextbcst.16	 x7, x11, #0x4
	vmov	bmlh1, bmlh2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmlh2, bmlh4
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x4, x11, #0x5
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x4, x7, r4
	vmov	bmhl2, bmhl4
	vconv.bf16.fp32	 x2, cmh1
	vups.4x	dm4, x1, s0, upssign0
	vadd	dm1, dm4, dm0, r0
	vextbcst.16	 x3, x11, #0x6
	vmov	bmhh2, bmhh4
	vsub.f	dm4, dm1, dm0, r5
	vextbcst.16	 x0, x11, #0x7
	vmac.f	dm4, dm3, x10, x4, r4
	vextbcst.16	 x4, x11, #0x9
	vconv.bf16.fp32	 x8, cml2
	vmov	wl3, wh8
	vmov.d	dm1, dm4
	vextbcst.16	 x7, x11, #0xa
	vmac.f	dm4, dm4, x8, x3, r4
	vextbcst.16	 x10, x11, #0xb
	vmov.d	dm3, dm1
	vmov	wl3, wh2
	vmac.f	dm4, dm4, x3, x0, r4
	nop
	vmov	wl5, wh5
	vmov	wl9, wh8
	vmac.f	dm4, dm4, x5, x9, r4
	vmov	bmhl1, bmhl3
	vmov	bmhh1, bmhh3
	vconv.bf16.fp32	 x1, cmh2
	vmov	bmll1, bmll3
	vmac.f	dm4, dm4, x5, x4, r4
	vconv.bf16.fp32	 x0, cmh1
	vups.4x	dm3, x1, s0, upssign0
	vadd	dm2, dm3, dm0, r0
	vextbcst.16	 x4, x11, #0xc
	vunpack	x4, wl6, unpacksign0
	vmov	bmlh1, bmlh3
	vmac.f	dm3, dm4, x2, x7, r4
	vunpack	x6, wh6, unpacksign0
	vextbcst.16	 x5, x11, #0xd
	vsub.f	dm2, dm2, dm0, r5
	vconv.bf16.fp32	 x2, cml1
	vextbcst.16	 x10, x11, #0xe
	vextbcst.16	 x3, x11, #0xf
	vmac.f	dm1, dm3, x3, x10, r4
	vextbcst.16	 x7, x11, #0x10
	vmov	wl8, wh1
	vextbcst.16	 x4, x11, #0x15
	vmac.f	dm3, dm1, x8, x4, r4
	vups.4x	dm4, x4, s0, upssign0
	vadd	dm4, dm4, dm0, r0
	vmov	wl5, wh2
	vextbcst.16	 x9, x11, #0x11
	vmac.f	dm3, dm3, x9, x5, r4
	vsub.f	dm4, dm4, dm0, r5
	vextbcst.16	 x1, x11, #0x12
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x1, x10, r4
	vextbcst.16	 x10, x11, #0x18
	vmov	wl3, wh0
	vextbcst.16	 x8, x11, #0x13
	vmac.f	dm3, dm3, x8, x3, r4
	vmov	bmlh1, bmlh2
	vextbcst.16	 x7, x11, #0x14
	vconv.bf16.fp32	 x2, cml1
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmhh1, bmhh2
	vups.4x	dm1, x6, s0, upssign0
	vadd	dm2, dm1, dm0, r0
	vconv.bf16.fp32	 x5, cmh1
	vextbcst.16	 x6, x11, #0x16
	vmac.f	dm3, dm3, x5, x9, r4
	vextbcst.16	 x9, x11, #0x17
	vsub.f	dm2, dm2, dm0, r5
	vextbcst.16	 x0, x11, #0x19
	vmov	bmll1, bmll4
	vmac.f	dm3, dm3, x0, x1, r4
	nop
	vmov	wl8, wh5
	vmov	bmlh1, bmlh4
	vmac.f	dm3, dm3, x3, x8, r4
	vmov	bmhh1, bmhh4
	vconv.bf16.fp32	 x3, cml1
	vmov	wl2, wh2
	vmov	bmhl1, bmhl4
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmll1, bmll2
	vconv.bf16.fp32	 x1, cmh1
	vmov	bmlh1, bmlh2
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x4, r4
	vmov	bmhh1, bmhh2
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x2, x11, #0x1a
	vconv.bf16.fp32	 x4, cmh1
	vextbcst.16	 x6, x11, #0x1b
	vmac.f	dm2, dm3, x5, x6, r4
	vextbcst.16	 x7, x11, #0x1c
	vmov	wl8, wh1
	vmov	wl9, wh5
	vmac.f	dm1, dm2, x8, x9, r4
	vmov	bmll4, lfh0
	lda.s16	 r7, [p3], #0x2
	vmov	wl3, wh3
	vldb	 wl2, [p5], #0x40
	vextbcst.16	 x10, x11, #0x1d
	vmac.f	dm1, dm1, x3, x10, r4
	mov	r21, p3
	vextbcst.16	 x0, x11, #0x1e
	vlda	 x8, [p0], #0x40
	vldb	 wl6, [p4], #0x40
	vextbcst.16	 x11, x11, #0x1f
	vmac.f	dm1, dm1, x3, x0, r4
	vmov	wl3, wh4
	vbcst.16	 x1, r7
	vldb	 x7, [p0], #0x40
	add.nc	p3, r21, #-0x10
	vmac.f	dm1, dm1, x1, x2, r4
	vunpack	x5, wl7, unpacksign0
	nop
	vunpack	x9, wl8, unpacksign0
	vmac.f	dm1, dm1, x8, x6, r4
	vunpack	x8, wh8, unpacksign0
	vunpack	x10, wh7, unpacksign0
	vldb	 x0, [p0], #0x40
	vmac.f	dm1, dm1, x5, x7, r4
	nop
	nop
.endm

.macro MYLM_Q4NX_DRAIN
	vldb	 x11, [p1], #0x40
	vmac.f	dm1, dm1, x9, x10, r4
	paddb	 [p1], #-0x200
	vups.4x	dm2, x9, s0, upssign0
	vadd	dm1, dm2, dm0, r0
	vups.4x	dm2, x8, s0, upssign0
	vmac.f	dm3, dm1, x4, x0, r4
	vsub.f	dm1, dm1, dm0, r5
	vadd	dm2, dm2, dm0, r0
	vmac.f	dm3, dm3, x3, x11, r4
	vextbcst.16	 x7, x11, #0x0
	vsub.f	dm3, dm2, dm0, r5
	vextbcst.16	 x9, x11, #0x1
	nop
	nop
	vunpack	x1, wl0, unpacksign0
	vmov	bmll2, bmll1
	vconv.bf16.fp32	 wl4, bmll3
	vmov	bmlh2, bmlh1
	vmov	bmhl2, bmhl1
	vconv.bf16.fp32	 x5, cml2
	vups.4x	dm1, x5, s0, upssign0
	vmac.f	dm4, dm4, x4, x2, r4
	nop
	vmov	bmhh2, bmhh1
	vmov	wl6, wh5
	vmac.f	dm4, dm4, x6, x1, r4
	vconv.bf16.fp32	 x3, cmh2
	nop
	vmov	wl2, wh3
	vadd	dm4, dm1, dm0, r0
	nop
	vmov	lfh0, bmll4
	vsub.f	dm2, dm4, dm0, r5
	nop
	vmov	bmhl1, bmhl3
	nop
	vldb	 x6, [p0], #0x40
	vmov	bmhh1, bmhh3
	vmov	bmll1, bmll3
	vconv.bf16.fp32	 x8, cmh1
	vups.4x	dm3, x10, s0, upssign0
	vadd	dm3, dm3, dm0, r0
	vextbcst.16	 x10, x11, #0x2
	vmul.f	dm3, x5, x7, r4
	vmov	bmlh1, bmlh3
	vsub.f	dm4, dm3, dm0, r5
	vextbcst.16	 x7, x11, #0x3
	vconv.bf16.fp32	 x4, cml1
	vmov	bmhh1, bmhh2
	vmac.f	dm3, dm3, x6, x9, r4
	vextbcst.16	 x9, x11, #0x8
	vmov	wl10, wh4
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x3, x10, r4
	vmov	bmll2, bmll4
	vunpack	x1, wh0, unpacksign0
	vextbcst.16	 x7, x11, #0x4
	vmov	bmlh1, bmlh2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmlh2, bmlh4
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x4, x11, #0x5
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x4, x7, r4
	vmov	bmhl2, bmhl4
	vconv.bf16.fp32	 x2, cmh1
	vups.4x	dm4, x1, s0, upssign0
	vadd	dm1, dm4, dm0, r0
	vextbcst.16	 x3, x11, #0x6
	vmov	bmhh2, bmhh4
	vsub.f	dm4, dm1, dm0, r5
	vextbcst.16	 x0, x11, #0x7
	vmac.f	dm4, dm3, x10, x4, r4
	vextbcst.16	 x4, x11, #0x9
	vconv.bf16.fp32	 x8, cml2
	vmov	wl3, wh8
	vmov.d	dm1, dm4
	vextbcst.16	 x7, x11, #0xa
	vmac.f	dm4, dm4, x8, x3, r4
	vextbcst.16	 x10, x11, #0xb
	vmov.d	dm3, dm1
	vmov	wl3, wh2
	vmac.f	dm4, dm4, x3, x0, r4
	nop
	vmov	wl5, wh5
	vmov	wl9, wh8
	vmac.f	dm4, dm4, x5, x9, r4
	vmov	bmhl1, bmhl3
	vmov	bmhh1, bmhh3
	vconv.bf16.fp32	 x1, cmh2
	vmov	bmll1, bmll3
	vmac.f	dm4, dm4, x5, x4, r4
	vconv.bf16.fp32	 x0, cmh1
	vups.4x	dm3, x1, s0, upssign0
	vadd	dm2, dm3, dm0, r0
	vextbcst.16	 x4, x11, #0xc
	vunpack	x4, wl6, unpacksign0
	vmov	bmlh1, bmlh3
	vmac.f	dm3, dm4, x2, x7, r4
	vunpack	x6, wh6, unpacksign0
	vextbcst.16	 x5, x11, #0xd
	vsub.f	dm2, dm2, dm0, r5
	vconv.bf16.fp32	 x2, cml1
	vextbcst.16	 x10, x11, #0xe
	vextbcst.16	 x3, x11, #0xf
	vmac.f	dm1, dm3, x3, x10, r4
	vextbcst.16	 x7, x11, #0x10
	vmov	wl8, wh1
	vextbcst.16	 x4, x11, #0x15
	vmac.f	dm3, dm1, x8, x4, r4
	vups.4x	dm4, x4, s0, upssign0
	vadd	dm4, dm4, dm0, r0
	vmov	wl5, wh2
	vextbcst.16	 x9, x11, #0x11
	vmac.f	dm3, dm3, x9, x5, r4
	vsub.f	dm4, dm4, dm0, r5
	vextbcst.16	 x1, x11, #0x12
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x1, x10, r4
	vextbcst.16	 x10, x11, #0x18
	vmov	wl3, wh0
	vextbcst.16	 x8, x11, #0x13
	vmac.f	dm3, dm3, x8, x3, r4
	vmov	bmlh1, bmlh2
	vextbcst.16	 x7, x11, #0x14
	vconv.bf16.fp32	 x2, cml1
	vmov	bmhl1, bmhl2
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmhh1, bmhh2
	vups.4x	dm1, x6, s0, upssign0
	vadd	dm2, dm1, dm0, r0
	vconv.bf16.fp32	 x5, cmh1
	vextbcst.16	 x6, x11, #0x16
	vmac.f	dm3, dm3, x5, x9, r4
	vextbcst.16	 x9, x11, #0x17
	vsub.f	dm2, dm2, dm0, r5
	vextbcst.16	 x0, x11, #0x19
	vmov	bmll1, bmll4
	vmac.f	dm3, dm3, x0, x1, r4
	vmov	bmll4, lfh0
	vmov	bmhh1, bmhh4
	vmov	wl8, wh5
	vmac.f	dm3, dm3, x3, x8, r4
	vmov	bmlh1, bmlh4
	vmov	wl2, wh2
	vconv.bf16.fp32	 x3, cml1
	vmov	bmhl1, bmhl4
	vmac.f	dm3, dm3, x2, x7, r4
	vmov	bmhl1, bmhl2
	vconv.bf16.fp32	 x1, cmh1
	vmov	bmhh1, bmhh2
	vmov	bmll1, bmll2
	vmac.f	dm3, dm3, x2, x4, r4
	vconv.bf16.fp32	 x4, cmh1
	vmov	bmlh1, bmlh2
	vextbcst.16	 x2, x11, #0x1a
	vconv.bf16.fp32	 x5, cml1
	vextbcst.16	 x6, x11, #0x1b
	vmac.f	dm2, dm3, x5, x6, r4
	vextbcst.16	 x7, x11, #0x1c
	vmov	wl8, wh1
	vmov	wl9, wh5
	vmac.f	dm1, dm2, x8, x9, r4
	nop
	vmov	wl3, wh3
	vldb	 wl2, [p5], #0x40
	vextbcst.16	 x10, x11, #0x1d
	vmac.f	dm1, dm1, x3, x10, r4
	mov	p5, r16
	paddb	 [p5], #-0x200
	vextbcst.16	 x0, x11, #0x1e
	vldb	 wl6, [p4], #0x40
	vextbcst.16	 x11, x11, #0x1f
	vmac.f	dm1, dm1, x3, x0, r4
	vmov	wl3, wh4
	vbcst.16	 x1, r7
	mov	r17, p5
	vmac.f	dm1, dm1, x1, x2, r4
	nop
	nop
	vmac.f	dm1, dm1, x8, x6, r4
	nop
	nop
	vmac.f	dm1, dm1, x5, x7, r4
	nop
	nop
	vmac.f	dm1, dm1, x9, x10, r4
	nop
	nop
	vmac.f	dm3, dm1, x4, x0, r4
	nop
	nop
	vmac.f	dm3, dm3, x3, x11, r4
	nop
	vconv.bf16.fp32	 wl4, bmll3
	nop
	vmac.f	dm4, dm4, x4, x2, r4
	nop
	nop
	vmac.f	dm4, dm4, x6, x1, r4
	nop
	vmov	lfh0, bmll4
	nopa
	nopb
.endm

// Replay shape:
//   MYLM_Q4NX_FILL
//   MYLM_Q4NX_STEADY x5
//   MYLM_Q4NX_PRE_DRAIN
//   MYLM_Q4NX_DRAIN
