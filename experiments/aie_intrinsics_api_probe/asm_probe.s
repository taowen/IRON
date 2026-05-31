	.text
	.globl	probe_asm_q4_group_shape
	.p2align	4
	.type	probe_asm_q4_group_shape,@function
probe_asm_q4_group_shape:
	// One Q4NX quant group for one 16-row lane:
	// 32 activation-lane broadcasts, 32 main MACs, and one offset correction MAC.
	// Registers are intentionally fixed here. This probe validates source-level
	// assembly control of the AIE2P hot-loop shape; it is not a callable C ABI
	// numerical kernel yet.
	vextbcst.16	 x0, x11, #0
	vmac.f	 dm1, dm1, x2, x0, r4
	vextbcst.16	 x1, x11, #1
	vmac.f	 dm1, dm1, x3, x1, r4
	vextbcst.16	 x4, x11, #2
	vmac.f	 dm1, dm1, x5, x4, r4
	vextbcst.16	 x6, x11, #3
	vmac.f	 dm1, dm1, x7, x6, r4
	vextbcst.16	 x8, x11, #4
	vmac.f	 dm1, dm1, x9, x8, r4
	vextbcst.16	 x10, x11, #5
	vmac.f	 dm1, dm1, x2, x10, r4
	vextbcst.16	 x0, x11, #6
	vmac.f	 dm1, dm1, x3, x0, r4
	vextbcst.16	 x1, x11, #7
	vmac.f	 dm1, dm1, x5, x1, r4
	vextbcst.16	 x4, x11, #8
	vmac.f	 dm1, dm1, x7, x4, r4
	vextbcst.16	 x6, x11, #9
	vmac.f	 dm1, dm1, x9, x6, r4
	vextbcst.16	 x8, x11, #10
	vmac.f	 dm1, dm1, x2, x8, r4
	vextbcst.16	 x10, x11, #11
	vmac.f	 dm1, dm1, x3, x10, r4
	vextbcst.16	 x0, x11, #12
	vmac.f	 dm1, dm1, x5, x0, r4
	vextbcst.16	 x1, x11, #13
	vmac.f	 dm1, dm1, x7, x1, r4
	vextbcst.16	 x4, x11, #14
	vmac.f	 dm1, dm1, x9, x4, r4
	vextbcst.16	 x6, x11, #15
	vmac.f	 dm1, dm1, x2, x6, r4
	vextbcst.16	 x8, x11, #16
	vmac.f	 dm1, dm1, x3, x8, r4
	vextbcst.16	 x10, x11, #17
	vmac.f	 dm1, dm1, x5, x10, r4
	vextbcst.16	 x0, x11, #18
	vmac.f	 dm1, dm1, x7, x0, r4
	vextbcst.16	 x1, x11, #19
	vmac.f	 dm1, dm1, x9, x1, r4
	vextbcst.16	 x4, x11, #20
	vmac.f	 dm1, dm1, x2, x4, r4
	vextbcst.16	 x6, x11, #21
	vmac.f	 dm1, dm1, x3, x6, r4
	vextbcst.16	 x8, x11, #22
	vmac.f	 dm1, dm1, x5, x8, r4
	vextbcst.16	 x10, x11, #23
	vmac.f	 dm1, dm1, x7, x10, r4
	vextbcst.16	 x0, x11, #24
	vmac.f	 dm1, dm1, x9, x0, r4
	vextbcst.16	 x1, x11, #25
	vmac.f	 dm1, dm1, x2, x1, r4
	vextbcst.16	 x4, x11, #26
	vmac.f	 dm1, dm1, x3, x4, r4
	vextbcst.16	 x6, x11, #27
	vmac.f	 dm1, dm1, x5, x6, r4
	vextbcst.16	 x8, x11, #28
	vmac.f	 dm1, dm1, x7, x8, r4
	vextbcst.16	 x10, x11, #29
	vmac.f	 dm1, dm1, x9, x10, r4
	vextbcst.16	 x0, x11, #30
	vmac.f	 dm1, dm1, x2, x0, r4
	vextbcst.16	 x1, x11, #31
	vmac.f	 dm1, dm1, x3, x1, r4
	vbcst.16	 x4, r7
	vmac.f	 dm1, dm1, x5, x4, r4
	ret	lr
	.size	probe_asm_q4_group_shape, .-probe_asm_q4_group_shape

	.globl	probe_asm_q4_group_with_prep_shape
	.p2align	4
	.type	probe_asm_q4_group_with_prep_shape,@function
probe_asm_q4_group_with_prep_shape:
	// Canonical middle-group Q4NX shape derived from MyLM:
	// 8 unpack stages, 8 ups stages, 32 lane broadcasts, 32 main MACs, and
	// one offset/group-sum correction MAC. This still validates instruction
	// ownership, not C ABI or numerical data placement.
	vldb	 x11, [p1], #0x40
	vldb	 wl2, [p5], #0x40
	vldb	 wl6, [p4], #0x40
	vlda	 x8, [p0], #0x40
	vunpack	 x1, wl0, unpacksign0
	vunpack	 x5, wl7, unpacksign0
	vunpack	 x9, wl8, unpacksign0
	vunpack	 x8, wh8, unpacksign0
	vunpack	 x10, wh7, unpacksign0
	vunpack	 x4, wl6, unpacksign0
	vunpack	 x6, wh6, unpacksign0
	vunpack	 x0, wh8, unpacksign0
	vups.4x	 dm2, x9, s0, upssign0
	vups.4x	 dm2, x8, s0, upssign0
	vups.4x	 dm1, x5, s0, upssign0
	vups.4x	 dm3, x10, s0, upssign0
	vups.4x	 dm4, x1, s0, upssign0
	vups.4x	 dm4, x4, s0, upssign0
	vups.4x	 dm1, x6, s0, upssign0
	vups.4x	 dm3, x1, s0, upssign0
	lda.s16	 r7, [p3], #0x2
	vextbcst.16	 x0, x11, #0
	vmac.f	 dm1, dm1, x2, x0, r4
	vextbcst.16	 x1, x11, #1
	vmac.f	 dm1, dm1, x3, x1, r4
	vextbcst.16	 x4, x11, #2
	vmac.f	 dm1, dm1, x5, x4, r4
	vextbcst.16	 x6, x11, #3
	vmac.f	 dm1, dm1, x7, x6, r4
	vextbcst.16	 x8, x11, #4
	vmac.f	 dm1, dm1, x9, x8, r4
	vextbcst.16	 x10, x11, #5
	vmac.f	 dm1, dm1, x2, x10, r4
	vextbcst.16	 x0, x11, #6
	vmac.f	 dm1, dm1, x3, x0, r4
	vextbcst.16	 x1, x11, #7
	vmac.f	 dm1, dm1, x5, x1, r4
	vextbcst.16	 x4, x11, #8
	vmac.f	 dm1, dm1, x7, x4, r4
	vextbcst.16	 x6, x11, #9
	vmac.f	 dm1, dm1, x9, x6, r4
	vextbcst.16	 x8, x11, #10
	vmac.f	 dm1, dm1, x2, x8, r4
	vextbcst.16	 x10, x11, #11
	vmac.f	 dm1, dm1, x3, x10, r4
	vextbcst.16	 x0, x11, #12
	vmac.f	 dm1, dm1, x5, x0, r4
	vextbcst.16	 x1, x11, #13
	vmac.f	 dm1, dm1, x7, x1, r4
	vextbcst.16	 x4, x11, #14
	vmac.f	 dm1, dm1, x9, x4, r4
	vextbcst.16	 x6, x11, #15
	vmac.f	 dm1, dm1, x2, x6, r4
	vextbcst.16	 x8, x11, #16
	vmac.f	 dm1, dm1, x3, x8, r4
	vextbcst.16	 x10, x11, #17
	vmac.f	 dm1, dm1, x5, x10, r4
	vextbcst.16	 x0, x11, #18
	vmac.f	 dm1, dm1, x7, x0, r4
	vextbcst.16	 x1, x11, #19
	vmac.f	 dm1, dm1, x9, x1, r4
	vextbcst.16	 x4, x11, #20
	vmac.f	 dm1, dm1, x2, x4, r4
	vextbcst.16	 x6, x11, #21
	vmac.f	 dm1, dm1, x3, x6, r4
	vextbcst.16	 x8, x11, #22
	vmac.f	 dm1, dm1, x5, x8, r4
	vextbcst.16	 x10, x11, #23
	vmac.f	 dm1, dm1, x7, x10, r4
	vextbcst.16	 x0, x11, #24
	vmac.f	 dm1, dm1, x9, x0, r4
	vextbcst.16	 x1, x11, #25
	vmac.f	 dm1, dm1, x2, x1, r4
	vextbcst.16	 x4, x11, #26
	vmac.f	 dm1, dm1, x3, x4, r4
	vextbcst.16	 x6, x11, #27
	vmac.f	 dm1, dm1, x5, x6, r4
	vextbcst.16	 x8, x11, #28
	vmac.f	 dm1, dm1, x7, x8, r4
	vextbcst.16	 x10, x11, #29
	vmac.f	 dm1, dm1, x9, x10, r4
	vextbcst.16	 x0, x11, #30
	vmac.f	 dm1, dm1, x2, x0, r4
	vextbcst.16	 x1, x11, #31
	vmac.f	 dm1, dm1, x3, x1, r4
	vbcst.16	 x4, r7
	vmac.f	 dm1, dm1, x5, x4, r4
	ret	lr
	.size	probe_asm_q4_group_with_prep_shape, .-probe_asm_q4_group_with_prep_shape
