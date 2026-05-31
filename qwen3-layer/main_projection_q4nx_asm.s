	.section	.text.q4nx_accum_lane_asm_group_shape,"ax",@progbits
	.globl	q4nx_accum_lane_asm_group_shape
	.p2align	4
	.type	q4nx_accum_lane_asm_group_shape,@function
q4nx_accum_lane_asm_group_shape:
	// MyLM-style canonical middle Q4NX group shape for one 16-row lane.
	// This is a production-build assembly shape probe. It is intentionally
	// unreferenced by the numerical qwen3 decode path until the full lane body
	// is made bit-compatible with the current Q4NX reference.
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
	.size	q4nx_accum_lane_asm_group_shape, .-q4nx_accum_lane_asm_group_shape
