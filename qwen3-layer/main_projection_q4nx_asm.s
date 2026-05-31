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

	.macro Q4_EXACT4_BLOCK pack0, pack1, lane0, lane1, lane2, lane3
	nopa	;		vldb.128	 wh0, [p5, #\pack0];		nops	;		nopxm	;		nopv
	vunpack	x6, wh0, unpacksign0
	vunpack	y3, x6, unpacksign0;		mov	crunpacksize, #0
	movxm	r0, #0x4b01
	mov	crupsmode, #0
	mova	r1, #0;		vbcst.16	 x8, r0
	mov	s0, r1
	mov	crunpacksize, #1
	vconv.fp32.bf16	cml0, x8
	vups.2x	cml2, x6, s0, upssign0;		vadd	dm2, dm2, dm0, r1
	mova	r0, #0x3c
	vsub.f	dm2, dm2, dm0, r0
	vldb.128	 wh0, [p5, #\pack1]
	nop
	nop
	nop
	vunpack	x6, wh0, unpacksign0
	vunpack	y3, x6, unpacksign0;		vconv.bf16.fp32	 x8, cml2
	nop
	vmul.f	dm2, x8, x0, r0
	nop
	mov	crunpacksize, #0
	mov	crunpacksize, #1
	nop
	vups.2x	cml2, x6, s0, upssign0;		vadd	dm3, dm2, dm0, r1
	vconv.bf16.fp32	 wh0, bmll2;		vmov	wl6, wh8
	vsub.f	dm0, dm3, dm0, r0
	vmul.f	dm0, x6, x0, r0
	nop
	nop
	nop
	nop
	vconv.bf16.fp32	 x8, cml0
	vconv.bf16.fp32	 wh0, bmll0;		vconv.fp32.bf16	bmll4, wh0
	vmov	wl6, wh8;		vadd.f	dm4, dm4, dm2, r0
	vconv.fp32.bf16	bmll2, wl2;		vmul.f	dm0, x8, x0, r0
	vmul.f	dm4, x6, x0, r0
	nop
	nop
	vadd.f	dm3, dm0, dm2, r0
	vconv.bf16.fp32	 wl2, bmll4;		vconv.fp32.bf16	bmll0, wh0
	vconv.bf16.fp32	 wh2, bmll0
	vconv.bf16.fp32	 wh2, bmll4;		vextbcst.16	 x0, x4, #\lane0;		vadd.f	dm0, dm0, dm2, r0
	mova	r1, #0x33c;		vconv.fp32.bf16	bmll0, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl2, bmll3;		vextbcst.16	 x0, x4, #\lane1;		vadd.f	dm2, dm3, dm2, r0
	vconv.fp32.bf16	bmll3, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl0, bmll0;		vextbcst.16	 x2, x4, #\lane2
	nop
	vmac.f	dm0, dm1, x0, x2, r1
	vconv.bf16.fp32	 wl0, bmll2;		vextbcst.16	 x2, x4, #\lane3
	nop
	vmac.f	dm0, dm0, x0, x2, r1
	.endm

	.macro Q4_RELOAD_ARGS_PTR
	vldb	wl0, [p1, #0]
	vldb	wl2, [p2, #0]
	vldb	x4, [p3, #0]
	.endm

	.macro Q4_SET_PACK_BASE_REG base_reg
	movs	m0, \base_reg
	mov	p5, p0
	padda	[p5], m0
	.endm

	.macro Q4_HANDOFF
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmov	bmll1, bmll0
	.endm

	.section	.text.q4nx_accum_lane_exact_body_shape,"ax",@progbits
	.globl	q4nx_accum_lane_exact_body_shape
	.p2align	4
	.type	q4nx_accum_lane_exact_body_shape,@function
q4nx_accum_lane_exact_body_shape:
	// Callable exact Q4NX lane candidate for the production role object.
	// ABI: p0=packed lane data, p1=scale lane, p2=offset lane,
	//      p3=activation bf16[256], p4=dst float[16].
	// It returns through the normal C ABI and does not release any lock.
	mova	r1, #0
	mov	crrnd, #0xc
	vbcst.32	x8, r1
	vmov	bmll1, x8
	mova	r10, #0
	mova	r15, #8
	mova	r14, #0x40
.Lq4_exact_lane_loop_body:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 0, 1, 2, 3
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 4, 5, 6, 7
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 8, 9, 10, 11
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 12, 13, 14, 15
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 16, 17, 18, 19
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 20, 21, 22, 23
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 24, 25, 26, 27
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x020, 28, 29, 30, 31
	Q4_HANDOFF
	add	r10, r10, r14
	mova	r7, #0x20
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
	mova	r7, #0x40
	movs	m0, r7
	padda	[p3], m0
	add	r15, r15, #-1
	jnz	r15, #.Lq4_exact_lane_loop_body
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	ret	lr
	.size	q4nx_accum_lane_exact_body_shape, .-q4nx_accum_lane_exact_body_shape
