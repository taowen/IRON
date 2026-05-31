	.text
	.globl	probe_asm_store_word
	.p2align	4
	.type	probe_asm_store_word,@function
probe_asm_store_word:
	mova	r1, #123
	st	r1, [p0, #0]
	ret	lr
	.size	probe_asm_store_word, .-probe_asm_store_word

	.globl	probe_asm_vector_mac_smoke
	.p2align	4
	.type	probe_asm_vector_mac_smoke,@function
probe_asm_vector_mac_smoke:
	// C ABI smoke test for a future callable Q4NX lane body:
	// p0=lhs pointer, p1=rhs pointer, p2=dst pointer.
	// The store intentionally writes lhs back; this probe proves the role object
	// can carry a vector-load/MAC/vector-store assembly function called by C++.
	vlda	x0, [p0, #0]
	vlda	x1, [p1, #0]
	mova	r4, #828
	vmac.f	dm1, dm1, x0, x1, r4
	vst	x0, [p2, #0]
	ret	lr
	.size	probe_asm_vector_mac_smoke, .-probe_asm_vector_mac_smoke

	.globl	probe_asm_vector_mac_local_asm
	.p2align	4
	.type	probe_asm_vector_mac_local_asm,@function
probe_asm_vector_mac_local_asm:
	// Direct MLIR/core-callable source assembly body:
	// p0=lhs local buffer, p1=rhs local buffer, p2=dst local buffer.
	movs	p3, p0
	movs	p4, p1
	movxm	r1, #4096
	st	r1, [p3], #4
	movxm	r1, #4097
	st	r1, [p3], #4
	movxm	r1, #4098
	st	r1, [p3], #4
	movxm	r1, #4099
	st	r1, [p3], #4
	movxm	r1, #4100
	st	r1, [p3], #4
	movxm	r1, #4101
	st	r1, [p3], #4
	movxm	r1, #4102
	st	r1, [p3], #4
	movxm	r1, #4103
	st	r1, [p3], #4
	movxm	r1, #4104
	st	r1, [p3], #4
	movxm	r1, #4105
	st	r1, [p3], #4
	movxm	r1, #4106
	st	r1, [p3], #4
	movxm	r1, #4107
	st	r1, [p3], #4
	movxm	r1, #4108
	st	r1, [p3], #4
	movxm	r1, #4109
	st	r1, [p3], #4
	movxm	r1, #4110
	st	r1, [p3], #4
	movxm	r1, #4111
	st	r1, [p3], #4
	movxm	r1, #8192
	st	r1, [p4], #4
	movxm	r1, #8193
	st	r1, [p4], #4
	movxm	r1, #8194
	st	r1, [p4], #4
	movxm	r1, #8195
	st	r1, [p4], #4
	movxm	r1, #8196
	st	r1, [p4], #4
	movxm	r1, #8197
	st	r1, [p4], #4
	movxm	r1, #8198
	st	r1, [p4], #4
	movxm	r1, #8199
	st	r1, [p4], #4
	movxm	r1, #8200
	st	r1, [p4], #4
	movxm	r1, #8201
	st	r1, [p4], #4
	movxm	r1, #8202
	st	r1, [p4], #4
	movxm	r1, #8203
	st	r1, [p4], #4
	movxm	r1, #8204
	st	r1, [p4], #4
	movxm	r1, #8205
	st	r1, [p4], #4
	movxm	r1, #8206
	st	r1, [p4], #4
	movxm	r1, #8207
	st	r1, [p4], #4
	movxm	r1, #4096
	vbcst.32	x0, r1
	mova	r4, #828
	vmac.f	dm1, dm1, x0, x0, r4
	vst	wl0, [p2, #0]
	vst	wh0, [p2, #0x20]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_vector_mac_local_asm, .-probe_asm_vector_mac_local_asm

	.globl	probe_asm_vector_copy_release
	.p2align	4
	.type	probe_asm_vector_copy_release,@function
probe_asm_vector_copy_release:
	// p0=src local buffer filled by tile DMA, p1=dst local buffer.
	vlda	x0, [p0, #0]
	// Match the load-use spacing Peano emits before consuming loaded vectors.
	// A naive vlda/vst pair can observe stale register contents on NPU.
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
	mova	r4, #828
	vmac.f	dm1, dm1, x0, x0, r4
	vst	wl0, [p1, #0]
	vst	wh0, [p1, #0x20]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_vector_copy_release, .-probe_asm_vector_copy_release

	.globl	probe_asm_bf16_mac_result_release
	.p2align	4
	.type	probe_asm_bf16_mac_result_release,@function
probe_asm_bf16_mac_result_release:
	// p0=source BF16 vector, p1=destination BF16 vector.
	// Compute dst[0:16] = src[0:16] * 1.0 using the native BF16 MAC path,
	// then store the accumulator as BF16. This is the smallest numerical
	// source-assembly boundary needed before writing a Q4NX exact lane body.
	mova	r0, #0
	vbcst.32	x4, r0
	vmov	bmll0, x4
	vlda	x0, [p0, #0]
	// Match the load-use spacing Peano emits before consuming loaded vectors.
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
	movxm	r1, #0x3f80
	vbcst.16	x2, r1
	mova	r4, #0x33c
	vmac.f	dm0, dm0, x0, x2, r4
	nop
	nop
	nop
	nop
	mov	crrnd, #0xc
	vst.conv.bf16.fp32	bmll0, [p1, #0]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_bf16_mac_result_release, .-probe_asm_bf16_mac_result_release

	.globl	probe_asm_q4_exact_unroll4_release
	.p2align	4
	.type	probe_asm_q4_exact_unroll4_release,@function
probe_asm_q4_exact_unroll4_release:
	// p0=source buffer:
	//   +0x000: packed uint4 columns for dims 0..3, q=1 in this smoke.
	//   +0x100: 16 bf16 scale lanes.
	//   +0x120: 16 bf16 offset lanes.
	//   +0x140: 32 bf16 activation lanes.
	// p1=destination bf16 vector. Computes the first four exact Q4NX dims
	// using the Peano exact-unroll4 instruction shape, then releases L53.
	mova	r1, #0
	mov	crrnd, #0xc
	vbcst.32	x0, r1
	movxm	r2, #0x4b01
	mov	s0, r1
	mova	r4, #0x33c;		mov	crupsmode, #0
	mova	r0, #0;		movx	r5, #1;		vbcst.16	 x2, r2
	mova	r2, #0x3c;		movx	r6, #0;		vconv.fp32.bf16	cml0, x2

	mov	p5, p0
	vldb.128	 wh2, [p5, #0]
	vunpack	x2, wh2, unpacksign0
	vunpack	y3, x2, unpacksign0
	nop
	nop
	nop
	mov	crunpacksize, #0
	mov	crunpacksize, #1
	nop
	vups.2x	cml1, x6, s0, upssign0;		vadd	dm2, dm1, dm0, r1
	lshl	r7, r6, r5
	mov	dj0, r7;		vsub.f	dm3, dm2, dm0, r2
	movxm	r7, #0x100
	mov	dj0, r7
	vldb	wl2, [p0, dj0]
	nop
	nop
	vldb.128	 wh8, [p5, #0x20]
	vunpack	x10, wh8, unpacksign0
	vunpack	y2, x10, unpacksign0;		vconv.bf16.fp32	 x1, cml3
	nop
	vmul.f	dm4, x1, x2, r2
	nop
	mov	crunpacksize, #0
	mov	crunpacksize, #1
	nop
	vups.2x	cml2, x4, s0, upssign0;		vadd	dm2, dm2, dm0, r1
	vconv.bf16.fp32	 wh2, bmll4;		vmov	wl4, wh1
	vsub.f	dm2, dm2, dm0, r2
	vmul.f	dm4, x4, x2, r2
	nop
	nop
	movxm	r7, #0x120
	mov	dj0, r7
	vlda.conv.fp32.bf16	 bmll1, [p0, dj0]
	nop
	vconv.bf16.fp32	 x11, cml2
	vconv.bf16.fp32	 wh2, bmll4
	vconv.fp32.bf16	bmll3, wh2;		vadd.f	dm3, dm3, dm1, r2
	movxm	r7, #0x140
	mov	dj0, r7
	vldb	x4, [p0, dj0];		vmul.f	dm3, x11, x2, r2
	vmov	wl8, wh11
	nop
	vmul.f	dm2, x8, x2, r2
	vadd.f	dm4, dm4, dm1, r2
	vconv.bf16.fp32	 wl9, bmll3;		vconv.fp32.bf16	bmll4, wh2
	vconv.bf16.fp32	 wh6, bmll3
	vextbcst.16	 x7, x4, #0;		vadd.f	dm3, dm3, dm1, r2
	vconv.fp32.bf16	bmll3, wh6
	vconv.bf16.fp32	 wl8, bmll2;		vextbcst.16	 x10, x4, #1;		vmac.f	dm2, dm2, x9, x7, r4
	vconv.bf16.fp32	 wl0, bmll4;		vmov	bmll2, x0;		vadd.f	dm4, dm4, dm1, r2
	vconv.fp32.bf16	bmll4, wl8
	vmac.f	dm2, dm2, x0, x10, r4
	vconv.bf16.fp32	 wl1, bmll3;		vextbcst.16	 x2, x4, #2
	nop
	vmac.f	dm2, dm2, x1, x2, r4
	vconv.bf16.fp32	 wl3, bmll4;		vextbcst.16	 x9, x4, #3
	nop
	vmac.f	dm1, dm2, x3, x9, r4
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
	vst.conv.bf16.fp32	bmll1, [p1, #0]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_q4_exact_unroll4_release, .-probe_asm_q4_exact_unroll4_release

	.globl	probe_asm_q4_exact_group8_release
	.p2align	4
	.type	probe_asm_q4_exact_group8_release,@function
probe_asm_q4_exact_group8_release:
	// p0=source buffer:
	//   +0x000: packed uint4 columns for dims 0..7, q=1 in this smoke.
	//   +0x100: 16 bf16 scale lanes.
	//   +0x120: 16 bf16 offset lanes.
	//   +0x140: 32 bf16 activation lanes.
	// p1=destination bf16 vector. This stitches two 4-dim exact kernels with
	// an explicit accumulator handoff, matching the production scaling problem.
	mova	r1, #0
	mov	crrnd, #0xc
	vbcst.32	x8, r1
	vmov	bmll1, x8
	movxm	r7, #0x100
	mov	dj0, r7
	vldb	wl0, [p0, dj0]
	movxm	r7, #0x120
	mov	dj0, r7
	vldb	wl2, [p0, dj0]
	movxm	r7, #0x140
	mov	dj0, r7
	vldb	x4, [p0, dj0]

	nopa	;		vldb.128	 wh0, [p0, #0];		nops	;		nopxm	;		nopv
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
	vldb.128	 wh0, [p0, #0x20]
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
	vconv.bf16.fp32	 wh2, bmll4;		vextbcst.16	 x0, x4, #0;		vadd.f	dm0, dm0, dm2, r0
	mova	r1, #0x33c;		vconv.fp32.bf16	bmll0, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl2, bmll3;		vextbcst.16	 x0, x4, #1;		vadd.f	dm2, dm3, dm2, r0
	vconv.fp32.bf16	bmll3, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl0, bmll0;		vextbcst.16	 x2, x4, #2
	nop
	vmac.f	dm0, dm1, x0, x2, r1
	vconv.bf16.fp32	 wl0, bmll2;		vextbcst.16	 x2, x4, #3
	nop
	vmac.f	dm0, dm0, x0, x2, r1
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
	movxm	r7, #0x100
	mov	dj0, r7
	vldb	wl0, [p0, dj0]
	movxm	r7, #0x120
	mov	dj0, r7
	vldb	wl2, [p0, dj0]
	movxm	r7, #0x140
	mov	dj0, r7
	vldb	x4, [p0, dj0]

	nopa	;		vldb.128	 wh0, [p0, #0x40];		nops	;		nopxm	;		nopv
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
	vldb.128	 wh0, [p0, #0x60]
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
	vconv.bf16.fp32	 wh2, bmll4;		vextbcst.16	 x0, x4, #4;		vadd.f	dm0, dm0, dm2, r0
	mova	r1, #0x33c;		vconv.fp32.bf16	bmll0, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl2, bmll3;		vextbcst.16	 x0, x4, #5;		vadd.f	dm2, dm3, dm2, r0
	vconv.fp32.bf16	bmll3, wh2
	vmac.f	dm1, dm1, x2, x0, r1
	vconv.bf16.fp32	 wl0, bmll0;		vextbcst.16	 x2, x4, #6
	nop
	vmac.f	dm0, dm1, x0, x2, r1
	vconv.bf16.fp32	 wl0, bmll2;		vextbcst.16	 x2, x4, #7
	nop
	vmac.f	dm0, dm0, x0, x2, r1
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32	bmll0, [p1, #0]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_q4_exact_group8_release, .-probe_asm_q4_exact_group8_release

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

	.macro Q4_SET_PACK_BASE base
	movxm	r7, #\base
	movs	m0, r7
	mov	p5, p0
	padda	[p5], m0
	.endm

	.macro Q4_RELOAD_ARGS scale_off, offset_off, activation_off
	movxm	r7, #\scale_off
	mov	dj0, r7
	vldb	wl0, [p0, dj0]
	movxm	r7, #\offset_off
	mov	dj0, r7
	vldb	wl2, [p0, dj0]
	movxm	r7, #\activation_off
	mov	dj0, r7
	vldb	x4, [p0, dj0]
	.endm

	.macro Q4_RELOAD_ARGS_REG scale_reg, offset_reg, activation_reg
	mov	dj0, \scale_reg
	vldb	wl0, [p0, dj0]
	mov	dj0, \offset_reg
	vldb	wl2, [p0, dj0]
	mov	dj0, \activation_reg
	vldb	x4, [p0, dj0]
	.endm

	.macro Q4_RELOAD_ARGS_PTR
	vldb	wl0, [p2, #0]
	vldb	wl2, [p3, #0]
	vldb	x4, [p4, #0]
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

	.globl	probe_asm_q4_exact_group32_release
	.p2align	4
	.type	probe_asm_q4_exact_group32_release,@function
probe_asm_q4_exact_group32_release:
	// p0=source buffer:
	//   +0x000: packed uint4 columns for dims 0..31, q=1 in this smoke.
	//   +0x200: 16 bf16 scale lanes.
	//   +0x220: 16 bf16 offset lanes.
	//   +0x240: 32 bf16 activation lanes.
	// p1=destination bf16 vector. This is one full 32-dim exact Q4NX group.
	mova	r1, #0
	mov	crrnd, #0xc
	vbcst.32	x8, r1
	vmov	bmll1, x8

	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x000
	Q4_EXACT4_BLOCK 0x000, 0x020, 0, 1, 2, 3
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x040
	Q4_EXACT4_BLOCK 0x000, 0x020, 4, 5, 6, 7
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x080
	Q4_EXACT4_BLOCK 0x000, 0x020, 8, 9, 10, 11
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x0c0
	Q4_EXACT4_BLOCK 0x000, 0x020, 12, 13, 14, 15
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x100
	Q4_EXACT4_BLOCK 0x000, 0x020, 16, 17, 18, 19
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x140
	Q4_EXACT4_BLOCK 0x000, 0x020, 20, 21, 22, 23
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x180
	Q4_EXACT4_BLOCK 0x000, 0x020, 24, 25, 26, 27
	Q4_HANDOFF
	Q4_RELOAD_ARGS 0x200, 0x220, 0x240
	Q4_SET_PACK_BASE 0x1c0
	Q4_EXACT4_BLOCK 0x000, 0x020, 28, 29, 30, 31
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
	vst.conv.bf16.fp32	bmll0, [p1, #0]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_q4_exact_group32_release, .-probe_asm_q4_exact_group32_release

	.macro Q4_EXACT32_GROUP pack_base, scale_off, offset_off, activation_off
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x000
	Q4_EXACT4_BLOCK 0x000, 0x020, 0, 1, 2, 3
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x040
	Q4_EXACT4_BLOCK 0x000, 0x020, 4, 5, 6, 7
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x080
	Q4_EXACT4_BLOCK 0x000, 0x020, 8, 9, 10, 11
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x0c0
	Q4_EXACT4_BLOCK 0x000, 0x020, 12, 13, 14, 15
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x100
	Q4_EXACT4_BLOCK 0x000, 0x020, 16, 17, 18, 19
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x140
	Q4_EXACT4_BLOCK 0x000, 0x020, 20, 21, 22, 23
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x180
	Q4_EXACT4_BLOCK 0x000, 0x020, 24, 25, 26, 27
	Q4_HANDOFF
	Q4_RELOAD_ARGS \scale_off, \offset_off, \activation_off
	Q4_SET_PACK_BASE \pack_base + 0x1c0
	Q4_EXACT4_BLOCK 0x000, 0x020, 28, 29, 30, 31
	.endm

	.globl	probe_asm_q4_exact_lane8_loop_release
	.p2align	4
	.type	probe_asm_q4_exact_lane8_loop_release,@function
probe_asm_q4_exact_lane8_loop_release:
	// p0=source buffer:
	//   +0x0000: packed uint4 columns for 8 groups, q=1 in this smoke.
	//   +0x1000: 8 * 16 bf16 scale lanes.
	//   +0x1100: 8 * 16 bf16 offset lanes.
	//   +0x1200: 8 * 32 bf16 activation lanes.
	// p1=destination bf16 vector. This reuses one 32-dim group body in an
	// 8-iteration hardware loop to avoid program-memory overflow.
	mova	r1, #0
	mov	crrnd, #0xc
	vbcst.32	x8, r1
	vmov	bmll1, x8
	mova	r10, #0
	mova	r15, #8
	mova	r14, #0x40
	mov	p2, p0
	movxm	r7, #0x1000
	movs	m0, r7
	padda	[p2], m0
	mov	p3, p0
	movxm	r7, #0x1100
	movs	m0, r7
	padda	[p3], m0
	mov	p4, p0
	movxm	r7, #0x1200
	movs	m0, r7
	padda	[p4], m0
.Lq4_lane8_loop_body:
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
	padda	[p2], m0
	padda	[p3], m0
	mova	r7, #0x40
	movs	m0, r7
	padda	[p4], m0
	add	r15, r15, #-1
	jnz	r15, #.Lq4_lane8_loop_body
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32	bmll1, [p1, #0]
	mova	r8, #1
	rel	#53, r8
	ret	lr
	.size	probe_asm_q4_exact_lane8_loop_release, .-probe_asm_q4_exact_lane8_loop_release
