	.text

	.macro SAVE_Q4_CALL_STATE
	paddxm	[sp], #0x80
	st	r10, [sp, #-0x80]
	st	r14, [sp, #-0x7c]
	st	r15, [sp, #-0x78]
	st	p0, [sp, #-0x74]
	st	p1, [sp, #-0x70]
	st	p2, [sp, #-0x6c]
	st	p3, [sp, #-0x68]
	st	p4, [sp, #-0x64]
	st	p5, [sp, #-0x60]
	st	lr, [sp, #-0x5c]
	.endm

	.macro RESTORE_Q4_CALL_STATE
	lda	lr, [sp, #-0x5c]
	lda	p5, [sp, #-0x60]
	lda	p4, [sp, #-0x64]
	lda	p3, [sp, #-0x68]
	lda	p2, [sp, #-0x6c]
	lda	p1, [sp, #-0x70]
	lda	p0, [sp, #-0x74]
	lda	r15, [sp, #-0x78]
	lda	r14, [sp, #-0x7c]
	lda	r10, [sp, #-0x80]
	paddxm	[sp], #-0x80
	.endm

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

	.macro Q4_EXACT32_GROUP_DIRECT
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 0, 1, 2, 3
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 4, 5, 6, 7
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 8, 9, 10, 11
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 12, 13, 14, 15
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 16, 17, 18, 19
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 20, 21, 22, 23
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 24, 25, 26, 27
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_RELOAD_ARGS_PTR
	Q4_SET_PACK_BASE_REG r10
	Q4_EXACT4_BLOCK 0x000, 0x010, 28, 29, 30, 31
	Q4_HANDOFF
	add	r10, r10, r14
	.endm

	.macro Q4_ADVANCE_BF16_GROUP
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
	padda	[p3], m0
	.endm

	.section	.text.asm_hwloop_smoke_lc2,"ax",@progbits
	.globl	asm_hwloop_smoke_lc2
	.p2align	4
	.type	asm_hwloop_smoke_lc2,@function
asm_hwloop_smoke_lc2:
	// p0=dst float[16]. Expected output is 2.0 if LC/LS/LE loops twice.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p0, #0]
	movxm	r10, #0x3f80
	vbcst.16	x2, r10
	mova	r14, #0x33c
	mova	r15, #0x2
	add.nc	lc, r15, #0
	movxm	ls, #.Lhwloop_smoke_lc2_start
	movxm	le, #.Lhwloop_smoke_lc2_end
.Lhwloop_smoke_lc2_start:
	nop
.Lhwloop_smoke_lc2_end:
	vmac.f	dm1, dm1, x2, x2, r14
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p0, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_hwloop_smoke_lc2, .-asm_hwloop_smoke_lc2

	.section	.text.asm_hwloop_smoke_lc2_aligned,"ax",@progbits
	.globl	asm_hwloop_smoke_lc2_aligned
	.p2align	4
	.type	asm_hwloop_smoke_lc2_aligned,@function
asm_hwloop_smoke_lc2_aligned:
	// Same smoke as above, but LS/LE are 16-byte aligned and LE is >112 bytes away.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p0, #0]
	movxm	r10, #0x3f80
	vbcst.16	x2, r10
	mova	r14, #0x33c
	mova	r15, #0x2
	add.nc	lc, r15, #0
	movxm	ls, #.Lhwloop_smoke_lc2_aligned_start
	movxm	le, #.Lhwloop_smoke_lc2_aligned_end
	.p2align	4
.Lhwloop_smoke_lc2_aligned_start:
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
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	.p2align	4
.Lhwloop_smoke_lc2_aligned_end:
	vmac.f	dm1, dm1, x2, x2, r14
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p0, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_hwloop_smoke_lc2_aligned, .-asm_hwloop_smoke_lc2_aligned

	.section	.text.asm_q4_direct_exact_group1,"ax",@progbits
	.globl	asm_q4_direct_exact_group1
	.p2align	4
	.type	asm_q4_direct_exact_group1,@function
asm_q4_direct_exact_group1:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	Q4_EXACT32_GROUP_DIRECT
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group1, .-asm_q4_direct_exact_group1

	.section	.text.asm_q4_direct_exact_group2,"ax",@progbits
	.globl	asm_q4_direct_exact_group2
	.p2align	4
	.type	asm_q4_direct_exact_group2,@function
asm_q4_direct_exact_group2:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r15, #2
	mova	r14, #0x20
.Lq4_direct_exact_group2_loop:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	add	r15, r15, #-1
	jnz	r15, #.Lq4_direct_exact_group2_loop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2, .-asm_q4_direct_exact_group2

	.section	.text.asm_q4_direct_exact_group2_hwloop,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_hwloop
	.p2align	4
	.type	asm_q4_direct_exact_group2_hwloop,@function
asm_q4_direct_exact_group2_hwloop:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	lc, #0x2
	movxm	ls, #.Lq4_direct_exact_group2_hwloop_start
	movxm	le, #.Lq4_direct_exact_group2_hwloop_end
.Lq4_direct_exact_group2_hwloop_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
.Lq4_direct_exact_group2_hwloop_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_hwloop, .-asm_q4_direct_exact_group2_hwloop

	.section	.text.asm_q4_direct_exact_group2_hwloop_addnc,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_hwloop_addnc
	.p2align	4
	.type	asm_q4_direct_exact_group2_hwloop_addnc,@function
asm_q4_direct_exact_group2_hwloop_addnc:
	// Same range as group2_hwloop, but match Peano's hardware-loop LC write.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x2
	add.nc	lc, r15, #0
	movxm	ls, #.Lq4_direct_exact_group2_hwloop_addnc_start
	movxm	le, #.Lq4_direct_exact_group2_hwloop_addnc_end
.Lq4_direct_exact_group2_hwloop_addnc_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
.Lq4_direct_exact_group2_hwloop_addnc_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_hwloop_addnc, .-asm_q4_direct_exact_group2_hwloop_addnc

	.section	.text.asm_q4_direct_exact_group2_hwloop_addnc_gap,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_hwloop_addnc_gap
	.p2align	4
	.type	asm_q4_direct_exact_group2_hwloop_addnc_gap,@function
asm_q4_direct_exact_group2_hwloop_addnc_gap:
	// Match Peano's LC write and leave a MyLM-style setup window before LS.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x2
	add.nc	lc, r15, #0
	movxm	ls, #.Lq4_direct_exact_group2_hwloop_addnc_gap_start
	movxm	le, #.Lq4_direct_exact_group2_hwloop_addnc_gap_end
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
.Lq4_direct_exact_group2_hwloop_addnc_gap_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
.Lq4_direct_exact_group2_hwloop_addnc_gap_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_hwloop_addnc_gap, .-asm_q4_direct_exact_group2_hwloop_addnc_gap

	.section	.text.asm_q4_direct_exact_group2_hwloop_addnc_lc3,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_hwloop_addnc_lc3
	.p2align	4
	.type	asm_q4_direct_exact_group2_hwloop_addnc_lc3,@function
asm_q4_direct_exact_group2_hwloop_addnc_lc3:
	// Test whether source-assembly LC is encoded as tripcount+1 in this path.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x3
	add.nc	lc, r15, #0
	movxm	ls, #.Lq4_direct_exact_group2_hwloop_addnc_lc3_start
	movxm	le, #.Lq4_direct_exact_group2_hwloop_addnc_lc3_end
.Lq4_direct_exact_group2_hwloop_addnc_lc3_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
.Lq4_direct_exact_group2_hwloop_addnc_lc3_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_hwloop_addnc_lc3, .-asm_q4_direct_exact_group2_hwloop_addnc_lc3

	.section	.text.asm_q4_direct_exact_group2_hwloop_addnc_aligned,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_hwloop_addnc_aligned
	.p2align	4
	.type	asm_q4_direct_exact_group2_hwloop_addnc_aligned,@function
asm_q4_direct_exact_group2_hwloop_addnc_aligned:
	// Q4 exact group loop with source-level LS/LE bundle alignment.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x2
	add.nc	lc, r15, #0
	movxm	ls, #.Lq4_direct_exact_group2_hwloop_addnc_aligned_start
	movxm	le, #.Lq4_direct_exact_group2_hwloop_addnc_aligned_end
	.p2align	4
.Lq4_direct_exact_group2_hwloop_addnc_aligned_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
	.p2align	4
.Lq4_direct_exact_group2_hwloop_addnc_aligned_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_hwloop_addnc_aligned, .-asm_q4_direct_exact_group2_hwloop_addnc_aligned

	.section	.text.asm_q4_direct_exact_group8_hwloop_addnc_aligned,"ax",@progbits
	.globl	asm_q4_direct_exact_group8_hwloop_addnc_aligned
	.p2align	4
	.type	asm_q4_direct_exact_group8_hwloop_addnc_aligned,@function
asm_q4_direct_exact_group8_hwloop_addnc_aligned:
	// Full 256-dim lane body: one 32-dim exact group repeated by aligned ZOL.
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x8
	add.nc	lc, r15, #0
	movxm	ls, #.Lq4_direct_exact_group8_hwloop_addnc_aligned_start
	movxm	le, #.Lq4_direct_exact_group8_hwloop_addnc_aligned_end
	.p2align	4
.Lq4_direct_exact_group8_hwloop_addnc_aligned_start:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
	.p2align	4
.Lq4_direct_exact_group8_hwloop_addnc_aligned_end:
	padda	[p3], m0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group8_hwloop_addnc_aligned, .-asm_q4_direct_exact_group8_hwloop_addnc_aligned

	.section	.text.asm_q4_direct_exact_group2_unrolled,"ax",@progbits
	.globl	asm_q4_direct_exact_group2_unrolled
	.p2align	4
	.type	asm_q4_direct_exact_group2_unrolled,@function
asm_q4_direct_exact_group2_unrolled:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	Q4_EXACT32_GROUP_DIRECT
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group2_unrolled, .-asm_q4_direct_exact_group2_unrolled

	.section	.text.asm_q4_direct_exact_group4_unrolled,"ax",@progbits
	.globl	asm_q4_direct_exact_group4_unrolled
	.p2align	4
	.type	asm_q4_direct_exact_group4_unrolled,@function
asm_q4_direct_exact_group4_unrolled:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group4_unrolled, .-asm_q4_direct_exact_group4_unrolled

	.section	.text.asm_q4_direct_exact_group4x2_asm_call,"ax",@progbits
	.globl	asm_q4_direct_exact_group4x2_asm_call
	.p2align	4
	.type	asm_q4_direct_exact_group4x2_asm_call,@function
asm_q4_direct_exact_group4x2_asm_call:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	jl	#asm_q4_direct_exact_group4_unrolled
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movxm	m0, #0x400
	padda	[p0], m0
	padda	[p1], #0x100
	padda	[p2], #0x100
	padda	[p3], #0x100
	jl	#asm_q4_direct_exact_group4_unrolled
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group4x2_asm_call, .-asm_q4_direct_exact_group4x2_asm_call

	.section	.text.asm_q4_direct_exact_group8_unrolled,"ax",@progbits
	.globl	asm_q4_direct_exact_group8_unrolled
	.p2align	4
	.type	asm_q4_direct_exact_group8_unrolled,@function
asm_q4_direct_exact_group8_unrolled:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	Q4_EXACT32_GROUP_DIRECT
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group8_unrolled, .-asm_q4_direct_exact_group8_unrolled

	.section	.text.asm_q4_direct_exact_group8,"ax",@progbits
	.globl	asm_q4_direct_exact_group8
	.p2align	4
	.type	asm_q4_direct_exact_group8,@function
asm_q4_direct_exact_group8:
	// p0=packed, p1=scales, p2=offsets, p3=activation, p4=dst float[16].
	SAVE_Q4_CALL_STATE
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r15, #8
	mova	r14, #0x20
.Lq4_direct_exact_group8_loop:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	Q4_EXACT32_GROUP_DIRECT
	Q4_ADVANCE_BF16_GROUP
	add	r15, r15, #-1
	jnz	r15, #.Lq4_direct_exact_group8_loop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	asm_q4_direct_exact_group8, .-asm_q4_direct_exact_group8
