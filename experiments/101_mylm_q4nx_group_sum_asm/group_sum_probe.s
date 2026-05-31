	.text

	.macro SAVE_GROUP_SUM_STATE
	paddxm	[sp], #0x80
	st	r0, [sp, #-0x80]
	st	r4, [sp, #-0x7c]
	st	r7, [sp, #-0x78]
	st	r10, [sp, #-0x74]
	st	r11, [sp, #-0x70]
	st	r12, [sp, #-0x6c]
	st	r13, [sp, #-0x68]
	st	r15, [sp, #-0x64]
	st	p0, [sp, #-0x60]
	st	p1, [sp, #-0x5c]
	st	p2, [sp, #-0x58]
	st	lr, [sp, #-0x54]
	.endm

	.macro RESTORE_GROUP_SUM_STATE
	lda	lr, [sp, #-0x54]
	lda	p2, [sp, #-0x58]
	lda	p1, [sp, #-0x5c]
	lda	p0, [sp, #-0x60]
	lda	r15, [sp, #-0x64]
	lda	r13, [sp, #-0x68]
	lda	r12, [sp, #-0x6c]
	lda	r11, [sp, #-0x70]
	lda	r10, [sp, #-0x74]
	lda	r7, [sp, #-0x78]
	lda	r4, [sp, #-0x7c]
	lda	r0, [sp, #-0x80]
	paddxm	[sp], #-0x80
	.endm

	.macro GS_MAIN_MAC lane, qreg, tmp
	vextbcst.16	\tmp, x11, #\lane
	nop
	nop
	nop
	nop
	vmac.f	dm1, dm1, \qreg, \tmp, r4
	.endm

	.macro MYLM_GROUP_SUM_GROUP
	// This numerical body feeds the MAC from pre-expanded q*scale registers so
	// the experiment isolates the group-sum schedule and accumulator lifetime.
	// `vups.4x` is deliberately not kept as a dead instruction here: the failed
	// draft showed it aliases real accumulator/control state unless it is part
	// of the same register plan as the MAC users.
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vbcst.16	x2, r10
	vbcst.16	x3, r10
	vbcst.16	x5, r10
	vbcst.16	x7, r10
	vbcst.16	x9, r10
	vldb	x11, [p1, #0]
	lda.s16	r7, [p0], #0x2
	GS_MAIN_MAC 0, x2, x0
	GS_MAIN_MAC 1, x3, x1
	GS_MAIN_MAC 2, x5, x4
	GS_MAIN_MAC 3, x7, x6
	GS_MAIN_MAC 4, x9, x8
	GS_MAIN_MAC 5, x2, x10
	GS_MAIN_MAC 6, x3, x0
	GS_MAIN_MAC 7, x5, x1
	GS_MAIN_MAC 8, x7, x4
	GS_MAIN_MAC 9, x9, x6
	GS_MAIN_MAC 10, x2, x8
	GS_MAIN_MAC 11, x3, x10
	GS_MAIN_MAC 12, x5, x0
	GS_MAIN_MAC 13, x7, x1
	GS_MAIN_MAC 14, x9, x4
	GS_MAIN_MAC 15, x2, x6
	GS_MAIN_MAC 16, x3, x8
	GS_MAIN_MAC 17, x5, x10
	GS_MAIN_MAC 18, x7, x0
	GS_MAIN_MAC 19, x9, x1
	GS_MAIN_MAC 20, x2, x4
	GS_MAIN_MAC 21, x3, x6
	GS_MAIN_MAC 22, x5, x8
	GS_MAIN_MAC 23, x7, x10
	GS_MAIN_MAC 24, x9, x0
	GS_MAIN_MAC 25, x2, x1
	GS_MAIN_MAC 26, x3, x4
	GS_MAIN_MAC 27, x5, x6
	GS_MAIN_MAC 28, x7, x8
	GS_MAIN_MAC 29, x9, x10
	GS_MAIN_MAC 30, x2, x0
	GS_MAIN_MAC 31, x3, x1
	vbcst.16	x2, r12
	vbcst.16	x0, r7
	vmac.f	dm1, dm1, x2, x0, r4
	.endm

	.section	.text.asm_mylm_q4_group_sum_body,"ax",@progbits
	.globl	asm_mylm_q4_group_sum_body
	.p2align	4
	.type	asm_mylm_q4_group_sum_body,@function
asm_mylm_q4_group_sum_body:
	// p0=workspace i32[128].
	// Derived layout:
	//   dst        = workspace + 0 dwords
	//   group_sum  = workspace + 16 dwords
	//   activation = workspace + 32 dwords
	// Synthetic contract:
	//   qscale vector = bf16(1.0)
	//   activation vector = bf16(1.0) x 32
	//   offset vector = bf16(2.0)
	//   group_sum[g] = bf16(32.0)
	// Each quant group contributes 32*1 + 2*32 = 96.
	SAVE_GROUP_SUM_STATE
	mov	p2, p0
	mov	p1, p0
	mova	r0, #0x80
	movs	m0, r0
	padda	[p1], m0
	mova	r0, #0x40
	movs	m0, r0
	padda	[p0], m0
	mov	crrnd, #0xc
	mov	crupsmode, #0
	movx	upssign0, #0
	mov	unpacksign0, upssign0
	mova	r0, #0
	mov	s0, r0
	mova	r4, #0x33c
	movxm	r10, #0x3f80
	movxm	r11, #0x3f80
	movxm	r12, #0x4000
	movxm	r13, #0x0f0f
	vlda	bmll1, [p2, #0]
	vbcst.16	x9, r13
	mova	r15, #0x8
	add.nc	lc, r15, #0
	movxm	ls, #.Lmylm_q4_group_sum_body_start
	movxm	le, #.Lmylm_q4_group_sum_body_end
	.p2align	4
.Lmylm_q4_group_sum_body_start:
	MYLM_GROUP_SUM_GROUP
	.p2align	4
.Lmylm_q4_group_sum_body_end:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p2, #0]
	RESTORE_GROUP_SUM_STATE
	ret	lr
	.size	asm_mylm_q4_group_sum_body, .-asm_mylm_q4_group_sum_body
