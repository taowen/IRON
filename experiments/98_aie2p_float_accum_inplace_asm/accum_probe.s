	.text
	.globl	asm_float_accum_inplace
	.p2align	4
	.type	asm_float_accum_inplace,@function
asm_float_accum_inplace:
	// p0=dst float[16], expected 64-byte aligned tile-local buffer.
	// Keep the normal AIE2P C ABI scalar state intact.
	paddxm	[sp], #0x40
	st	r8, [sp, #-0x40]
	st	r10, [sp, #-0x3c]
	st	r14, [sp, #-0x38]
	st	r15, [sp, #-0x34]
	st	p6, [sp, #-0x30]
	st	p7, [sp, #-0x2c]
	st	lr, [sp, #-0x28]
	mov	crrnd, #0xc
	vlda	bmll1, [p0, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movxm	r8, #0x3f80
	vbcst.16	x2, r8
	mova	r14, #0x33c
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
	lda	lr, [sp, #-0x28]
	lda	p7, [sp, #-0x2c]
	lda	p6, [sp, #-0x30]
	lda	r15, [sp, #-0x34]
	lda	r14, [sp, #-0x38]
	lda	r10, [sp, #-0x3c]
	lda	r8, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	.size	asm_float_accum_inplace, .-asm_float_accum_inplace
