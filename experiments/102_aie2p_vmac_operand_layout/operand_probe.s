	.text

	.macro SAVE_OPERAND_STATE
	paddxm	[sp], #0x80
	st	r0, [sp, #-0x80]
	st	r4, [sp, #-0x7c]
	st	r7, [sp, #-0x78]
	st	r10, [sp, #-0x74]
	st	r11, [sp, #-0x70]
	st	r12, [sp, #-0x6c]
	st	r15, [sp, #-0x68]
	st	p0, [sp, #-0x64]
	st	p1, [sp, #-0x60]
	st	p2, [sp, #-0x5c]
	st	p3, [sp, #-0x58]
	st	p4, [sp, #-0x54]
	st	lr, [sp, #-0x50]
	.endm

	.macro RESTORE_OPERAND_STATE
	lda	lr, [sp, #-0x50]
	lda	p4, [sp, #-0x54]
	lda	p3, [sp, #-0x58]
	lda	p2, [sp, #-0x5c]
	lda	p1, [sp, #-0x60]
	lda	p0, [sp, #-0x64]
	lda	r15, [sp, #-0x68]
	lda	r12, [sp, #-0x6c]
	lda	r11, [sp, #-0x70]
	lda	r10, [sp, #-0x74]
	lda	r7, [sp, #-0x78]
	lda	r4, [sp, #-0x7c]
	lda	r0, [sp, #-0x80]
	paddxm	[sp], #-0x80
	.endm

	.macro RESET_ACC
	vlda	bmll1, [p3, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	.endm

	.macro STORE_ACC
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst	bmll1, [p4, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mova	r0, #0x40
	movs	m0, r0
	padda	[p4], m0
	.endm

	.macro MAC_EXT lane
	vextbcst.16	x0, x11, #\lane
	nop
	nop
	nop
	nop
	vmac.f	dm1, dm1, x2, x0, r4
	.endm

	.section	.text.asm_vmac_operand_layout,"ax",@progbits
	.globl	asm_vmac_operand_layout
	.p2align	4
	.type	asm_vmac_operand_layout,@function
asm_vmac_operand_layout:
	// p0=workspace i32[1024]
	// output blocks are 16 float lanes each:
	//   block0..4: q operand x2/x3/x5/x7/x9 times vbcst(1)
	//   block5..36: vextbcst activation lane 0..31
	//   block37: group-sum correction vbcst(2) * vbcst(32)
	//   block38: 32 vextbcst lane MACs
	SAVE_OPERAND_STATE
	mov	p4, p0
	mov	p1, p0
	movxm	r0, #0xa00
	movs	m0, r0
	padda	[p1], m0
	mov	p3, p0
	movxm	r0, #0xc00
	movs	m0, r0
	padda	[p3], m0
	mov	crrnd, #0xc
	mova	r4, #0x33c
	movxm	r10, #0x3f80
	movxm	r11, #0x4000
	movxm	r12, #0x4200
	vbcst.16	x2, r10
	vbcst.16	x3, r10
	vbcst.16	x5, r10
	vbcst.16	x7, r10
	vbcst.16	x9, r10
	vldb	x11, [p1, #0]

	RESET_ACC
	vmac.f	dm1, dm1, x2, x2, r4
	STORE_ACC

	RESET_ACC
	vmac.f	dm1, dm1, x3, x2, r4
	STORE_ACC

	RESET_ACC
	vmac.f	dm1, dm1, x5, x2, r4
	STORE_ACC

	RESET_ACC
	vmac.f	dm1, dm1, x7, x2, r4
	STORE_ACC

	RESET_ACC
	vmac.f	dm1, dm1, x9, x2, r4
	STORE_ACC

	RESET_ACC
	MAC_EXT 0
	STORE_ACC

	RESET_ACC
	MAC_EXT 1
	STORE_ACC

	RESET_ACC
	MAC_EXT 2
	STORE_ACC

	RESET_ACC
	MAC_EXT 3
	STORE_ACC

	RESET_ACC
	MAC_EXT 4
	STORE_ACC

	RESET_ACC
	MAC_EXT 5
	STORE_ACC

	RESET_ACC
	MAC_EXT 6
	STORE_ACC

	RESET_ACC
	MAC_EXT 7
	STORE_ACC

	RESET_ACC
	MAC_EXT 8
	STORE_ACC

	RESET_ACC
	MAC_EXT 9
	STORE_ACC

	RESET_ACC
	MAC_EXT 10
	STORE_ACC

	RESET_ACC
	MAC_EXT 11
	STORE_ACC

	RESET_ACC
	MAC_EXT 12
	STORE_ACC

	RESET_ACC
	MAC_EXT 13
	STORE_ACC

	RESET_ACC
	MAC_EXT 14
	STORE_ACC

	RESET_ACC
	MAC_EXT 15
	STORE_ACC

	RESET_ACC
	MAC_EXT 16
	STORE_ACC

	RESET_ACC
	MAC_EXT 17
	STORE_ACC

	RESET_ACC
	MAC_EXT 18
	STORE_ACC

	RESET_ACC
	MAC_EXT 19
	STORE_ACC

	RESET_ACC
	MAC_EXT 20
	STORE_ACC

	RESET_ACC
	MAC_EXT 21
	STORE_ACC

	RESET_ACC
	MAC_EXT 22
	STORE_ACC

	RESET_ACC
	MAC_EXT 23
	STORE_ACC

	RESET_ACC
	MAC_EXT 24
	STORE_ACC

	RESET_ACC
	MAC_EXT 25
	STORE_ACC

	RESET_ACC
	MAC_EXT 26
	STORE_ACC

	RESET_ACC
	MAC_EXT 27
	STORE_ACC

	RESET_ACC
	MAC_EXT 28
	STORE_ACC

	RESET_ACC
	MAC_EXT 29
	STORE_ACC

	RESET_ACC
	MAC_EXT 30
	STORE_ACC

	RESET_ACC
	MAC_EXT 31
	STORE_ACC

	RESET_ACC
	vbcst.16	x2, r11
	vbcst.16	x0, r12
	vmac.f	dm1, dm1, x2, x0, r4
	STORE_ACC

	RESET_ACC
	vbcst.16	x2, r10
	MAC_EXT 0
	MAC_EXT 1
	MAC_EXT 2
	MAC_EXT 3
	MAC_EXT 4
	MAC_EXT 5
	MAC_EXT 6
	MAC_EXT 7
	MAC_EXT 8
	MAC_EXT 9
	MAC_EXT 10
	MAC_EXT 11
	MAC_EXT 12
	MAC_EXT 13
	MAC_EXT 14
	MAC_EXT 15
	MAC_EXT 16
	MAC_EXT 17
	MAC_EXT 18
	MAC_EXT 19
	MAC_EXT 20
	MAC_EXT 21
	MAC_EXT 22
	MAC_EXT 23
	MAC_EXT 24
	MAC_EXT 25
	MAC_EXT 26
	MAC_EXT 27
	MAC_EXT 28
	MAC_EXT 29
	MAC_EXT 30
	MAC_EXT 31
	STORE_ACC

	RESTORE_OPERAND_STATE
	ret	lr
	.size	asm_vmac_operand_layout, .-asm_vmac_operand_layout
