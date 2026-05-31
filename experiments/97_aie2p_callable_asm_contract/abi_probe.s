	.text
	.globl	asm_scalar_store
	.p2align	4
	.type	asm_scalar_store,@function
asm_scalar_store:
	movxm	r1, #0x1234
	st	r1, [p0, #0]
	ret	lr
	.size	asm_scalar_store, .-asm_scalar_store

	.globl	asm_vmac_bf16_store
	.p2align	4
	.type	asm_vmac_bf16_store,@function
asm_vmac_bf16_store:
	// p0=src bf16 vector, p1=dst bf16 vector.
	mov	crrnd, #0xc
	mova	r0, #0
	vbcst.32	x4, r0
	vmov	bmll0, x4
	vlda	x0, [p0, #0]
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
	vst.conv.bf16.fp32	bmll0, [p1, #0]
	ret	lr
	.size	asm_vmac_bf16_store, .-asm_vmac_bf16_store

	.globl	asm_vmac_float_store
	.p2align	4
	.type	asm_vmac_float_store,@function
asm_vmac_float_store:
	// p0=dst float vector. This deliberately mirrors the Peano float
	// accumulator writeback shape, without claiming this is production-ready.
	mova	r1, #0
	vbcst.32	x0, r1
	vmov	bmll0, x0
	movxm	r0, #0x3f80
	vbcst.16	x2, r0
	mova	r4, #0x33c
	vmac.f	dm0, dm0, x2, x2, r4
	nop
	nop
	nop
	nop
	vmov	x0, bmll0
	nop
	nop
	vst	x0, [p0, #0]
	ret	lr
	.size	asm_vmac_float_store, .-asm_vmac_float_store

	.globl	asm_vector_clobber_probe
	.p2align	4
	.type	asm_vector_clobber_probe,@function
asm_vector_clobber_probe:
	// p0=src bf16 vector, p1=dst bf16 vector. Clobbers x0 and bmll0 on
	// purpose so the C++ callsite disassembly can show whether normal calls
	// preserve vector state around external source assembly.
	vlda	x0, [p0, #0]
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
	vmov	bmll0, x0
	vst	x0, [p1, #0]
	ret	lr
	.size	asm_vector_clobber_probe, .-asm_vector_clobber_probe
