.macro SAVE_Q4_CALL_STATE
	paddxm	[sp], #0xc0
	st	r10, [sp, #-0xc0]
	st	r14, [sp, #-0xbc]
	st	r15, [sp, #-0xb8]
	st	p0, [sp, #-0xb4]
	st	p1, [sp, #-0xb0]
	st	p2, [sp, #-0xac]
	st	p3, [sp, #-0xa8]
	st	p4, [sp, #-0xa4]
	st	p5, [sp, #-0xa0]
	st	p6, [sp, #-0x9c]
	st	p7, [sp, #-0x98]
	st	lr, [sp, #-0x94]
	.endm

	.macro RESTORE_Q4_CALL_STATE
	lda	lr, [sp, #-0x94]
	lda	p7, [sp, #-0x98]
	lda	p6, [sp, #-0x9c]
	lda	p5, [sp, #-0xa0]
	lda	p4, [sp, #-0xa4]
	lda	p3, [sp, #-0xa8]
	lda	p2, [sp, #-0xac]
	lda	p1, [sp, #-0xb0]
	lda	p0, [sp, #-0xb4]
	lda	r15, [sp, #-0xb8]
	lda	r14, [sp, #-0xbc]
	lda	r10, [sp, #-0xc0]
	paddxm	[sp], #-0xc0
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

	.macro Q4_LOAD_GROUP_ACTIVATION
	vldb	x4, [p3, #0]
	.endm

	.macro Q4_RELOAD_BLOCK_SCALE_OFFSET
	vldb	wl0, [p1, #0]
	vldb	wl2, [p2, #0]
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
	vmov	bmll1, bmll0
	.endm

	.macro Q4_EXACT32_GROUP_DIRECT
	Q4_LOAD_GROUP_ACTIVATION
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 0, 1, 2, 3
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 4, 5, 6, 7
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 8, 9, 10, 11
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 12, 13, 14, 15
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 16, 17, 18, 19
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 20, 21, 22, 23
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 24, 25, 26, 27
	Q4_HANDOFF
	add	r10, r10, r14
	Q4_SET_PACK_BASE_REG r10
	Q4_RELOAD_BLOCK_SCALE_OFFSET
	Q4_EXACT4_BLOCK 0x000, 0x010, 28, 29, 30, 31
	Q4_HANDOFF
	add	r10, r10, r14
	.endm

	.macro Q4_EXACT_LANE_ZOL start_label, end_label
	mov	crrnd, #0xc
	vlda	bmll1, [p4, #0]
	mova	r10, #0
	mova	r14, #0x20
	mova	r15, #0x8
	add.nc	lc, r15, #0
	movxm	ls, #\start_label
	movxm	le, #\end_label
	.p2align	4
\start_label:
	Q4_EXACT32_GROUP_DIRECT
	mova	r7, #0x40
	movs	m0, r7
	padda	[p1], m0
	padda	[p2], m0
	.p2align	4
\end_label:
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
	.endm

.section	.text.q4nx_chunk_accum_asm_zol,"ax",@progbits
	.globl	q4nx_chunk_accum_asm_zol
	.p2align	4
	.type	q4nx_chunk_accum_asm_zol,@function
q4nx_chunk_accum_asm_zol:
	// p0=target float[32], p1=packed Q4NX chunk, p2=activation bf16[256].
	// packed layout: scales[256 bf16], offsets[256 bf16], data[4096 bytes].
	SAVE_Q4_CALL_STATE
mov	p4, p0
	mov	p7, p1
	mov	p6, p2

	mov	p0, p7
	movxm	m0, #0x400
	padda	[p0], m0
	mov	p1, p7
	mov	p2, p7
	movxm	m0, #0x200
	padda	[p2], m0
	mov	p3, p6
	Q4_EXACT_LANE_ZOL .Lq4nx_chunk_accum_asm_zol_lane0_start, .Lq4nx_chunk_accum_asm_zol_lane0_end

	movxm	m0, #0x40
	padda	[p4], m0
	mov	p0, p7
	movxm	m0, #0xc00
	padda	[p0], m0
	mov	p1, p7
	movxm	m0, #0x20
	padda	[p1], m0
	mov	p2, p7
	movxm	m0, #0x220
	padda	[p2], m0
	mov	p3, p6
	Q4_EXACT_LANE_ZOL .Lq4nx_chunk_accum_asm_zol_lane1_start, .Lq4nx_chunk_accum_asm_zol_lane1_end
	RESTORE_Q4_CALL_STATE
	ret	lr
	.size	q4nx_chunk_accum_asm_zol, .-q4nx_chunk_accum_asm_zol

	.section	.text.q4nx_main16_layer_scheduler,"ax",@progbits
	.globl	q4nx_main16_layer_scheduler
	.p2align	4
	.type	q4nx_main16_layer_scheduler,@function
q4nx_main16_layer_scheduler:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	mov	r16, r0
	mov	r17, r1
	mov	r31, r3
	mov	r19, p0
	mov	r20, p1
	mov	r21, p2
	mov	r22, p3
	movxm	r30, #0x72000
	mova	r29, #0
	movs	p6, r30
	movxm	m0, #0x20
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	jl	#q4nx_main16_layer_dispatcher
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_scheduler, .-q4nx_main16_layer_scheduler

	.section	.text.q4nx_main16_layer_dispatcher,"ax",@progbits
	.globl	q4nx_main16_layer_dispatcher
	.p2align	4
	.type	q4nx_main16_layer_dispatcher,@function
q4nx_main16_layer_dispatcher:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	st	r31, [sp, #-0x3c]
	jl	#q4nx_main16_layer_body_qkv
	nop
	nop
	nop
	nop
	nop
	lda	r31, [sp, #-0x3c]
	mova	r15, #3
	nop
	nop
	nop
	eq	r27, r31, r15
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
	jnz	r27, #.Llayer_dispatch_done
	nop
	nop
	nop
	nop
	nop
	jl	#q4nx_main16_layer_body_o
	nop
	nop
	nop
	nop
	nop
	lda	r31, [sp, #-0x3c]
	mova	r15, #4
	nop
	nop
	nop
	eq	r27, r31, r15
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
	jnz	r27, #.Llayer_dispatch_done
	nop
	nop
	nop
	nop
	nop
	jl	#q4nx_main16_layer_body_upgate
	nop
	nop
	nop
	nop
	nop
	lda	r31, [sp, #-0x3c]
	mova	r15, #6
	nop
	nop
	nop
	eq	r27, r31, r15
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
	jnz	r27, #.Llayer_dispatch_done
	nop
	nop
	nop
	nop
	nop
	jl	#q4nx_main16_layer_body_down
	nop
	nop
	nop
	nop
	nop
.Llayer_dispatch_done:
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_dispatcher, .-q4nx_main16_layer_dispatcher

	.section	.text.q4nx_main16_layer_q4_body,"ax",@progbits
	.globl	q4nx_main16_layer_q4_body
	.p2align	4
	.type	q4nx_main16_layer_q4_body,@function
q4nx_main16_layer_q4_body:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
// Shared Q4 run body.
// Phase config lives in tile-local control words at 0x3d00.
	mova	r18, #0
	movxm	p6, #0x73d00
	st	r18, [p6, #0]
	nop
	nop
	nop
.Llayer_q4_record_loop:
	mova	r5, #0
.Llayer_q4_chunk_loop:
	movx	r14, #-1
	acq	#49, r14
	nop
	nop
	nop
	acq	#51, r14
	nop
	nop
	nop
// Select DMA1 weight and DMA0 activation ping/pong from pinned main16 buffers.
	mova	r15, #1
	and	r27, r5, r15
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
	mova	r2, #12
	lshl	r8, r27, r2
	nop
	nop
	nop
	mova	r2, #11
	lshl	r9, r27, r2
	nop
	nop
	nop
	add	r8, r8, r9
	nop
	nop
	nop
	movxm	r9, #0x72800
	add	r8, r8, r9
	nop
	nop
	nop
	mova	r2, #14
	lshl	r9, r27, r2
	nop
	nop
	nop
	movxm	r6, #0x78000
	add	r9, r9, r6
	nop
	nop
	nop
	movs	p1, r8
	movs	p2, r9
	movs	p0, r30
mov	p4, p0
	mov	p7, p1
	mov	p6, p2

	mov	p0, p7
	movxm	m0, #0x400
	padda	[p0], m0
	mov	p1, p7
	mov	p2, p7
	movxm	m0, #0x200
	padda	[p2], m0
	mov	p3, p6
	Q4_EXACT_LANE_ZOL .Lq4nx_main16_layer_q4_body_lane0_start, .Lq4nx_main16_layer_q4_body_lane0_end

	movxm	m0, #0x40
	padda	[p4], m0
	mov	p0, p7
	movxm	m0, #0xc00
	padda	[p0], m0
	mov	p1, p7
	movxm	m0, #0x20
	padda	[p1], m0
	mov	p2, p7
	movxm	m0, #0x220
	padda	[p2], m0
	mov	p3, p6
	Q4_EXACT_LANE_ZOL .Lq4nx_main16_layer_q4_body_lane1_start, .Lq4nx_main16_layer_q4_body_lane1_end
	mova	r15, #1
	rel	#48, r15
	nop
	nop
	nop
	rel	#50, r15
	nop
	nop
	nop
	add	r5, r5, #1
	nop
	nop
	nop
	movxm	p6, #0x73d00
	lda	r13, [p6, #8]
	nop
	nop
	nop
	eq	r0, r5, r13
	nop
	nop
	nop
	jz	r0, #.Llayer_q4_chunk_loop
	nop
	nop
	nop
	nop
	nop
	movx	r14, #-1
	acq	#52, r14
	nop
	nop
	nop
	movxm	p6, #0x73d00
	lda	r18, [p6, #0]
	nop
	nop
	nop
	mova	r15, #1
	and	r0, r18, r15
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
	jz	r0, #.Llayer_record_ping_path
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
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
.Llayer_record_pong_path:
	mova	r5, #1
	movxm	p0, #0x7541c
	movxm	p6, #0x73d00
	lda	r23, [p6, #12]
	lda	r24, [p6, #16]
	lda	r25, [p6, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	and	r7, r24, r5
	nop
	nop
	nop
	add	r0, r25, r7
	nop
	nop
	nop
	mov	r1, r18
	mov	r10, r23
	mov	r3, r16
	mov	r4, r17
	movxm	p6, #0x72000
	jl	#q4nx_main16_layer_emit_record
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
	mova	r15, #1
	rel	#53, r15
	nop
	nop
	nop
	add	r18, r18, #1
	nop
	nop
	nop
	movxm	p6, #0x73d00
	st	r18, [p6, #0]
	lda	r12, [p6, #4]
	nop
	nop
	nop
	eq	r0, r18, r12
	nop
	nop
	nop
	jz	r0, #.Llayer_q4_record_loop
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Llayer_record_ping_path:
	mova	r5, #0
	movxm	p0, #0x73c1c
	movxm	p6, #0x73d00
	lda	r23, [p6, #12]
	lda	r24, [p6, #16]
	lda	r25, [p6, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	and	r7, r24, r5
	nop
	nop
	nop
	add	r0, r25, r7
	nop
	nop
	nop
	mov	r1, r18
	mov	r10, r23
	mov	r3, r16
	mov	r4, r17
	movxm	p6, #0x72000
	jl	#q4nx_main16_layer_emit_record
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
	mova	r15, #1
	rel	#53, r15
	nop
	nop
	nop
	add	r18, r18, #1
	nop
	nop
	nop
	movxm	p6, #0x73d00
	st	r18, [p6, #0]
	lda	r12, [p6, #4]
	nop
	nop
	nop
	eq	r0, r18, r12
	nop
	nop
	nop
	jz	r0, #.Llayer_q4_record_loop
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_q4_body, .-q4nx_main16_layer_q4_body

	.section	.text.q4nx_main16_layer_emit_record,"ax",@progbits
	.globl	q4nx_main16_layer_emit_record
	.p2align	4
	.type	q4nx_main16_layer_emit_record,@function
q4nx_main16_layer_emit_record:
// p0=selected record buffer, r0=phase, r1=block, r3=group, r4=row, r10=packet id.
// p6=32-lane FP32 projection accumulator.
	mova	r2, #24
	lshl	r8, r0, r2
	nop
	nop
	nop
	mova	r2, #20
	lshl	r9, r1, r2
	nop
	nop
	nop
	or	r8, r8, r9
	nop
	nop
	nop
	mova	r2, #16
	lshl	r9, r3, r2
	nop
	nop
	nop
	or	r8, r8, r9
	nop
	nop
	nop
	mova	r2, #8
	lshl	r9, r4, r2
	nop
	nop
	nop
	or	r8, r8, r9
	nop
	nop
	nop
	or	r8, r8, r10
	nop
	nop
	nop
	mov	p7, p0
	st	r8, [p7], #4
	vlda	bmll0, [p6, #0]
	vst.conv.bf16.fp32	bmll0, [p7, #0]
	vlda	bmll0, [p6, #0x40]
	vst.conv.bf16.fp32	bmll0, [p7, #0x20]
	movs	p6, r30
	movxm	m0, #0x20
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	padda	[p6], m0
	st	r29, [p6, #0]
	st	r29, [p6, #4]
	st	r29, [p6, #8]
	st	r29, [p6, #12]
	st	r29, [p6, #16]
	st	r29, [p6, #20]
	st	r29, [p6, #24]
	st	r29, [p6, #28]
	nop
	nop
	nop
	nop
	nop
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_emit_record, .-q4nx_main16_layer_emit_record

	.section	.text.q4nx_main16_layer_body_qkv,"ax",@progbits
	.globl	q4nx_main16_layer_body_qkv
	.p2align	4
	.type	q4nx_main16_layer_body_qkv,@function
q4nx_main16_layer_body_qkv:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	movxm	p6, #0x73d00
	mova	r12, #8
	st	r12, [p6, #4]
	mova	r12, #16
	st	r12, [p6, #8]
	mova	r12, #10
	st	r12, [p6, #12]
	mova	r12, #0
	st	r12, [p6, #16]
	mova	r12, #0
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	movxm	p6, #0x73d00
	mova	r12, #2
	st	r12, [p6, #4]
	mova	r12, #16
	st	r12, [p6, #8]
	mova	r12, #11
	st	r12, [p6, #12]
	mova	r12, #0
	st	r12, [p6, #16]
	mova	r12, #1
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	movxm	p6, #0x73d00
	mova	r12, #2
	st	r12, [p6, #4]
	mova	r12, #16
	st	r12, [p6, #8]
	mova	r12, #12
	st	r12, [p6, #12]
	mova	r12, #0
	st	r12, [p6, #16]
	mova	r12, #2
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_body_qkv, .-q4nx_main16_layer_body_qkv

	.section	.text.q4nx_main16_layer_body_o,"ax",@progbits
	.globl	q4nx_main16_layer_body_o
	.p2align	4
	.type	q4nx_main16_layer_body_o,@function
q4nx_main16_layer_body_o:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	movxm	p6, #0x73d00
	mova	r12, #8
	st	r12, [p6, #4]
	mova	r12, #16
	st	r12, [p6, #8]
	mova	r12, #13
	st	r12, [p6, #12]
	mova	r12, #0
	st	r12, [p6, #16]
	mova	r12, #3
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_body_o, .-q4nx_main16_layer_body_o

	.section	.text.q4nx_main16_layer_body_upgate,"ax",@progbits
	.globl	q4nx_main16_layer_body_upgate
	.p2align	4
	.type	q4nx_main16_layer_body_upgate,@function
q4nx_main16_layer_body_upgate:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	movxm	p6, #0x73d00
	mova	r12, #48
	st	r12, [p6, #4]
	mova	r12, #16
	st	r12, [p6, #8]
	mova	r12, #14
	st	r12, [p6, #12]
	mova	r12, #1
	st	r12, [p6, #16]
	mova	r12, #4
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_body_upgate, .-q4nx_main16_layer_body_upgate

	.section	.text.q4nx_main16_layer_body_down,"ax",@progbits
	.globl	q4nx_main16_layer_body_down
	.p2align	4
	.type	q4nx_main16_layer_body_down,@function
q4nx_main16_layer_body_down:
	paddxm	[sp], #0x40
	st	lr, [sp, #-0x40]
	movxm	p6, #0x73d00
	mova	r12, #8
	st	r12, [p6, #4]
	mova	r12, #48
	st	r12, [p6, #8]
	mova	r12, #15
	st	r12, [p6, #12]
	mova	r12, #0
	st	r12, [p6, #16]
	mova	r12, #6
	st	r12, [p6, #20]
	nop
	nop
	nop
	jl	#q4nx_main16_layer_q4_body
	nop
	nop
	nop
	nop
	nop
	lda	lr, [sp, #-0x40]
	paddxm	[sp], #-0x40
	ret	lr
	nop
	nop
	nop
	nop
	nop
	.size	q4nx_main16_layer_body_down, .-q4nx_main16_layer_body_down
