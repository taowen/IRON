#!/usr/bin/env python3
"""Shared source-assembly templates for the main16 Q4NX body."""

from __future__ import annotations


MAIN16_DATA_MEMORY_BASE = 0x70000
MAIN16_ACCUM_ADDR = 0x2000
MAIN16_WT_PING_ADDR = 0x2800
MAIN16_RECORD_PING_ADDR = 0x3C1C
MAIN16_CONTROL_ADDR = 0x3D00
MAIN16_WT_PONG_ADDR = 0x4000
MAIN16_RECORD_PONG_ADDR = 0x541C
MAIN16_CHUNK_PING_ADDR = 0x8000
MAIN16_CHUNK_PONG_ADDR = 0xC000
MAIN16_ACCUM_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_ACCUM_ADDR
MAIN16_WT_PING_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_WT_PING_ADDR
MAIN16_RECORD_PING_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_RECORD_PING_ADDR
MAIN16_CONTROL_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_CONTROL_ADDR
MAIN16_WT_PONG_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_WT_PONG_ADDR
MAIN16_RECORD_PONG_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_RECORD_PONG_ADDR
MAIN16_CHUNK_PING_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_CHUNK_PING_ADDR
MAIN16_CHUNK_PONG_ABS_ADDR = MAIN16_DATA_MEMORY_BASE + MAIN16_CHUNK_PONG_ADDR
CONTROL_RECORD_COUNTER_OFFSET = 0
CONTROL_RECORDS_OFFSET = 4
CONTROL_CHUNKS_PER_RECORD_OFFSET = 8
CONTROL_PACKET_OFFSET = 12
CONTROL_PROJECTION_FLAG_OFFSET = 16
CONTROL_BASE_PHASE_OFFSET = 20


Q4_EXACT_MACROS = r"""
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
"""


def q4_exact_chunk_body_asm(label_prefix: str) -> str:
    return (
        f"""
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
	Q4_EXACT_LANE_ZOL .L{label_prefix}_lane0_start, .L{label_prefix}_lane0_end

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
	Q4_EXACT_LANE_ZOL .L{label_prefix}_lane1_start, .L{label_prefix}_lane1_end
"""
    ).strip()


def q4_exact_function_asm(symbol: str, save_restore: bool) -> str:
    save_line = "\tSAVE_Q4_CALL_STATE\n" if save_restore else ""
    restore_line = "\tRESTORE_Q4_CALL_STATE\n" if save_restore else ""
    return (
        f"""
	.section	.text.{symbol},"ax",@progbits
	.globl	{symbol}
	.p2align	4
	.type	{symbol},@function
{symbol}:
	// p0=target float[32], p1=packed Q4NX chunk, p2=activation bf16[256].
	// packed layout: scales[256 bf16], offsets[256 bf16], data[4096 bytes].
{save_line}{q4_exact_chunk_body_asm(symbol)}
{restore_line}	ret	lr
	.size	{symbol}, .-{symbol}
"""
    ).strip()


def q4_exact_assembly(symbol: str, save_restore: bool) -> str:
    return Q4_EXACT_MACROS.strip() + "\n\n" + q4_exact_function_asm(symbol, save_restore) + "\n"


def call_with_delay(target: str) -> list[str]:
    return [
        f"\tjl\t#{target}",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
    ]


def jump_delay_lines() -> list[str]:
    return [
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
    ]


def long_branch_delay_lines() -> list[str]:
    return ["\tnop"] * 64


def lock_delay_lines() -> list[str]:
    return [
        "\tnop",
        "\tnop",
        "\tnop",
    ]


def scalar_delay_lines() -> list[str]:
    return [
        "\tnop",
        "\tnop",
        "\tnop",
    ]


def predicate_delay_lines() -> list[str]:
    return ["\tnop"] * 16


def record_store_delay_lines() -> list[str]:
    return ["\tnop"] * 32


def save_lr_frame_lines() -> list[str]:
    return [
        "\tpaddxm\t[sp], #0x40",
        "\tst\tlr, [sp, #-0x40]",
    ]


def restore_lr_frame_lines() -> list[str]:
    return [
        "\tlda\tlr, [sp, #-0x40]",
        "\tpaddxm\t[sp], #-0x40",
    ]


def return_with_delay(restore_lr: bool = False) -> list[str]:
    lines: list[str] = []
    if restore_lr:
        lines.extend(restore_lr_frame_lines())
    lines.extend(
        [
            "\tret\tlr",
            "\tnop",
            "\tnop",
            "\tnop",
            "\tnop",
            "\tnop",
        ]
    )
    return lines


def clear_accum_lines(accum_register: str = "r30") -> list[str]:
    lines = [
        f"\tmovs\tp6, {accum_register}",
        "\tmovxm\tm0, #0x20",
    ]
    for group in range(4):
        lines.extend(f"\tst\tr29, [p6, #{offset}]" for offset in range(0, 8 * 4, 4))
        if group != 3:
            lines.append("\tpadda\t[p6], m0")
    return lines


def load_phase_config_lines(load_records: bool = False) -> list[str]:
    lines = [f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}"]
    if load_records:
        lines.append(f"\tlda\tr12, [p6, #{CONTROL_RECORDS_OFFSET}]")
    lines.extend(
        [
            f"\tlda\tr23, [p6, #{CONTROL_PACKET_OFFSET}]",
            f"\tlda\tr24, [p6, #{CONTROL_PROJECTION_FLAG_OFFSET}]",
            f"\tlda\tr25, [p6, #{CONTROL_BASE_PHASE_OFFSET}]",
            *scalar_delay_lines(),
        ]
    )
    return lines


def store_phase_config_lines(
    records: int,
    chunks_per_record: int,
    packet: int,
    projection_flag: int,
    base_phase: int,
) -> list[str]:
    return [
        f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}",
        f"\tmova\tr12, #{records}",
        f"\tst\tr12, [p6, #{CONTROL_RECORDS_OFFSET}]",
        f"\tmova\tr12, #{chunks_per_record}",
        f"\tst\tr12, [p6, #{CONTROL_CHUNKS_PER_RECORD_OFFSET}]",
        f"\tmova\tr12, #{packet}",
        f"\tst\tr12, [p6, #{CONTROL_PACKET_OFFSET}]",
        f"\tmova\tr12, #{projection_flag}",
        f"\tst\tr12, [p6, #{CONTROL_PROJECTION_FLAG_OFFSET}]",
        f"\tmova\tr12, #{base_phase}",
        f"\tst\tr12, [p6, #{CONTROL_BASE_PHASE_OFFSET}]",
        *scalar_delay_lines(),
    ]


def emit_symbol(symbol: str, lines: list[str]) -> list[str]:
    return [
        f"\t.section\t.text.{symbol},\"ax\",@progbits",
        f"\t.globl\t{symbol}",
        "\t.p2align\t4",
        f"\t.type\t{symbol},@function",
        f"{symbol}:",
        *lines,
        f"\t.size\t{symbol}, .-{symbol}",
        "",
    ]


def q4_run_call_lines(q4_body_symbol: str, chunk_counter: str) -> list[str]:
    return [
        "// Select DMA1 weight and DMA0 activation ping/pong from pinned main16 buffers.",
        "\tmova\tr15, #1",
        f"\tand\tr27, {chunk_counter}, r15",
        *predicate_delay_lines(),
        "\tmova\tr2, #12",
        "\tlshl\tr8, r27, r2",
        *scalar_delay_lines(),
        "\tmova\tr2, #11",
        "\tlshl\tr9, r27, r2",
        *scalar_delay_lines(),
        "\tadd\tr8, r8, r9",
        *scalar_delay_lines(),
        f"\tmovxm\tr9, #0x{MAIN16_WT_PING_ABS_ADDR:x}",
        "\tadd\tr8, r8, r9",
        *scalar_delay_lines(),
        "\tmova\tr2, #14",
        "\tlshl\tr9, r27, r2",
        *scalar_delay_lines(),
        f"\tmovxm\tr6, #0x{MAIN16_CHUNK_PING_ABS_ADDR:x}",
        "\tadd\tr9, r9, r6",
        *scalar_delay_lines(),
        "\tmovs\tp1, r8",
        "\tmovs\tp2, r9",
        "\tmovs\tp0, r30",
        *q4_exact_chunk_body_asm(q4_body_symbol).splitlines(),
    ]


def record_emit_path_lines(
    emit_record_symbol: str,
    record_abs_addr: int,
    record_parity: int,
) -> list[str]:
    return [
        f"\tmova\tr5, #{record_parity}",
        f"\tmovxm\tp0, #0x{record_abs_addr:x}",
        *load_phase_config_lines(load_records=False),
        *scalar_delay_lines(),
        "\tand\tr7, r24, r5",
        *scalar_delay_lines(),
        "\tadd\tr0, r25, r7",
        *scalar_delay_lines(),
        "\tmov\tr1, r18",
        "\tmov\tr10, r23",
        "\tmov\tr3, r16",
        "\tmov\tr4, r17",
        f"\tmovxm\tp6, #0x{MAIN16_ACCUM_ABS_ADDR:x}",
        *call_with_delay(emit_record_symbol),
        *record_store_delay_lines(),
        "\tmova\tr15, #1",
        "\trel\t#53, r15",
        *lock_delay_lines(),
        "\tadd\tr18, r18, #1",
        *scalar_delay_lines(),
        f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}",
        f"\tst\tr18, [p6, #{CONTROL_RECORD_COUNTER_OFFSET}]",
        f"\tlda\tr12, [p6, #{CONTROL_RECORDS_OFFSET}]",
        *scalar_delay_lines(),
        "\teq\tr0, r18, r12",
        *scalar_delay_lines(),
        "\tjz\tr0, #.Llayer_q4_record_loop",
        *jump_delay_lines(),
        *return_with_delay(restore_lr=True),
    ]


def main16_layer_q4_run_body_asm(q4_body_symbol: str, emit_record_symbol: str) -> list[str]:
    return [
        *save_lr_frame_lines(),
        "// Shared Q4 run body.",
        f"// Phase config lives in tile-local control words at 0x{MAIN16_CONTROL_ADDR:04x}.",
        "\tmova\tr18, #0",
        f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}",
        f"\tst\tr18, [p6, #{CONTROL_RECORD_COUNTER_OFFSET}]",
        *scalar_delay_lines(),
        ".Llayer_q4_record_loop:",
        "\tmova\tr5, #0",
        ".Llayer_q4_chunk_loop:",
        "\tmovx\tr14, #-1",
        "\tacq\t#49, r14",
        *lock_delay_lines(),
        "\tacq\t#51, r14",
        *lock_delay_lines(),
        *q4_run_call_lines(q4_body_symbol, "r5"),
        "\tmova\tr15, #1",
        "\trel\t#48, r15",
        *lock_delay_lines(),
        "\trel\t#50, r15",
        *lock_delay_lines(),
        "\tadd\tr5, r5, #1",
        *scalar_delay_lines(),
        f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}",
        f"\tlda\tr13, [p6, #{CONTROL_CHUNKS_PER_RECORD_OFFSET}]",
        *scalar_delay_lines(),
        "\teq\tr0, r5, r13",
        *scalar_delay_lines(),
        "\tjz\tr0, #.Llayer_q4_chunk_loop",
        *jump_delay_lines(),
        "\tmovx\tr14, #-1",
        "\tacq\t#52, r14",
        *lock_delay_lines(),
        f"\tmovxm\tp6, #0x{MAIN16_CONTROL_ABS_ADDR:x}",
        f"\tlda\tr18, [p6, #{CONTROL_RECORD_COUNTER_OFFSET}]",
        *scalar_delay_lines(),
        "\tmova\tr15, #1",
        "\tand\tr0, r18, r15",
        *predicate_delay_lines(),
        "\tjz\tr0, #.Llayer_record_ping_path",
        *long_branch_delay_lines(),
        ".Llayer_record_pong_path:",
        *record_emit_path_lines(emit_record_symbol, MAIN16_RECORD_PONG_ABS_ADDR, 1),
        ".Llayer_record_ping_path:",
        *record_emit_path_lines(emit_record_symbol, MAIN16_RECORD_PING_ABS_ADDR, 0),
    ]


def main16_layer_emit_record_asm() -> list[str]:
    return [
        "// p0=selected record buffer, r0=phase, r1=block, r3=group, r4=row, r10=packet id.",
        "// p6=32-lane FP32 projection accumulator.",
        "\tmova\tr2, #24",
        "\tlshl\tr8, r0, r2",
        *scalar_delay_lines(),
        "\tmova\tr2, #20",
        "\tlshl\tr9, r1, r2",
        *scalar_delay_lines(),
        "\tor\tr8, r8, r9",
        *scalar_delay_lines(),
        "\tmova\tr2, #16",
        "\tlshl\tr9, r3, r2",
        *scalar_delay_lines(),
        "\tor\tr8, r8, r9",
        *scalar_delay_lines(),
        "\tmova\tr2, #8",
        "\tlshl\tr9, r4, r2",
        *scalar_delay_lines(),
        "\tor\tr8, r8, r9",
        *scalar_delay_lines(),
        "\tor\tr8, r8, r10",
        *scalar_delay_lines(),
        "\tmov\tp7, p0",
        "\tst\tr8, [p7], #4",
        "\tvlda\tbmll0, [p6, #0]",
        "\tvst.conv.bf16.fp32\tbmll0, [p7, #0]",
        "\tvlda\tbmll0, [p6, #0x40]",
        "\tvst.conv.bf16.fp32\tbmll0, [p7, #0x20]",
        *clear_accum_lines(),
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        "\tnop",
        *return_with_delay(),
    ]


def main16_phase_body_asm(
    q4_body_symbol: str,
    runs: tuple[tuple[int, int, int, int, int], ...],
) -> list[str]:
    lines = [*save_lr_frame_lines()]
    for records, chunks_per_record, packet, projection_flag, base_phase in runs:
        lines.extend(
            [
                *store_phase_config_lines(records, chunks_per_record, packet, projection_flag, base_phase),
                *call_with_delay(q4_body_symbol),
            ]
        )
    lines.extend(return_with_delay(restore_lr=True))
    return lines


def main16_layer_scheduler_asm(
    entry_symbol: str = "q4nx_main16_layer_scheduler",
    accum_addr: int = MAIN16_ACCUM_ABS_ADDR,
) -> str:
    dispatcher_symbol = "q4nx_main16_layer_dispatcher"
    q4_body_symbol = "q4nx_main16_layer_q4_body"
    emit_record_symbol = "q4nx_main16_layer_emit_record"
    body_qkv_symbol = "q4nx_main16_layer_body_qkv"
    body_o_symbol = "q4nx_main16_layer_body_o"
    body_upgate_symbol = "q4nx_main16_layer_body_upgate"
    body_down_symbol = "q4nx_main16_layer_body_down"
    lines: list[str] = []
    lines.extend(
        emit_symbol(
            entry_symbol,
            [
                *save_lr_frame_lines(),
                "\tmov\tr16, r0",
                "\tmov\tr17, r1",
                "\tmov\tr31, r3",
                "\tmov\tr19, p0",
                "\tmov\tr20, p1",
                "\tmov\tr21, p2",
                "\tmov\tr22, p3",
                f"\tmovxm\tr30, #0x{accum_addr:x}",
                "\tmova\tr29, #0",
                *clear_accum_lines(),
                *call_with_delay(dispatcher_symbol),
                *return_with_delay(restore_lr=True),
            ],
        )
    )
    lines.extend(
        emit_symbol(
            dispatcher_symbol,
            [
                *save_lr_frame_lines(),
                "\tst\tr31, [sp, #-0x3c]",
                *call_with_delay(body_qkv_symbol),
                "\tlda\tr31, [sp, #-0x3c]",
                "\tmova\tr15, #3",
                *scalar_delay_lines(),
                "\teq\tr27, r31, r15",
                *predicate_delay_lines(),
                "\tjnz\tr27, #.Llayer_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(body_o_symbol),
                "\tlda\tr31, [sp, #-0x3c]",
                "\tmova\tr15, #4",
                *scalar_delay_lines(),
                "\teq\tr27, r31, r15",
                *predicate_delay_lines(),
                "\tjnz\tr27, #.Llayer_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(body_upgate_symbol),
                "\tlda\tr31, [sp, #-0x3c]",
                "\tmova\tr15, #6",
                *scalar_delay_lines(),
                "\teq\tr27, r31, r15",
                *predicate_delay_lines(),
                "\tjnz\tr27, #.Llayer_dispatch_done",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                "\tnop",
                *call_with_delay(body_down_symbol),
                ".Llayer_dispatch_done:",
                *return_with_delay(restore_lr=True),
            ],
        )
    )
    lines.extend(emit_symbol(q4_body_symbol, main16_layer_q4_run_body_asm(q4_body_symbol, emit_record_symbol)))
    lines.extend(emit_symbol(emit_record_symbol, main16_layer_emit_record_asm()))
    lines.extend(
        emit_symbol(
            body_qkv_symbol,
            main16_phase_body_asm(
                q4_body_symbol,
                (
                    (8, 16, 10, 0, 0),
                    (2, 16, 11, 0, 1),
                    (2, 16, 12, 0, 2),
                ),
            ),
        )
    )
    lines.extend(emit_symbol(body_o_symbol, main16_phase_body_asm(q4_body_symbol, ((8, 16, 13, 0, 3),))))
    lines.extend(emit_symbol(body_upgate_symbol, main16_phase_body_asm(q4_body_symbol, ((48, 16, 14, 1, 4),))))
    lines.extend(emit_symbol(body_down_symbol, main16_phase_body_asm(q4_body_symbol, ((8, 48, 15, 0, 6),))))
    return "\n".join(lines).rstrip() + "\n"
