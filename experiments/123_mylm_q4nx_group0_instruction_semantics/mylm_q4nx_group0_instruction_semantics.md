# MyLM Q4NX Group0 Instruction Semantics

This experiment annotates the first MyLM Q4NX hot-loop section at
instruction-slot granularity. It is a learning artifact for writing a
future generator; it does not change the active IRON kernel.

## Checks

- Group range: `0x260..0x52a`
- Parsed instruction slots: `192`
- `vmac.f`: `28`
- `vextbcst.16`: `32`
- `vups.4x`: `8`
- `vconv.bf16.fp32`: `16`
- Boundary1 data cells: `27`
- Boundary1 control cells: `12`
- Boundary1 cells produced by group0: `27`
- Boundary1 entry cells carried through: `12`
- Boundary1 group0-produced data/control cells: `23` / `4`
- Boundary1 entry data/control cells: `4` / `8`

## Interpretation

- Group0 is pipeline fill. It does not close over a complete Q4NX group by itself.
- The useful learning unit is the slot-level def/use trace plus the boundary1 live state.
- A production generator must preserve the live cells crossing into group1 before changing the arithmetic body.

## Role Counts

| Role | Slots |
| --- | ---: |
| `register_move` | 51 |
| `activation_lane_broadcast` | 32 |
| `mac_accumulate` | 28 |
| `vector_arith` | 17 |
| `bf16_coeff` | 16 |
| `unpacked_q4` | 12 |
| `vector_load` | 11 |
| `expanded_q4` | 8 |
| `nop` | 7 |
| `add.nc` | 3 |
| `lshl` | 2 |
| `lda.s16` | 2 |
| `movx` | 1 |
| `mov` | 1 |
| `scalar_broadcast` | 1 |

## Key Opcode Counts

| Op | Slots |
| --- | ---: |
| `vmac.f` | 28 |
| `vextbcst.16` | 32 |
| `vups.4x` | 8 |
| `vunpack` | 12 |
| `vconv.bf16.fp32` | 16 |
| `vlda` | 5 |
| `vldb` | 6 |
| `lda.s16` | 2 |
| `vbcst.16` | 1 |
| `vst` | 0 |

## Boundary1 Live State

| Producer Kind | Cells |
| --- | ---: |
| `entry` | 12 |
| `unpacked_q4` | 6 |
| `accumulator_carry` | 4 |
| `vector_arith` | 3 |
| `bf16_coeff` | 3 |
| `vector_load` | 3 |
| `add.nc` | 3 |
| `register_move` | 2 |
| `activation_lane` | 2 |
| `lda.s16` | 1 |

- Cells produced inside group0 and live into group1: `27`

| Cell | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | ---: | ---: |
| `r16` | `g0@0x260.3:add.nc` | `g0@0x284.2:add.nc` | `g7@0x17d4.0:mov` | 2 | 1493 |
| `p5` | `g0@0x26c.3:add.nc` | `g0@0x4dc.0:vldb` | `g7@0x17ca.0:vldb` | 8 | 1486 |
| `p4` | `g0@0x284.2:add.nc` | `g0@0x4ec.1:vldb` | `g7@0x17de.0:vldb` | 8 | 1485 |
| `r4` | `entry` | `g0@0x30a.1:vmul.f` | `g7@0x1838.0:vmac.f` | 272 | 1481 |
| `acc0.bmhh` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `r5` | `entry` | `g0@0x2ae.1:vsub.f` | `g7@0x1754.1:vsub.f` | 64 | 1438 |
| `r0` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1742.1:vadd` | 64 | 1437 |
| `s0` | `entry` | `g0@0x29e.0:vups.4x` | `g7@0x1742.0:vups.4x` | 64 | 1437 |
| `upssign0` | `entry` | `g0@0x29e.0:vups.4x` | `g7@0x1742.0:vups.4x` | 64 | 1437 |
| `unpacksign0` | `entry` | `g0@0x26c.1:vunpack` | `g7@0x16ce.0:vunpack` | 64 | 1422 |
| `p0` | `entry` | `g0@0x260.0:vlda` | `g7@0x15e6.0:vldb` | 32 | 1364 |
| `p1` | `entry` | `g0@0x260.1:vldb` | `g7@0x156e.0:paddb` | 9 | 1328 |
| `p3` | `g0@0x4d6.0:lda.s16` | `g1@0x78a.0:lda.s16` | `g1@0x78a.0:lda.s16` | 1 | 189 |

## Group0 Instruction Table

| Address | Slot | Op | Role | Def Cells | Use Cells And Producers | Instruction |
| --- | ---: | --- | --- | --- | --- | --- |
| `0x260` | 0 | `vlda` | `vector_load` | `vec8.lo, vec8.hi` | `p0<-entry:entry` | `vlda	 x8, [p0], #0x40` |
| `0x260` | 1 | `vldb` | `vector_load` | `vec11.lo, vec11.hi` | `p1<-entry:entry` | `vldb	 x11, [p1], #0x40` |
| `0x260` | 2 | `lshl` | `lshl` | `` | `r18<-entry:entry; r19<-entry:entry; r2<-entry:entry` | `lshl	 r18, r19, r2` |
| `0x260` | 3 | `add.nc` | `add.nc` | `r16` | `r16<-entry:entry; r17<-entry:entry; r1<-entry:entry` | `add.nc	r16, r17, r1` |
| `0x26c` | 0 | `vlda` | `vector_load` | `vec7.lo, vec7.hi` | `p0<-entry:entry` | `vlda	 x7, [p0], #0x40` |
| `0x26c` | 1 | `vunpack` | `unpacked_q4` | `vec9.lo, vec9.hi` | `vec8.lo<-vector_load:g0@0x260.0:vlda; unpacksign0<-entry:entry` | `vunpack	x9, wl8, unpacksign0` |
| `0x26c` | 2 | `lshl` | `lshl` | `` | `r20<-entry:entry; r19<-entry:entry; r3<-entry:entry` | `lshl	 r20, r19, r3` |
| `0x26c` | 3 | `add.nc` | `add.nc` | `p5` | `p5<-entry:entry; r17<-entry:entry; r18<-entry:entry` | `add.nc	p5, r17, r18` |
| `0x278` | 0 | `vlda` | `vector_load` | `vec0.lo, vec0.hi` | `p0<-entry:entry` | `vlda	 x0, [p0], #0x40` |
| `0x278` | 1 | `vunpack` | `unpacked_q4` | `vec8.lo, vec8.hi` | `vec8.hi<-vector_load:g0@0x260.0:vlda; unpacksign0<-entry:entry` | `vunpack	x8, wh8, unpacksign0` |
| `0x278` | 2 | `movx` | `movx` | `r19` | `` | `movx	r19, #0x10` |
| `0x278` | 3 | `mov` | `mov` | `dj0` | `r20<-entry:entry` | `mov	dj0, r20` |
| `0x284` | 0 | `vlda` | `vector_load` | `lfh0` | `p2<-entry:entry; dj0<-mov:g0@0x278.3:mov` | `vlda	 lfh0, [p2, dj0]` |
| `0x284` | 1 | `vunpack` | `unpacked_q4` | `vec5.lo, vec5.hi` | `vec7.lo<-vector_load:g0@0x26c.0:vlda; unpacksign0<-entry:entry` | `vunpack	x5, wl7, unpacksign0` |
| `0x284` | 2 | `add.nc` | `add.nc` | `p4` | `p4<-entry:entry; r18<-entry:entry; r16<-add.nc:g0@0x260.3:add.nc` | `add.nc	p4, r18, r16` |
| `0x28e` | 0 | `lda.s16` | `lda.s16` | `r7, p3` | `p3<-entry:entry` | `lda.s16	 r7, [p3], #0x2` |
| `0x28e` | 1 | `vunpack` | `unpacked_q4` | `vec10.lo, vec10.hi` | `vec7.hi<-vector_load:g0@0x26c.0:vlda; unpacksign0<-entry:entry` | `vunpack	x10, wh7, unpacksign0` |
| `0x294` | 0 | `vunpack` | `unpacked_q4` | `vec1.lo, vec1.hi` | `vec0.lo<-vector_load:g0@0x278.0:vlda; unpacksign0<-entry:entry` | `vunpack	x1, wl0, unpacksign0` |
| `0x298` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x29a` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec9.lo, vec9.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x9, x11, #0x1` |
| `0x29e` | 0 | `vups.4x` | `expanded_q4` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `vec9.lo<-activation_lane_broadcast:g0@0x29a.0:vextbcst.16; vec9.hi<-activation_lane_broadcast:g0@0x29a.0:vextbcst.16; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm2, x9, s0, upssign0` |
| `0x29e` | 1 | `vadd` | `vector_arith` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc2.bmll<-expanded_q4:g0@0x29e.0:vups.4x; acc2.bmlh<-expanded_q4:g0@0x29e.0:vups.4x; acc2.bmhl<-expanded_q4:g0@0x29e.0:vups.4x; acc2.bmhh<-expanded_q4:g0@0x29e.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm1, dm2, dm0, r0` |
| `0x2a6` | 0 | `vups.4x` | `expanded_q4` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `vec8.lo<-unpacked_q4:g0@0x278.1:vunpack; vec8.hi<-unpacked_q4:g0@0x278.1:vunpack; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm2, x8, s0, upssign0` |
| `0x2a6` | 1 | `vadd` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc2.bmll<-expanded_q4:g0@0x2a6.0:vups.4x; acc2.bmlh<-expanded_q4:g0@0x2a6.0:vups.4x; acc2.bmhl<-expanded_q4:g0@0x2a6.0:vups.4x; acc2.bmhh<-expanded_q4:g0@0x2a6.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm2, dm2, dm0, r0` |
| `0x2ae` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x0` |
| `0x2ae` | 1 | `vsub.f` | `vector_arith` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-vector_arith:g0@0x29e.1:vadd; acc1.bmlh<-vector_arith:g0@0x29e.1:vadd; acc1.bmhl<-vector_arith:g0@0x29e.1:vadd; acc1.bmhh<-vector_arith:g0@0x29e.1:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm1, dm1, dm0, r5` |
| `0x2b6` | 0 | `vsub.f` | `vector_arith` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc2.bmll<-vector_arith:g0@0x2a6.1:vadd; acc2.bmlh<-vector_arith:g0@0x2a6.1:vadd; acc2.bmhl<-vector_arith:g0@0x2a6.1:vadd; acc2.bmhh<-vector_arith:g0@0x2a6.1:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm3, dm2, dm0, r5` |
| `0x2c2` | 0 | `vmov` | `register_move` | `acc2.bmll` | `acc1.bmll<-vector_arith:g0@0x2ae.1:vsub.f` | `vmov	bmll2, bmll1` |
| `0x2c6` | 0 | `vmov` | `register_move` | `acc2.bmlh` | `acc1.bmlh<-vector_arith:g0@0x2ae.1:vsub.f` | `vmov	bmlh2, bmlh1` |
| `0x2ca` | 0 | `vmov` | `register_move` | `acc2.bmhl` | `acc1.bmhl<-vector_arith:g0@0x2ae.1:vsub.f` | `vmov	bmhl2, bmhl1` |
| `0x2ce` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec5.lo, vec5.hi` | `acc2.bmll<-register_move:g0@0x2c2.0:vmov; acc2.bmlh<-register_move:g0@0x2c6.0:vmov` | `vconv.bf16.fp32	 x5, cml2` |
| `0x2ce` | 1 | `vmov` | `register_move` | `acc2.bmhh` | `acc1.bmhh<-vector_arith:g0@0x2ae.1:vsub.f` | `vmov	bmhh2, bmhh1` |
| `0x2d6` | 0 | `vups.4x` | `expanded_q4` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `vec5.lo<-bf16_coeff:g0@0x2ce.0:vconv.bf16.fp32; vec5.hi<-bf16_coeff:g0@0x2ce.0:vconv.bf16.fp32; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm1, x5, s0, upssign0` |
| `0x2d6` | 1 | `vadd` | `vector_arith` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc1.bmll<-expanded_q4:g0@0x2d6.0:vups.4x; acc1.bmlh<-expanded_q4:g0@0x2d6.0:vups.4x; acc1.bmhl<-expanded_q4:g0@0x2d6.0:vups.4x; acc1.bmhh<-expanded_q4:g0@0x2d6.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm4, dm1, dm0, r0` |
| `0x2de` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec3.lo, vec3.hi` | `acc2.bmhl<-register_move:g0@0x2ca.0:vmov; acc2.bmhh<-register_move:g0@0x2ce.1:vmov` | `vconv.bf16.fp32	 x3, cmh2` |
| `0x2de` | 1 | `vmov` | `register_move` | `vec6.lo` | `vec5.hi<-bf16_coeff:g0@0x2ce.0:vconv.bf16.fp32` | `vmov	wl6, wh5` |
| `0x2e6` | 0 | `vmov` | `register_move` | `acc1.bmhl` | `acc3.bmhl<-vector_arith:g0@0x2b6.0:vsub.f` | `vmov	bmhl1, bmhl3` |
| `0x2e6` | 1 | `vsub.f` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc4.bmll<-vector_arith:g0@0x2d6.1:vadd; acc4.bmlh<-vector_arith:g0@0x2d6.1:vadd; acc4.bmhl<-vector_arith:g0@0x2d6.1:vadd; acc4.bmhh<-vector_arith:g0@0x2d6.1:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm2, dm4, dm0, r5` |
| `0x2ee` | 0 | `vmov` | `register_move` | `vec2.lo` | `vec3.hi<-bf16_coeff:g0@0x2de.0:vconv.bf16.fp32` | `vmov	wl2, wh3` |
| `0x2f2` | 0 | `vmov` | `register_move` | `acc1.bmhh` | `acc3.bmhh<-vector_arith:g0@0x2b6.0:vsub.f` | `vmov	bmhh1, bmhh3` |
| `0x2f6` | 0 | `vldb` | `vector_load` | `vec6.lo, vec6.hi` | `p0<-entry:entry` | `vldb	 x6, [p0], #0x40` |
| `0x2fa` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec8.lo, vec8.hi` | `acc1.bmhl<-register_move:g0@0x2e6.0:vmov; acc1.bmhh<-register_move:g0@0x2f2.0:vmov` | `vconv.bf16.fp32	 x8, cmh1` |
| `0x2fa` | 1 | `vmov` | `register_move` | `acc1.bmll` | `acc3.bmll<-vector_arith:g0@0x2b6.0:vsub.f` | `vmov	bmll1, bmll3` |
| `0x302` | 0 | `vups.4x` | `expanded_q4` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `vec10.lo<-unpacked_q4:g0@0x28e.1:vunpack; vec10.hi<-unpacked_q4:g0@0x28e.1:vunpack; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm3, x10, s0, upssign0` |
| `0x302` | 1 | `vadd` | `vector_arith` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-expanded_q4:g0@0x302.0:vups.4x; acc3.bmlh<-expanded_q4:g0@0x302.0:vups.4x; acc3.bmhl<-expanded_q4:g0@0x302.0:vups.4x; acc3.bmhh<-expanded_q4:g0@0x302.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm3, dm3, dm0, r0` |
| `0x30a` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec10.lo, vec10.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x10, x11, #0x2` |
| `0x30a` | 1 | `vmul.f` | `vector_arith` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `vec5.lo<-bf16_coeff:g0@0x2ce.0:vconv.bf16.fp32; vec5.hi<-bf16_coeff:g0@0x2ce.0:vconv.bf16.fp32; vec7.lo<-activation_lane_broadcast:g0@0x2ae.0:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x2ae.0:vextbcst.16; r4<-entry:entry` | `vmul.f	dm3, x5, x7, r4` |
| `0x312` | 0 | `vmov` | `register_move` | `acc1.bmlh` | `acc3.bmlh<-vector_arith:g0@0x30a.1:vmul.f` | `vmov	bmlh1, bmlh3` |
| `0x312` | 1 | `vsub.f` | `vector_arith` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc3.bmll<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmlh<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmhl<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmhh<-vector_arith:g0@0x30a.1:vmul.f; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm4, dm3, dm0, r5` |
| `0x31a` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x3` |
| `0x31e` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec4.lo, vec4.hi` | `acc1.bmll<-register_move:g0@0x2fa.1:vmov; acc1.bmlh<-register_move:g0@0x312.0:vmov` | `vconv.bf16.fp32	 x4, cml1` |
| `0x31e` | 1 | `vmov` | `register_move` | `acc1.bmhh` | `acc2.bmhh<-vector_arith:g0@0x2e6.1:vsub.f` | `vmov	bmhh1, bmhh2` |
| `0x31e` | 2 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmlh<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmhl<-vector_arith:g0@0x30a.1:vmul.f; acc3.bmhh<-vector_arith:g0@0x30a.1:vmul.f; vec6.lo<-vector_load:g0@0x2f6.0:vldb; vec6.hi<-vector_load:g0@0x2f6.0:vldb; vec9.lo<-activation_lane_broadcast:g0@0x29a.0:vextbcst.16; vec9.hi<-activation_lane_broadcast:g0@0x29a.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x6, x9, r4` |
| `0x328` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec9.lo, vec9.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x9, x11, #0x8` |
| `0x32c` | 0 | `vmov` | `register_move` | `vec10.lo` | `vec4.hi<-bf16_coeff:g0@0x31e.0:vconv.bf16.fp32` | `vmov	wl10, wh4` |
| `0x330` | 0 | `vmov` | `register_move` | `acc1.bmll` | `acc2.bmll<-vector_arith:g0@0x2e6.1:vsub.f` | `vmov	bmll1, bmll2` |
| `0x330` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x31e.2:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x31e.2:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x31e.2:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x31e.2:vmac.f; vec3.lo<-bf16_coeff:g0@0x2de.0:vconv.bf16.fp32; vec3.hi<-bf16_coeff:g0@0x2de.0:vconv.bf16.fp32; vec10.lo<-register_move:g0@0x32c.0:vmov; vec10.hi<-activation_lane_broadcast:g0@0x30a.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x3, x10, r4` |
| `0x338` | 0 | `vmov` | `register_move` | `acc2.bmll` | `acc4.bmll<-vector_arith:g0@0x312.1:vsub.f` | `vmov	bmll2, bmll4` |
| `0x33c` | 0 | `vunpack` | `unpacked_q4` | `vec1.lo, vec1.hi` | `vec0.hi<-vector_load:g0@0x278.0:vlda; unpacksign0<-entry:entry` | `vunpack	x1, wh0, unpacksign0` |
| `0x33c` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x4` |
| `0x342` | 0 | `vmov` | `register_move` | `acc1.bmlh` | `acc2.bmlh<-vector_arith:g0@0x2e6.1:vsub.f` | `vmov	bmlh1, bmlh2` |
| `0x342` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x330.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x330.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x330.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x330.1:vmac.f; vec2.lo<-register_move:g0@0x2ee.0:vmov; vec2.hi<-entry:entry; vec7.lo<-activation_lane_broadcast:g0@0x33c.1:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x33c.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x2, x7, r4` |
| `0x34a` | 0 | `vmov` | `register_move` | `acc2.bmlh` | `acc4.bmlh<-vector_arith:g0@0x312.1:vsub.f` | `vmov	bmlh2, bmlh4` |
| `0x34e` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec5.lo, vec5.hi` | `acc1.bmll<-register_move:g0@0x330.0:vmov; acc1.bmlh<-register_move:g0@0x342.0:vmov` | `vconv.bf16.fp32	 x5, cml1` |
| `0x34e` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec4.lo, vec4.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x4, x11, #0x5` |
| `0x356` | 0 | `vmov` | `register_move` | `acc1.bmhl` | `acc2.bmhl<-vector_arith:g0@0x2e6.1:vsub.f` | `vmov	bmhl1, bmhl2` |
| `0x356` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x342.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x342.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x342.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x342.1:vmac.f; vec4.lo<-activation_lane_broadcast:g0@0x34e.1:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x34e.1:vextbcst.16; vec7.lo<-activation_lane_broadcast:g0@0x33c.1:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x33c.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x4, x7, r4` |
| `0x35e` | 0 | `vmov` | `register_move` | `acc2.bmhl` | `acc4.bmhl<-vector_arith:g0@0x312.1:vsub.f` | `vmov	bmhl2, bmhl4` |
| `0x362` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec2.lo, vec2.hi` | `acc1.bmhl<-register_move:g0@0x356.0:vmov; acc1.bmhh<-register_move:g0@0x31e.1:vmov` | `vconv.bf16.fp32	 x2, cmh1` |
| `0x362` | 1 | `vups.4x` | `expanded_q4` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `vec1.lo<-unpacked_q4:g0@0x33c.0:vunpack; vec1.hi<-unpacked_q4:g0@0x33c.0:vunpack; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm4, x1, s0, upssign0` |
| `0x362` | 2 | `vadd` | `vector_arith` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc4.bmll<-expanded_q4:g0@0x362.1:vups.4x; acc4.bmlh<-expanded_q4:g0@0x362.1:vups.4x; acc4.bmhl<-expanded_q4:g0@0x362.1:vups.4x; acc4.bmhh<-expanded_q4:g0@0x362.1:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm1, dm4, dm0, r0` |
| `0x36c` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec3.lo, vec3.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x3, x11, #0x6` |
| `0x370` | 0 | `vmov` | `register_move` | `acc2.bmhh` | `acc4.bmhh<-expanded_q4:g0@0x362.1:vups.4x` | `vmov	bmhh2, bmhh4` |
| `0x370` | 1 | `vsub.f` | `vector_arith` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc1.bmll<-vector_arith:g0@0x362.2:vadd; acc1.bmlh<-vector_arith:g0@0x362.2:vadd; acc1.bmhl<-vector_arith:g0@0x362.2:vadd; acc1.bmhh<-vector_arith:g0@0x362.2:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm4, dm1, dm0, r5` |
| `0x378` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec0.lo, vec0.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x0, x11, #0x7` |
| `0x378` | 1 | `vmac.f` | `mac_accumulate` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x356.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x356.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x356.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x356.1:vmac.f; vec10.lo<-register_move:g0@0x32c.0:vmov; vec10.hi<-activation_lane_broadcast:g0@0x30a.0:vextbcst.16; vec4.lo<-activation_lane_broadcast:g0@0x34e.1:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x34e.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm4, dm3, x10, x4, r4` |
| `0x380` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec4.lo, vec4.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x4, x11, #0x9` |
| `0x384` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec8.lo, vec8.hi` | `acc2.bmll<-register_move:g0@0x338.0:vmov; acc2.bmlh<-register_move:g0@0x34a.0:vmov` | `vconv.bf16.fp32	 x8, cml2` |
| `0x384` | 1 | `vmov` | `register_move` | `vec3.lo` | `vec8.hi<-bf16_coeff:g0@0x384.0:vconv.bf16.fp32` | `vmov	wl3, wh8` |
| `0x384` | 2 | `vmov.d` | `register_move` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x378.1:vmac.f` | `vmov.d	dm1, dm4` |
| `0x38e` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0xa` |
| `0x38e` | 1 | `vmac.f` | `mac_accumulate` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x378.1:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x378.1:vmac.f; vec8.lo<-bf16_coeff:g0@0x384.0:vconv.bf16.fp32; vec8.hi<-bf16_coeff:g0@0x384.0:vconv.bf16.fp32; vec3.lo<-register_move:g0@0x384.1:vmov; vec3.hi<-activation_lane_broadcast:g0@0x36c.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm4, dm4, x8, x3, r4` |
| `0x396` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec10.lo, vec10.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x10, x11, #0xb` |
| `0x396` | 1 | `vmov.d` | `register_move` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc1.bmll<-register_move:g0@0x384.2:vmov.d; acc1.bmlh<-register_move:g0@0x384.2:vmov.d; acc1.bmhl<-register_move:g0@0x384.2:vmov.d; acc1.bmhh<-register_move:g0@0x384.2:vmov.d` | `vmov.d	dm3, dm1` |
| `0x39e` | 0 | `vmov` | `register_move` | `vec3.lo` | `vec2.hi<-bf16_coeff:g0@0x362.0:vconv.bf16.fp32` | `vmov	wl3, wh2` |
| `0x3a2` | 0 | `vmac.f` | `mac_accumulate` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x38e.1:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x38e.1:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x38e.1:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x38e.1:vmac.f; vec3.lo<-register_move:g0@0x39e.0:vmov; vec3.hi<-activation_lane_broadcast:g0@0x36c.0:vextbcst.16; vec0.lo<-activation_lane_broadcast:g0@0x378.0:vextbcst.16; vec0.hi<-activation_lane_broadcast:g0@0x378.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm4, dm4, x3, x0, r4` |
| `0x3a6` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x3a8` | 0 | `vmov` | `register_move` | `vec5.lo` | `vec5.hi<-bf16_coeff:g0@0x34e.0:vconv.bf16.fp32` | `vmov	wl5, wh5` |
| `0x3ac` | 0 | `vmov` | `register_move` | `vec9.lo` | `vec8.hi<-bf16_coeff:g0@0x384.0:vconv.bf16.fp32` | `vmov	wl9, wh8` |
| `0x3ac` | 1 | `vmac.f` | `mac_accumulate` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x3a2.0:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x3a2.0:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x3a2.0:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x3a2.0:vmac.f; vec5.lo<-register_move:g0@0x3a8.0:vmov; vec5.hi<-bf16_coeff:g0@0x34e.0:vconv.bf16.fp32; vec9.lo<-register_move:g0@0x3ac.0:vmov; vec9.hi<-activation_lane_broadcast:g0@0x328.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm4, dm4, x5, x9, r4` |
| `0x3b4` | 0 | `vmov` | `register_move` | `acc1.bmhl` | `acc3.bmhl<-register_move:g0@0x396.1:vmov.d` | `vmov	bmhl1, bmhl3` |
| `0x3b8` | 0 | `vmov` | `register_move` | `acc1.bmhh` | `acc3.bmhh<-register_move:g0@0x396.1:vmov.d` | `vmov	bmhh1, bmhh3` |
| `0x3bc` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec1.lo, vec1.hi` | `acc2.bmhl<-register_move:g0@0x35e.0:vmov; acc2.bmhh<-register_move:g0@0x370.0:vmov` | `vconv.bf16.fp32	 x1, cmh2` |
| `0x3bc` | 1 | `vmov` | `register_move` | `acc1.bmll` | `acc3.bmll<-register_move:g0@0x396.1:vmov.d` | `vmov	bmll1, bmll3` |
| `0x3bc` | 2 | `vmac.f` | `mac_accumulate` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x3ac.1:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x3ac.1:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x3ac.1:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x3ac.1:vmac.f; vec5.lo<-register_move:g0@0x3a8.0:vmov; vec5.hi<-bf16_coeff:g0@0x34e.0:vconv.bf16.fp32; vec4.lo<-activation_lane_broadcast:g0@0x380.0:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x380.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm4, dm4, x5, x4, r4` |
| `0x3c6` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec0.lo, vec0.hi` | `acc1.bmhl<-register_move:g0@0x3b4.0:vmov; acc1.bmhh<-register_move:g0@0x3b8.0:vmov` | `vconv.bf16.fp32	 x0, cmh1` |
| `0x3c6` | 1 | `vups.4x` | `expanded_q4` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `vec1.lo<-bf16_coeff:g0@0x3bc.0:vconv.bf16.fp32; vec1.hi<-bf16_coeff:g0@0x3bc.0:vconv.bf16.fp32; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm3, x1, s0, upssign0` |
| `0x3c6` | 2 | `vadd` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc3.bmll<-expanded_q4:g0@0x3c6.1:vups.4x; acc3.bmlh<-expanded_q4:g0@0x3c6.1:vups.4x; acc3.bmhl<-expanded_q4:g0@0x3c6.1:vups.4x; acc3.bmhh<-expanded_q4:g0@0x3c6.1:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm2, dm3, dm0, r0` |
| `0x3d0` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec4.lo, vec4.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x4, x11, #0xc` |
| `0x3d4` | 0 | `vunpack` | `unpacked_q4` | `vec4.lo, vec4.hi` | `vec6.lo<-vector_load:g0@0x2f6.0:vldb; unpacksign0<-entry:entry` | `vunpack	x4, wl6, unpacksign0` |
| `0x3d4` | 1 | `vmov` | `register_move` | `acc1.bmlh` | `acc3.bmlh<-expanded_q4:g0@0x3c6.1:vups.4x` | `vmov	bmlh1, bmlh3` |
| `0x3d4` | 2 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc4.bmll<-mac_accumulate:g0@0x3bc.2:vmac.f; acc4.bmlh<-mac_accumulate:g0@0x3bc.2:vmac.f; acc4.bmhl<-mac_accumulate:g0@0x3bc.2:vmac.f; acc4.bmhh<-mac_accumulate:g0@0x3bc.2:vmac.f; vec2.lo<-bf16_coeff:g0@0x362.0:vconv.bf16.fp32; vec2.hi<-bf16_coeff:g0@0x362.0:vconv.bf16.fp32; vec7.lo<-activation_lane_broadcast:g0@0x38e.0:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x38e.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm4, x2, x7, r4` |
| `0x3de` | 0 | `vunpack` | `unpacked_q4` | `vec6.lo, vec6.hi` | `vec6.hi<-vector_load:g0@0x2f6.0:vldb; unpacksign0<-entry:entry` | `vunpack	x6, wh6, unpacksign0` |
| `0x3de` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec5.lo, vec5.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x5, x11, #0xd` |
| `0x3de` | 2 | `vsub.f` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc2.bmll<-vector_arith:g0@0x3c6.2:vadd; acc2.bmlh<-vector_arith:g0@0x3c6.2:vadd; acc2.bmhl<-vector_arith:g0@0x3c6.2:vadd; acc2.bmhh<-vector_arith:g0@0x3c6.2:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm2, dm2, dm0, r5` |
| `0x3e8` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec2.lo, vec2.hi` | `acc1.bmll<-register_move:g0@0x3bc.1:vmov; acc1.bmlh<-register_move:g0@0x3d4.1:vmov` | `vconv.bf16.fp32	 x2, cml1` |
| `0x3e8` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec10.lo, vec10.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x10, x11, #0xe` |
| `0x3f0` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec3.lo, vec3.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x3, x11, #0xf` |
| `0x3f0` | 1 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x3d4.2:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x3d4.2:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x3d4.2:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x3d4.2:vmac.f; vec3.lo<-activation_lane_broadcast:g0@0x3f0.0:vextbcst.16; vec3.hi<-activation_lane_broadcast:g0@0x3f0.0:vextbcst.16; vec10.lo<-activation_lane_broadcast:g0@0x3e8.1:vextbcst.16; vec10.hi<-activation_lane_broadcast:g0@0x3e8.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm3, x3, x10, r4` |
| `0x3f8` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x10` |
| `0x3fc` | 0 | `vmov` | `register_move` | `vec8.lo` | `vec1.hi<-bf16_coeff:g0@0x3bc.0:vconv.bf16.fp32` | `vmov	wl8, wh1` |
| `0x400` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec4.lo, vec4.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x4, x11, #0x15` |
| `0x400` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x3f0.1:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x3f0.1:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x3f0.1:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x3f0.1:vmac.f; vec8.lo<-register_move:g0@0x3fc.0:vmov; vec8.hi<-bf16_coeff:g0@0x384.0:vconv.bf16.fp32; vec4.lo<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm1, x8, x4, r4` |
| `0x408` | 0 | `vups.4x` | `expanded_q4` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `vec4.lo<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm4, x4, s0, upssign0` |
| `0x408` | 1 | `vadd` | `vector_arith` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-expanded_q4:g0@0x408.0:vups.4x; acc4.bmlh<-expanded_q4:g0@0x408.0:vups.4x; acc4.bmhl<-expanded_q4:g0@0x408.0:vups.4x; acc4.bmhh<-expanded_q4:g0@0x408.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm4, dm4, dm0, r0` |
| `0x410` | 0 | `vmov` | `register_move` | `vec5.lo` | `vec2.hi<-bf16_coeff:g0@0x3e8.0:vconv.bf16.fp32` | `vmov	wl5, wh2` |
| `0x414` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec9.lo, vec9.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x9, x11, #0x11` |
| `0x414` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x400.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x400.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x400.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x400.1:vmac.f; vec9.lo<-activation_lane_broadcast:g0@0x414.0:vextbcst.16; vec9.hi<-activation_lane_broadcast:g0@0x414.0:vextbcst.16; vec5.lo<-register_move:g0@0x410.0:vmov; vec5.hi<-activation_lane_broadcast:g0@0x3de.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x9, x5, r4` |
| `0x41c` | 0 | `vsub.f` | `vector_arith` | `acc4.bmll, acc4.bmlh, acc4.bmhl, acc4.bmhh` | `acc4.bmll<-vector_arith:g0@0x408.1:vadd; acc4.bmlh<-vector_arith:g0@0x408.1:vadd; acc4.bmhl<-vector_arith:g0@0x408.1:vadd; acc4.bmhh<-vector_arith:g0@0x408.1:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm4, dm4, dm0, r5` |
| `0x420` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec1.lo, vec1.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x1, x11, #0x12` |
| `0x424` | 0 | `vmov` | `register_move` | `acc1.bmll` | `acc2.bmll<-vector_arith:g0@0x3de.2:vsub.f` | `vmov	bmll1, bmll2` |
| `0x424` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x414.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x414.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x414.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x414.1:vmac.f; vec1.lo<-activation_lane_broadcast:g0@0x420.0:vextbcst.16; vec1.hi<-activation_lane_broadcast:g0@0x420.0:vextbcst.16; vec10.lo<-activation_lane_broadcast:g0@0x3e8.1:vextbcst.16; vec10.hi<-activation_lane_broadcast:g0@0x3e8.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x1, x10, r4` |
| `0x42c` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec10.lo, vec10.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x10, x11, #0x18` |
| `0x430` | 0 | `vmov` | `register_move` | `vec3.lo` | `vec0.hi<-bf16_coeff:g0@0x3c6.0:vconv.bf16.fp32` | `vmov	wl3, wh0` |
| `0x434` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec8.lo, vec8.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x8, x11, #0x13` |
| `0x434` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x424.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x424.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x424.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x424.1:vmac.f; vec8.lo<-activation_lane_broadcast:g0@0x434.0:vextbcst.16; vec8.hi<-activation_lane_broadcast:g0@0x434.0:vextbcst.16; vec3.lo<-register_move:g0@0x430.0:vmov; vec3.hi<-activation_lane_broadcast:g0@0x3f0.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x8, x3, r4` |
| `0x43c` | 0 | `vmov` | `register_move` | `acc1.bmlh` | `acc2.bmlh<-vector_arith:g0@0x3de.2:vsub.f` | `vmov	bmlh1, bmlh2` |
| `0x440` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x14` |
| `0x444` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec2.lo, vec2.hi` | `acc1.bmll<-register_move:g0@0x424.0:vmov; acc1.bmlh<-register_move:g0@0x43c.0:vmov` | `vconv.bf16.fp32	 x2, cml1` |
| `0x444` | 1 | `vmov` | `register_move` | `acc1.bmhl` | `acc2.bmhl<-vector_arith:g0@0x3de.2:vsub.f` | `vmov	bmhl1, bmhl2` |
| `0x444` | 2 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x434.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x434.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x434.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x434.1:vmac.f; vec2.lo<-bf16_coeff:g0@0x444.0:vconv.bf16.fp32; vec2.hi<-bf16_coeff:g0@0x444.0:vconv.bf16.fp32; vec7.lo<-activation_lane_broadcast:g0@0x440.0:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x440.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x2, x7, r4` |
| `0x44e` | 0 | `vmov` | `register_move` | `acc1.bmhh` | `acc2.bmhh<-vector_arith:g0@0x3de.2:vsub.f` | `vmov	bmhh1, bmhh2` |
| `0x452` | 0 | `vups.4x` | `expanded_q4` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `vec6.lo<-unpacked_q4:g0@0x3de.0:vunpack; vec6.hi<-unpacked_q4:g0@0x3de.0:vunpack; s0<-entry:entry; upssign0<-entry:entry` | `vups.4x	dm1, x6, s0, upssign0` |
| `0x452` | 1 | `vadd` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc1.bmll<-expanded_q4:g0@0x452.0:vups.4x; acc1.bmlh<-expanded_q4:g0@0x452.0:vups.4x; acc1.bmhl<-expanded_q4:g0@0x452.0:vups.4x; acc1.bmhh<-expanded_q4:g0@0x452.0:vups.4x; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r0<-entry:entry` | `vadd	dm2, dm1, dm0, r0` |
| `0x45a` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec5.lo, vec5.hi` | `acc1.bmhl<-expanded_q4:g0@0x452.0:vups.4x; acc1.bmhh<-expanded_q4:g0@0x452.0:vups.4x` | `vconv.bf16.fp32	 x5, cmh1` |
| `0x45a` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec6.lo, vec6.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x6, x11, #0x16` |
| `0x45a` | 2 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x444.2:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x444.2:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x444.2:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x444.2:vmac.f; vec5.lo<-bf16_coeff:g0@0x45a.0:vconv.bf16.fp32; vec5.hi<-bf16_coeff:g0@0x45a.0:vconv.bf16.fp32; vec9.lo<-activation_lane_broadcast:g0@0x414.0:vextbcst.16; vec9.hi<-activation_lane_broadcast:g0@0x414.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x5, x9, r4` |
| `0x464` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec9.lo, vec9.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x9, x11, #0x17` |
| `0x464` | 1 | `vsub.f` | `vector_arith` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc2.bmll<-vector_arith:g0@0x452.1:vadd; acc2.bmlh<-vector_arith:g0@0x452.1:vadd; acc2.bmhl<-vector_arith:g0@0x452.1:vadd; acc2.bmhh<-vector_arith:g0@0x452.1:vadd; acc0.bmll<-entry:entry; acc0.bmlh<-entry:entry; acc0.bmhl<-entry:entry; acc0.bmhh<-entry:entry; r5<-entry:entry` | `vsub.f	dm2, dm2, dm0, r5` |
| `0x46c` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec0.lo, vec0.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x0, x11, #0x19` |
| `0x470` | 0 | `vmov` | `register_move` | `acc1.bmll` | `acc4.bmll<-vector_arith:g0@0x41c.0:vsub.f` | `vmov	bmll1, bmll4` |
| `0x470` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x45a.2:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x45a.2:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x45a.2:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x45a.2:vmac.f; vec0.lo<-activation_lane_broadcast:g0@0x46c.0:vextbcst.16; vec0.hi<-activation_lane_broadcast:g0@0x46c.0:vextbcst.16; vec1.lo<-activation_lane_broadcast:g0@0x420.0:vextbcst.16; vec1.hi<-activation_lane_broadcast:g0@0x420.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x0, x1, r4` |
| `0x478` | 0 | `vmov` | `register_move` | `acc4.bmll` | `lfh0<-vector_load:g0@0x284.0:vlda` | `vmov	bmll4, lfh0` |
| `0x47c` | 0 | `vmov` | `register_move` | `acc1.bmhh` | `acc4.bmhh<-vector_arith:g0@0x41c.0:vsub.f` | `vmov	bmhh1, bmhh4` |
| `0x480` | 0 | `vmov` | `register_move` | `vec8.lo` | `vec5.hi<-bf16_coeff:g0@0x45a.0:vconv.bf16.fp32` | `vmov	wl8, wh5` |
| `0x480` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x470.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x470.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x470.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x470.1:vmac.f; vec3.lo<-register_move:g0@0x430.0:vmov; vec3.hi<-activation_lane_broadcast:g0@0x3f0.0:vextbcst.16; vec8.lo<-register_move:g0@0x480.0:vmov; vec8.hi<-activation_lane_broadcast:g0@0x434.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x3, x8, r4` |
| `0x488` | 0 | `vmov` | `register_move` | `acc1.bmlh` | `acc4.bmlh<-vector_arith:g0@0x41c.0:vsub.f` | `vmov	bmlh1, bmlh4` |
| `0x48c` | 0 | `vmov` | `register_move` | `vec2.lo` | `vec2.hi<-bf16_coeff:g0@0x444.0:vconv.bf16.fp32` | `vmov	wl2, wh2` |
| `0x490` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec3.lo, vec3.hi` | `acc1.bmll<-register_move:g0@0x470.0:vmov; acc1.bmlh<-register_move:g0@0x488.0:vmov` | `vconv.bf16.fp32	 x3, cml1` |
| `0x490` | 1 | `vmov` | `register_move` | `acc1.bmhl` | `acc4.bmhl<-vector_arith:g0@0x41c.0:vsub.f` | `vmov	bmhl1, bmhl4` |
| `0x490` | 2 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x480.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x480.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x480.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x480.1:vmac.f; vec2.lo<-register_move:g0@0x48c.0:vmov; vec2.hi<-bf16_coeff:g0@0x444.0:vconv.bf16.fp32; vec7.lo<-activation_lane_broadcast:g0@0x440.0:vextbcst.16; vec7.hi<-activation_lane_broadcast:g0@0x440.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x2, x7, r4` |
| `0x49a` | 0 | `vmov` | `register_move` | `acc1.bmll` | `acc2.bmll<-vector_arith:g0@0x464.1:vsub.f` | `vmov	bmll1, bmll2` |
| `0x49e` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec1.lo, vec1.hi` | `acc1.bmhl<-register_move:g0@0x490.1:vmov; acc1.bmhh<-register_move:g0@0x47c.0:vmov` | `vconv.bf16.fp32	 x1, cmh1` |
| `0x49e` | 1 | `vmov` | `register_move` | `acc1.bmlh` | `acc2.bmlh<-vector_arith:g0@0x464.1:vsub.f` | `vmov	bmlh1, bmlh2` |
| `0x4a6` | 0 | `vmov` | `register_move` | `acc1.bmhl` | `acc2.bmhl<-vector_arith:g0@0x464.1:vsub.f` | `vmov	bmhl1, bmhl2` |
| `0x4a6` | 1 | `vmac.f` | `mac_accumulate` | `acc3.bmll, acc3.bmlh, acc3.bmhl, acc3.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x490.2:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x490.2:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x490.2:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x490.2:vmac.f; vec2.lo<-register_move:g0@0x48c.0:vmov; vec2.hi<-bf16_coeff:g0@0x444.0:vconv.bf16.fp32; vec4.lo<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; vec4.hi<-activation_lane_broadcast:g0@0x400.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm3, dm3, x2, x4, r4` |
| `0x4ae` | 0 | `vmov` | `register_move` | `acc1.bmhh` | `acc2.bmhh<-vector_arith:g0@0x464.1:vsub.f` | `vmov	bmhh1, bmhh2` |
| `0x4b2` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec5.lo, vec5.hi` | `acc1.bmll<-register_move:g0@0x49a.0:vmov; acc1.bmlh<-register_move:g0@0x49e.1:vmov` | `vconv.bf16.fp32	 x5, cml1` |
| `0x4b2` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec2.lo, vec2.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x2, x11, #0x1a` |
| `0x4ba` | 0 | `vconv.bf16.fp32` | `bf16_coeff` | `vec4.lo, vec4.hi` | `acc1.bmhl<-register_move:g0@0x4a6.0:vmov; acc1.bmhh<-register_move:g0@0x4ae.0:vmov` | `vconv.bf16.fp32	 x4, cmh1` |
| `0x4ba` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec6.lo, vec6.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x6, x11, #0x1b` |
| `0x4ba` | 2 | `vmac.f` | `mac_accumulate` | `acc2.bmll, acc2.bmlh, acc2.bmhl, acc2.bmhh` | `acc3.bmll<-mac_accumulate:g0@0x4a6.1:vmac.f; acc3.bmlh<-mac_accumulate:g0@0x4a6.1:vmac.f; acc3.bmhl<-mac_accumulate:g0@0x4a6.1:vmac.f; acc3.bmhh<-mac_accumulate:g0@0x4a6.1:vmac.f; vec5.lo<-bf16_coeff:g0@0x4b2.0:vconv.bf16.fp32; vec5.hi<-bf16_coeff:g0@0x4b2.0:vconv.bf16.fp32; vec6.lo<-activation_lane_broadcast:g0@0x4ba.1:vextbcst.16; vec6.hi<-activation_lane_broadcast:g0@0x4ba.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm2, dm3, x5, x6, r4` |
| `0x4c4` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec7.lo, vec7.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x7, x11, #0x1c` |
| `0x4c8` | 0 | `vmov` | `register_move` | `vec8.lo` | `vec1.hi<-bf16_coeff:g0@0x49e.0:vconv.bf16.fp32` | `vmov	wl8, wh1` |
| `0x4cc` | 0 | `vmov` | `register_move` | `vec9.lo` | `vec5.hi<-bf16_coeff:g0@0x4b2.0:vconv.bf16.fp32` | `vmov	wl9, wh5` |
| `0x4cc` | 1 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc2.bmll<-mac_accumulate:g0@0x4ba.2:vmac.f; acc2.bmlh<-mac_accumulate:g0@0x4ba.2:vmac.f; acc2.bmhl<-mac_accumulate:g0@0x4ba.2:vmac.f; acc2.bmhh<-mac_accumulate:g0@0x4ba.2:vmac.f; vec8.lo<-register_move:g0@0x4c8.0:vmov; vec8.hi<-activation_lane_broadcast:g0@0x434.0:vextbcst.16; vec9.lo<-register_move:g0@0x4cc.0:vmov; vec9.hi<-activation_lane_broadcast:g0@0x464.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm2, x8, x9, r4` |
| `0x4d4` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x4d6` | 0 | `lda.s16` | `lda.s16` | `r7, p3` | `p3<-lda.s16:g0@0x28e.0:lda.s16` | `lda.s16	 r7, [p3], #0x2` |
| `0x4d6` | 1 | `vmov` | `register_move` | `vec3.lo` | `vec3.hi<-bf16_coeff:g0@0x490.0:vconv.bf16.fp32` | `vmov	wl3, wh3` |
| `0x4dc` | 0 | `vldb` | `vector_load` | `vec2.lo` | `p5<-add.nc:g0@0x26c.3:add.nc` | `vldb	 wl2, [p5], #0x40` |
| `0x4dc` | 1 | `vextbcst.16` | `activation_lane_broadcast` | `vec10.lo, vec10.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x10, x11, #0x1d` |
| `0x4dc` | 2 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x4cc.1:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x4cc.1:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x4cc.1:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x4cc.1:vmac.f; vec3.lo<-register_move:g0@0x4d6.1:vmov; vec3.hi<-bf16_coeff:g0@0x490.0:vconv.bf16.fp32; vec10.lo<-activation_lane_broadcast:g0@0x4dc.1:vextbcst.16; vec10.hi<-activation_lane_broadcast:g0@0x4dc.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm1, x3, x10, r4` |
| `0x4e6` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x4e8` | 0 | `vextbcst.16` | `activation_lane_broadcast` | `vec0.lo, vec0.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x0, x11, #0x1e` |
| `0x4ec` | 0 | `vlda` | `vector_load` | `vec8.lo, vec8.hi` | `p0<-entry:entry` | `vlda	 x8, [p0], #0x40` |
| `0x4ec` | 1 | `vldb` | `vector_load` | `vec6.lo` | `p4<-add.nc:g0@0x284.2:add.nc` | `vldb	 wl6, [p4], #0x40` |
| `0x4ec` | 2 | `vextbcst.16` | `activation_lane_broadcast` | `vec11.lo, vec11.hi` | `vec11.lo<-vector_load:g0@0x260.1:vldb; vec11.hi<-vector_load:g0@0x260.1:vldb` | `vextbcst.16	 x11, x11, #0x1f` |
| `0x4ec` | 3 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x4dc.2:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x4dc.2:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x4dc.2:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x4dc.2:vmac.f; vec3.lo<-register_move:g0@0x4d6.1:vmov; vec3.hi<-bf16_coeff:g0@0x490.0:vconv.bf16.fp32; vec0.lo<-activation_lane_broadcast:g0@0x4e8.0:vextbcst.16; vec0.hi<-activation_lane_broadcast:g0@0x4e8.0:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm1, x3, x0, r4` |
| `0x4f8` | 0 | `vmov` | `register_move` | `vec3.lo` | `vec4.hi<-bf16_coeff:g0@0x4ba.0:vconv.bf16.fp32` | `vmov	wl3, wh4` |
| `0x4fc` | 0 | `vbcst.16` | `scalar_broadcast` | `vec1.lo, vec1.hi` | `r7<-lda.s16:g0@0x4d6.0:lda.s16` | `vbcst.16	 x1, r7` |
| `0x500` | 0 | `vldb` | `vector_load` | `vec7.lo, vec7.hi` | `p0<-entry:entry` | `vldb	 x7, [p0], #0x40` |
| `0x500` | 1 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x4ec.3:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x4ec.3:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x4ec.3:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x4ec.3:vmac.f; vec1.lo<-scalar_broadcast:g0@0x4fc.0:vbcst.16; vec1.hi<-scalar_broadcast:g0@0x4fc.0:vbcst.16; vec2.lo<-vector_load:g0@0x4dc.0:vldb; vec2.hi<-activation_lane_broadcast:g0@0x4b2.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm1, x1, x2, r4` |
| `0x508` | 0 | `vunpack` | `unpacked_q4` | `vec5.lo, vec5.hi` | `vec7.lo<-vector_load:g0@0x500.0:vldb; unpacksign0<-entry:entry` | `vunpack	x5, wl7, unpacksign0` |
| `0x50c` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x50e` | 0 | `vunpack` | `unpacked_q4` | `vec9.lo, vec9.hi` | `vec8.lo<-vector_load:g0@0x4ec.0:vlda; unpacksign0<-entry:entry` | `vunpack	x9, wl8, unpacksign0` |
| `0x50e` | 1 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x500.1:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x500.1:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x500.1:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x500.1:vmac.f; vec8.lo<-vector_load:g0@0x4ec.0:vlda; vec8.hi<-vector_load:g0@0x4ec.0:vlda; vec6.lo<-vector_load:g0@0x4ec.1:vldb; vec6.hi<-activation_lane_broadcast:g0@0x4ba.1:vextbcst.16; r4<-entry:entry` | `vmac.f	dm1, dm1, x8, x6, r4` |
| `0x516` | 0 | `vunpack` | `unpacked_q4` | `vec8.lo, vec8.hi` | `vec8.hi<-vector_load:g0@0x4ec.0:vlda; unpacksign0<-entry:entry` | `vunpack	x8, wh8, unpacksign0` |
| `0x51a` | 0 | `vunpack` | `unpacked_q4` | `vec10.lo, vec10.hi` | `vec7.hi<-vector_load:g0@0x500.0:vldb; unpacksign0<-entry:entry` | `vunpack	x10, wh7, unpacksign0` |
| `0x51e` | 0 | `vldb` | `vector_load` | `vec0.lo, vec0.hi` | `p0<-entry:entry` | `vldb	 x0, [p0], #0x40` |
| `0x51e` | 1 | `vmac.f` | `mac_accumulate` | `acc1.bmll, acc1.bmlh, acc1.bmhl, acc1.bmhh` | `acc1.bmll<-mac_accumulate:g0@0x50e.1:vmac.f; acc1.bmlh<-mac_accumulate:g0@0x50e.1:vmac.f; acc1.bmhl<-mac_accumulate:g0@0x50e.1:vmac.f; acc1.bmhh<-mac_accumulate:g0@0x50e.1:vmac.f; vec5.lo<-unpacked_q4:g0@0x508.0:vunpack; vec5.hi<-unpacked_q4:g0@0x508.0:vunpack; vec7.lo<-vector_load:g0@0x500.0:vldb; vec7.hi<-vector_load:g0@0x500.0:vldb; r4<-entry:entry` | `vmac.f	dm1, dm1, x5, x7, r4` |
| `0x526` | 0 | `nop` | `nop` | `` | `` | `nop` |
| `0x528` | 0 | `nop` | `nop` | `` | `` | `nop` |
