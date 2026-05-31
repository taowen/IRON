# MyLM Q4NX Cell Liveness

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This report tracks the live ranges of vector-half, accumulator-quadrant,
pointer, and scalar cells across the full `0x260..0x1850` MyLM Q4NX hot
loop. It is a learning artifact for assembly codegen; it does not change
the active IRON kernel.

## Summary

- Parsed slots: `1532`
- Live ranges with at least one use: `2922`
- Data live ranges: `2872`
- Cross-group live ranges: `196`
- Unused definitions: `467`

## Boundary Pressure

| Boundary Into Group | Data Cells | Pointer/Scalar Cells |
| ---: | ---: | ---: |
| 1 | 27 | 12 |
| 2 | 27 | 12 |
| 3 | 27 | 12 |
| 4 | 27 | 12 |
| 5 | 27 | 12 |
| 6 | 27 | 12 |
| 7 | 27 | 12 |

## Producer Kinds

| Producer Kind | Live Ranges |
| --- | ---: |
| `accumulator_carry` | 966 |
| `vector_arith` | 457 |
| `register_move` | 437 |
| `activation_lane` | 340 |
| `expanded_q4` | 256 |
| `bf16_coeff` | 200 |
| `unpacked_q4` | 106 |
| `vector_load` | 97 |
| `entry` | 25 |
| `scalar_broadcast` | 16 |
| `lda.s16` | 15 |
| `add.nc` | 3 |
| `mov` | 3 |
| `paddb` | 1 |

## Key Opcode Counts

| Op | Count |
| --- | ---: |
| `vmac.f` | 264 |
| `vextbcst.16` | 256 |
| `vups.4x` | 64 |
| `vunpack` | 64 |
| `vconv.bf16.fp32` | 136 |
| `vlda` | 11 |
| `vldb` | 46 |
| `lda.s16` | 8 |
| `vbcst.16` | 8 |
| `vst` | 0 |

## Longest Data Live Ranges

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `vec11.hi` | `vector_load` | `g0@0x260.1:vldb` | `g0@0x29a.0:vextbcst.16` | `g0@0x4ec.2:vextbcst.16` | 32 | 175 |
| `vec11.lo` | `vector_load` | `g0@0x260.1:vldb` | `g0@0x29a.0:vextbcst.16` | `g0@0x4ec.2:vextbcst.16` | 32 | 175 |
| `vec11.hi` | `vector_load` | `g1@0x52a.0:vldb` | `g1@0x54a.0:vmac.f` | `g1@0x7a0.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g2@0x7de.0:vldb` | `g2@0x7fe.0:vmac.f` | `g2@0xa54.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g3@0xa92.0:vldb` | `g3@0xab2.0:vmac.f` | `g3@0xd08.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g4@0xd46.0:vldb` | `g4@0xd66.0:vmac.f` | `g4@0xfbc.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g5@0xffa.0:vldb` | `g5@0x101a.0:vmac.f` | `g5@0x1270.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g6@0x12ae.0:vldb` | `g6@0x12ce.0:vmac.f` | `g6@0x1526.2:vextbcst.16` | 33 | 173 |
| `vec11.hi` | `vector_load` | `g7@0x1566.0:vldb` | `g7@0x1588.0:vmac.f` | `g7@0x17de.1:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g1@0x52a.0:vldb` | `g1@0x54a.0:vmac.f` | `g1@0x7a0.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g2@0x7de.0:vldb` | `g2@0x7fe.0:vmac.f` | `g2@0xa54.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g3@0xa92.0:vldb` | `g3@0xab2.0:vmac.f` | `g3@0xd08.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g4@0xd46.0:vldb` | `g4@0xd66.0:vmac.f` | `g4@0xfbc.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g5@0xffa.0:vldb` | `g5@0x101a.0:vmac.f` | `g5@0x1270.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g6@0x12ae.0:vldb` | `g6@0x12ce.0:vmac.f` | `g6@0x1526.2:vextbcst.16` | 33 | 173 |
| `vec11.lo` | `vector_load` | `g7@0x1566.0:vldb` | `g7@0x1588.0:vmac.f` | `g7@0x17de.1:vextbcst.16` | 33 | 173 |
| `acc4.bmhh` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14bc.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `acc4.bmhl` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14c8.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `acc4.bmlh` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14b4.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `acc4.bmhh` | `vector_arith` | `g0@0x41c.0:vsub.f` | `g0@0x47c.0:vmov` | `g1@0x570.2:vmac.f` | 2 | 95 |
| `acc4.bmhh` | `vector_arith` | `g1@0x6ce.0:vsub.f` | `g1@0x738.0:vmov` | `g2@0x824.2:vmac.f` | 2 | 95 |
| `acc4.bmhh` | `vector_arith` | `g2@0x982.0:vsub.f` | `g2@0x9ec.0:vmov` | `g3@0xad8.2:vmac.f` | 2 | 95 |
| `acc4.bmhh` | `vector_arith` | `g3@0xc36.0:vsub.f` | `g3@0xca0.0:vmov` | `g4@0xd8c.2:vmac.f` | 2 | 95 |
| `acc4.bmhh` | `vector_arith` | `g4@0xeea.0:vsub.f` | `g4@0xf54.0:vmov` | `g5@0x1040.2:vmac.f` | 2 | 95 |
| `acc4.bmhh` | `vector_arith` | `g5@0x119e.0:vsub.f` | `g5@0x1208.0:vmov` | `g6@0x12f4.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g0@0x41c.0:vsub.f` | `g0@0x490.1:vmov` | `g1@0x570.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g1@0x6ce.0:vsub.f` | `g1@0x744.0:vmov` | `g2@0x824.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g2@0x982.0:vsub.f` | `g2@0x9f8.0:vmov` | `g3@0xad8.2:vmac.f` | 2 | 95 |

## Boundary Samples

### Boundary Into Group 1

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g0@0x41c.0:vsub.f` | `g0@0x47c.0:vmov` | `g1@0x570.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g0@0x41c.0:vsub.f` | `g0@0x490.1:vmov` | `g1@0x570.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g0@0x41c.0:vsub.f` | `g0@0x488.0:vmov` | `g1@0x570.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g0@0x4b2.1:vextbcst.16` | `g0@0x500.1:vmac.f` | `g1@0x5f4.1:vmac.f` | 3 | 92 |
| `acc4.bmll` | `register_move` | `g0@0x478.0:vmov` | `g1@0x570.2:vmac.f` | `g1@0x570.2:vmac.f` | 1 | 71 |
| `vec0.hi` | `vector_load` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` | `g1@0x5ee.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g0@0x4ba.1:vextbcst.16` | `g0@0x4ba.2:vmac.f` | `g1@0x580.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g0@0x4ba.0:vconv.bf16.fp32` | `g0@0x4f8.0:vmov` | `g1@0x570.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g0@0x490.0:vconv.bf16.fp32` | `g0@0x4d6.1:vmov` | `g1@0x54a.0:vmac.f` | 4 | 52 |
| `vec10.hi` | `unpacked_q4` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` | `g1@0x5b2.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` | `g1@0x5b2.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g0@0x4dc.0:vldb` | `g0@0x500.1:vmac.f` | `g1@0x570.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` | `g1@0x53a.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g0@0x4f8.0:vmov` | `g1@0x54a.0:vmac.f` | `g1@0x54a.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` | `g1@0x55e.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` | `g1@0x53a.0:vups.4x` | 1 | 10 |

### Boundary Into Group 2

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g1@0x6ce.0:vsub.f` | `g1@0x738.0:vmov` | `g2@0x824.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g1@0x6ce.0:vsub.f` | `g1@0x744.0:vmov` | `g2@0x824.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g1@0x6ce.0:vsub.f` | `g1@0x730.0:vmov` | `g2@0x824.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g1@0x764.1:vextbcst.16` | `g1@0x7b4.1:vmac.f` | `g2@0x8a8.1:vmac.f` | 3 | 92 |
| `vec0.hi` | `vector_load` | `g1@0x7d2.0:vldb` | `g2@0x7ee.1:vmac.f` | `g2@0x8a2.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g1@0x76c.1:vextbcst.16` | `g1@0x76c.2:vmac.f` | `g2@0x834.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g1@0x76c.0:vconv.bf16.fp32` | `g1@0x7ac.0:vmov` | `g2@0x824.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g1@0x73c.0:vconv.bf16.fp32` | `g1@0x78a.1:vmov` | `g2@0x7fe.0:vmac.f` | 4 | 53 |
| `acc4.bmll` | `register_move` | `g1@0x786.0:vmov` | `g2@0x824.2:vmac.f` | `g2@0x824.2:vmac.f` | 1 | 47 |
| `vec10.hi` | `unpacked_q4` | `g1@0x7ce.0:vunpack` | `g2@0x7de.1:vmac.f` | `g2@0x866.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g1@0x7ce.0:vunpack` | `g2@0x7de.1:vmac.f` | `g2@0x866.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g1@0x790.0:vldb` | `g1@0x7b4.1:vmac.f` | `g2@0x824.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g1@0x76c.0:vconv.bf16.fp32` | `g2@0x7ee.1:vmac.f` | `g2@0x7ee.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g1@0x7ac.0:vmov` | `g2@0x7fe.0:vmac.f` | `g2@0x7fe.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g1@0x7d2.0:vldb` | `g2@0x7ee.1:vmac.f` | `g2@0x812.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g1@0x7ca.0:vunpack` | `g2@0x7ee.0:vups.4x` | `g2@0x7ee.0:vups.4x` | 1 | 10 |

### Boundary Into Group 3

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g2@0x982.0:vsub.f` | `g2@0x9ec.0:vmov` | `g3@0xad8.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g2@0x982.0:vsub.f` | `g2@0x9f8.0:vmov` | `g3@0xad8.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g2@0x982.0:vsub.f` | `g2@0x9e4.0:vmov` | `g3@0xad8.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g2@0xa18.1:vextbcst.16` | `g2@0xa68.1:vmac.f` | `g3@0xb5c.1:vmac.f` | 3 | 92 |
| `vec0.hi` | `vector_load` | `g2@0xa86.0:vldb` | `g3@0xaa2.1:vmac.f` | `g3@0xb56.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g2@0xa20.1:vextbcst.16` | `g2@0xa20.2:vmac.f` | `g3@0xae8.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g2@0xa20.0:vconv.bf16.fp32` | `g2@0xa60.0:vmov` | `g3@0xad8.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g2@0x9f0.0:vconv.bf16.fp32` | `g2@0xa3e.1:vmov` | `g3@0xab2.0:vmac.f` | 4 | 53 |
| `acc4.bmll` | `register_move` | `g2@0xa3a.0:vmov` | `g3@0xad8.2:vmac.f` | `g3@0xad8.2:vmac.f` | 1 | 47 |
| `vec10.hi` | `unpacked_q4` | `g2@0xa82.0:vunpack` | `g3@0xa92.1:vmac.f` | `g3@0xb1a.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g2@0xa82.0:vunpack` | `g3@0xa92.1:vmac.f` | `g3@0xb1a.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g2@0xa44.0:vldb` | `g2@0xa68.1:vmac.f` | `g3@0xad8.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g2@0xa20.0:vconv.bf16.fp32` | `g3@0xaa2.1:vmac.f` | `g3@0xaa2.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g2@0xa60.0:vmov` | `g3@0xab2.0:vmac.f` | `g3@0xab2.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g2@0xa86.0:vldb` | `g3@0xaa2.1:vmac.f` | `g3@0xac6.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g2@0xa7e.0:vunpack` | `g3@0xaa2.0:vups.4x` | `g3@0xaa2.0:vups.4x` | 1 | 10 |

### Boundary Into Group 4

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g3@0xc36.0:vsub.f` | `g3@0xca0.0:vmov` | `g4@0xd8c.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g3@0xc36.0:vsub.f` | `g3@0xcac.0:vmov` | `g4@0xd8c.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g3@0xc36.0:vsub.f` | `g3@0xc98.0:vmov` | `g4@0xd8c.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g3@0xccc.1:vextbcst.16` | `g3@0xd1c.1:vmac.f` | `g4@0xe10.1:vmac.f` | 3 | 92 |
| `vec0.hi` | `vector_load` | `g3@0xd3a.0:vldb` | `g4@0xd56.1:vmac.f` | `g4@0xe0a.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g3@0xcd4.1:vextbcst.16` | `g3@0xcd4.2:vmac.f` | `g4@0xd9c.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g3@0xcd4.0:vconv.bf16.fp32` | `g3@0xd14.0:vmov` | `g4@0xd8c.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g3@0xca4.0:vconv.bf16.fp32` | `g3@0xcf2.1:vmov` | `g4@0xd66.0:vmac.f` | 4 | 53 |
| `acc4.bmll` | `register_move` | `g3@0xcee.0:vmov` | `g4@0xd8c.2:vmac.f` | `g4@0xd8c.2:vmac.f` | 1 | 47 |
| `vec10.hi` | `unpacked_q4` | `g3@0xd36.0:vunpack` | `g4@0xd46.1:vmac.f` | `g4@0xdce.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g3@0xd36.0:vunpack` | `g4@0xd46.1:vmac.f` | `g4@0xdce.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g3@0xcf8.0:vldb` | `g3@0xd1c.1:vmac.f` | `g4@0xd8c.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g3@0xcd4.0:vconv.bf16.fp32` | `g4@0xd56.1:vmac.f` | `g4@0xd56.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g3@0xd14.0:vmov` | `g4@0xd66.0:vmac.f` | `g4@0xd66.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g3@0xd3a.0:vldb` | `g4@0xd56.1:vmac.f` | `g4@0xd7a.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g3@0xd32.0:vunpack` | `g4@0xd56.0:vups.4x` | `g4@0xd56.0:vups.4x` | 1 | 10 |

### Boundary Into Group 5

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g4@0xeea.0:vsub.f` | `g4@0xf54.0:vmov` | `g5@0x1040.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g4@0xeea.0:vsub.f` | `g4@0xf60.0:vmov` | `g5@0x1040.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g4@0xeea.0:vsub.f` | `g4@0xf4c.0:vmov` | `g5@0x1040.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g4@0xf80.1:vextbcst.16` | `g4@0xfd0.1:vmac.f` | `g5@0x10c4.1:vmac.f` | 3 | 92 |
| `vec0.hi` | `vector_load` | `g4@0xfee.0:vldb` | `g5@0x100a.1:vmac.f` | `g5@0x10be.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g4@0xf88.1:vextbcst.16` | `g4@0xf88.2:vmac.f` | `g5@0x1050.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g4@0xf88.0:vconv.bf16.fp32` | `g4@0xfc8.0:vmov` | `g5@0x1040.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g4@0xf58.0:vconv.bf16.fp32` | `g4@0xfa6.1:vmov` | `g5@0x101a.0:vmac.f` | 4 | 53 |
| `acc4.bmll` | `register_move` | `g4@0xfa2.0:vmov` | `g5@0x1040.2:vmac.f` | `g5@0x1040.2:vmac.f` | 1 | 47 |
| `vec10.hi` | `unpacked_q4` | `g4@0xfea.0:vunpack` | `g5@0xffa.1:vmac.f` | `g5@0x1082.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g4@0xfea.0:vunpack` | `g5@0xffa.1:vmac.f` | `g5@0x1082.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g4@0xfac.0:vldb` | `g4@0xfd0.1:vmac.f` | `g5@0x1040.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g4@0xf88.0:vconv.bf16.fp32` | `g5@0x100a.1:vmac.f` | `g5@0x100a.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g4@0xfc8.0:vmov` | `g5@0x101a.0:vmac.f` | `g5@0x101a.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g4@0xfee.0:vldb` | `g5@0x100a.1:vmac.f` | `g5@0x102e.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g4@0xfe6.0:vunpack` | `g5@0x100a.0:vups.4x` | `g5@0x100a.0:vups.4x` | 1 | 10 |

### Boundary Into Group 6

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g5@0x119e.0:vsub.f` | `g5@0x1208.0:vmov` | `g6@0x12f4.2:vmac.f` | 2 | 95 |
| `acc4.bmhl` | `vector_arith` | `g5@0x119e.0:vsub.f` | `g5@0x1214.0:vmov` | `g6@0x12f4.2:vmac.f` | 2 | 95 |
| `acc4.bmlh` | `vector_arith` | `g5@0x119e.0:vsub.f` | `g5@0x1200.0:vmov` | `g6@0x12f4.2:vmac.f` | 2 | 95 |
| `vec2.hi` | `activation_lane` | `g5@0x1234.1:vextbcst.16` | `g5@0x1284.1:vmac.f` | `g6@0x1378.1:vmac.f` | 3 | 92 |
| `vec0.hi` | `vector_load` | `g5@0x12a2.0:vldb` | `g6@0x12be.1:vmac.f` | `g6@0x1372.0:vunpack` | 2 | 59 |
| `vec6.hi` | `activation_lane` | `g5@0x123c.1:vextbcst.16` | `g5@0x123c.2:vmac.f` | `g6@0x1304.1:vmac.f` | 3 | 57 |
| `vec4.hi` | `bf16_coeff` | `g5@0x123c.0:vconv.bf16.fp32` | `g5@0x127c.0:vmov` | `g6@0x12f4.2:vmac.f` | 3 | 54 |
| `vec3.hi` | `bf16_coeff` | `g5@0x120c.0:vconv.bf16.fp32` | `g5@0x125a.1:vmov` | `g6@0x12ce.0:vmac.f` | 4 | 53 |
| `acc4.bmll` | `register_move` | `g5@0x1256.0:vmov` | `g6@0x12f4.2:vmac.f` | `g6@0x12f4.2:vmac.f` | 1 | 47 |
| `vec10.hi` | `unpacked_q4` | `g5@0x129e.0:vunpack` | `g6@0x12ae.1:vmac.f` | `g6@0x1336.1:vups.4x` | 2 | 45 |
| `vec10.lo` | `unpacked_q4` | `g5@0x129e.0:vunpack` | `g6@0x12ae.1:vmac.f` | `g6@0x1336.1:vups.4x` | 2 | 45 |
| `vec2.lo` | `vector_load` | `g5@0x1260.0:vldb` | `g5@0x1284.1:vmac.f` | `g6@0x12f4.2:vmac.f` | 2 | 44 |
| `vec4.lo` | `bf16_coeff` | `g5@0x123c.0:vconv.bf16.fp32` | `g6@0x12be.1:vmac.f` | `g6@0x12be.1:vmac.f` | 1 | 38 |
| `vec3.lo` | `register_move` | `g5@0x127c.0:vmov` | `g6@0x12ce.0:vmac.f` | `g6@0x12ce.0:vmac.f` | 1 | 22 |
| `vec0.lo` | `vector_load` | `g5@0x12a2.0:vldb` | `g6@0x12be.1:vmac.f` | `g6@0x12e2.0:vunpack` | 2 | 18 |
| `vec8.hi` | `unpacked_q4` | `g5@0x129a.0:vunpack` | `g6@0x12be.0:vups.4x` | `g6@0x12be.0:vups.4x` | 1 | 10 |

### Boundary Into Group 7

- Data cells live across boundary: `27`
- Pointer/scalar cells live across boundary: `12`

| Cell | Producer Kind | Producer | First Use | Last Use | Uses | Span Slots |
| --- | --- | --- | --- | --- | ---: | ---: |
| `acc0.bmhh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmhl` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmlh` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc0.bmll` | `entry` | `entry` | `g0@0x29e.1:vadd` | `g7@0x1754.1:vsub.f` | 128 | 1442 |
| `acc4.bmhh` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14bc.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `acc4.bmhl` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14c8.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `acc4.bmlh` | `vector_arith` | `g6@0x1452.0:vsub.f` | `g6@0x14b4.0:vmov` | `g7@0x15ae.2:vmac.f` | 2 | 97 |
| `vec2.hi` | `activation_lane` | `g6@0x14e8.1:vextbcst.16` | `g6@0x153a.2:vmac.f` | `g7@0x1632.1:vmac.f` | 3 | 94 |
| `vec0.hi` | `vector_load` | `g6@0x155a.0:vldb` | `g7@0x1578.1:vmac.f` | `g7@0x162c.0:vunpack` | 2 | 60 |
| `vec6.hi` | `activation_lane` | `g6@0x14f0.1:vextbcst.16` | `g6@0x14f0.2:vmac.f` | `g7@0x15be.1:vmac.f` | 3 | 59 |
| `vec4.hi` | `bf16_coeff` | `g6@0x14f0.0:vconv.bf16.fp32` | `g6@0x1532.0:vmov` | `g7@0x15ae.2:vmac.f` | 3 | 56 |
| `vec3.hi` | `bf16_coeff` | `g6@0x14c0.0:vconv.bf16.fp32` | `g6@0x150e.1:vmov` | `g7@0x1588.0:vmac.f` | 4 | 55 |
| `acc4.bmll` | `register_move` | `g6@0x150a.0:vmov` | `g7@0x15ae.2:vmac.f` | `g7@0x15ae.2:vmac.f` | 1 | 49 |
| `vec10.hi` | `unpacked_q4` | `g6@0x1556.0:vunpack` | `g7@0x1566.1:vmac.f` | `g7@0x15f0.1:vups.4x` | 2 | 46 |
| `vec10.lo` | `unpacked_q4` | `g6@0x1556.0:vunpack` | `g7@0x1566.1:vmac.f` | `g7@0x15f0.1:vups.4x` | 2 | 46 |
| `vec2.lo` | `vector_load` | `g6@0x1514.0:vldb` | `g6@0x153a.2:vmac.f` | `g7@0x15ae.2:vmac.f` | 2 | 46 |
| `vec4.lo` | `bf16_coeff` | `g6@0x14f0.0:vconv.bf16.fp32` | `g7@0x1578.1:vmac.f` | `g7@0x1578.1:vmac.f` | 1 | 40 |
| `vec3.lo` | `register_move` | `g6@0x1532.0:vmov` | `g7@0x1588.0:vmac.f` | `g7@0x1588.0:vmac.f` | 1 | 24 |
| `vec0.lo` | `vector_load` | `g6@0x155a.0:vldb` | `g7@0x1578.1:vmac.f` | `g7@0x159c.0:vunpack` | 2 | 19 |
| `vec8.hi` | `unpacked_q4` | `g6@0x1552.0:vunpack` | `g7@0x1578.0:vups.4x` | `g7@0x1578.0:vups.4x` | 1 | 11 |

## What This Teaches

- MyLM's fast body is a register-residency schedule, not an opcode list.
- The steady groups keep dozens of data cells live across group boundaries, so a generator must model live state explicitly.
- A production rewrite should first reproduce these live ranges on a small numeric probe, then move the graph-derived body into main16.
