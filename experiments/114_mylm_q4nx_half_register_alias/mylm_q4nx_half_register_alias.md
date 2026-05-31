# MyLM Q4NX Half-Register Alias Trace

Source disasm: `/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s`

This report tracks producer state at vector-half and accumulator-quadrant granularity.

## Whole Hot Loop

- `vmac.f`: `264`
- Vector operands: `528`
- Mixed-half vector operands: `202`
- Vector operands with cross-group cells: `63`

| Vector Operand Producer Shape | Count |
| --- | ---: |
| `activation_lane` | 196 |
| `register_move+activation_lane` | 104 |
| `bf16_coeff` | 65 |
| `register_move+bf16_coeff` | 64 |
| `vector_load` | 36 |
| `unpacked_q4` | 28 |
| `vector_load+activation_lane` | 25 |
| `scalar_broadcast` | 9 |
| `register_move+entry` | 1 |

## Relation To Exp102

Exp102 already proved the primitive `vmac.f #0x33c` operand model on real NPU:

- q operand registers `x2/x3/x5/x7/x9` all produced first lane `1` when multiplied by a broadcast one vector;
- `vextbcst.16` lanes `0..31` produced first lanes `1..32`;
- the group-sum correction produced `64`, and summing all 32 lanes produced `528`.

So the remaining problem is not whether `#0x33c` is a valid BF16 MAC configuration. The remaining problem is preserving the scheduled half-register state that MyLM feeds into those MACs.

## Steady Group1

- `vmac.f`: `33`
- Mixed-half vector operands: `25` / `66`
- Vector operands with cross-group cells: `9` / `66`
- Accumulator operands with cross-group cells: `2` / `33`

| Vector Producer Pair | MAC Count |
| --- | ---: |
| `activation_lane + register_move+bf16_coeff` | 6 |
| `activation_lane + register_move+activation_lane` | 5 |
| `activation_lane + activation_lane` | 4 |
| `activation_lane + bf16_coeff` | 4 |
| `bf16_coeff + register_move+activation_lane` | 2 |
| `register_move+activation_lane + register_move+activation_lane` | 2 |
| `unpacked_q4 + unpacked_q4` | 1 |
| `bf16_coeff + vector_load` | 1 |
| `register_move+bf16_coeff + vector_load` | 1 |
| `bf16_coeff + vector_load+activation_lane` | 1 |
| `register_move+activation_lane + unpacked_q4` | 1 |
| `activation_lane + vector_load` | 1 |
| `register_move+activation_lane + register_move+bf16_coeff` | 1 |
| `scalar_broadcast + vector_load+activation_lane` | 1 |
| `vector_load + vector_load+activation_lane` | 1 |
| `unpacked_q4 + vector_load` | 1 |

## Boundary Cells Into Group1

| Cell | Producer | First Group1 Use |
| --- | --- | --- |
| `acc1.bmhh` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmhl` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmlh` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc1.bmll` | `g0@0x51e.1:vmac.f` | `g1@0x52a.1:vmac.f` |
| `acc4.bmhh` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmhl` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmlh` | `g0@0x41c.0:vsub.f` | `g1@0x570.2:vmac.f` |
| `acc4.bmll` | `g0@0x478.0:vmov` | `g1@0x570.2:vmac.f` |
| `p3` | `g0@0x4d6.0:lda.s16` | `g1@0x78a.0:lda.s16` |
| `p4` | `g0@0x284.2:add.nc` | `g1@0x7a0.1:vldb` |
| `p5` | `g0@0x26c.3:add.nc` | `g1@0x790.0:vldb` |
| `vec0.hi` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` |
| `vec0.lo` | `g0@0x51e.0:vldb` | `g1@0x53a.1:vmac.f` |
| `vec10.hi` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec10.lo` | `g0@0x51a.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec2.hi` | `g0@0x4b2.1:vextbcst.16` | `g1@0x570.2:vmac.f` |
| `vec2.lo` | `g0@0x4dc.0:vldb` | `g1@0x570.2:vmac.f` |
| `vec3.hi` | `g0@0x490.0:vconv.bf16.fp32` | `g1@0x54a.0:vmac.f` |
| `vec3.lo` | `g0@0x4f8.0:vmov` | `g1@0x54a.0:vmac.f` |
| `vec4.hi` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` |
| `vec4.lo` | `g0@0x4ba.0:vconv.bf16.fp32` | `g1@0x53a.1:vmac.f` |
| `vec6.hi` | `g0@0x4ba.1:vextbcst.16` | `g1@0x580.1:vmac.f` |
| `vec8.hi` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` |
| `vec8.lo` | `g0@0x516.0:vunpack` | `g1@0x53a.0:vups.4x` |
| `vec9.hi` | `g0@0x50e.0:vunpack` | `g1@0x52a.1:vmac.f` |
| `vec9.lo` | `g0@0x50e.0:vunpack` | `g1@0x52a.1:vmac.f` |

## Group1 MAC Operand Table

| MAC | Accumulator Source | Left Vector | Right Vector |
| --- | --- | --- | --- |
| `g1@0x52a.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry carry `g0@0x51e.1:vmac.f` (-4) | `x9` `vec9.lo+vec9.hi` <- unpacked_q4 carry `g0@0x50e.0:vunpack` (-9) | `x10` `vec10.lo+vec10.hi` <- unpacked_q4 carry `g0@0x51a.0:vunpack` (-6) |
| `g1@0x53a.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- vector_arith `g1@0x536.0:vadd` (-2) | `x4` `vec4.lo+vec4.hi` <- bf16_coeff carry `g0@0x4ba.0:vconv.bf16.fp32` (-38) | `x0` `vec0.lo+vec0.hi` <- vector_load carry `g0@0x51e.0:vldb` (-9) |
| `g1@0x54a.0:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x53a.1:vmac.f` (-3) | `x3` mixed:<br>`vec3.lo` <- register_move carry `g0@0x4f8.0:vmov` (-22)<br>`vec3.hi` <- bf16_coeff carry `g0@0x490.0:vconv.bf16.fp32` (-52) | `x11` `vec11.lo+vec11.hi` <- vector_load `g1@0x52a.0:vldb` (-8) |
| `g1@0x570.2:vmac.f` | `dm4` mixed:<br>`acc4.bmll` <- register_move carry `g0@0x478.0:vmov` (-71)<br>`acc4.bmlh` <- vector_arith carry `g0@0x41c.0:vsub.f` (-95)<br>`acc4.bmhl` <- vector_arith carry `g0@0x41c.0:vsub.f` (-95)<br>`acc4.bmhh` <- vector_arith carry `g0@0x41c.0:vsub.f` (-95) | `x4` mixed:<br>`vec4.lo` <- bf16_coeff `g1@0x564.0:vconv.bf16.fp32` (-5)<br>`vec4.hi` <- bf16_coeff carry `g0@0x4ba.0:vconv.bf16.fp32` (-54) | `x2` mixed:<br>`vec2.lo` <- vector_load carry `g0@0x4dc.0:vldb` (-44)<br>`vec2.hi` <- activation_lane #0x1a carry `g0@0x4b2.1:vextbcst.16` (-55) |
| `g1@0x580.1:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x570.2:vmac.f` (-4) | `x6` mixed:<br>`vec6.lo` <- register_move `g1@0x580.0:vmov` (-1)<br>`vec6.hi` <- activation_lane #0x1b carry `g0@0x4ba.1:vextbcst.16` (-57) | `x1` `vec1.lo+vec1.hi` <- unpacked_q4 `g1@0x55e.0:vunpack` (-11) |
| `g1@0x5d0.2:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- vector_arith `g1@0x5bc.1:vmul.f` (-6) | `x6` `vec6.lo+vec6.hi` <- vector_load `g1@0x5a8.0:vldb` (-13) | `x9` `vec9.lo+vec9.hi` <- activation_lane #0x1 `g1@0x556.0:vextbcst.16` (-38) |
| `g1@0x5e2.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x5d0.2:vmac.f` (-4) | `x3` `vec3.lo+vec3.hi` <- bf16_coeff `g1@0x588.0:vconv.bf16.fp32` (-27) | `x10` mixed:<br>`vec10.lo` <- register_move `g1@0x5de.0:vmov` (-2)<br>`vec10.hi` <- activation_lane #0x2 `g1@0x5bc.0:vextbcst.16` (-11) |
| `g1@0x5f4.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x5e2.1:vmac.f` (-5) | `x2` mixed:<br>`vec2.lo` <- register_move `g1@0x58e.0:vmov` (-30)<br>`vec2.hi` <- activation_lane #0x1a carry `g0@0x4b2.1:vextbcst.16` (-92) | `x7` `vec7.lo+vec7.hi` <- activation_lane #0x4 `g1@0x5ee.1:vextbcst.16` (-2) |
| `g1@0x608.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x5f4.1:vmac.f` (-5) | `x4` `vec4.lo+vec4.hi` <- activation_lane #0x5 `g1@0x600.1:vextbcst.16` (-2) | `x7` `vec7.lo+vec7.hi` <- activation_lane #0x4 `g1@0x5ee.1:vextbcst.16` (-7) |
| `g1@0x62a.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x608.1:vmac.f` (-9) | `x10` mixed:<br>`vec10.lo` <- register_move `g1@0x5de.0:vmov` (-21)<br>`vec10.hi` <- activation_lane #0x2 `g1@0x5bc.0:vextbcst.16` (-30) | `x4` `vec4.lo+vec4.hi` <- activation_lane #0x5 `g1@0x600.1:vextbcst.16` (-11) |
| `g1@0x640.1:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x62a.1:vmac.f` (-6) | `x8` `vec8.lo+vec8.hi` <- bf16_coeff `g1@0x636.0:vconv.bf16.fp32` (-4) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x636.1:vmov` (-3)<br>`vec3.hi` <- activation_lane #0x6 `g1@0x61e.0:vextbcst.16` (-10) |
| `g1@0x654.0:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x640.1:vmac.f` (-4) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x650.0:vmov` (-1)<br>`vec3.hi` <- activation_lane #0x6 `g1@0x61e.0:vextbcst.16` (-14) | `x0` `vec0.lo+vec0.hi` <- activation_lane #0x7 `g1@0x62a.0:vextbcst.16` (-11) |
| `g1@0x65e.1:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x654.0:vmac.f` (-4) | `x5` mixed:<br>`vec5.lo` <- register_move `g1@0x65a.0:vmov` (-2)<br>`vec5.hi` <- bf16_coeff `g1@0x600.0:vconv.bf16.fp32` (-26) | `x9` mixed:<br>`vec9.lo` <- register_move `g1@0x65e.0:vmov` (-1)<br>`vec9.hi` <- activation_lane #0x8 `g1@0x5da.0:vextbcst.16` (-36) |
| `g1@0x66e.2:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x65e.1:vmac.f` (-5) | `x5` mixed:<br>`vec5.lo` <- register_move `g1@0x65a.0:vmov` (-7)<br>`vec5.hi` <- bf16_coeff `g1@0x600.0:vconv.bf16.fp32` (-31) | `x4` `vec4.lo+vec4.hi` <- activation_lane #0x9 `g1@0x632.0:vextbcst.16` (-18) |
| `g1@0x686.2:vmac.f` | `dm4` `acc4.bmll+acc4.bmlh+acc4.bmhl+acc4.bmhh` <- accumulator_carry `g1@0x66e.2:vmac.f` (-7) | `x2` `vec2.lo+vec2.hi` <- bf16_coeff `g1@0x614.0:vconv.bf16.fp32` (-33) | `x7` `vec7.lo+vec7.hi` <- activation_lane #0xa `g1@0x640.0:vextbcst.16` (-21) |
| `g1@0x6a2.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x686.2:vmac.f` (-7) | `x3` `vec3.lo+vec3.hi` <- activation_lane #0xf `g1@0x6a2.0:vextbcst.16` (-1) | `x10` `vec10.lo+vec10.hi` <- activation_lane #0xe `g1@0x69a.1:vextbcst.16` (-2) |
| `g1@0x6b2.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x6a2.1:vmac.f` (-4) | `x8` mixed:<br>`vec8.lo` <- register_move `g1@0x6ae.0:vmov` (-2)<br>`vec8.hi` <- bf16_coeff `g1@0x636.0:vconv.bf16.fp32` (-35) | `x4` `vec4.lo+vec4.hi` <- activation_lane #0x15 `g1@0x6b2.0:vextbcst.16` (-1) |
| `g1@0x6c6.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x6b2.1:vmac.f` (-5) | `x9` `vec9.lo+vec9.hi` <- activation_lane #0x11 `g1@0x6c6.0:vextbcst.16` (-1) | `x5` mixed:<br>`vec5.lo` <- register_move `g1@0x6c2.0:vmov` (-2)<br>`vec5.hi` <- activation_lane #0xd `g1@0x690.1:vextbcst.16` (-14) |
| `g1@0x6d6.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x6c6.1:vmac.f` (-4) | `x1` `vec1.lo+vec1.hi` <- activation_lane #0x12 `g1@0x6d2.0:vextbcst.16` (-2) | `x10` `vec10.lo+vec10.hi` <- activation_lane #0xe `g1@0x69a.1:vextbcst.16` (-15) |
| `g1@0x6e6.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x6d6.1:vmac.f` (-4) | `x8` `vec8.lo+vec8.hi` <- activation_lane #0x13 `g1@0x6e6.0:vextbcst.16` (-1) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x6e2.0:vmov` (-2)<br>`vec3.hi` <- activation_lane #0xf `g1@0x6a2.0:vextbcst.16` (-18) |
| `g1@0x6f6.2:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x6e6.1:vmac.f` (-5) | `x2` `vec2.lo+vec2.hi` <- bf16_coeff `g1@0x6f6.0:vconv.bf16.fp32` (-2) | `x7` `vec7.lo+vec7.hi` <- activation_lane #0x14 `g1@0x6f2.0:vextbcst.16` (-3) |
| `g1@0x70c.2:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x6f6.2:vmac.f` (-6) | `x5` `vec5.lo+vec5.hi` <- bf16_coeff `g1@0x70c.0:vconv.bf16.fp32` (-2) | `x9` `vec9.lo+vec9.hi` <- activation_lane #0x11 `g1@0x6c6.0:vextbcst.16` (-20) |
| `g1@0x722.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x70c.2:vmac.f` (-5) | `x0` `vec0.lo+vec0.hi` <- activation_lane #0x19 `g1@0x71e.0:vextbcst.16` (-2) | `x1` `vec1.lo+vec1.hi` <- activation_lane #0x12 `g1@0x6d2.0:vextbcst.16` (-22) |
| `g1@0x730.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x722.1:vmac.f` (-4) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x6e2.0:vmov` (-22)<br>`vec3.hi` <- activation_lane #0xf `g1@0x6a2.0:vextbcst.16` (-38) | `x8` mixed:<br>`vec8.lo` <- register_move `g1@0x72c.0:vmov` (-2)<br>`vec8.hi` <- activation_lane #0x13 `g1@0x6e6.0:vextbcst.16` (-21) |
| `g1@0x744.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x730.1:vmac.f` (-5) | `x2` mixed:<br>`vec2.lo` <- register_move `g1@0x73c.1:vmov` (-2)<br>`vec2.hi` <- bf16_coeff `g1@0x6f6.0:vconv.bf16.fp32` (-22) | `x7` `vec7.lo+vec7.hi` <- activation_lane #0x14 `g1@0x6f2.0:vextbcst.16` (-23) |
| `g1@0x758.1:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x744.1:vmac.f` (-5) | `x2` mixed:<br>`vec2.lo` <- register_move `g1@0x73c.1:vmov` (-7)<br>`vec2.hi` <- bf16_coeff `g1@0x6f6.0:vconv.bf16.fp32` (-27) | `x4` `vec4.lo+vec4.hi` <- activation_lane #0x15 `g1@0x6b2.0:vextbcst.16` (-44) |
| `g1@0x76c.2:vmac.f` | `dm3` `acc3.bmll+acc3.bmlh+acc3.bmhl+acc3.bmhh` <- accumulator_carry `g1@0x758.1:vmac.f` (-6) | `x5` `vec5.lo+vec5.hi` <- bf16_coeff `g1@0x764.0:vconv.bf16.fp32` (-4) | `x6` `vec6.lo+vec6.hi` <- activation_lane #0x1b `g1@0x76c.1:vextbcst.16` (-1) |
| `g1@0x77e.1:vmac.f` | `dm2` `acc2.bmll+acc2.bmlh+acc2.bmhl+acc2.bmhh` <- accumulator_carry `g1@0x76c.2:vmac.f` (-4) | `x8` mixed:<br>`vec8.lo` <- register_move `g1@0x77a.0:vmov` (-2)<br>`vec8.hi` <- activation_lane #0x13 `g1@0x6e6.0:vextbcst.16` (-41) | `x9` mixed:<br>`vec9.lo` <- register_move `g1@0x77e.0:vmov` (-1)<br>`vec9.hi` <- activation_lane #0x17 `g1@0x716.0:vextbcst.16` (-28) |
| `g1@0x790.2:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x77e.1:vmac.f` (-6) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x78a.1:vmov` (-3)<br>`vec3.hi` <- bf16_coeff `g1@0x73c.0:vconv.bf16.fp32` (-24) | `x10` `vec10.lo+vec10.hi` <- activation_lane #0x1d `g1@0x790.1:vextbcst.16` (-1) |
| `g1@0x7a0.3:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x790.2:vmac.f` (-6) | `x3` mixed:<br>`vec3.lo` <- register_move `g1@0x78a.1:vmov` (-9)<br>`vec3.hi` <- bf16_coeff `g1@0x73c.0:vconv.bf16.fp32` (-30) | `x0` `vec0.lo+vec0.hi` <- activation_lane #0x1e `g1@0x79c.0:vextbcst.16` (-4) |
| `g1@0x7b4.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x7a0.3:vmac.f` (-4) | `x1` `vec1.lo+vec1.hi` <- scalar_broadcast `g1@0x7b0.0:vbcst.16` (-2) | `x2` mixed:<br>`vec2.lo` <- vector_load `g1@0x790.0:vldb` (-12)<br>`vec2.hi` <- activation_lane #0x1a `g1@0x764.1:vextbcst.16` (-23) |
| `g1@0x7c2.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x7b4.1:vmac.f` (-4) | `x8` `vec8.lo+vec8.hi` <- vector_load `g1@0x7a0.0:vlda` (-11) | `x6` mixed:<br>`vec6.lo` <- vector_load `g1@0x7a0.1:vldb` (-10)<br>`vec6.hi` <- activation_lane #0x1b `g1@0x76c.1:vextbcst.16` (-25) |
| `g1@0x7d2.1:vmac.f` | `dm1` `acc1.bmll+acc1.bmlh+acc1.bmhl+acc1.bmhh` <- accumulator_carry `g1@0x7c2.1:vmac.f` (-4) | `x5` `vec5.lo+vec5.hi` <- unpacked_q4 `g1@0x7bc.0:vunpack` (-7) | `x7` `vec7.lo+vec7.hi` <- vector_load `g1@0x7b4.0:vldb` (-9) |

## Conclusion

- Whole-vector producer tracking hides real composition: many `xN` operands are assembled from separately scheduled halves.
- Group1 starts with live half-register cells from group0, including packed Q4 halves, coefficient halves, loaded vectors, and accumulator quadrants.
- The next Q4NX generator should carry explicit cell state across `fill -> steady` and `steady -> steady` boundaries before trying to emit a new exact or MyLM-like body.
